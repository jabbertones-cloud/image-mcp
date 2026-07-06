"""Client-side session: AutoAbortWatcher + GenerationQueue.

Two long-lived helpers shared across every MCP tool call:

* ``AutoAbortWatcher`` — daemon thread polling ``/v1/server_status`` every
  ``poll_interval_s``. If the remote pod reports ``compromised=True``,
  becomes unreachable, or signs its responses with a different identity
  than the one we pinned, the watcher:
    1. Marks our own ``Session.compromised`` flag.
    2. Cancels every pending queued job.
    3. Logs the trigger.
  All subsequent ``remote_*`` calls raise :class:`PodCompromisedError`.

* ``GenerationQueue`` — FIFO queue with a single worker thread that
  serializes remote-pod calls. Each job carries the destination canvas
  (auto-created if None). Status flow:
    ``queued`` → ``running`` → ``done`` | ``failed`` | ``cancelled``
  Results are held in process RAM until ``get_result(job_id)`` consumes
  them.

Both are module-level singletons constructed lazily on first use, then
stopped via an ``atexit`` hook.
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Optional

log = logging.getLogger("remote_session")


# ─── errors ──────────────────────────────────────────────────────────


class PodCompromisedError(RuntimeError):
    """Raised by every submission path once the watcher has detected an
    intrusion on the paired pod. The pod's identity material is no longer
    trustworthy; discard it and provision a fresh one."""
    def __init__(self, trigger: str, detail: str = ""):
        super().__init__(f"pod compromised: {trigger}" + (f" — {detail}" if detail else ""))
        self.trigger = trigger
        self.detail = detail


# ─── session state ──────────────────────────────────────────────────


class JobStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id:    str
    kind:      str                # "txt2img" | "edit"
    params:    dict[str, Any]
    canvas_id: Optional[str]      # if set, place result on this canvas
    submitted_at: float
    status:    JobStatus = JobStatus.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    error:      str | None = None
    # Result is a PIL.Image.Image when status == DONE. Stored here until
    # the caller retrieves it via get_result(). Caller is responsible for
    # consuming results in a timely fashion — we cap with MAX_HELD_RESULTS.
    result_image: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id":       self.job_id,
            "kind":         self.kind,
            "status":       self.status.value,
            "canvas_id":    self.canvas_id,
            "submitted_at": self.submitted_at,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "error":        self.error,
            # never expose the result PIL through to_dict — it's a heavy
            # object, callers fetch via get_result() if they need it.
            "params":       {k: v for k, v in self.params.items()
                             if k not in ("control_image",)},
        }


MAX_HELD_RESULTS = 32      # LRU-evict completed jobs older than this


# ─── auto-abort watcher ─────────────────────────────────────────────


@dataclass
class WatcherConfig:
    poll_interval_s:   float = 10.0
    request_timeout_s: float = 8.0


class AutoAbortWatcher:
    """Polls the remote pod's ``/v1/server_status`` on a daemon thread.

    Trip conditions:
      * server returns ``compromised=True``
      * server returns 503 with "compromised" detail
      * any unhandled HTTPError on N consecutive polls
      * fingerprint of returned pubkey no longer matches the pinned one
    """

    def __init__(self, session: "Session", cfg: WatcherConfig | None = None):
        self._session = session
        self._cfg = cfg or WatcherConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._ever_reached = False  # gate unreachable-trip on prior contact
        self._last_status: dict | None = None
        self._last_polled_at: float = 0.0
        self._lock = threading.Lock()

    # ─── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="remote-watcher",
        )
        self._thread.start()
        log.info(f"AutoAbortWatcher armed (poll={self._cfg.poll_interval_s}s)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    @property
    def last_status(self) -> dict | None:
        with self._lock:
            return self._last_status

    @property
    def last_polled_at(self) -> float:
        return self._last_polled_at

    # ─── poll loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                log.exception("watcher poll raised")
            self._stop.wait(self._cfg.poll_interval_s)

    def _poll_once(self) -> None:
        # Lazy import to avoid circular: remote_gen → remote_session.
        from . import remote_gen
        try:
            status = remote_gen.remote_server_status()
        except Exception as e:
            self._note_failure(f"status fetch raised: {e!r}")
            return
        self._last_polled_at = time.time()

        # remote_server_status returns a structured dict (not an exception) for
        # HTTP >=400, so an erroring/unreachable pod must be counted here — else
        # the failure counter resets every poll and the watcher never trips.
        if status.get("unreachable_or_compromised") is True:
            detail = str(status.get("detail", ""))
            if "compromised" in detail.lower():
                self._trip("server returned compromised", detail)  # affirmative signal
            else:
                self._note_failure(f"server error {status.get('status')}: {detail}")
            return

        # healthy response
        self._ever_reached = True
        self._consecutive_failures = 0
        with self._lock:
            self._last_status = status

        if status.get("compromised") is True:
            self._trip(
                trigger=status.get("compromise_trigger") or "server reports compromised",
                detail="server-side tripwire fired",
            )

    def _note_failure(self, msg: str) -> None:
        self._consecutive_failures += 1
        log.warning(
            f"watcher: {msg} ({self._consecutive_failures}/{self._max_consecutive_failures})"
        )
        # Only latch "unreachable" after the pod has been reachable at least
        # once, so a not-yet-booted pod or missing config doesn't permanently
        # mark the session compromised before it ever comes up.
        if self._ever_reached and self._consecutive_failures >= self._max_consecutive_failures:
            self._trip("unreachable", f"{self._consecutive_failures} consecutive failures: {msg}")

    def _trip(self, trigger: str, detail: str) -> None:
        self._session.mark_compromised(trigger=trigger, detail=detail)
        # stop polling — we've done our job
        self._stop.set()


# ─── queue ──────────────────────────────────────────────────────────


class GenerationQueue:
    """FIFO + single-worker. Serializes remote_pod calls because the pod's
    pipeline is single-tenant anyway. Multiple producers (MCP tools) are
    safe via the inbound Queue."""

    def __init__(self, session: "Session"):
        self._session = session
        self._inbox: Queue[Job] = Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="remote-queue-worker",
        )
        self._worker.start()
        log.info("GenerationQueue worker started")

    def stop(self) -> None:
        self._stop.set()
        # poke the queue with a sentinel so the worker wakes up
        try: self._inbox.put_nowait(None)        # type: ignore[arg-type]
        except Exception: pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)
        self._worker = None

    # ─── public api ────────────────────────────────────────────────

    def submit(self, kind: str, params: dict[str, Any],
               canvas_id: str | None) -> str:
        if self._session.compromised:
            raise PodCompromisedError(
                trigger=self._session.compromise_trigger or "unknown",
                detail=self._session.compromise_detail or "",
            )
        if kind not in ("txt2img", "edit"):
            raise ValueError(f"unknown kind {kind!r}")
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            kind=kind, params=dict(params), canvas_id=canvas_id,
            submitted_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._gc_completed()
        self._inbox.put(job)
        return job.job_id

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = {s.value: 0 for s in JobStatus}
            for j in self._jobs.values():
                counts[j.status.value] += 1
            return {
                "queue_size":  self._inbox.qsize(),
                "counts":      counts,
                "running_job": next(
                    (j.job_id for j in self._jobs.values()
                     if j.status == JobStatus.RUNNING), None,
                ),
                "compromised": self._session.compromised,
                "jobs":        [j.to_dict() for j in self._jobs.values()],
            }

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Running jobs cannot be cancelled mid-flight
        (the remote pod has accepted the request); they'll be marked
        ``cancelled`` once the in-flight call returns, and the result is
        discarded. Returns True if the job was queued and is now cancelled."""
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return False
            if j.status == JobStatus.QUEUED:
                j.status = JobStatus.CANCELLED
                j.finished_at = time.time()
                return True
            if j.status == JobStatus.RUNNING:
                # mark; worker will drop the result on completion
                j.status = JobStatus.CANCELLED
                return False
            return False

    def get_result(self, job_id: str, *, consume: bool = True) -> Any:
        """Returns the PIL.Image for ``DONE`` jobs. ``consume=True`` drops
        the result from the queue's RAM after returning it; a second
        call (or one after ``clear()``) raises rather than returning None."""
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                raise KeyError(f"no job {job_id!r}")
            if j.status != JobStatus.DONE:
                raise RuntimeError(f"job {job_id} is {j.status.value}, not done")
            if j.result_image is None:
                raise RuntimeError(f"job {job_id}: result already consumed or cleared")
            img = j.result_image
            if consume:
                j.result_image = None
            return img

    def clear(self) -> int:
        """Cancel every QUEUED job, drop every completed result. Returns
        the number of jobs cancelled. Does NOT interrupt the running one."""
        cancelled = 0
        with self._lock:
            for j in self._jobs.values():
                if j.status == JobStatus.QUEUED:
                    j.status = JobStatus.CANCELLED
                    j.finished_at = time.time()
                    cancelled += 1
                if j.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
                    j.result_image = None
        return cancelled

    def cancel_all_pending_due_to_compromise(self) -> int:
        """Called by the watcher when it trips. Returns count cancelled."""
        cancelled = 0
        with self._lock:
            for j in self._jobs.values():
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    j.status = JobStatus.CANCELLED
                    j.error = "pod compromised"
                    j.finished_at = time.time()
                    j.result_image = None
                    cancelled += 1
        return cancelled

    # ─── worker ───────────────────────────────────────────────────

    def _run(self) -> None:
        from . import remote_gen
        while not self._stop.is_set():
            try:
                job = self._inbox.get(timeout=1.0)
            except Empty:
                continue
            if job is None:        # stop sentinel
                break
            if self._session.compromised:
                with self._lock:
                    job.status = JobStatus.CANCELLED
                    job.error = "pod compromised"
                    job.finished_at = time.time()
                continue
            # honor pre-emptive cancellation
            with self._lock:
                if job.status == JobStatus.CANCELLED:
                    continue
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
            try:
                img = self._run_job(remote_gen, job)
                with self._lock:
                    if job.status == JobStatus.CANCELLED:
                        # cancelled while running — drop the result
                        log.info(f"job {job.job_id}: cancelled mid-flight, dropping result")
                    else:
                        job.status = JobStatus.DONE
                        job.result_image = img
                        job.finished_at = time.time()
                        # auto-publish to canvas if requested
                        if job.canvas_id is not None:
                            self._publish(job, img)
            except PodCompromisedError as e:
                # the call itself raised compromise — mark + halt
                self._session.mark_compromised(e.trigger, e.detail)
                with self._lock:
                    job.status = JobStatus.CANCELLED
                    job.error = str(e)
                    job.finished_at = time.time()
                self.cancel_all_pending_due_to_compromise()
                self._stop.set()
            except Exception as e:
                log.exception(f"job {job.job_id} failed")
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = f"{type(e).__name__}: {e}"
                    job.finished_at = time.time()
        log.info("queue worker exiting")

    def _run_job(self, remote_gen, job: Job):
        if job.kind == "txt2img":
            return remote_gen.remote_qwen_txt2img(**job.params)
        if job.kind == "edit":
            return remote_gen.remote_qwen_edit(**job.params)
        raise ValueError(f"unknown kind {job.kind!r}")

    def _publish(self, job: Job, img) -> None:
        try:
            from .canvas import store
            rgba = img.convert("RGBA")
            prompt = str(job.params.get("prompt", ""))[:40]
            name = f"remote {job.kind}: {prompt}"
            # Add as a new, undoable layer — never overwrite the source layer
            # (which for an edit job is the user's control image). Fall back to
            # a fresh canvas if the target is gone or was never given.
            if job.canvas_id is not None and store.has(job.canvas_id):
                store.snapshot(job.canvas_id)
                store.add_layer(job.canvas_id, name=name)
                store.replace_active_image(job.canvas_id, rgba)
            else:
                store.put_image(rgba, canvas_id=job.canvas_id, layer_name=name)
        except Exception:
            log.exception(f"job {job.job_id}: canvas publish failed")

    def _gc_completed(self) -> None:
        """Drop the oldest DONE/FAILED/CANCELLED jobs if we have too many."""
        completed = sorted(
            (j for j in self._jobs.values()
             if j.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)),
            key=lambda j: j.finished_at or 0,
        )
        excess = len(completed) - MAX_HELD_RESULTS
        for j in completed[:max(0, excess)]:
            self._jobs.pop(j.job_id, None)


# ─── session singleton ─────────────────────────────────────────────


@dataclass
class Session:
    """Process-wide state. Lazy-init via ``get_session()``."""
    watcher:  AutoAbortWatcher | None = None
    queue:    GenerationQueue | None = None
    compromised: bool = False
    compromise_trigger: str | None = None
    compromise_detail:  str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_compromised(self, trigger: str, detail: str = "") -> None:
        with self._lock:
            if self.compromised:
                return
            self.compromised = True
            self.compromise_trigger = trigger
            self.compromise_detail = detail
        log.error(f"SESSION COMPROMISED: {trigger} — {detail}")
        if self.queue is not None:
            try: self.queue.cancel_all_pending_due_to_compromise()
            except Exception: pass


_session_lock = threading.Lock()
_session: Session | None = None
_atexit_registered = False


def get_session(start_watcher: bool = True, start_queue: bool = True) -> Session:
    global _session, _atexit_registered
    with _session_lock:
        if _session is None:
            _session = Session()
            _session.queue = GenerationQueue(_session)
            _session.watcher = AutoAbortWatcher(_session)
            if not _atexit_registered:
                atexit.register(_atexit_cleanup)
                _atexit_registered = True
        if start_watcher and _session.watcher and _session.watcher._thread is None:
            _session.watcher.start()
        if start_queue and _session.queue and _session.queue._worker is None:
            _session.queue.start()
        return _session


def reset_session_for_tests() -> None:
    """Tests call this between scenarios to get a fresh Session."""
    global _session
    with _session_lock:
        if _session is not None:
            if _session.watcher: _session.watcher.stop()
            if _session.queue:   _session.queue.stop()
        _session = None


def _atexit_cleanup() -> None:
    s = _session
    if not s: return
    try:
        if s.watcher: s.watcher.stop()
        if s.queue:   s.queue.stop()
    except Exception: pass


# ─── public guard used by remote_gen submission sites ──────────────


def check_not_compromised() -> None:
    """Called by every remote_gen entry point before sending. Idempotent
    and cheap; reads a module-level boolean."""
    if _session is not None and _session.compromised:
        raise PodCompromisedError(
            trigger=_session.compromise_trigger or "unknown",
            detail=_session.compromise_detail or "",
        )
