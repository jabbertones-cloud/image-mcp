"""Intrusion tripwire — detect docker exec / nsenter / ptrace / gcore-suspend.

Runs as a background daemon thread started in the FastAPI lifespan. Polls
three signals every ``poll_interval_s``:

1. ``/proc/self/status`` line ``TracerPid:`` — non-zero means something has
   ptrace-attached to us (gdb, strace, py-spy, anyone with PTRACE_ATTACH).
2. Direct children of PID 1 — anything beyond our known-good subprocesses
   means someone ran ``docker exec``, ``nsenter``, or spawned an extra
   process via the container runtime.
3. Heartbeat staleness — the main asyncio loop writes a monotonic timestamp
   every N seconds. If the tripwire sees > ``stale_after_s`` of staleness
   it suspects the process was suspended (gcore freezes the target for
   the duration of the dump).

On any trip:
  * call every registered ``on_trip`` callback (e.g. wipe secrets)
  * set ``compromised`` flag → FastAPI route guards return 503
  * log the trigger; DO NOT include any client data in the log

Limits:
  * Detection is *probabilistic* for the suspend signal — a healthy host
    sometimes pauses processes for cgroup throttling. The default 5-second
    staleness window is tuned to avoid false positives on a 1Hz heartbeat.
  * Cannot detect ``docker cp`` (host-side FS read), ``docker logs``, or
    confidential-memory introspection on the host kernel.

This file is Linux-only by intent — non-Linux platforms get a no-op shim.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger(__name__)


@dataclass
class TripwireConfig:
    poll_interval_s:     float = 1.0
    stale_after_s:       float = 5.0
    # Children of PID 1 that are EXPECTED. Populate at startup with our own
    # PIDs (tailscaled, the main python, anything we know we spawned).
    expected_pid1_children: set[int] = field(default_factory=set)


@dataclass
class TripwireReport:
    triggered_at:    float
    trigger:         str         # "tracer" | "stranger_child" | "suspend"
    details:         dict


class Tripwire:
    def __init__(self, cfg: TripwireConfig | None = None):
        self._cfg = cfg or TripwireConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_trip: list[Callable[[TripwireReport], None]] = []
        self._compromised: bool = False
        self._report:      TripwireReport | None = None
        self._heartbeat = time.monotonic()
        self._lock = threading.Lock()
        # Snapshot the "before" set of PID-1 children so the daemon doesn't
        # treat siblings present at boot as intruders.
        self._cfg.expected_pid1_children |= self._snapshot_pid1_children()

    # ─── lifecycle ─────────────────────────────────────────────────

    def register_on_trip(self, cb: Callable[[TripwireReport], None]) -> None:
        with self._lock:
            self._on_trip.append(cb)

    def start(self) -> None:
        if sys.platform != "linux":
            log.info("tripwire: non-Linux platform, daemon disabled")
            return
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="qcl-tripwire",
        )
        self._thread.start()
        log.info(
            f"tripwire armed (poll={self._cfg.poll_interval_s}s, "
            f"stale_after={self._cfg.stale_after_s}s, "
            f"trusted_pid1_children={sorted(self._cfg.expected_pid1_children)})"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    # ─── state queries ─────────────────────────────────────────────

    @property
    def compromised(self) -> bool:
        return self._compromised

    @property
    def report(self) -> TripwireReport | None:
        return self._report

    def heartbeat(self) -> None:
        """Call from the main asyncio loop on a regular cadence; tripwire
        compares the freshness against ``stale_after_s``."""
        self._heartbeat = time.monotonic()

    # ─── poller ────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_once()
            except Exception:
                log.exception("tripwire poll raised")
            time.sleep(self._cfg.poll_interval_s)

    def _check_once(self) -> None:
        # 1) TracerPid in /proc/self/status
        tracer = self._tracer_pid()
        if tracer is not None and tracer != 0:
            return self._trip("tracer", {"tracer_pid": tracer})

        # 2) Strangers among PID 1's children
        children = self._snapshot_pid1_children()
        strangers = children - self._cfg.expected_pid1_children
        # filter out kernel threads / tini-like internals that some runtimes
        # spawn after boot. We allow processes we've adopted via ``trust_pid``.
        if strangers:
            return self._trip("stranger_child", {"strangers": sorted(strangers),
                                                 "expected":  sorted(self._cfg.expected_pid1_children)})

        # 3) Heartbeat staleness — main loop frozen?
        gap = time.monotonic() - self._heartbeat
        if gap > self._cfg.stale_after_s:
            return self._trip("suspend", {"heartbeat_gap_s": round(gap, 2)})

    def trust_pid(self, pid: int) -> None:
        """Whitelist a PID after-the-fact (e.g. a subprocess we just spawned)."""
        with self._lock:
            self._cfg.expected_pid1_children.add(int(pid))

    # ─── trip handling ─────────────────────────────────────────────

    def _trip(self, kind: str, details: dict) -> None:
        with self._lock:
            if self._compromised:
                return
            self._compromised = True
            self._report = TripwireReport(
                triggered_at=time.time(), trigger=kind, details=details,
            )
            cbs = list(self._on_trip)
        log.error(f"TRIPWIRE: {kind} {details}")
        for cb in cbs:
            try: cb(self._report)
            except Exception:
                log.exception("on_trip callback raised")

    # ─── /proc helpers ────────────────────────────────────────────

    @staticmethod
    def _tracer_pid() -> int | None:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("TracerPid:"):
                        return int(line.split()[1])
        except OSError:
            return None
        return None

    @staticmethod
    def _snapshot_pid1_children() -> set[int]:
        """Return the set of process IDs whose parent is PID 1."""
        out: set[int] = set()
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == 1:
                    continue
                try:
                    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                        # stat fields: pid (comm) state ppid ...
                        line = f.read()
                    # comm can contain spaces + parens; safest parse: take the
                    # text after the LAST ')'.
                    rest = line.rsplit(")", 1)[1].split()
                    # rest[0] = state; rest[1] = ppid
                    if int(rest[1]) == 1:
                        out.add(pid)
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            pass
        return out


# ─── singleton accessor (used by app.py) ────────────────────────────


_tripwire: Tripwire | None = None


def get_tripwire() -> Tripwire | None:
    return _tripwire


def install_tripwire(cfg: TripwireConfig | None = None) -> Tripwire:
    global _tripwire
    if _tripwire is not None:
        return _tripwire
    _tripwire = Tripwire(cfg)
    return _tripwire
