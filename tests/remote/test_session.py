"""Watcher + queue tests with a faked ``remote_gen``.

We replace the module-level helpers (``remote_server_status``,
``remote_qwen_txt2img``, ``remote_qwen_edit``) with controllable
test doubles so we never need a live pod for these scenarios.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Make ``server`` importable as a package — same pattern as the other tests.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

# The remote-session module is shipped inside the existing ``server`` package.
# Import via that path so the canvas + PIL dependencies are wired up.
from server import remote_session
from server.remote_session import (
    AutoAbortWatcher, GenerationQueue, JobStatus, PodCompromisedError,
    Session, WatcherConfig, get_session, reset_session_for_tests,
)


@pytest.fixture(autouse=True)
def _fresh_session():
    """Each test gets a fresh module singleton."""
    reset_session_for_tests()
    yield
    reset_session_for_tests()


# ─── watcher ────────────────────────────────────────────────────────


def test_watcher_trips_on_compromised_true():
    """remote_server_status returns compromised=True → session is marked."""
    sess = Session()
    sess.queue = GenerationQueue(sess)
    sess.queue.start()
    sess.watcher = AutoAbortWatcher(sess, WatcherConfig(poll_interval_s=0.05))

    statuses = iter([
        {"tripwire_armed": True, "compromised": False, "pipeline_ready": True},
        {"tripwire_armed": True, "compromised": True,  "pipeline_ready": True,
         "compromise_trigger": "stranger_child"},
    ])
    with patch("server.remote_gen.remote_server_status",
               side_effect=lambda: next(statuses)):
        sess.watcher.start()
        # wait for trip
        deadline = time.monotonic() + 3.0
        while not sess.compromised and time.monotonic() < deadline:
            time.sleep(0.05)
        sess.watcher.stop()
    assert sess.compromised
    assert sess.compromise_trigger == "stranger_child"


def test_watcher_trips_on_consecutive_failures():
    """Five consecutive httpx exceptions trip the watcher as 'unreachable'."""
    sess = Session()
    sess.queue = GenerationQueue(sess)
    sess.queue.start()
    sess.watcher = AutoAbortWatcher(sess, WatcherConfig(poll_interval_s=0.02))
    sess.watcher._max_consecutive_failures = 3
    sess.watcher._ever_reached = True  # the unreachable trip is gated on prior contact

    with patch("server.remote_gen.remote_server_status",
               side_effect=RuntimeError("connection refused")):
        sess.watcher.start()
        deadline = time.monotonic() + 3.0
        while not sess.compromised and time.monotonic() < deadline:
            time.sleep(0.05)
        sess.watcher.stop()
    assert sess.compromised
    assert sess.compromise_trigger == "unreachable"


def test_watcher_cancels_pending_jobs_on_trip():
    """Pending jobs go to CANCELLED when the watcher trips."""
    sess = Session()
    sess.queue = GenerationQueue(sess)
    sess.queue.start()
    sess.watcher = AutoAbortWatcher(sess, WatcherConfig(poll_interval_s=0.05))

    # Stuff a generation that takes long enough to be QUEUED when trip fires
    def slow_txt2img(**kw):
        time.sleep(10)
        from PIL import Image
        return Image.new("RGB", (4, 4))

    statuses = iter([
        {"compromised": False, "pipeline_ready": True},
        {"compromised": False, "pipeline_ready": True},
        {"compromised": True, "compromise_trigger": "tracer", "pipeline_ready": True},
    ])

    with patch("server.remote_gen.remote_server_status",
               side_effect=lambda: next(statuses)), \
         patch("server.remote_gen.remote_qwen_txt2img", side_effect=slow_txt2img):
        # submit 5 jobs; the worker takes the first, the rest sit QUEUED
        ids = [sess.queue.submit("txt2img", {"prompt": str(i)}, canvas_id=None)
               for i in range(5)]
        sess.watcher.start()
        deadline = time.monotonic() + 3.0
        while not sess.compromised and time.monotonic() < deadline:
            time.sleep(0.05)
        sess.watcher.stop()

    # After trip: every job should be CANCELLED (running too)
    s = sess.queue.status()
    cancelled = s["counts"][JobStatus.CANCELLED.value]
    assert cancelled == len(ids), f"expected all 5 cancelled, got counts={s['counts']}"
    # subsequent submit must reject
    with pytest.raises(PodCompromisedError):
        sess.queue.submit("txt2img", {"prompt": "x"}, canvas_id=None)

    sess.queue.stop()


# ─── queue ──────────────────────────────────────────────────────────


def test_queue_submits_run_in_fifo_order():
    sess = Session()
    sess.queue = GenerationQueue(sess)
    seen: list[str] = []

    def fake_gen(**kw):
        seen.append(kw["prompt"])
        from PIL import Image
        return Image.new("RGB", (4, 4))

    with patch("server.remote_gen.remote_qwen_txt2img", side_effect=fake_gen):
        sess.queue.start()
        ids = [sess.queue.submit("txt2img", {"prompt": p}, canvas_id=None)
               for p in ("a", "b", "c", "d")]
        # wait for done
        deadline = time.monotonic() + 5.0
        while True:
            s = sess.queue.status()
            if s["counts"][JobStatus.DONE.value] == 4: break
            if time.monotonic() > deadline: break
            time.sleep(0.05)
        sess.queue.stop()
    assert seen == ["a", "b", "c", "d"], f"out of order: {seen}"
    # each job ended in DONE
    for jid in ids:
        img = sess.queue.get_result(jid)
        assert img is not None
        # second get with default consume=True wipes it
        with pytest.raises(RuntimeError):
            sess.queue.get_result(jid)


def test_queue_cancel_pending():
    sess = Session()
    sess.queue = GenerationQueue(sess)

    def slow(**kw):
        time.sleep(2.0)
        from PIL import Image
        return Image.new("RGB", (4, 4))

    with patch("server.remote_gen.remote_qwen_txt2img", side_effect=slow):
        sess.queue.start()
        a = sess.queue.submit("txt2img", {"prompt": "first"}, canvas_id=None)
        b = sess.queue.submit("txt2img", {"prompt": "second"}, canvas_id=None)
        # b should be QUEUED still — worker is busy on a
        time.sleep(0.05)
        assert sess.queue.cancel(b) is True
        # a is RUNNING — cancel returns False (we'll mark cancelled but
        # cannot stop the in-flight call)
        cancelled_a_immediately = sess.queue.cancel(a)
        assert cancelled_a_immediately is False
        sess.queue.stop()
    s = sess.queue.status()
    states = {j["job_id"]: j["status"] for j in s["jobs"]}
    assert states[b] == JobStatus.CANCELLED.value
    # a may be CANCELLED (worker dropped its result) or DONE (finished
    # before our cancel arrived). Either is acceptable.
    assert states[a] in (JobStatus.CANCELLED.value, JobStatus.DONE.value)


def test_queue_clear_drops_results():
    sess = Session()
    sess.queue = GenerationQueue(sess)

    def fake(**kw):
        from PIL import Image
        return Image.new("RGB", (4, 4))

    with patch("server.remote_gen.remote_qwen_txt2img", side_effect=fake):
        sess.queue.start()
        ids = [sess.queue.submit("txt2img", {"prompt": str(i)}, canvas_id=None)
               for i in range(3)]
        deadline = time.monotonic() + 3.0
        while sess.queue.status()["counts"][JobStatus.DONE.value] < 3:
            if time.monotonic() > deadline: break
            time.sleep(0.05)
        sess.queue.stop()
    cleared = sess.queue.clear()
    # nothing was queued at this point (all DONE) — but the held images
    # should be wiped.
    s = sess.queue.status()
    for j in s["jobs"]:
        # result_image is internal — verified indirectly: get_result should
        # now raise because the image was wiped
        if j["status"] == JobStatus.DONE.value:
            with pytest.raises((RuntimeError, KeyError)):
                sess.queue.get_result(j["job_id"])


def test_queue_submit_rejected_when_compromised():
    sess = Session()
    sess.queue = GenerationQueue(sess)
    sess.mark_compromised("test_trigger", "test_detail")
    with pytest.raises(PodCompromisedError):
        sess.queue.submit("txt2img", {"prompt": "x"}, canvas_id=None)


def test_queue_running_job_marked_cancelled_drops_result():
    """A long-running call ongoing when compromise fires has its result dropped."""
    sess = Session()
    sess.queue = GenerationQueue(sess)
    sess.queue.start()

    def slow(**kw):
        time.sleep(0.3)
        from PIL import Image
        return Image.new("RGB", (4, 4))

    with patch("server.remote_gen.remote_qwen_txt2img", side_effect=slow):
        jid = sess.queue.submit("txt2img", {"prompt": "x"}, canvas_id=None)
        time.sleep(0.05)
        # marking compromise pre-emptively cancels the in-flight job
        sess.mark_compromised("test", "test")
        time.sleep(0.5)
        sess.queue.stop()
    s = sess.queue.status()
    states = {j["job_id"]: j["status"] for j in s["jobs"]}
    # status is CANCELLED whether the worker hit the slow() return or not
    assert states[jid] in (JobStatus.CANCELLED.value,)


def test_get_session_starts_singletons():
    s1 = get_session(start_watcher=False, start_queue=False)
    s2 = get_session(start_watcher=False, start_queue=False)
    assert s1 is s2
    assert s1.watcher is not None
    assert s1.queue is not None


def test_check_not_compromised_raises():
    sess = get_session(start_watcher=False, start_queue=False)
    assert not sess.compromised
    # no-op while clean
    remote_session.check_not_compromised()
    sess.mark_compromised("test", "test")
    with pytest.raises(PodCompromisedError):
        remote_session.check_not_compromised()
