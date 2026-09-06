"""The single-thread import gate that fixed the 2026-09-07 event-loop deadlock."""
import threading
import time

import pytest

from server import prewarm


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(prewarm, "_thread", None)
    monkeypatch.setattr(prewarm, "_started_at", None)
    monkeypatch.setattr(prewarm, "_finished_in", None)
    monkeypatch.setattr(prewarm, "_errors", [])
    monkeypatch.setattr(prewarm, "_done", threading.Event())
    yield


def test_wait_is_a_noop_when_never_started():
    t0 = time.perf_counter()
    prewarm.wait(timeout=5)
    assert time.perf_counter() - t0 < 0.5
    assert prewarm.status() == {"started": False, "done": False, "seconds": None, "skipped": []}


def test_sync_start_runs_on_the_calling_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(prewarm, "_import_all", lambda: seen.append(threading.current_thread().name))
    prewarm.start()  # default: synchronous, main thread (Windows DLL-loader rule)
    prewarm.start()  # idempotent
    assert seen == [threading.current_thread().name] and prewarm.status()["done"]
    prewarm.wait(timeout=1)


def test_wait_blocks_until_the_import_thread_finishes(monkeypatch):
    gate = threading.Event()

    def fake_import_all():
        gate.wait(5)

    monkeypatch.setattr(prewarm, "_import_all", fake_import_all)
    prewarm.start(background=True)
    prewarm.start(background=True)  # idempotent
    assert prewarm.status()["started"] and not prewarm.status()["done"]
    with pytest.raises(RuntimeError, match="still loading"):
        prewarm.wait(timeout=0.2)
    gate.set()
    prewarm.wait(timeout=5)
    assert prewarm.status()["done"] and prewarm.status()["seconds"] is not None


def test_missing_optional_imports_are_recorded_not_fatal(monkeypatch):
    monkeypatch.setattr(prewarm, "_import_all", lambda: prewarm._errors.append("x: ImportError: no"))
    prewarm.start(background=True)
    prewarm.wait(timeout=5)
    assert prewarm.status()["skipped"] == ["x: ImportError: no"]


@pytest.mark.parametrize("module", ["sd", "qwen", "birefnet", "clipseg", "sam", "sam1", "yolo_seg", "gguf_io"])
def test_every_ai_module_waits_for_the_prewarm(module, monkeypatch):
    import importlib

    mod = importlib.import_module(f"server.{module}")
    calls = []
    monkeypatch.setattr(prewarm, "wait", lambda timeout=None: calls.append(module))
    try:
        mod._check_available()
    except RuntimeError:
        pass  # optional extra missing on this box - the wait still had to come first
    assert calls == [module]


def test_sd_never_loads_the_safety_checker():
    from server import sd

    assert sd._NO_SAFETY == {"safety_checker": None, "requires_safety_checker": False}
    src = open(sd.__file__, encoding="utf-8").read()
    assert src.count("**_NO_SAFETY") >= 4, "every from_pretrained path must pass _NO_SAFETY"
    assert sd.DEFAULT_INPAINT_MODEL == "Lykon/dreamshaper-8-inpainting"
