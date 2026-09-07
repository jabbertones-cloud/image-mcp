"""One-time, single-threaded import of the heavy AI stack.

Why this exists (root cause found 2026-09-07 with py-spy on the live server):
FastMCP runs sync tools on its event-loop thread. ``run_server.py`` used to
import the diffusers/transformers pipeline classes on a background thread so
MCP ``initialize`` could be answered immediately. When a tool call then did
its own lazy import (``sd._get_pipe`` -> ``AutoPipelineForText2Image`` ->
controlnet -> ``diffusers.loaders.peft`` -> ``peft`` -> ``transformers.models.auto``)
while the prewarm thread was still inside ``transformers.generation.utils``
(-> sklearn -> scipy), the two threads blocked on each other's module locks.
The event loop was stuck inside an import, so every later request — even
``sd_status`` — hung forever. That is the "SD engine is broken" symptom.

Second finding, same day: even with the main thread doing nothing but
waiting, a prewarm *thread* froze inside ``scipy.special`` (native extension
load under the Windows DLL loader lock; OpenBLAS thread init). scipy / sklearn
C extensions must be first-imported on the MAIN thread on Windows. So the
default is now a synchronous main-thread prewarm before FastMCP starts;
MCPManager's ``startup_timeout_s`` for this server is raised to cover it (the
child stays resident, so it is paid once). ``IMAGETOOLS_PREWARM_BACKGROUND=1``
restores the thread for non-Windows hosts.

Rule: heavy imports happen on exactly one thread. Every module that lazily
imports diffusers / transformers / peft / sam2 / ultralytics calls
:func:`wait` first, so tool calls block (bounded) until the prewarm finished
instead of importing concurrently with it. After that, their own imports are
cache hits.
"""
from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("imagetools.prewarm")

_done = threading.Event()
_thread: threading.Thread | None = None
_started_at: float | None = None
_finished_in: float | None = None
_errors: list[str] = []

# How long a tool call waits for the prewarm before giving up (cold NVMe
# cache on the live box is ~60 s; a spinning dev disk can take minutes).
DEFAULT_WAIT_S = 600.0


def _import_all() -> None:
    steps = (
        ("torch", "import torch"),
        # scipy / sklearn native extensions first and explicitly (see module docstring).
        ("scipy", "import scipy, scipy.special, scipy.linalg, scipy.sparse, scipy.stats, scipy.interpolate, scipy.optimize"),
        ("sklearn", "import sklearn"),
        ("transformers", "import transformers"),
        ("diffusers", "import diffusers"),
        # transformers: everything the AI modules reach for lazily.
        ("transformers.auto",
         "from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM, "
         "BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel"),
        # transformers.generation pulls sklearn -> scipy.stats -> scipy.interpolate; this is the
        # slow chain the deadlock was sitting in.
        ("transformers.generation", "import transformers.generation.utils"),
        # diffusers: SD auto pipelines + every ControlNet pipeline/model sd.py can pick.
        ("diffusers.sd",
         "from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, "
         "StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, "
         "StableDiffusionControlNetInpaintPipeline, StableDiffusionXLControlNetPipeline, "
         "StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLControlNetInpaintPipeline, "
         "StableDiffusion3ControlNetPipeline, StableDiffusion3ControlNetInpaintingPipeline, "
         "FluxControlNetPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetInpaintPipeline, "
         "ControlNetModel, SD3ControlNetModel, FluxControlNetModel, "
         "FluxTransformer2DModel, SD3Transformer2DModel"),
        ("diffusers.qwen",
         "from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline, QwenImageTransformer2DModel, "
         "AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, "
         "QwenImageControlNetPipeline, QwenImageControlNetInpaintPipeline, QwenImageControlNetModel"),
        ("diffusers.llada", "from diffusers import AutoencoderKLFlux2"),
        ("llada_vendor", "from server import llada_vendor"),
        ("diffusers.loaders", "import diffusers.loaders.peft, diffusers.hooks"),
        ("peft", "import peft"),
    )
    for name, stmt in steps:
        t0 = time.perf_counter()
        try:
            exec(stmt, {})
            log.info("prewarm %-24s %.1fs", name, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001 - optional extras may be missing
            _errors.append(f"{name}: {type(e).__name__}: {e}")
            log.info("prewarm %-24s skipped (%s)", name, type(e).__name__)


def _run() -> None:
    global _finished_in
    t0 = time.perf_counter()
    try:
        _import_all()
    finally:
        _finished_in = time.perf_counter() - t0
        _done.set()
        log.info("prewarm finished in %.1fs (%d skipped)", _finished_in, len(_errors))


def start(background: bool | None = None) -> None:
    """Run the imports once. Default (Windows-safe): synchronously on the calling
    thread, which must be the main thread. ``background=True`` (or env
    ``IMAGETOOLS_PREWARM_BACKGROUND=1``) uses a daemon thread instead."""
    global _thread, _started_at
    if _thread is not None:
        return
    if background is None:
        background = os.environ.get("IMAGETOOLS_PREWARM_BACKGROUND", "") == "1"
    _started_at = time.perf_counter()
    if background:
        _thread = threading.Thread(target=_run, daemon=True, name="prewarm")
        _thread.start()
    else:
        _thread = threading.current_thread()
        _run()


def wait(timeout: float | None = DEFAULT_WAIT_S) -> None:
    """Block until the heavy imports are loaded. Call before any lazy diffusers /
    transformers / peft import in a tool path. If :func:`start` was never called
    (tests, scripts) this returns immediately."""
    if _thread is None:
        return
    if not _done.wait(timeout):
        raise RuntimeError(
            f"the AI libraries are still loading after {timeout:.0f}s "
            f"(started {time.perf_counter() - (_started_at or 0):.0f}s ago); try again shortly"
        )


def status() -> dict:
    return {
        "started": _thread is not None,
        "done": _done.is_set(),
        "seconds": round(_finished_in, 1) if _finished_in is not None
        else (round(time.perf_counter() - _started_at, 1) if _started_at else None),
        "skipped": list(_errors),
    }
