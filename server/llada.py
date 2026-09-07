"""LLaDA-Image (inclusionAI) text-to-image and instruction-guided editing.

Two-phase GPU scheduling, because the FP8 checkpoint does not fit a 24 GB card
in one piece (text encoder ~17 GB, denoiser side ~10 GB) and this box has no
RAM to spare for CPU offload:

* **Phase A (encode)** — the LLaDA2 MoE text encoder is loaded onto the GPU
  together with the small queryformer / text_projection, the prompt (and the
  negative prompt when guidance > 1) is encoded, the embeddings are moved to
  CPU and cached, and the text encoder is freed again.
* **Phase B (denoise)** — the transformer (FP8 block-quantised on disk,
  dequantised on load and stored as float8 via Diffusers layerwise casting)
  plus SigVQ and the Flux2 VAE stay resident; generation runs from the cached
  embeddings. Editing mode also encodes the reference image here (SigVQ + VAE),
  never with the text encoder.

A prompt that was already encoded costs no reload at all. A new prompt costs
one text-encoder load + one denoiser reload (~25 GB read from the NVMe volume).
Weights are read through safetensors mmap one shard at a time.

Models are directories in ``server_config['llada_model_dir']`` (default the
ComfyUI ``models/diffusers`` folder): ``LLaDA-Image-Turbo-FP8`` (4 steps,
guidance 1.0) and ``LLaDA-Image-FP8`` (50 steps, guidance 5.0).
"""
from __future__ import annotations

import gc
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image

from . import server_config

log = logging.getLogger("imagetools.llada")

VARIANTS: dict[str, dict[str, Any]] = {
    "turbo": {"dir": "LLaDA-Image-Turbo-FP8", "steps": 4, "guidance": 1.0},
    "base": {"dir": "LLaDA-Image-FP8", "steps": 50, "guidance": 5.0},
}
MODES = ("text", "editing")
_EMBED_CACHE_MAX = 32

_lock = threading.RLock()
_variant: str | None = None
_model_dir: Path | None = None
_fp8_storage = True
_small: dict[str, Any] = {}          # scheduler, vae, tokenizer, queryformer, text_projection (resident)
_denoiser: dict[str, Any] = {}       # transformer, sigvq (resident when loaded)
_embed_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None
_stats: dict[str, Any] = {"encoder_loads": 0, "denoiser_loads": 0, "generations": 0, "cache_hits": 0}


# ---- availability / device ---------------------------------------------------------------------

def _check_available() -> None:
    from . import prewarm
    prewarm.wait()
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from . import llada_vendor  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "LLaDA-Image is unavailable: install the [qwen] extra (diffusers >= 0.38, transformers 4.57.x, torch)."
        ) from e


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    import torch
    return torch.bfloat16 if _device() == "cuda" else torch.float32


def _touch() -> None:
    global _last_used
    with _lock:
        _last_used = time.monotonic()


def _vram() -> dict[str, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
    except Exception:  # noqa: BLE001
        return {}


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---- model directory resolution ---------------------------------------------------------------

def resolve_model_dir(variant_or_name: str) -> tuple[str, Path]:
    """Map ``turbo`` / ``base`` / a folder name under ``llada_model_dir`` to a directory.
    Rejects anything path-like: the model root is the only place we read from."""
    key = (variant_or_name or "turbo").strip()
    root = server_config.get_path("llada_model_dir")
    if key.lower() in VARIANTS:
        name = VARIANTS[key.lower()]["dir"]
        variant = key.lower()
    else:
        if any(ch in key for ch in "/\\:") or key in (".", "..") or key.startswith("."):
            raise ValueError(
                f"model must be 'turbo', 'base' or a folder name inside {root} (got {variant_or_name!r})"
            )
        name = key
        variant = next((v for v, spec in VARIANTS.items() if spec["dir"] == name), name)
    path = root / name
    if not (path / "model_index.json").is_file():
        have = sorted(p.name for p in root.iterdir() if (p / "model_index.json").is_file()) if root.is_dir() else []
        raise ValueError(
            f"no LLaDA-Image model at {path} (model_index.json missing). Available in {root}: {have or 'none'}; "
            f"pull inclusionAI/LLaDA-Image-Turbo-FP8 or LLaDA-Image-FP8 into that folder"
        )
    return variant, path


# ---- FP8 block dequantisation ------------------------------------------------------------------

def _dequant_block_fp8(weight, scale_inv, block: int = 128, dtype=None, transposed: bool = False):
    """Block-FP8 -> dtype. ``transposed`` says the scale grid is stored as [ceil(K/b), ceil(N/b)]
    instead of the DeepSeek [ceil(N/b), ceil(K/b)]; the caller detects that once per checkpoint
    from a non-square tensor, because square tensors cannot tell."""
    import math
    import torch
    n, k = weight.shape
    nb, kb = math.ceil(n / block), math.ceil(k / block)
    s = scale_inv.to(torch.float32)
    if transposed:
        s = s.t()
    if tuple(s.shape) != (nb, kb):
        raise RuntimeError(f"scale grid {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)} (block {block}, transposed={transposed})")
    s = s.repeat_interleave(block, dim=0)[:n].repeat_interleave(block, dim=1)[:, :k]
    return (weight.to(torch.float32) * s).to(dtype or torch.bfloat16)


def _detect_scale_orientation(raw: dict, block: int = 128) -> bool:
    """True when scale grids are stored [ceil(K/b), ceil(N/b)] (transposed). Decided from the first
    non-square weight/scale pair; a checkpoint with only square tensors defaults to DeepSeek order."""
    import math
    for k, v in raw.items():
        if not k.endswith(".weight") or v.ndim != 2:
            continue
        sk = k[: -len(".weight")] + ".weight_scale_inv"
        if sk not in raw:
            continue
        n, kk = v.shape
        nb, kb = math.ceil(n / block), math.ceil(kk / block)
        if nb == kb:
            continue
        shape = tuple(raw[sk].shape)
        if shape == (nb, kb):
            return False
        if shape == (kb, nb):
            return True
        raise RuntimeError(f"{k}: scale grid {shape} matches neither orientation for {tuple(v.shape)}")
    return False


def _split_fused_keys(state: dict) -> int:
    """The FP8 export fuses ``attention.to_qkv`` (= [to_q; to_k; to_v]) and ``feed_forward.w13``
    (= [w1; w3]) SGLang-style; the Diffusers model keeps them separate. Split in place."""
    n_split = 0
    for k in list(state):
        if k.endswith(".attention.to_qkv.weight"):
            w = state.pop(k)
            d = w.shape[0] // 3
            base = k[: -len("to_qkv.weight")]
            state[base + "to_q.weight"], state[base + "to_k.weight"], state[base + "to_v.weight"] = w[:d], w[d:2 * d], w[2 * d:]
            n_split += 1
        elif k.endswith(".feed_forward.w13.weight"):
            w = state.pop(k)
            h = w.shape[0] // 2
            base = k[: -len("w13.weight")]
            state[base + "w1.weight"], state[base + "w3.weight"] = w[:h], w[h:]
            n_split += 1
    return n_split


def _load_fp8_state_dict(model_dir: Path, dtype) -> tuple[dict, int]:
    """Read the shards one at a time, dequantise block-FP8 pairs and split fused keys. Every tensor
    is returned on the CPU in ``dtype``; the caller moves the assembled model to its device.
    Returns (state_dict, n_dequantised)."""
    from safetensors import safe_open
    index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        shards = sorted({v for v in json.load(open(index, encoding="utf-8"))["weight_map"].values()})
    else:
        shards = ["diffusion_pytorch_model.safetensors"]
    out: dict = {}
    n_dq = 0
    transposed: bool | None = None
    pending_w: dict = {}   # weights whose scale is in a later shard
    pending_s: dict = {}   # scales whose weight is in a later shard

    def place(k, t):
        out[k] = t.to(dtype) if t.is_floating_point() else t

    def dequant_pair(k, w, sc):
        nonlocal n_dq
        place(k, _dequant_block_fp8(w, sc, dtype=dtype, transposed=bool(transposed)))
        n_dq += 1

    for shard in shards:
        with safe_open(str(model_dir / shard), "pt", device="cpu") as f:
            raw = {k: f.get_tensor(k) for k in f.keys()}
        if transposed is None:
            transposed = _detect_scale_orientation({**pending_w, **pending_s, **raw})
        for k, v in raw.items():
            if k.endswith(".weight_scale_inv"):
                wk = k[: -len(".weight_scale_inv")] + ".weight"
                if wk in raw:
                    continue  # handled with its weight below
                if wk in pending_w:
                    dequant_pair(wk, pending_w.pop(wk), v)
                else:
                    pending_s[k] = v
                continue
            sk = k[: -len(".weight")] + ".weight_scale_inv" if k.endswith(".weight") else None
            if sk and sk in raw:
                dequant_pair(k, v, raw[sk])
            elif sk and sk in pending_s:
                dequant_pair(k, v, pending_s.pop(sk))
            elif sk and _might_have_scale(k):
                pending_w[k] = v  # decide when the last shard is read
            else:
                place(k, v)
        del raw
    for k, v in pending_w.items():  # never got a scale: plain tensor
        place(k, v)
    if pending_s:
        raise RuntimeError(f"scale tensors without weights: {sorted(pending_s)[:3]}")
    _split_fused_keys(out)
    return out, n_dq


def _might_have_scale(key: str) -> bool:
    """Only 2-D linear weights of attention / feed-forward blocks are block-quantised."""
    return any(part in key for part in (".attention.", ".feed_forward."))


def _load_transformer(model_dir: Path, dtype, device: str, fp8_storage: bool):
    """Build the DiT from config + dequantised weights. With ``fp8_storage`` the weights are re-stored
    as float8 on the GPU through Diffusers layerwise casting (bf16 compute)."""
    import torch
    from .llada_vendor import LLaDAImageTransformer2DModel

    cfg = json.load(open(model_dir / "transformer" / "config.json", encoding="utf-8"))
    quantised = "quantization_config" in cfg
    if not quantised:
        model = LLaDAImageTransformer2DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype)
        return model.to(device).eval()
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "quantization_config"}
    with torch.device("meta"):
        model = LLaDAImageTransformer2DModel.from_config(cfg)
    storage = torch.float8_e4m3fn if (fp8_storage and device == "cuda") else None
    state, n_dq = _load_fp8_state_dict(model_dir / "transformer", dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    if unexpected or missing:
        raise RuntimeError(f"transformer checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    log.info("llada transformer: %d FP8 block tensors dequantised (storage %s)", n_dq, storage or dtype)
    model = model.to(device, dtype)
    if storage is not None:
        model.enable_layerwise_casting(storage_dtype=storage, compute_dtype=dtype)
    return model.eval()


def _load_scheduler(model_dir: Path):
    """Diffusers ignores config keys its scheduler does not declare (``use_uniform_sigmas`` for the
    Turbo checkpoint), but the vendored pipeline reads it from ``scheduler.config``. Re-register it."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler")
    raw = json.load(open(model_dir / "scheduler" / "scheduler_config.json", encoding="utf-8"))
    if "use_uniform_sigmas" in raw and "use_uniform_sigmas" not in sched.config:
        sched.register_to_config(use_uniform_sigmas=bool(raw["use_uniform_sigmas"]))
    return sched


# ---- loading phases ----------------------------------------------------------------------------

def _load_small(model_dir: Path) -> None:
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer
    from .llada_vendor import LLaDAImageQueryFormerModel, LLaDAImageTextProjectionModel
    dev, dt = _device(), _dtype()
    _small.clear()
    _small["scheduler"] = _load_scheduler(model_dir)
    _small["vae"] = AutoencoderKLFlux2.from_pretrained(model_dir / "vae", torch_dtype=dt).to(dev).eval()
    _small["tokenizer"] = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    _small["queryformer"] = LLaDAImageQueryFormerModel.from_pretrained(model_dir / "queryformer", torch_dtype=dt).to(dev).eval()
    _small["text_projection"] = LLaDAImageTextProjectionModel.from_pretrained(model_dir / "text_projection", torch_dtype=dt).to(dev).eval()


def _load_denoiser() -> None:
    from .llada_vendor import LLaDAImageSigVQModel
    assert _model_dir is not None
    if _denoiser:
        return
    t0 = time.perf_counter()
    dev, dt = _device(), _dtype()
    sigvq = LLaDAImageSigVQModel.from_pretrained(_model_dir / "sigvq", torch_dtype=dt).to(dev).eval()
    transformer = _load_transformer(_model_dir, dt, dev, _fp8_storage)
    _denoiser["sigvq"] = sigvq
    _denoiser["transformer"] = transformer
    _stats["denoiser_loads"] += 1
    log.info("llada denoiser loaded in %.1fs %s", time.perf_counter() - t0, _vram())


def _drop_denoiser() -> None:
    if _denoiser:
        _denoiser.clear()
        _free_cuda()


def _pipeline(text_encoder=None):
    from .llada_vendor import LLaDAImagePipeline
    return LLaDAImagePipeline(
        scheduler=_small["scheduler"], vae=_small["vae"], text_encoder=text_encoder,
        tokenizer=_small["tokenizer"], queryformer=_small["queryformer"],
        text_projection=_small["text_projection"],
        sigvq=_denoiser.get("sigvq"), transformer=_denoiser.get("transformer"),
    )


def _encode_prompts(prompt: str, negative: str | None, cfg: bool, max_len: int) -> tuple:
    """Phase A. Returns (pe, pm, ne, nm) on CPU."""
    import torch
    from transformers import AutoModel
    assert _model_dir is not None
    key = (str(_model_dir), prompt, negative if cfg else None, max_len)
    hit = _embed_cache.get(key)
    if hit is not None:
        _embed_cache.move_to_end(key)
        _stats["cache_hits"] += 1
        return hit
    dev, dt = _device(), _dtype()
    _drop_denoiser()  # the text encoder needs the room
    t0 = time.perf_counter()
    text_encoder = AutoModel.from_pretrained(
        _model_dir / "text_encoder", dtype=dt, trust_remote_code=True,
        device_map={"": dev} if dev == "cuda" else None,
    ).eval()
    _stats["encoder_loads"] += 1
    _patch_fp8_experts(text_encoder)
    log.info("llada text encoder loaded in %.1fs %s", time.perf_counter() - t0, _vram())
    try:
        pipe = _pipeline(text_encoder=text_encoder)
        with torch.inference_mode():
            pe, pm, ne, nm = pipe.encode_prompt(
                prompt, negative or "", cfg, 1, max_sequence_length=max_len, device=torch.device(dev),
            )
        out = (pe.to("cpu"), pm.to("cpu"), None if ne is None else ne.to("cpu"), None if nm is None else nm.to("cpu"))
    finally:
        del text_encoder
        _free_cuda()
    _embed_cache[key] = out
    while len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)
    return out


def _patch_fp8_experts(text_encoder) -> int:
    """The LLaDA2 MoE text encoder computes its FP8 experts with ``torch._scaled_mm`` using
    per-row scales, which needs sm_90 (on Ada it raises "Per-row scaling is not supported for this
    platform"). Swap the per-expert matmul for the modeling file's own dequantise-then-bf16-matmul."""
    import torch
    import torch.nn.functional as F

    def _scaled_mm_expert(self, xq, x_scale, weight, weight_scale, expert_id, out_dtype):
        x = (xq.float() * x_scale).to(out_dtype)
        return F.linear(x, self._dequant_expert(weight, weight_scale, expert_id, out_dtype))

    seen: set = set()
    for m in text_encoder.modules():
        cls = type(m)
        if getattr(m, "use_fp8", False) and hasattr(m, "_scaled_mm_expert") and cls not in seen:
            cls._scaled_mm_expert = _scaled_mm_expert
            seen.add(cls)
    return len(seen)


# ---- idle sweeper ------------------------------------------------------------------------------

def _start_sweeper_if_needed() -> None:
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_thread = threading.Thread(target=_sweep_loop, daemon=True, name="llada-idle-sweeper")
    _sweeper_thread.start()


def _sweep_loop() -> None:
    while True:
        time.sleep(_SWEEP_INTERVAL_S)
        try:
            if _idle_timeout_s <= 0:
                continue
            with _lock:
                if not _small or _last_used is None or time.monotonic() - _last_used <= _idle_timeout_s:
                    continue
            unload()
        except Exception:  # noqa: BLE001
            pass


def set_idle_timeout(seconds: float) -> dict[str, Any]:
    global _idle_timeout_s
    _idle_timeout_s = max(0.0, float(seconds))
    return {"idle_timeout_s": _idle_timeout_s, "sweep_interval_s": _SWEEP_INTERVAL_S,
            "auto_evict_enabled": _idle_timeout_s > 0}


# ---- public API --------------------------------------------------------------------------------

def status() -> dict[str, Any]:
    try:
        _check_available()
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
    with _lock:
        loaded = bool(_small)
        info = {
            "available": True,
            "device": _device(),
            "loaded": loaded,
            "variant": _variant,
            "model_dir": str(_model_dir) if _model_dir else None,
            "fp8_storage": _fp8_storage,
            "denoiser_resident": bool(_denoiser),
            "cached_prompts": len(_embed_cache),
            "idle_s": None if _last_used is None else round(time.monotonic() - _last_used, 1),
            "idle_timeout_s": _idle_timeout_s,
            "stats": dict(_stats),
            "vram": _vram(),
            "variants": {k: {"folder": v["dir"], "default_steps": v["steps"], "default_guidance": v["guidance"]}
                         for k, v in VARIANTS.items()},
            "model_root": str(server_config.get_path("llada_model_dir")),
        }
    return info


def load(model: str = "turbo", *, fp8_storage: bool = True) -> dict[str, Any]:
    """Load the resident parts (VAE, tokenizer, queryformer, text_projection, SigVQ, transformer).
    Idempotent for the same model."""
    global _variant, _model_dir, _fp8_storage
    _check_available()
    variant, path = resolve_model_dir(model)
    with _lock:
        if _small and _model_dir == path and _fp8_storage == fp8_storage:
            _touch()
            return {"was_already_loaded": True, **status()}
        t0 = time.perf_counter()
        if _small:
            unload()
        _variant, _model_dir, _fp8_storage = variant, path, fp8_storage
        try:
            _load_small(path)
            _load_denoiser()
        except Exception:
            unload()
            raise
        _touch()
        _start_sweeper_if_needed()
        elapsed = time.perf_counter() - t0
    out = status()
    out.update({"was_already_loaded": False, "load_time_s": round(elapsed, 1)})
    return out


def unload() -> dict[str, Any]:
    global _variant, _model_dir, _last_used
    with _lock:
        had = bool(_small)
        _denoiser.clear()
        _small.clear()
        _embed_cache.clear()
        _variant, _model_dir, _last_used = None, None, None
    _free_cuda()
    return {"unloaded": had, "vram": _vram()}


def _resolve_defaults(steps: int | None, guidance: float | None) -> tuple[int, float]:
    spec = VARIANTS.get(_variant or "", {"steps": 20, "guidance": 4.5})
    return (int(steps) if steps is not None else spec["steps"],
            float(guidance) if guidance is not None else spec["guidance"])


def generate(prompt: str, *, negative_prompt: str | None = None, width: int = 1024, height: int = 1024,
             steps: int | None = None, guidance: float | None = None, seed: int | None = None,
             image: Image.Image | None = None, max_sequence_length: int = 2048) -> Image.Image:
    """Text-to-image (``image=None``) or instruction-guided editing (``image`` given)."""
    import torch
    _check_available()
    if not (prompt or "").strip():
        raise ValueError("prompt is required")
    with _lock:
        if not _small:
            load("turbo")
        n_steps, g = _resolve_defaults(steps, guidance)
        cfg = g > 1.0
        pe, pm, ne, nm = _encode_prompts(prompt.strip(), negative_prompt, cfg, max_sequence_length)
        _load_denoiser()
        dev = _device()
        generator = torch.Generator(dev).manual_seed(int(seed)) if seed is not None else None
        pipe = _pipeline()
        kwargs: dict[str, Any] = dict(
            prompt=None, prompt_embeds=pe.to(dev), prompt_attention_mask=pm.to(dev),
            negative_prompt_embeds=None if ne is None else ne.to(dev),
            negative_prompt_attention_mask=None if nm is None else nm.to(dev),
            generation_mode="editing" if image is not None else "text",
            height=int(height), width=int(width), num_inference_steps=n_steps, guidance_scale=g,
            generator=generator, max_sequence_length=max_sequence_length,
        )
        if image is not None:
            kwargs["image"] = image.convert("RGB")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pipe(**kwargs).images[0]
        _stats["generations"] += 1
        _touch()
        log.info("llada %s %dx%d %d steps in %.1fs %s", kwargs["generation_mode"], width, height, n_steps,
                 time.perf_counter() - t0, _vram())
        return out
