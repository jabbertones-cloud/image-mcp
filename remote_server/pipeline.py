"""Diffusers pipeline loader — eager-loads ONE full Qwen model per pod.

Two supported configurations, picked by ``REMOTE_MODEL``:

  qwen-image            → Qwen/Qwen-Image  (txt2img, QwenImagePipeline)
  qwen-image-edit-2511  → Qwen/Qwen-Image-Edit-2511  (img2img, QwenImageEditPlusPipeline)

Both are the **full bf16 versions**. No GGUF quantization, no fp8_scaled
text encoder — the remote pod is sized for it. The H100 80 GB target has
ample headroom; the cost is ~38 GB on the persistent volume.

The loader follows the same component-by-component pattern the existing
local ``server/qwen.py`` uses, because it sidesteps two known issues:

  1. ``Qwen2_5_VLForConditionalGeneration.from_pretrained`` access-violates
     on certain Windows transformers builds during ``_materialize_copy``.
     The remote pod is Linux so this is less acute, but the AutoModel path
     is robust either way.
  2. Single-file VAE / DiT loaders lose the JSON config (scaling factor,
     schedulers). Using ``from_pretrained(subfolder=...)`` gets it right.

Each component is loaded as bf16. The pipeline returns the LATENT — the
client decodes locally with its own VAE.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

from remote_server.settings import ServerSettings
from remote_server.transfer import TransferWorkspace

log = logging.getLogger(__name__)


# HuggingFace model IDs. Override per-env if you mirror them.
QWEN_IMAGE_REPO      = os.environ.get("REMOTE_QWEN_IMAGE_REPO", "Qwen/Qwen-Image")
QWEN_IMAGE_EDIT_REPO = os.environ.get("REMOTE_QWEN_EDIT_REPO",  "Qwen/Qwen-Image-Edit-2511")


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _dtype():
    import torch
    return torch.bfloat16 if _device() == "cuda" else torch.float32


# ─── base interface ──────────────────────────────────────────────────


class Pipeline:
    """All pipelines implement this. The base class supplies LoRA application
    helpers (apply_loras / clear_loras) that route to diffusers'
    ``load_lora_weights`` / ``set_adapters`` / ``unload_lora_weights`` API."""

    name: str = ""
    vae_required: str = ""             # filename hint for the client decoder

    def generate(
        self, payload: dict[str, Any], ws: TransferWorkspace, mode: str,
    ) -> tuple["torch.Tensor", dict[str, Any]]:
        raise NotImplementedError

    def unload(self) -> None:
        pass

    # ─── LoRA hot-swap surface ───────────────────────────────────

    def apply_loras(self, loras: list, weights: list[float]) -> list[str]:
        """Reconcile the pipeline to exactly the requested LoRA set.

        We pass each LoRA's in-RAM ``state_dict`` (DICTS, never paths — the
        bytes never touch tmpfs) to diffusers' ``load_lora_weights``. Adapter
        names are deterministic: ``lora_<sha-12>``.

        The currently-applied ``(sha256, weight)`` set is remembered, so a
        repeated request with the same LoRAs (e.g. a queued batch) is a no-op
        instead of a full unload + reload of a several-hundred-MB adapter.
        Passing an empty list clears any resident adapters."""
        if not hasattr(self, "_pipe"):
            raise RuntimeError("pipeline not initialised")
        desired = [(l.sha256, float(w)) for l, w in zip(loras, weights)]
        if desired == getattr(self, "_applied_loras", None):
            return [f"lora_{sha[:12]}" for sha, _ in desired]  # already resident

        # State differs → rebuild from a clean slate.
        try: self._pipe.unload_lora_weights()
        except Exception: pass
        self._applied_loras = []
        if not loras:
            return []

        names: list[str] = []
        for lora in loras:
            adapter = f"lora_{lora.sha256[:12]}"
            sd = getattr(lora, "state_dict", None)
            if sd is None:
                sd = str(getattr(lora, "path", ""))
            try:
                self._pipe.load_lora_weights(sd, adapter_name=adapter)
            except Exception as e:
                try: self._pipe.unload_lora_weights()
                except Exception: pass
                raise RuntimeError(
                    f"load_lora_weights failed for {lora.sha256[:12]}…: {e!r}"
                ) from e
            names.append(adapter)
        try:
            self._pipe.set_adapters(names, weights)
        except Exception as e:
            try: self._pipe.unload_lora_weights()
            except Exception: pass
            raise RuntimeError(f"set_adapters failed: {e!r}") from e
        self._applied_loras = desired
        return names

    def clear_loras(self) -> None:
        if not hasattr(self, "_pipe"):
            return
        try: self._pipe.unload_lora_weights()
        except Exception: pass
        self._applied_loras = []


# ─── Qwen-Image (txt2img) ────────────────────────────────────────────


class QwenImagePipeline(Pipeline):
    name = "qwen-image"
    vae_required = "qwen_image_vae.safetensors"
    _stride = 16        # latent scale factor for Wan2.1 VAE → 16× downsample

    def __init__(self, settings: ServerSettings):
        import torch
        from diffusers import (
            AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
            QwenImagePipeline as _QwenImagePipeline,
            QwenImageTransformer2DModel,
        )
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self._device = _device()
        dtype = _dtype()
        repo = QWEN_IMAGE_REPO
        cache_dir = self._hf_cache_dir(settings)
        log.info(f"[qwen-image] repo={repo}  cache_dir={cache_dir}")

        # Component-by-component load (see header docstring for why).
        # Text encoder first — its 16 GB mmap is the largest and goes onto
        # GPU directly to avoid double-buffering during the next load.
        log.info("[qwen-image] loading text encoder")
        text_encoder = AutoModel.from_pretrained(
            repo, subfolder="text_encoder",
            torch_dtype=dtype, low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            device_map=("cuda:0" if self._device == "cuda" else None),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo, subfolder="tokenizer", cache_dir=cache_dir,
        )
        # Qwen2.5-VL ships a processor for vision conditioning; T2I doesn't
        # use the image side but the pipeline still expects it to exist.
        try:
            processor = AutoProcessor.from_pretrained(
                repo, subfolder="processor", cache_dir=cache_dir,
            )
        except Exception:
            processor = None

        log.info("[qwen-image] loading VAE")
        vae = AutoencoderKLQwenImage.from_pretrained(
            repo, subfolder="vae", torch_dtype=dtype, cache_dir=cache_dir,
        )
        # We don't decode here — keep VAE on CPU to free VRAM.
        vae.to("cpu")

        log.info("[qwen-image] loading scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            repo, subfolder="scheduler", cache_dir=cache_dir,
        )

        log.info("[qwen-image] loading transformer")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            repo, subfolder="transformer",
            torch_dtype=dtype, cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        if self._device == "cuda":
            transformer = transformer.to("cuda")

        kwargs: dict[str, Any] = dict(
            transformer=transformer, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            scheduler=scheduler,
        )
        if processor is not None:
            kwargs["processor"] = processor
        self._pipe = _QwenImagePipeline(**kwargs)
        # The component-wise constructor doesn't auto-`.to(device)` everything;
        # text_encoder + transformer are already on cuda above. VAE is on CPU
        # deliberately. tokenizer/scheduler are CPU-only.
        log.info(f"[qwen-image] ready on {self._device}")

    def generate(
        self, payload: dict[str, Any], ws: TransferWorkspace, mode: str,
    ) -> tuple["torch.Tensor", dict[str, Any]]:
        import torch
        if mode != "txt2img":
            raise ValueError(f"qwen-image only supports txt2img, got {mode!r}")

        prompt   = payload["prompt"]
        negative = payload.get("negative_prompt") or " "    # diffusers needs a non-empty cond
        steps    = int(payload.get("steps", 30))
        # Qwen-Image uses ``true_cfg_scale`` (CFG via paired uncond pass), not
        # ``guidance_scale`` — guidance_scale is the embedded-guidance knob
        # set by the model card, usually ~3.5.
        cfg      = float(payload.get("cfg", 4.0))
        seed     = int(payload.get("seed", 0))
        w = self._clamp(int(payload.get("width", 1024)))
        h = self._clamp(int(payload.get("height", 1024)))

        gen = torch.Generator(device=self._device).manual_seed(seed)
        with torch.inference_mode():
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                width=w, height=h,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                generator=gen,
                output_type="latent",
            )
        latent = out.images if hasattr(out, "images") else out[0]
        if isinstance(latent, (list, tuple)):
            latent = latent[0]

        scaling = self._vae_scaling()
        metadata = {
            "model":        self.name,
            "seed":         seed,
            "steps":        steps,
            "cfg":          cfg,
            "height":       h, "width": w,
            "scaling":      scaling,
            "vae_required": self.vae_required,
        }
        return latent, metadata

    def unload(self) -> None:
        try:
            del self._pipe
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception: pass

    # ─── helpers ──────────────────────────────────────────────────

    def _vae_scaling(self) -> float:
        try:
            return float(self._pipe.vae.config.scaling_factor)
        except Exception:
            return 0.18215

    def _clamp(self, n: int) -> int:
        return max(self._stride, (n // self._stride) * self._stride)

    @staticmethod
    def _hf_cache_dir(s: ServerSettings) -> str:
        """Force HF downloads into the persistent models_root so they survive
        pod restarts (and so a future agent can pre-stage them via the
        download_weights.py helper)."""
        cache = s.models_root / "hf-cache"
        cache.mkdir(parents=True, exist_ok=True)
        # Make sure huggingface_hub also writes there
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_HUB_CACHE", str(cache))
        return str(cache)


# ─── Qwen-Image-Edit-2511 (img2img) ──────────────────────────────────


class QwenImageEditPipeline(QwenImagePipeline):
    name = "qwen-image-edit-2511"
    # Qwen-Image-Edit-2511 ships its own VAE that's a Wan2.1 variant; the
    # client decoder should use the matching file. The metadata key tells it.
    vae_required = "qwen_image_vae.safetensors"

    def __init__(self, settings: ServerSettings):
        import torch
        from diffusers import (
            AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
            QwenImageEditPlusPipeline, QwenImageTransformer2DModel,
        )
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self._device = _device()
        dtype = _dtype()
        repo = QWEN_IMAGE_EDIT_REPO
        cache_dir = self._hf_cache_dir(settings)
        log.info(f"[qwen-image-edit-2511] repo={repo}  cache_dir={cache_dir}")

        log.info("[qwen-image-edit-2511] loading text encoder")
        text_encoder = AutoModel.from_pretrained(
            repo, subfolder="text_encoder",
            torch_dtype=dtype, low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            device_map=("cuda:0" if self._device == "cuda" else None),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo, subfolder="tokenizer", cache_dir=cache_dir,
        )
        processor = AutoProcessor.from_pretrained(
            repo, subfolder="processor", cache_dir=cache_dir,
        )

        log.info("[qwen-image-edit-2511] loading VAE")
        vae = AutoencoderKLQwenImage.from_pretrained(
            repo, subfolder="vae", torch_dtype=dtype, cache_dir=cache_dir,
        )
        # Edit pipeline needs the VAE encoder for the control image. Keep
        # on CPU and call ``.to(cuda)`` only for the brief encode pass.
        vae.to("cpu")

        log.info("[qwen-image-edit-2511] loading scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            repo, subfolder="scheduler", cache_dir=cache_dir,
        )

        log.info("[qwen-image-edit-2511] loading transformer")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            repo, subfolder="transformer",
            torch_dtype=dtype, cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        if self._device == "cuda":
            transformer = transformer.to("cuda")

        self._pipe = QwenImageEditPlusPipeline(
            transformer=transformer, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            processor=processor, scheduler=scheduler,
        )
        log.info(f"[qwen-image-edit-2511] ready on {self._device}")

    def generate(
        self, payload: dict[str, Any], ws: TransferWorkspace, mode: str,
    ) -> tuple["torch.Tensor", dict[str, Any]]:
        import torch
        from PIL import Image

        if mode != "img2img":
            raise ValueError(f"edit-2511 requires img2img, got {mode!r}")
        ctrl_b64 = payload.get("control_image_b64")
        if not ctrl_b64:
            raise ValueError("img2img requires payload.control_image_b64")
        ctrl_path = ws.write_image_bytes(base64.b64decode(ctrl_b64.encode("ascii")))
        control_image = Image.open(ctrl_path).convert("RGB")

        prompt   = payload["prompt"]
        negative = payload.get("negative_prompt") or " "
        steps    = int(payload.get("steps", 30))
        cfg      = float(payload.get("cfg", 4.0))
        seed     = int(payload.get("seed", 0))
        w = self._clamp(int(payload.get("width", control_image.width)))
        h = self._clamp(int(payload.get("height", control_image.height)))

        # The control image needs to go through the VAE encoder once. Move
        # VAE to GPU briefly, encode, move back.
        if self._device == "cuda":
            self._pipe.vae.to("cuda")
        try:
            gen = torch.Generator(device=self._device).manual_seed(seed)
            with torch.inference_mode():
                out = self._pipe(
                    image=control_image,
                    prompt=prompt,
                    negative_prompt=negative,
                    width=w, height=h,
                    num_inference_steps=steps,
                    true_cfg_scale=cfg,
                    generator=gen,
                    output_type="latent",
                )
        finally:
            if self._device == "cuda":
                self._pipe.vae.to("cpu")

        latent = out.images if hasattr(out, "images") else out[0]
        if isinstance(latent, (list, tuple)):
            latent = latent[0]

        metadata = {
            "model":        self.name,
            "seed":         seed,
            "steps":        steps,
            "cfg":          cfg,
            "height":       h, "width": w,
            "scaling":      self._vae_scaling(),
            "vae_required": self.vae_required,
        }
        return latent, metadata


# ─── loader ──────────────────────────────────────────────────────────


def load_pipeline(settings: ServerSettings) -> Pipeline:
    if settings.model == "qwen-image":
        return QwenImagePipeline(settings)
    if settings.model == "qwen-image-edit-2511":
        return QwenImageEditPipeline(settings)
    raise ValueError(f"unknown REMOTE_MODEL={settings.model!r}")
