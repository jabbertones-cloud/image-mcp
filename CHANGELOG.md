# Changelog

All notable changes to ImageTools MCP. Versions follow [Semantic Versioning](https://semver.org/);
dates are ISO 8601.

## [1.1.0] — 2026-09-12

242 tools (was 216).

### Added
- **LLaDA-Image backend** (`llada_status` / `llada_load` / `llada_unload` / `llada_set_idle_timeout` /
  `llada_generate` / `llada_edit`): inclusionAI's 6B DiT with a frozen 16B-A1B LLaDA2.0-Mini MoE
  text encoder, one checkpoint for text-to-image and instruction editing. Variants `turbo`
  (4 steps, guidance 1.0) and `base` (50 steps, guidance 5.0), resolved only inside
  `llada_model_dir`. The official model + pipeline code is vendored in `server/llada_vendor/`
  (Apache-2.0). The loader reads the DeepSeek-style block-FP8 checkpoints that stock Diffusers
  cannot, and schedules the 17 GB text encoder and the 10 GB denoiser in two phases with a
  32-entry prompt-embedding cache so the pair fits a 24 GB card without CPU offload.
- **Remote generation**: `remote_server/` (FastAPI app for a rented GPU pod) plus 15 client tools
  (`remote_status`, `remote_qwen_txt2img` / `remote_qwen_edit`, `remote_lora_*`,
  `remote_server_status`, `remote_watcher_*`, `remote_queue_*`). The pod returns a sealed latent;
  the local server decodes it with its own VAE. `[remote]` extra; `docs/REMOTE_GEN.md`.
- **Perception tools**: `inspect_region` (1:1 zoom that composites only the requested rectangle)
  and `canvases_overview` (labelled contact sheet of every open canvas). Previews are encoded
  within a transport budget: PNG for small, alpha or lossless requests, JPEG for photographic
  content, shrinking until under ~1.2 MB; callers report the true returned size.
- **Server config**: `get_config` / `set_config` for the model directories (`llada_model_dir`,
  `comfyui_unet_dir`, `comfyui_vae_dir`, `lora_dirs`, `insightface_root`, ...) and the scratch folder.
- `deploy.py` (dev → live copy via shutil with hash verification), `tools/smoke_stdio.py`
  (drives `run_server.py` over real stdio JSON-RPC for the SD / Qwen / LLaDA paths),
  `tools/sd_direct_check.py`. Tests: `test_llada.py`, `test_prewarm.py`, `test_previews.py`,
  `test_remote_*.py`, `tests/remote/`.

### Fixed
- **Every tool hung after the first SD call** ("the SD engine is broken"). Root cause was an import
  deadlock, not SD: FastMCP runs sync tools on the event-loop thread, and a lazy diffusers import
  there collided with the background prewarm thread on module locks; scipy's native extensions
  also freeze when first imported off the main thread on Windows. `server/prewarm.py` now imports
  the whole AI stack once, synchronously, before FastMCP starts, and every AI module waits for it.
- **LLaDA FP8 loader**, four bugs found while bringing the model up:
  - the expanded 128×128 scale grid was sliced `[:n]...[:k]`, trimming rows twice; it only worked on
    square tensors whose size is a multiple of 128 and broke every rectangular linear;
  - the export fuses `attention.to_qkv` and `feed_forward.w13` while the model keeps them separate;
    they are split before assign-loading;
  - on torch 2.5.1 the text encoder's FP8 experts crash on `index_select` for float8 activations
    (and `torch._scaled_mm` per-row scaling is Hopper-only); the experts run on bf16 activations;
  - dequantised weights were staged as bf16 on the GPU before layerwise casting, which OOMed a
    24 GB card; they now stream to the device tensor by tensor directly as float8.
- **Stable Diffusion**: `sd_img2img` returned solid black images because the SD 1.x safety
  checker fired on ordinary content; it is never loaded now. The inpaint default is
  `Lykon/dreamshaper-8-inpainting` (safetensors; the runwayml repo ships pickles that torch 2.5.1
  refuses) and inpaint keeps the source size instead of the pipeline's 512×512.
- Generated results placed on an existing canvas (`_place_generated`) are added as a new layer
  after a snapshot instead of replacing the document and its history.
- Undo history is now capped by bytes across **all** canvases (`IMAGETOOLS_UNDO_MAX_MB`, default 512)
  instead of per stack; redo stacks are no longer trimmed from the wrong end.
- `face_list_models` docstring no longer mentions fields that were removed in 1.0.x.

### Changed
- Tool count 216 → 242; README, banner and tool table regenerated; face tools listed without
  further promotion.
- Version is now tracked in `pyproject.toml` and `remote_server/__init__.py` together.

## [1.0.0] — 2026-05-25

Initial release: 216 tools over an in-memory canvas with a Photoshop-style layer stack — drawing,
transforms, adjustments, gradients, channels, painting brushes, layer effects, patterns, warp /
liquify / distort, PS-style blurs, five segmentation models (SAM 2, SAM 1, YOLOv8-seg, BiRefNet,
CLIPSeg) with mask tooling, Stable Diffusion and Qwen-Image-Edit (Q4 GGUF + Lightning LoRA) with
ControlNet for both, format conversion, animation / ICO / PDF, and utilities (watermark, QR,
perceptual hash, compare, histogram, annotate, glitch). Same-day follow-ups added the server
config module, saved face model listing, and the full attribution / license audit.
