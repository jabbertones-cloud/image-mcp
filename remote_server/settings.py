"""Server runtime configuration — env-driven, overridable for tests."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# Supported models. Each maps to a pipeline loader in pipeline.py.
SupportedModel = Literal["qwen-image", "qwen-image-edit-2511"]


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass
class ServerSettings:
    """Path layout — read carefully:

    * ``secrets_dir`` (persistent OK) — server's own identity + PSK halves.
      Loaded once at boot, then held in process RAM. Disk image is non-PII
      (only your server's pubkey + its half of the secret it shares with one
      pre-paired client).
    * ``models_root`` (persistent) — public Qwen-Image / Qwen-Image-Edit /
      Wan2.1 VAE weights. No PII. Mount a Network Volume here for fast
      cold-starts.
    * ``transfer_dir`` (**tmpfs ONLY** — enforced at boot) — every byte of
      data-in-flight: request bodies, decoded control images, latents
      before sealing. Nothing client-supplied ever lands on the persistent
      volume. The entrypoint refuses to start if ``transfer_dir`` is not a
      tmpfs mountpoint.
    """

    # Bootstrap material — persistent OK.
    secrets_dir: Path = field(default_factory=lambda: _env_path(
        "REMOTE_SECRETS_DIR", "/etc/remote-gen"))

    # Public model weights — persistent volume.
    models_root: Path = field(default_factory=lambda: _env_path(
        "REMOTE_MODELS_ROOT", "/var/remote-gen/models"))

    # Data in flight — tmpfs ONLY. Workspace for control images that PIL/HF
    # processors must read from a path (occasional). Cleared each request.
    transfer_dir: Path = field(default_factory=lambda: _env_path(
        "REMOTE_TRANSFER_DIR", "/run/remote-gen/transfer"))

    # Which model the pod is dedicated to. Picked at boot, NEVER swapped at runtime.
    model: SupportedModel = field(
        default_factory=lambda: os.environ.get("REMOTE_MODEL", "qwen-image")
    )  # type: ignore[assignment]

    # Per-request body cap (bytes). Generation requests are small; even
    # img2img with a 4 MB control image fits in 16 MB.
    max_upload_bytes: int = int(os.environ.get("REMOTE_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))

    # Listen address — behind RunPod's TLS proxy / Tailscale serve.
    host: str = os.environ.get("REMOTE_HOST", "0.0.0.0")
    port: int = int(os.environ.get("REMOTE_PORT", "8443"))

    # Generation knobs the operator may want to clamp.
    max_steps: int = int(os.environ.get("REMOTE_MAX_STEPS", "100"))
    max_pixels: int = int(os.environ.get("REMOTE_MAX_PIXELS", str(2048 * 2048)))

    # Replay-cache window (seconds). Same defaults as QwenCharLoRA.
    replay_window_s: float = float(os.environ.get("REMOTE_REPLAY_WINDOW", "300"))


def load() -> ServerSettings:
    s = ServerSettings()
    s.secrets_dir.mkdir(parents=True, exist_ok=True)
    s.models_root.mkdir(parents=True, exist_ok=True)
    s.transfer_dir.mkdir(parents=True, exist_ok=True)
    if s.model not in ("qwen-image", "qwen-image-edit-2511"):
        raise ValueError(
            f"REMOTE_MODEL={s.model!r} not supported. "
            "Use 'qwen-image' or 'qwen-image-edit-2511'."
        )
    return s


def assert_transfer_is_tmpfs(transfer_dir: Path) -> None:
    """Refuse to serve if the transfer dir isn't a tmpfs mountpoint.

    Called from the FastAPI lifespan after settings load. On Linux we use
    ``/proc/mounts``; on other platforms we skip the check (the server is
    only ever meant to run inside a Linux container)."""
    import sys
    if sys.platform != "linux":
        return                              # local dev on Win/macOS, fine
    target = str(transfer_dir.resolve())
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == target and parts[2] == "tmpfs":
                    return
    except OSError:
        pass
    raise RuntimeError(
        f"transfer_dir {transfer_dir} is not a tmpfs mountpoint. "
        "Refusing to handle transfer files on persistent disk. "
        "Mount via `--tmpfs /run/remote-gen/transfer:size=2g,mode=1700` "
        "or set REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK=1 for local testing."
    )
