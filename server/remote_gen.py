"""Client-side bridge to the remote_server.

Public API (what the MCP tools call):

    remote_qwen_txt2img(prompt, negative_prompt, width, height, steps, cfg, seed)
        → PIL.Image.Image

    remote_qwen_edit(control_image, prompt, negative_prompt, width, height,
                     steps, cfg, seed) → PIL.Image.Image

Flow:
  1. Load client config from ~/.image-tools-remote/config.json
     (server_url, server_fingerprint, paths).
  2. Build signed manifest + JSON envelope.
  3. Seal with server X25519 pubkey, POST to /v1/generate/*.
  4. Decrypt response with client X25519 priv, unpack latent.
  5. Lazy-load local VAE (qwen_image_vae.safetensors) on first call.
  6. Decode latent → PIL.Image, return.

The local VAE is held in process RAM across calls — first call pays the
~3-5s load, subsequent calls decode in ~0.5s on a 4090.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image


log = logging.getLogger("remote_gen")


# ─── client config ────────────────────────────────────────────────────


def _config_dir() -> Path:
    env = os.environ.get("IMAGE_TOOLS_REMOTE_DIR")
    if env: return Path(env)
    return Path.home() / ".image-tools-remote"


@dataclass
class ClientConfig:
    server_url: str
    server_fingerprint: str
    config_dir: Path = field(default_factory=_config_dir)
    request_timeout_s: float = 600.0      # generation runs up to a few minutes

    @classmethod
    def load(cls) -> "ClientConfig":
        d = _config_dir()
        p = d / "config.json"
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}. Run the bootstrap script + copy "
                "client-secrets/* into the directory.")
        body = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            server_url=body["server_url"],
            server_fingerprint=body["server_fingerprint"],
            config_dir=d,
            request_timeout_s=float(body.get("request_timeout_s", 600.0)),
        )


# ─── crypto context — held in process RAM ────────────────────────────


_lock = threading.Lock()
_session_state: dict[str, Any] = {}


def _ensure_session() -> dict[str, Any]:
    """Idempotent: loads ClientConfig, identity, halves, derives combined PSK.
    Caches everything in module-level dict so subsequent calls are free."""
    with _lock:
        if _session_state:
            return _session_state

        try:
            import httpx  # noqa: F401  (used by every transport call)
            from remote_server.crypto import (
                Identity, PublicIdentity, combine_psk, read_half,
            )
        except ImportError as e:
            raise RuntimeError(
                "remote generation requires the optional 'remote' dependencies "
                "(httpx, pynacl). Install them with: pip install "
                "image-tools-mcp[remote]"
            ) from e

        cfg = ClientConfig.load()

        # Load both halves
        my_half_path = cfg.config_dir / "half.bin"
        srv_half_path = cfg.config_dir / "server_half.bin"
        if not my_half_path.exists() or not srv_half_path.exists():
            raise FileNotFoundError(
                f"missing half.bin or server_half.bin under {cfg.config_dir}")
        my_half = read_half(my_half_path)
        srv_half = read_half(srv_half_path)
        combined = combine_psk(srv_half, my_half)

        # Identity + paired server pubkey
        ident_path = cfg.config_dir / "identity.json"
        peer_path  = cfg.config_dir / "server_peer.json"
        ident = Identity.from_dict(json.loads(ident_path.read_text(encoding="utf-8")))
        peer  = PublicIdentity.from_dict(json.loads(peer_path.read_text(encoding="utf-8")))
        if peer.fingerprint() != cfg.server_fingerprint:
            raise RuntimeError(
                f"server_peer.json fingerprint mismatch with config.json: "
                f"{peer.fingerprint()} vs {cfg.server_fingerprint}")

        _session_state.update(
            cfg=cfg, identity=ident, server=peer, combined_psk=combined,
        )
        return _session_state


# ─── lazy local VAE ───────────────────────────────────────────────────


_vae_lock = threading.Lock()
_vae_cache: dict[str, Any] = {}        # vae_filename → loaded vae


def _load_local_vae(filename: str):
    """Find the VAE locally and load it. Looks under common ComfyUI paths."""
    import torch
    with _vae_lock:
        if filename in _vae_cache:
            return _vae_cache[filename]

        # Resolve the VAE dir through the shared config surface (overridable
        # via configure_paths / IMAGETOOLS_COMFYUI_VAE_DIR) rather than a
        # hardcoded path, so moving the ComfyUI tree only needs one change.
        from . import server_config
        candidates = [
            server_config.get_path("comfyui_vae_dir") / filename,
            Path.home() / ".cache" / "huggingface" / filename,
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"local VAE {filename} not found. Looked in: "
                + ", ".join(str(c) for c in candidates))

        log.info(f"loading local VAE: {path}")
        # Must match the VAE the server generated with — the pod uses
        # AutoencoderKLQwenImage (a 3D/temporal VAE), NOT AutoencoderKLWan.
        # Decoding Qwen latents with the Wan VAE produces shape errors / garbage.
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_single_file(str(path), torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = vae.to(device)
        vae.eval()
        _vae_cache[filename] = vae
        return vae


def _unpack_qwen_latents(latent, height: int, width: int, scale_factor: int):
    """Reverse QwenImagePipeline._pack_latents: (B, num_patches, C*4) packed
    latents → (B, C, T=1, H, W) ready for the VAE. Mirrors diffusers so the
    client decode matches server generation."""
    b, num_patches, c4 = latent.shape
    h = 2 * (int(height) // (scale_factor * 2))
    w = 2 * (int(width) // (scale_factor * 2))
    latent = latent.view(b, h // 2, w // 2, c4 // 4, 1, 2, 2)
    latent = latent.permute(0, 3, 4, 1, 5, 2, 6)
    return latent.reshape(b, c4 // 4, 1, h, w)


def _decode_latent(latent, metadata: dict[str, Any]) -> Image.Image:
    # NOTE: this decode path (Qwen packed-latent unpack + per-channel
    # mean/std denormalization + temporal-VAE decode) mirrors diffusers'
    # QwenImagePipeline but has not yet been smoke-tested against a live pod.
    import torch
    vae = _load_local_vae(metadata.get("vae_required", "qwen_image_vae.safetensors"))
    cfg = vae.config
    with torch.inference_mode():
        x = latent.to(vae.device, dtype=vae.dtype)
        # Packed latents arrive as (B, num_patches, C*4); unpack to 5D.
        if x.ndim == 3:
            scale_factor = 2 ** len(getattr(cfg, "temperal_downsample", [0, 0, 0]))
            x = _unpack_qwen_latents(
                x, metadata.get("height", 1024), metadata.get("width", 1024), scale_factor)
        elif x.ndim == 4:                      # (B, C, H, W) → add temporal axis
            x = x.unsqueeze(2)

        # QwenImage denormalizes with per-channel latents_mean/std; fall back to
        # the single scaling factor only if the config doesn't provide them.
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if mean is not None and std is not None:
            z = cfg.z_dim
            lm = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, z, 1, 1, 1)
            ls = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, z, 1, 1, 1)
            x = x / ls + lm
        else:
            x = x / float(metadata.get("scaling", 0.18215))

        decoded = vae.decode(x, return_dict=False)[0]

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    frame = decoded[0, :, 0] if decoded.ndim == 5 else decoded[0]   # first frame
    arr = (frame.permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


# ─── transport ────────────────────────────────────────────────────────


def _signed_headers(combined_psk: bytes, body: bytes) -> dict[str, str]:
    from remote_server.crypto import compute_request_proof, sha256_hex
    rid = uuid.uuid4().hex
    h = sha256_hex(body)
    proof = compute_request_proof(combined_psk, rid, h)
    return {"X-RGEN-Request-Id": rid, "X-RGEN-Proof": proof,
            "Content-Type": "application/octet-stream"}


def _post_raw(path: str, body: bytes, timeout: float) -> tuple[bytes, int]:
    """Lower-level POST that returns (bytes, status_code) without raising."""
    import httpx
    sess = _ensure_session()
    headers = _signed_headers(sess["combined_psk"], body)
    with httpx.Client(base_url=sess["cfg"].server_url, timeout=timeout) as cli:
        r = cli.post(path, content=body, headers=headers)
    return r.content, r.status_code


def _signed_request(method: str, path: str, *, timeout: float = 30.0,
                    raise_for_status: bool = True):
    """Signed GET/DELETE with an empty body. One place owns the client +
    proof-header + error-raise ritual so every bodyless remote call behaves
    the same (timeout, auth, status handling)."""
    import httpx
    sess = _ensure_session()
    headers = _signed_headers(sess["combined_psk"], b"")
    with httpx.Client(base_url=sess["cfg"].server_url, timeout=timeout) as cli:
        r = cli.request(method, path, headers=headers)
    if raise_for_status and r.status_code >= 400:
        raise RuntimeError(f"server returned {r.status_code}: {r.text}")
    return r


def _build_envelope(payload: dict[str, Any]) -> bytes:
    """Sign + seal an envelope for the server."""
    from remote_server.crypto import (
        WrappedBlob, make_job_manifest, sha256_hex, sign_json,
    )
    sess = _ensure_session()
    # Canonical-JSON hash binds the payload to the signed manifest.
    payload_canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_sha = sha256_hex(payload_canonical)
    manifest = make_job_manifest(
        sess["identity"].public(),
        job_params={"kind": payload.get("_kind", "generate")},
        dataset_sha256=payload_sha,
    )
    manifest["payload_sha256"] = payload_sha       # also surfaced top-level for server check
    manifest_signed = sign_json(sess["identity"].sig_priv, manifest)
    envelope = {
        "manifest_signed_hex": manifest_signed.hex(),
        "payload":             payload,
    }
    return WrappedBlob.seal(
        sess["server"].enc_pub,
        json.dumps(envelope).encode("utf-8"),
    ).blob


def _open_sealed_latent(sealed: bytes) -> tuple[Any, dict[str, Any]]:
    from remote_server.crypto import WrappedBlob
    from remote_server.latent_blob import unpack_latent_blob
    sess = _ensure_session()
    blob = WrappedBlob(sealed).open(sess["identity"].enc_priv)
    return unpack_latent_blob(blob)


# ─── LoRA helpers (client side) ──────────────────────────────────────


@dataclass
class LoRASpec:
    """A LoRA the client wants applied. Either ``path`` OR ``bytes`` must be set."""
    weight: float = 1.0
    path: Optional[Path] = None
    bytes: Optional[bytes] = None       # raw .safetensors bytes
    sha256: Optional[str] = None        # filled in by _prepare_loras


_LORA_SUFFIXES = (".safetensors", ".pt")
# path.resolve() → (size, mtime, sha256); skip the re-read+re-hash of an
# unchanged multi-hundred-MB LoRA on every generation call.
_lora_hash_cache: dict[str, tuple[int, float, str]] = {}


def _lora_roots() -> list[Path]:
    from . import server_config
    raw = server_config.get("lora_dirs")
    return [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]


def _validate_lora_path(path: Path) -> Path:
    """Chokepoint for LOCAL LoRA file reads. Prevents a prompt-injected path
    from exfiltrating arbitrary on-disk files (credentials, key halves) to the
    remote pod: the file must be a real .safetensors/.pt under a configured
    lora_dirs root."""
    p = path.expanduser().resolve()
    if p.suffix.lower() not in _LORA_SUFFIXES:
        raise ValueError(
            f"LoRA path must be a {' or '.join(_LORA_SUFFIXES)} file, got {p.name!r}.")
    roots = _lora_roots()
    if not roots:
        raise PermissionError(
            "local LoRA uploads are disabled (lora_dirs is empty). Set the "
            "'lora_dirs' config to a directory of LoRAs, or reference a "
            "server-cached LoRA by sha256 instead.")
    if not any(p == r or r in p.parents for r in roots):
        raise PermissionError(
            f"LoRA path {p} is outside the allowed lora_dirs roots "
            f"({', '.join(map(str, roots))}). Move the file there or widen "
            "the 'lora_dirs' config.")
    if not p.is_file():
        raise FileNotFoundError(f"LoRA file missing: {p}")
    return p


def _hash_lora_file(p: Path) -> str:
    from remote_server.crypto import sha256_hex
    st = p.stat()
    key = str(p)
    cached = _lora_hash_cache.get(key)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime:
        return cached[2]
    digest = sha256_hex(p)  # streams the file; no full read into RAM
    _lora_hash_cache[key] = (st.st_size, st.st_mtime, digest)
    return digest


def _prepare_loras(specs: list[Any]) -> list[LoRASpec]:
    """Normalize LoRA specs and resolve each to a sha256 handle.

    Accepts (LoRASpec, dict, str/Path). Three reference modes:
      * local file  — ``{"path": "...safetensors", "weight": w}`` or a bare
        path string; validated, hashed (bytes deferred until the server 404s).
      * raw bytes   — ``{"bytes": b"...", "weight": w}``; hashed directly.
      * by-reference — ``{"sha256": "<hex>", "weight": w}`` for a LoRA already
        cached on the pod; no local file needed and no bytes leave the PC.

    A caller-supplied sha256 is verified against the file/bytes (mismatch is an
    error) rather than silently overwritten.
    """
    from remote_server.crypto import sha256_hex

    out: list[LoRASpec] = []
    for s in specs or []:
        if isinstance(s, LoRASpec):
            spec = s
        elif isinstance(s, (str, Path)):
            spec = LoRASpec(path=Path(s))
        elif isinstance(s, dict):
            spec = LoRASpec(
                weight=float(s.get("weight", 1.0)),
                path=Path(s["path"]) if s.get("path") else None,
                bytes=s.get("bytes"),
                sha256=s.get("sha256"),
            )
        else:
            raise TypeError(f"unsupported lora spec type: {type(s)!r}")

        if spec.bytes is not None:
            computed = sha256_hex(spec.bytes)
        elif spec.path is not None:
            spec.path = _validate_lora_path(spec.path)
            computed = _hash_lora_file(spec.path)
        elif spec.sha256:
            out.append(spec)          # server-cached by-reference; nothing to read
            continue
        else:
            raise ValueError(
                "LoRA spec needs one of: 'path', 'bytes', or 'sha256' "
                "(a server-cached reference).")

        if spec.sha256 and spec.sha256.lower() != computed.lower():
            raise ValueError(
                f"LoRA sha256 mismatch: caller gave {spec.sha256}, "
                f"content hashes to {computed}.")
        spec.sha256 = computed
        out.append(spec)
    return out


def _build_loras_payload(specs: list[LoRASpec], include_bytes_for: set[str]) -> list[dict]:
    """Render the ``payload.loras`` list. For each sha in ``include_bytes_for``
    we attach the base64 bytes (loading from disk here, only when the server
    actually asked); others are by-reference only."""
    out = []
    for s in specs:
        entry: dict[str, Any] = {"sha256": s.sha256, "weight": s.weight}
        if s.sha256 in include_bytes_for:
            data = s.bytes
            if data is None and s.path is not None:
                data = s.path.read_bytes()   # deferred: read only on server cache-miss
            if not data:
                raise RuntimeError(
                    f"server needs the bytes for LoRA {s.sha256} but it was "
                    "referenced by sha256 only and isn't cached on the pod. "
                    "Pass a local 'path'/'bytes', or cache it server-side via "
                    "remote_lora_download_hf / remote_lora_download_civitai.")
            entry["bytes_b64"] = base64.b64encode(data).decode("ascii")
        out.append(entry)
    return out


def _submit_with_lora_retry(
    path: str, build_payload, lora_specs: list[LoRASpec],
) -> bytes:
    """Generic submit driver with the 404-then-retry-with-bytes flow.

    ``build_payload(loras_field)`` is a callable returning the full payload
    dict given the rendered ``loras`` list — lets the caller (txt2img /
    img2img) keep the rest of the payload static while we vary just the
    LoRA section."""
    sess = _ensure_session()
    timeout = sess["cfg"].request_timeout_s

    # Attempt 1: by-sha only
    include_bytes_for: set[str] = set()
    while True:
        payload = build_payload(_build_loras_payload(lora_specs, include_bytes_for))
        body = _build_envelope(payload)
        raw, status = _post_raw(path, body, timeout=timeout)
        if status == 200:
            return raw
        if status == 404:
            try: detail = json.loads(raw)
            except Exception: detail = {}
            missing = detail.get("missing_loras") or []
            if not missing:
                raise RuntimeError("server returned 404 with no missing_loras list")
            new = set(missing) - include_bytes_for
            if not new:
                # we already supplied these and the server still wants them
                raise RuntimeError(f"server keeps requesting bytes we sent: {missing}")
            log.info(f"server missing {len(new)} loras; retrying with bytes")
            include_bytes_for |= new
            continue
        # any other non-2xx → raise
        try: msg = json.loads(raw).get("detail", "")
        except Exception: msg = raw[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned {status}: {msg}")


# ─── public entry points used by the MCP tools ───────────────────────


def remote_qwen_txt2img(
    prompt: str,
    *,
    negative_prompt: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 4.0,
    seed: int = 0,
    loras: list[Any] | None = None,
) -> Image.Image:
    """``loras`` accepts str paths, dicts {path, weight}, or LoRASpec objects."""
    from . import remote_session
    remote_session.check_not_compromised()
    specs = _prepare_loras(loras or [])

    def build_payload(loras_field: list[dict]) -> dict:
        return {
            "_kind":           "txt2img",
            "prompt":          prompt,
            "negative_prompt": negative_prompt or "",
            "width":           int(width),
            "height":          int(height),
            "steps":           int(steps),
            "cfg":             float(cfg),
            "seed":            int(seed),
            "loras":           loras_field,
        }

    sealed = _submit_with_lora_retry("/v1/generate/txt2img", build_payload, specs)
    latent, metadata = _open_sealed_latent(sealed)
    return _decode_latent(latent, metadata)


def remote_qwen_edit(
    control_image: Image.Image,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int = 25,
    cfg: float = 4.0,
    seed: int = 0,
    loras: list[Any] | None = None,
) -> Image.Image:
    from . import remote_session
    remote_session.check_not_compromised()
    specs = _prepare_loras(loras or [])
    buf = io.BytesIO()
    control_image.save(buf, format="PNG")
    control_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    def build_payload(loras_field: list[dict]) -> dict:
        return {
            "_kind":             "img2img",
            "prompt":            prompt,
            "negative_prompt":   negative_prompt or "",
            "width":             int(width or control_image.width),
            "height":            int(height or control_image.height),
            "steps":             int(steps),
            "cfg":               float(cfg),
            "seed":              int(seed),
            "control_image_b64": control_b64,
            "loras":             loras_field,
        }

    sealed = _submit_with_lora_retry("/v1/generate/img2img", build_payload, specs)
    latent, metadata = _open_sealed_latent(sealed)
    return _decode_latent(latent, metadata)


# ─── LoRA management helpers ─────────────────────────────────────────


def remote_list_loras() -> dict[str, Any]:
    """GET /v1/loras — see what the remote has cached."""
    return _signed_request("GET", "/v1/loras").json()


def remote_evict_lora(sha256: str) -> dict[str, Any]:
    """DELETE /v1/loras/{sha} — force-clear one entry. Returns {evicted, sha256}."""
    return _signed_request("DELETE", f"/v1/loras/{sha256.lower()}").json()


def remote_download_lora(
    *,
    source:        str,
    repo:          str | None = None,
    filename:      str | None = None,
    revision:      str | None = None,
    hf_token:      str | None = None,
    model_id:      int | None = None,
    version_id:    int | None = None,
    civitai_url:   str | None = None,
    civitai_token: str | None = None,
) -> dict[str, Any]:
    """Tell the SERVER to fetch a LoRA from HF or Civitai and cache it
    locally. Returns ``{sha256, bytes_cached, source}``.  The bytes never
    cross the client's uplink — only the sha256 + a small JSON envelope."""
    if source not in ("hf", "civitai"):
        raise ValueError(f"source must be 'hf' or 'civitai', got {source!r}")
    payload: dict[str, Any] = {"source": source}
    if source == "hf":
        if not repo or not filename:
            raise ValueError("hf download requires repo + filename")
        payload.update({"repo": repo, "filename": filename})
        if revision: payload["revision"] = revision
        if hf_token: payload["hf_token"] = hf_token
    else:
        if not (model_id or version_id or civitai_url):
            raise ValueError("civitai download requires model_id, version_id, or civitai_url")
        if model_id:    payload["model_id"]    = int(model_id)
        if version_id:  payload["version_id"]  = int(version_id)
        if civitai_url: payload["civitai_url"] = civitai_url
        if civitai_token: payload["civitai_token"] = civitai_token

    sess = _ensure_session()
    body = _build_envelope(payload)
    raw, status = _post_raw("/v1/loras/download", body,
                            timeout=sess["cfg"].request_timeout_s)
    if status >= 400:
        try: msg = json.loads(raw).get("detail", "")
        except Exception: msg = raw[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned {status}: {msg}")
    return json.loads(raw)


def remote_server_status() -> dict[str, Any]:
    """GET /v1/server_status — tripwire + lockdown state.

    Returns ``{tripwire_armed, compromised, compromise_trigger, loras_cached,
    pipeline_ready}``. If ``compromised`` is True, the pod has detected an
    intrusion (docker exec, ptrace, or suspend) and is refusing to serve;
    you should treat it as untrusted and provision a fresh pod."""
    r = _signed_request("GET", "/v1/server_status", raise_for_status=False)
    if r.status_code >= 400:
        # Surface an error/503 as a structured response (not an exception) so
        # the watcher and callers can distinguish unreachable from compromised.
        try: detail = r.json().get("detail", "")
        except Exception: detail = r.text
        return {"unreachable_or_compromised": True, "detail": detail,
                "status": r.status_code}
    return r.json()


# ─── verify_server (for setup CLI / status calls) ─────────────────────


def verify_server() -> dict[str, str]:
    """Fetch /v1/pubkey, verify Ed25519 signature, compare fingerprint."""
    import httpx
    from remote_server.crypto import b64d, verify_signed
    sess = _ensure_session()
    with httpx.Client(base_url=sess["cfg"].server_url, timeout=30.0) as cli:
        r = cli.get("/v1/pubkey")
        r.raise_for_status()
        body = r.json()
    signed = b64d(body["signed_b64"])
    payload = verify_signed(sess["server"].sig_pub, signed)
    fp_expect = sess["server"].fingerprint()
    if payload["fingerprint"] != fp_expect:
        raise RuntimeError(
            f"server fingerprint mismatch: pinned={fp_expect}, "
            f"server={payload['fingerprint']}")
    return {
        "label":       payload["label"],
        "fingerprint": payload["fingerprint"],
        "model":       payload.get("model", "<unknown>"),
    }
