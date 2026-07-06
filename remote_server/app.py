"""FastAPI app — ImageTools remote generation server.

Routes:
  GET   /healthz                       unauth
  GET   /v1/pubkey                     unauth — signed server pubkey
  GET   /v1/model                      protected — which model this pod loaded
  POST  /v1/generate/txt2img           protected — returns sealed latent blob
  POST  /v1/generate/img2img           protected — returns sealed latent blob

Pipeline is loaded once at lifespan startup and held in process RAM. Cold
start is governed by REMOTE_MODEL + the model weights on the persistent
volume.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from remote_server import __version__
from remote_server.auth import ProofedRequest, require_proof
from remote_server.crypto import (
    Identity, PublicIdentity, WrappedBlob, b64e, combine_psk, sign_json,
    verify_signed,
)
from remote_server.identity_store import ServerIdentityStore, load_store
from remote_server.lockdown import apply_lockdowns
from remote_server.settings import (
    ServerSettings, assert_transfer_is_tmpfs, load as load_settings,
)
from remote_server.transfer import TransferWorkspace
from remote_server.tripwire import (
    TripwireConfig, get_tripwire, install_tripwire,
)

log = logging.getLogger(__name__)


# ─── module-level singletons (set in lifespan) ───────────────────────


_settings: Optional[ServerSettings] = None
_identity: Optional[ServerIdentityStore] = None
_combined_psk: Optional[bytes] = None
_pipeline: Any = None              # set by pipeline.load_pipeline()
_lora_cache: Any = None            # set in lifespan

# Serialise all generate() calls — diffusers pipelines aren't thread-safe and
# the LoRA hot-swap modifies the in-memory state.
import threading as _threading
_GENERATION_LOCK = _threading.Lock()


def get_settings() -> ServerSettings:
    if _settings is None:
        raise RuntimeError("server not started")
    return _settings


def get_identity() -> ServerIdentityStore:
    if _identity is None:
        raise RuntimeError("identity not loaded")
    return _identity


def get_combined_psk() -> bytes:
    if _combined_psk is None:
        raise RuntimeError("PSK halves not loaded")
    return _combined_psk


def get_pipeline() -> Any:
    if _pipeline is None:
        raise HTTPException(503, "pipeline not yet loaded")
    return _pipeline


def get_lora_cache() -> Any:
    if _lora_cache is None:
        raise HTTPException(503, "lora cache not initialised")
    return _lora_cache


def _check_not_compromised() -> None:
    """Hard-fail every protected route if the tripwire has tripped.

    The error body deliberately does NOT include the trip details (those
    are in the server log), only that the session is unsafe. The client
    sees 503 and should treat the pod as untrusted."""
    tw = get_tripwire()
    if tw is not None and tw.compromised:
        raise HTTPException(
            status_code=503,
            detail="server session was marked compromised; refusing to serve",
        )


# ─── lifespan ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _identity, _combined_psk, _pipeline, _lora_cache

    _settings = load_settings()
    log.info(f"loading server identity from {_settings.secrets_dir}")
    _identity = load_store(_settings.secrets_dir)
    _combined_psk = combine_psk(_identity.server_half, _identity.client_half)
    log.info("PSK halves combined; protected routes active")

    # ─── lockdowns + tripwire ──────────────────────────────────────
    # 1. apply best-effort process hardening (PR_SET_DUMPABLE=0,
    #    PR_SET_NO_NEW_PRIVS=1, RLIMIT_CORE=0, mlock secrets, env scrub)
    if os.environ.get("REMOTE_SKIP_LOCKDOWN") != "1":
        apply_lockdowns(secrets_to_mlock=[_combined_psk, _identity.self_identity.enc_priv,
                                          _identity.self_identity.sig_priv])
    # 2. install the tripwire daemon. It can wipe secrets + flip the
    #    compromised flag on any of: ptrace attach, stranger child of PID 1,
    #    long heartbeat staleness (gcore-style suspend).
    tw = install_tripwire(TripwireConfig())
    tw.register_on_trip(_on_tripwire_trip)
    tw.start()

    # Heartbeat — the tripwire uses this to detect a suspended process.
    async def _heartbeat_loop():
        while True:
            try: tw.heartbeat()
            except Exception: pass
            await asyncio.sleep(1.0)
    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    # Enforce tmpfs for the transfer dir (skipable in tests).
    if os.environ.get("REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK") != "1":
        try:
            assert_transfer_is_tmpfs(_settings.transfer_dir)
            log.info(f"transfer_dir is tmpfs: {_settings.transfer_dir}")
        except RuntimeError as e:
            log.error(str(e))
            raise

    # LoRA cache lives under transfer_dir (tmpfs) — wiped with the rest of
    # the workspace on shutdown.
    from remote_server.lora_cache import LoRACache
    _lora_cache = LoRACache(_settings.transfer_dir)

    # Eager-load the pipeline. NOT an async import — diffusers loads on
    # the asyncio worker thread, which is fine.
    if os.environ.get("REMOTE_SKIP_PIPELINE_LOAD") == "1":
        log.warning("REMOTE_SKIP_PIPELINE_LOAD=1 — server runs without a pipeline")
    else:
        from remote_server.pipeline import load_pipeline
        log.info(f"loading pipeline: {_settings.model}")
        _pipeline = await asyncio.to_thread(load_pipeline, _settings)
        log.info("pipeline loaded; ready for /v1/generate/*")

    yield

    # ─── teardown ─────────────────────────────────────────────────
    heartbeat_task.cancel()
    try: await heartbeat_task
    except (asyncio.CancelledError, Exception): pass
    try: tw.stop()
    except Exception: pass

    if _pipeline is not None:
        try: _pipeline.unload()
        except Exception: log.exception("pipeline unload failed")
        _pipeline = None

    # final wipe of secrets — last gasp before process exits
    _wipe_secrets()


def _on_tripwire_trip(report) -> None:
    """Tripwire callback. Wipe everything sensitive from process RAM and
    flip the compromised flag so all subsequent requests 503."""
    log.error(f"compromise detected ({report.trigger}); wiping secrets")
    _wipe_secrets()


def _wipe_secrets() -> None:
    """Best-effort scrub of in-RAM secret material. Python bytes are
    immutable; the actual overwrite only works if libsodium's sodium_munlock
    exposes a backing pointer (PyNaCl provides ``sodium_memzero``). For
    plain bytes we drop the reference + suggest GC, which is what we can
    do from pure-Python."""
    global _combined_psk, _identity, _lora_cache
    try:
        if _combined_psk is not None:
            try:
                import nacl.bindings as nbi
                nbi.sodium_memzero(_combined_psk)
            except Exception: pass
            _combined_psk = None
    except Exception: pass
    try:
        if _identity is not None:
            for attr in ("enc_priv", "sig_priv"):
                k = getattr(_identity.self_identity, attr, None)
                if k:
                    try:
                        import nacl.bindings as nbi
                        nbi.sodium_memzero(k)
                    except Exception: pass
            _identity = None
    except Exception: pass
    try:
        if _lora_cache is not None:
            for sha in list(_lora_cache._entries):
                _lora_cache._entries.pop(sha, None)
    except Exception: pass
    import gc; gc.collect()


app = FastAPI(
    title="ImageTools remote-generation server",
    version=__version__,
    lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,
)


# Explicit allowlist of routes that may be reached without a PSK proof. Every
# other path is default-denied by the middleware below, so a newly-added route
# that forgets ``Depends(require_proof)`` still can't be hit unauthenticated —
# and the tripwire compromise gate is enforced in one place rather than relying
# on each handler to remember it.
_UNAUTH_PATHS = frozenset({"/healthz", "/v1/pubkey"})


@app.middleware("http")
async def _default_deny(request, call_next):
    if request.method == "OPTIONS" or request.url.path in _UNAUTH_PATHS:
        return await call_next(request)
    # 1. central compromise gate — untrusted pod refuses every protected route.
    tw = get_tripwire()
    if tw is not None and tw.compromised:
        return JSONResponse(
            status_code=503,
            content={"detail": "server session was marked compromised; refusing to serve"},
        )
    # 2. default-deny: proof headers must be present. Full cryptographic
    #    verification still happens in require_proof (defense in depth); this
    #    just guarantees no route is reachable without presenting a proof.
    if not request.headers.get("X-RGEN-Request-Id") or not request.headers.get("X-RGEN-Proof"):
        return JSONResponse(status_code=401, content={"detail": "missing PSK proof headers"})
    return await call_next(request)


# ─── unauth routes ───────────────────────────────────────────────────


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status":  "ok",
        "version": __version__,
        "model":   (_settings.model if _settings else None),
        "ready":   _pipeline is not None,
    }


@app.get("/v1/pubkey")
async def pubkey() -> dict:
    ident = get_identity()
    payload = ident.self_identity.public().to_dict()
    payload["model"] = get_settings().model
    signed = sign_json(ident.self_identity.sig_priv, payload)
    return JSONResponse({
        "payload":    payload,
        "signed_b64": b64e(signed),
    })


# ─── protected routes ────────────────────────────────────────────────


@app.get("/v1/model")
async def get_model(req: ProofedRequest = Depends(require_proof)) -> dict:
    s = get_settings()
    return {
        "model":      s.model,
        "max_steps":  s.max_steps,
        "max_pixels": s.max_pixels,
        "ready":      _pipeline is not None,
    }


@app.get("/v1/server_status")
async def server_status(req: ProofedRequest = Depends(require_proof)) -> dict:
    """Tripwire / lockdown / cache state at a glance. The client polls this
    BEFORE the first generation; if ``compromised`` is true or the trigger
    field is set, the client refuses to upload."""
    tw = get_tripwire()
    return {
        "tripwire_armed":      tw is not None and tw._thread is not None,
        "compromised":         bool(tw and tw.compromised),
        "compromise_trigger":  (tw.report.trigger if tw and tw.report else None),
        "loras_cached":        len(_lora_cache._entries) if _lora_cache else 0,
        "pipeline_ready":      _pipeline is not None,
    }


@app.get("/v1/loras")
async def list_loras(req: ProofedRequest = Depends(require_proof)) -> dict:
    """List every LoRA currently in the in-RAM cache, sorted by last-used
    DESC. Used to confirm which adapters survived LRU eviction without
    having to re-upload to test."""
    cache = get_lora_cache()
    entries = cache.list_entries()
    entries.sort(key=lambda e: e.last_used, reverse=True)
    return {
        "count": len(entries),
        "entries": [
            {"sha256":    e.sha256,
             "size":      e.size,
             "last_used": e.last_used}
            for e in entries
        ],
    }


@app.delete("/v1/loras/{sha256}")
async def evict_lora(sha256: str,
                     req: ProofedRequest = Depends(require_proof)) -> dict:
    """Force-evict a cached LoRA. Useful for rotating credentials or wiping
    a specific adapter without restarting the pod."""
    cache = get_lora_cache()
    ok = cache.evict(sha256.lower())
    return {"evicted": ok, "sha256": sha256.lower()}


@app.post("/v1/loras/download")
async def download_lora(req: ProofedRequest = Depends(require_proof)) -> dict:
    """Pull a LoRA file from HuggingFace or Civitai DIRECTLY to the pod,
    cache it in RAM, and return the sha256 the client can reference from
    later ``generate`` calls. The bytes never traverse the client's uplink.

    Body is a sealed JSON envelope of shape::

        {
            "manifest_signed_hex": "<Ed25519 signature>",
            "payload": {
                "source": "hf" | "civitai",
                # HF:
                "repo":     "user/repo",
                "filename": "pytorch_lora_weights.safetensors",
                "revision": "main",            # optional
                "hf_token": "hf_xxxxx",        # optional override
                # Civitai (any one of):
                "model_id":    12345,
                "version_id":  67890,
                "civitai_url": "https://civitai.com/models/12345?modelVersionId=67890",
                "civitai_token": "civitai_api_key"   # optional override
            }
        }

    Tokens are read once, used for the upstream call, then dropped from
    process memory after the response is sent."""
    import hashlib
    import json
    from remote_server.lora_fetch import (
        LoRAFetchError, fetch_from_civitai, fetch_from_hf,
    )

    ident = get_identity()
    try:
        opened = WrappedBlob(req.body).open(ident.self_identity.enc_priv)
    except Exception:
        raise HTTPException(400, "ciphertext could not be opened by server pubkey")
    try:
        envelope = json.loads(opened)
        manifest_signed = bytes.fromhex(envelope["manifest_signed_hex"])
        payload = envelope["payload"]
    except Exception as e:
        raise HTTPException(400, f"malformed envelope: {e!r}")
    try:
        manifest = verify_signed(ident.paired_client.sig_pub, manifest_signed)
    except Exception:
        raise HTTPException(401, "manifest signature invalid")
    # bind manifest to payload
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("payload_sha256") != expected:
        raise HTTPException(400, "payload_sha256 mismatch")

    source = payload.get("source")
    cache  = get_lora_cache()
    try:
        if source == "hf":
            res = await asyncio.to_thread(
                fetch_from_hf, cache,
                repo=payload.get("repo", ""),
                filename=payload.get("filename", ""),
                revision=payload.get("revision"),
                token=payload.get("hf_token"),
            )
        elif source == "civitai":
            res = await asyncio.to_thread(
                fetch_from_civitai, cache,
                model_id=payload.get("model_id"),
                version_id=payload.get("version_id"),
                civitai_url=payload.get("civitai_url"),
                token=payload.get("civitai_token"),
            )
        else:
            raise HTTPException(400, f"unknown source {source!r}; use 'hf' or 'civitai'")
    except LoRAFetchError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("lora download failed")
        raise HTTPException(500, f"download failed: {e!r}")
    finally:
        # scrub the tokens from the parsed payload so they don't linger in
        # the envelope JSON's holding memory longer than necessary.
        payload["hf_token"] = ""
        payload["civitai_token"] = ""

    return {
        "sha256":       res.sha256,
        "bytes_cached": res.bytes_read,
        "source":       res.source,
    }


@app.post("/v1/generate/txt2img")
async def txt2img(req: ProofedRequest = Depends(require_proof)) -> Response:
    """Decrypt the request envelope, run the pipeline, return a sealed latent.

    Request envelope (decrypted via server X25519 priv): JSON

        {
            "manifest_signed_hex": "<Ed25519-signed manifest>",
            "payload":             {
                "prompt": "...",
                "negative_prompt": "...",
                "width": 1024, "height": 1024,
                "steps": 20, "cfg": 4.0, "seed": 12345,
                "loras": [...]                                # optional
            }
        }

    Response: sealed binary (WrappedBlob to client.enc_pub) wrapping the
    safetensors latent blob.
    """
    s = get_settings()
    ident = get_identity()
    if s.model != "qwen-image":
        raise HTTPException(409, f"this pod is loaded for {s.model!r}; "
                                  "use /v1/generate/img2img for edit models")
    return await _run_generation(req, ident, mode="txt2img")


@app.post("/v1/generate/img2img")
async def img2img(req: ProofedRequest = Depends(require_proof)) -> Response:
    """Same envelope shape as txt2img, plus a ``control_image_b64`` field
    holding the PNG bytes of the source image (base64-encoded)."""
    s = get_settings()
    ident = get_identity()
    if s.model != "qwen-image-edit-2511":
        raise HTTPException(409, f"this pod is loaded for {s.model!r}; "
                                  "use /v1/generate/txt2img for the base model")
    return await _run_generation(req, ident, mode="img2img")


# ─── generation runner ───────────────────────────────────────────────


async def _run_generation(
    req: ProofedRequest, ident: ServerIdentityStore, mode: str,
) -> Response:
    """Shared driver for txt2img/img2img: opens the envelope, validates
    the signed manifest, hands off to the pipeline, seals the latent.

    LoRA handling:
      payload.loras = [{"sha256":..., "weight":..., "bytes_b64":?}]
      - cached sha → used as-is
      - bytes_b64 supplied → verified + cached + used
      - neither → 404 with {"missing_loras": [...]} so client re-sends bytes
    """
    from remote_server.latent_blob import pack_latent_blob
    from remote_server.lora_cache import (
        LoRACacheError, parse_loras_payload, resolve_against_cache,
    )

    try:
        opened = WrappedBlob(req.body).open(ident.self_identity.enc_priv)
    except Exception as e:
        log.warning(f"sealed envelope open failed: {e!r}")
        raise HTTPException(400, "ciphertext could not be opened by server pubkey")

    try:
        envelope = json.loads(opened)
        manifest_signed = bytes.fromhex(envelope["manifest_signed_hex"])
        payload = envelope["payload"]
    except Exception as e:
        raise HTTPException(400, f"malformed envelope: {e!r}")

    try:
        manifest = verify_signed(ident.paired_client.sig_pub, manifest_signed)
    except Exception:
        raise HTTPException(401, "manifest signature invalid")

    # bind manifest to payload — prevents the encrypted-payload-with-different-
    # signed-manifest attack
    import hashlib
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("payload_sha256") != expected:
        raise HTTPException(400, "payload_sha256 mismatch")

    pipeline = get_pipeline()
    settings = get_settings()
    cache    = get_lora_cache()

    # Validate clamps
    steps = int(payload.get("steps", 20))
    if steps < 1 or steps > settings.max_steps:
        raise HTTPException(400, f"steps {steps} outside [1, {settings.max_steps}]")
    w = int(payload.get("width", 1024)); h = int(payload.get("height", 1024))
    if w * h > settings.max_pixels:
        raise HTTPException(400, f"width*height={w*h} exceeds max_pixels={settings.max_pixels}")

    # ─── LoRA resolution (BEFORE we acquire the generation lock so
    #     cache misses fail fast without blocking other requests) ────
    try:
        lora_reqs = parse_loras_payload(payload.get("loras") or [])
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        resolved, weights, missing = resolve_against_cache(
            cache, payload.get("loras") or [], lora_reqs,
        )
    except LoRACacheError as e:
        raise HTTPException(400, str(e))

    if missing:
        # 404 — client should re-issue the same request with bytes_b64
        # filled in for these shas.
        return JSONResponse(
            status_code=404,
            content={
                "detail":        "lora cache miss",
                "missing_loras": missing,
            },
        )

    # ─── exclusive generate() — diffusers + LoRA hot-swap aren't thread-safe ──
    def _do() -> tuple[Any, dict[str, Any]]:
        with _GENERATION_LOCK:
            # Reconcile the pipeline to exactly this request's LoRA set. An
            # empty ``resolved`` clears any adapters left by a prior request;
            # an unchanged set is a no-op (no reload). We intentionally do NOT
            # clear on the happy path so a queued batch sharing one LoRA pays
            # the load cost once instead of per job.
            try:
                pipeline.apply_loras(resolved, weights)
            except Exception as e:
                try: pipeline.clear_loras()
                except Exception: pass
                raise HTTPException(500, f"lora apply failed: {e}") from e
            with TransferWorkspace(settings.transfer_dir) as ws:
                latent, gen_meta = pipeline.generate(payload, ws, mode)
            if resolved:
                gen_meta["applied_loras"] = [
                    {"sha256": r.sha256, "weight": w}
                    for r, w in zip(resolved, weights)
                ]
            return latent, gen_meta

    try:
        latent, gen_meta = await asyncio.to_thread(_do)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("generation failed")
        raise HTTPException(500, f"generation failed: {e!r}")

    blob = pack_latent_blob(latent, gen_meta)
    client_pub = ident.paired_client.enc_pub
    sealed = WrappedBlob.seal(client_pub, blob).blob

    return Response(content=sealed, media_type="application/octet-stream")
