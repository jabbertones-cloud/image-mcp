"""End-to-end FastAPI tests with a stub pipeline.

The stub pipeline returns a deterministic small tensor instead of running
diffusers. This exercises the full crypto chain, route wiring, and
sealed-latent return without needing the GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# project root on path so 'remote_server' imports
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from remote_server.crypto import (
    Identity, WrappedBlob, b64d, b64e, compute_request_proof, generate_psk,
    make_job_manifest, sha256_hex, sign_json, split_psk, verify_signed,
    write_half,
)


# ─── fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    secrets_dir = tmp_path / "secrets"; secrets_dir.mkdir()
    models_root = tmp_path / "models"; models_root.mkdir()
    transfer = tmp_path / "transfer"; transfer.mkdir()

    psk = generate_psk()
    s_half, c_half = split_psk(psk)
    write_half(secrets_dir / "half.bin", s_half)
    write_half(secrets_dir / "client_half.bin", c_half)

    server_id = Identity.generate("test-server")
    client_id = Identity.generate("test-client")
    (secrets_dir / "identity.json").write_text(
        json.dumps(server_id.to_dict()), encoding="utf-8")
    (secrets_dir / "client_peer.json").write_text(
        json.dumps(client_id.public().to_dict()), encoding="utf-8")

    monkeypatch.setenv("REMOTE_SECRETS_DIR", str(secrets_dir))
    monkeypatch.setenv("REMOTE_MODELS_ROOT", str(models_root))
    monkeypatch.setenv("REMOTE_TRANSFER_DIR", str(transfer))
    monkeypatch.setenv("REMOTE_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))
    monkeypatch.setenv("REMOTE_MODEL", "qwen-image")
    monkeypatch.setenv("REMOTE_SKIP_PIPELINE_LOAD", "1")
    monkeypatch.setenv("REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK", "1")

    # purge cached app module so it picks up the env vars
    for m in list(sys.modules):
        if m.startswith("remote_server"):
            sys.modules.pop(m)

    from remote_server.app import app
    with TestClient(app) as client:
        yield {
            "client":    client,
            "server_id": server_id,
            "client_id": client_id,
            "psk":       psk,
        }


def signed_headers(psk, method, path, body):
    rid = uuid.uuid4().hex
    proof = compute_request_proof(psk, rid, hashlib.sha256(body).hexdigest())
    return {"X-RGEN-Request-Id": rid, "X-RGEN-Proof": proof}


# ─── tests ──────────────────────────────────────────────────────────


def test_healthz_unauth(provisioned):
    r = provisioned["client"].get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "qwen-image"
    assert body["ready"] is False           # we skipped pipeline load


def test_pubkey_signed_and_carries_model(provisioned):
    r = provisioned["client"].get("/v1/pubkey")
    assert r.status_code == 200
    body = r.json()
    signed = b64d(body["signed_b64"])
    payload = verify_signed(provisioned["server_id"].sig_pub, signed)
    assert payload["label"] == provisioned["server_id"].label
    assert payload["model"] == "qwen-image"


def test_get_model_requires_proof(provisioned):
    r = provisioned["client"].get("/v1/model")
    assert r.status_code == 401


def test_get_model_with_valid_proof(provisioned):
    psk = provisioned["psk"]
    headers = signed_headers(psk, "GET", "/v1/model", b"")
    r = provisioned["client"].get("/v1/model", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "qwen-image"
    assert body["ready"] is False


def test_txt2img_503_without_pipeline(provisioned):
    """With REMOTE_SKIP_PIPELINE_LOAD=1 the pipeline is None → 503."""
    psk        = provisioned["psk"]
    server_id  = provisioned["server_id"]
    client_id  = provisioned["client_id"]

    payload = {
        "_kind": "txt2img", "prompt": "a duck",
        "negative_prompt": "", "width": 64, "height": 64,
        "steps": 1, "cfg": 1.0, "seed": 0,
    }
    payload_canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(payload_canonical).hexdigest()
    manifest = make_job_manifest(client_id.public(), {"kind": "txt2img"},
                                 dataset_sha256=payload_sha)
    manifest["payload_sha256"] = payload_sha
    manifest_signed = sign_json(client_id.sig_priv, manifest)
    envelope = {"manifest_signed_hex": manifest_signed.hex(), "payload": payload}
    body = WrappedBlob.seal(server_id.enc_pub,
                            json.dumps(envelope).encode()).blob

    headers = signed_headers(psk, "POST", "/v1/generate/txt2img", body)
    r = provisioned["client"].post("/v1/generate/txt2img",
                                   content=body, headers=headers)
    assert r.status_code == 503


def test_txt2img_with_stub_pipeline_returns_sealed_latent(provisioned, monkeypatch):
    """Plug a stub pipeline into the app, send a real txt2img, get a sealed
    latent back, decrypt + unpack, verify metadata."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import torch
    from remote_server import app as app_mod
    from remote_server.latent_blob import unpack_latent_blob

    class StubPipeline:
        def apply_loras(self, loras, weights):
            return []

        def clear_loras(self):
            pass

        def generate(self, payload, ws, mode):
            lat = torch.zeros(1, 16, 64, 64, dtype=torch.bfloat16)
            return lat, {
                "model": "qwen-image", "seed": payload["seed"],
                "steps": payload["steps"], "cfg": payload["cfg"],
                "height": payload["height"], "width": payload["width"],
                "scaling": 0.18215, "vae_required": "qwen_image_vae.safetensors",
            }
        def unload(self): pass

    monkeypatch.setattr(app_mod, "_pipeline", StubPipeline())

    psk        = provisioned["psk"]
    server_id  = provisioned["server_id"]
    client_id  = provisioned["client_id"]

    payload = {
        "_kind": "txt2img", "prompt": "a duck wearing a hat",
        "negative_prompt": "", "width": 1024, "height": 1024,
        "steps": 20, "cfg": 4.0, "seed": 42,
    }
    payload_canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(payload_canonical).hexdigest()
    manifest = make_job_manifest(client_id.public(), {"kind": "txt2img"},
                                 dataset_sha256=payload_sha)
    manifest["payload_sha256"] = payload_sha
    manifest_signed = sign_json(client_id.sig_priv, manifest)
    envelope = {"manifest_signed_hex": manifest_signed.hex(), "payload": payload}
    body = WrappedBlob.seal(server_id.enc_pub,
                            json.dumps(envelope).encode()).blob

    headers = signed_headers(psk, "POST", "/v1/generate/txt2img", body)
    r = provisioned["client"].post("/v1/generate/txt2img",
                                   content=body, headers=headers)
    assert r.status_code == 200, r.text

    sealed = r.content
    # Client-side: decrypt, unpack
    plain = WrappedBlob(sealed).open(client_id.enc_priv)
    latent, md = unpack_latent_blob(plain)
    assert latent.shape == (1, 16, 64, 64)
    assert md["seed"] == 42
    assert md["model"] == "qwen-image"
    assert md["vae_required"] == "qwen_image_vae.safetensors"


def test_payload_hash_mismatch_rejected(provisioned, monkeypatch):
    """Tamper the encrypted payload bytes after computing the signed
    manifest's payload_sha256 — server rejects."""
    pytest.importorskip("torch")
    from remote_server import app as app_mod
    monkeypatch.setattr(app_mod, "_pipeline", type("S", (), {
        "generate": lambda self, p, w, m: (None, {})})())

    psk        = provisioned["psk"]
    server_id  = provisioned["server_id"]
    client_id  = provisioned["client_id"]

    real_payload = {"_kind": "txt2img", "prompt": "real", "negative_prompt": "",
                    "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0}
    real_sha = hashlib.sha256(json.dumps(real_payload, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    manifest = make_job_manifest(client_id.public(), {"kind": "txt2img"},
                                 dataset_sha256=real_sha)
    manifest["payload_sha256"] = real_sha
    manifest_signed = sign_json(client_id.sig_priv, manifest)
    # ship a DIFFERENT payload but with the signed manifest above
    bad_payload = dict(real_payload, prompt="EVIL")
    envelope = {"manifest_signed_hex": manifest_signed.hex(),
                "payload": bad_payload}
    body = WrappedBlob.seal(server_id.enc_pub,
                            json.dumps(envelope).encode()).blob
    headers = signed_headers(psk, "POST", "/v1/generate/txt2img", body)
    r = provisioned["client"].post("/v1/generate/txt2img",
                                   content=body, headers=headers)
    assert r.status_code == 400
    assert "payload_sha256" in r.json()["detail"]


def test_wrong_model_endpoint_rejected(provisioned, monkeypatch):
    """REMOTE_MODEL=qwen-image; hitting /v1/generate/img2img must 409."""
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]
    client_id = provisioned["client_id"]

    payload = {"_kind": "img2img", "prompt": "x", "negative_prompt": "",
               "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
               "control_image_b64": ""}
    payload_canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(payload_canonical).hexdigest()
    manifest = make_job_manifest(client_id.public(), {}, dataset_sha256=payload_sha)
    manifest["payload_sha256"] = payload_sha
    manifest_signed = sign_json(client_id.sig_priv, manifest)
    envelope = {"manifest_signed_hex": manifest_signed.hex(), "payload": payload}
    body = WrappedBlob.seal(server_id.enc_pub,
                            json.dumps(envelope).encode()).blob
    headers = signed_headers(psk, "POST", "/v1/generate/img2img", body)
    r = provisioned["client"].post("/v1/generate/img2img",
                                   content=body, headers=headers)
    assert r.status_code == 409
