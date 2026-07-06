"""Tests for the LoRA cache + the 404-retry-with-bytes flow.

We avoid loading real diffusers — instead the stub pipeline's
``apply_loras`` records what it was asked to apply, ``clear_loras`` is a
no-op. This exercises the full app.py LoRA-handling path including the
cache miss → 404 → client-bytes upload cycle.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from remote_server.crypto import (
    Identity, WrappedBlob, b64d, b64e, compute_request_proof, generate_psk,
    make_job_manifest, sha256_hex, sign_json, split_psk, verify_signed,
    write_half,
)
# NOTE — lora_cache is imported FRESH inside each unit test below.
# The provisioned fixture purges remote_server.* modules; tests that capture
# top-level imports here would see stale references. Tests that need
# constants from this module (MAX_ENTRIES, MAX_BYTES_PER_LORA) read them
# from a current-call-time import.


# ─── unit tests: LoRACache ───────────────────────────────────────────


def _import_cache():
    """Always returns the current module — defeats the provisioned-fixture
    module purge that would otherwise leave us holding a stale class."""
    import importlib, remote_server.lora_cache as lc
    return importlib.reload(lc) if "remote_server.lora_cache" in sys.modules else lc


def test_cache_put_round_trip(tmp_path):
    lc = _import_cache()
    cache = lc.LoRACache(tmp_path)
    data = b"FAKE-LORA-CONTENT-" + b"0" * 200
    sha = hashlib.sha256(data).hexdigest()
    e = cache.put(data, claimed_sha=sha)
    assert e.sha256 == sha
    # In-RAM only — raw_bytes is the original, no path on disk.
    assert e.raw_bytes == data
    assert e.size == len(data)
    got = cache.get(sha)
    assert got is not None and got.raw_bytes == e.raw_bytes
    assert cache.has(sha)


def test_cache_sha_mismatch_rejected(tmp_path):
    lc = _import_cache()
    cache = lc.LoRACache(tmp_path)
    with pytest.raises(lc.LoRACacheError):
        cache.put(b"foo", claimed_sha="a" * 64)


def test_cache_size_cap(tmp_path, monkeypatch):
    lc = _import_cache()
    monkeypatch.setattr(lc, "MAX_BYTES_PER_LORA", 10)
    cache = lc.LoRACache(tmp_path)
    with pytest.raises(lc.LoRACacheError):
        cache.put(b"X" * 32)


def test_cache_lru_eviction(tmp_path, monkeypatch):
    lc = _import_cache()
    monkeypatch.setattr(lc, "MAX_ENTRIES", 2)
    cache = lc.LoRACache(tmp_path)
    a = cache.put(b"AAAA")
    import time; time.sleep(0.05)
    b = cache.put(b"BBBB")
    import time; time.sleep(0.05)
    c = cache.put(b"CCCC")
    assert cache.get(a.sha256) is None, "oldest entry should have been evicted"
    assert cache.get(b.sha256) is not None
    assert cache.get(c.sha256) is not None


def test_evict_explicit(tmp_path):
    lc = _import_cache()
    cache = lc.LoRACache(tmp_path)
    e = cache.put(b"X")
    assert cache.evict(e.sha256) is True
    assert not cache.has(e.sha256)
    assert cache.evict(e.sha256) is False


def test_invalid_sha_string_rejected(tmp_path):
    lc = _import_cache()
    cache = lc.LoRACache(tmp_path)
    with pytest.raises(lc.LoRACacheError):
        cache.get("not-a-real-sha")
    with pytest.raises(lc.LoRACacheError):
        cache._path_for("zz" * 32)


# ─── unit tests: parse_loras_payload ─────────────────────────────────


def test_parse_loras_validates_shape():
    lc = _import_cache()
    out = lc.parse_loras_payload([
        {"sha256": "a" * 64, "weight": 0.8},
        {"sha256": "b" * 64, "weight": 1.0, "bytes_b64": "Zg=="},
    ])
    assert len(out) == 2
    assert out[0].weight == 0.8
    assert out[1].bytes_provided is True
    with pytest.raises(ValueError):
        lc.parse_loras_payload([{"sha256": "too-short"}])
    with pytest.raises(ValueError):
        lc.parse_loras_payload([{"sha256": "a" * 64, "weight": 5.0}])
    with pytest.raises(ValueError):
        lc.parse_loras_payload([{"sha256": "a" * 64}, {"sha256": "a" * 64}])
    with pytest.raises(ValueError):
        lc.parse_loras_payload([{"sha256": "a" * 64}] * 9)


# ─── integration: full app with LoRA cache + stub pipeline ──────────


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
    (secrets_dir / "identity.json").write_text(json.dumps(server_id.to_dict()))
    (secrets_dir / "client_peer.json").write_text(json.dumps(client_id.public().to_dict()))

    monkeypatch.setenv("REMOTE_SECRETS_DIR", str(secrets_dir))
    monkeypatch.setenv("REMOTE_MODELS_ROOT", str(models_root))
    monkeypatch.setenv("REMOTE_TRANSFER_DIR", str(transfer))
    monkeypatch.setenv("REMOTE_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024))
    monkeypatch.setenv("REMOTE_MODEL", "qwen-image")
    monkeypatch.setenv("REMOTE_SKIP_PIPELINE_LOAD", "1")
    monkeypatch.setenv("REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK", "1")

    for m in list(sys.modules):
        if m.startswith("remote_server"):
            sys.modules.pop(m)

    from remote_server import app as app_mod
    pytest.importorskip("torch")
    import torch

    class StubPipeline:
        def __init__(self):
            self.applied_calls: list = []
            self.cleared_count = 0
        def apply_loras(self, loras, weights):
            self.applied_calls.append([(l.sha256, w) for l, w in zip(loras, weights)])
            return [f"lora_{l.sha256[:12]}" for l in loras]
        def clear_loras(self):
            self.cleared_count += 1
        def generate(self, payload, ws, mode):
            lat = torch.zeros(1, 16, 64, 64, dtype=torch.bfloat16)
            return lat, {
                "model": "qwen-image", "seed": payload["seed"],
                "steps": payload["steps"], "cfg": payload["cfg"],
                "height": payload["height"], "width": payload["width"],
                "scaling": 0.18215, "vae_required": "qwen_image_vae.safetensors",
            }
        def unload(self): pass

    stub = StubPipeline()
    with TestClient(app_mod.app) as client:
        # inject AFTER lifespan so we don't clobber _lora_cache init
        app_mod._pipeline = stub
        yield {"client": client, "stub": stub,
               "server_id": server_id, "client_id": client_id, "psk": psk}


def _signed_headers(psk, body):
    rid = uuid.uuid4().hex
    proof = compute_request_proof(psk, rid, hashlib.sha256(body).hexdigest())
    return {"X-RGEN-Request-Id": rid, "X-RGEN-Proof": proof}


def _build_request_body(server_id, client_id, payload):
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(canon).hexdigest()
    manifest = make_job_manifest(client_id.public(), {"kind": payload["_kind"]},
                                 dataset_sha256=sha)
    manifest["payload_sha256"] = sha
    manifest_signed = sign_json(client_id.sig_priv, manifest)
    envelope = {"manifest_signed_hex": manifest_signed.hex(), "payload": payload}
    return WrappedBlob.seal(server_id.enc_pub,
                            json.dumps(envelope).encode()).blob


def test_lora_cache_miss_returns_404_with_missing_list(provisioned):
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]

    fake_sha = hashlib.sha256(b"never-uploaded").hexdigest()
    payload = {
        "_kind": "txt2img", "prompt": "x", "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
        "loras": [{"sha256": fake_sha, "weight": 1.0}],
    }
    body = _build_request_body(server_id, client_id, payload)
    r = provisioned["client"].post(
        "/v1/generate/txt2img", content=body,
        headers=_signed_headers(psk, body),
    )
    assert r.status_code == 404
    body_json = r.json()
    assert fake_sha in body_json["missing_loras"]


def test_lora_cache_hit_via_bytes_then_by_sha_alone(provisioned):
    """First request supplies bytes → cached. Second request with the same
    sha but NO bytes succeeds and runs the pipeline with that LoRA."""
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]
    stub = provisioned["stub"]

    lora_bytes = b"FAKE-LORA-WEIGHTS-" + b"\x00" * 300
    lora_sha = hashlib.sha256(lora_bytes).hexdigest()

    # 1) supply bytes
    payload1 = {
        "_kind": "txt2img", "prompt": "first", "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
        "loras": [{
            "sha256":    lora_sha, "weight": 0.7,
            "bytes_b64": base64.b64encode(lora_bytes).decode("ascii"),
        }],
    }
    body1 = _build_request_body(server_id, client_id, payload1)
    r1 = provisioned["client"].post(
        "/v1/generate/txt2img", content=body1,
        headers=_signed_headers(psk, body1),
    )
    assert r1.status_code == 200, r1.text

    # 2) by-sha alone (cache HIT)
    payload2 = dict(payload1, prompt="second",
                    loras=[{"sha256": lora_sha, "weight": 1.0}])
    body2 = _build_request_body(server_id, client_id, payload2)
    r2 = provisioned["client"].post(
        "/v1/generate/txt2img", content=body2,
        headers=_signed_headers(psk, body2),
    )
    assert r2.status_code == 200, r2.text

    # stub received both applies, the second with weight=1.0
    assert len(stub.applied_calls) == 2
    assert stub.applied_calls[1] == [(lora_sha, 1.0)]
    # clear was called both times — once per request
    assert stub.cleared_count == 2


def test_lora_list_endpoint_lists_cached_entries(provisioned):
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]

    # seed two LoRAs via real requests
    for tag in (b"A", b"B"):
        lora_bytes = tag * 256
        lora_sha = hashlib.sha256(lora_bytes).hexdigest()
        payload = {
            "_kind": "txt2img", "prompt": "x", "negative_prompt": "",
            "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
            "loras": [{
                "sha256": lora_sha, "weight": 1.0,
                "bytes_b64": base64.b64encode(lora_bytes).decode("ascii"),
            }],
        }
        body = _build_request_body(server_id, client_id, payload)
        r = provisioned["client"].post(
            "/v1/generate/txt2img", content=body,
            headers=_signed_headers(psk, body),
        )
        assert r.status_code == 200, r.text

    # GET /v1/loras
    headers = _signed_headers(psk, b"")
    r = provisioned["client"].get("/v1/loras", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    seen = {e["sha256"] for e in body["entries"]}
    assert hashlib.sha256(b"A" * 256).hexdigest() in seen
    assert hashlib.sha256(b"B" * 256).hexdigest() in seen


def test_lora_evict_endpoint(provisioned):
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]

    lora_bytes = b"Z" * 128
    lora_sha = hashlib.sha256(lora_bytes).hexdigest()
    payload = {
        "_kind": "txt2img", "prompt": "x", "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
        "loras": [{
            "sha256": lora_sha, "weight": 1.0,
            "bytes_b64": base64.b64encode(lora_bytes).decode("ascii"),
        }],
    }
    body = _build_request_body(server_id, client_id, payload)
    r = provisioned["client"].post(
        "/v1/generate/txt2img", content=body,
        headers=_signed_headers(psk, body),
    )
    assert r.status_code == 200

    # evict
    r = provisioned["client"].delete(
        f"/v1/loras/{lora_sha}",
        headers=_signed_headers(psk, b""),
    )
    assert r.status_code == 200
    assert r.json() == {"evicted": True, "sha256": lora_sha}

    # second eviction is no-op
    r = provisioned["client"].delete(
        f"/v1/loras/{lora_sha}",
        headers=_signed_headers(psk, b""),
    )
    assert r.json() == {"evicted": False, "sha256": lora_sha}


def test_lora_weight_cap_validated(provisioned):
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]

    lora_bytes = b"X" * 256
    payload = {
        "_kind": "txt2img", "prompt": "x", "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
        "loras": [{
            "sha256": hashlib.sha256(lora_bytes).hexdigest(),
            "weight": 5.0,        # outside [-2, 2]
            "bytes_b64": base64.b64encode(lora_bytes).decode("ascii"),
        }],
    }
    body = _build_request_body(server_id, client_id, payload)
    r = provisioned["client"].post(
        "/v1/generate/txt2img", content=body,
        headers=_signed_headers(psk, body),
    )
    assert r.status_code == 400
    assert "weight" in r.json()["detail"]
