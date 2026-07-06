"""Hardening tests:

* tripwire compromise blocks every protected route with 503
* RAM-only LoRA cache leaves zero files in the transfer dir between requests
* lockdown apply path is no-op-safe outside Linux + with REMOTE_SKIP_LOCKDOWN
"""
from __future__ import annotations

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
    Identity, WrappedBlob, compute_request_proof, generate_psk,
    make_job_manifest, sign_json, split_psk, write_half,
)


def _signed_headers(psk, body):
    rid = uuid.uuid4().hex
    proof = compute_request_proof(psk, rid, hashlib.sha256(body).hexdigest())
    return {"X-RGEN-Request-Id": rid, "X-RGEN-Proof": proof}


def _make_body(server_id, client_id, payload):
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(canon).hexdigest()
    manifest = make_job_manifest(client_id.public(), {"kind": payload["_kind"]},
                                 dataset_sha256=sha)
    manifest["payload_sha256"] = sha
    manifest_signed = sign_json(client_id.sig_priv, manifest)
    envelope = {"manifest_signed_hex": manifest_signed.hex(), "payload": payload}
    return WrappedBlob.seal(server_id.enc_pub,
                            json.dumps(envelope).encode()).blob


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    secrets_dir = tmp_path / "secrets"; secrets_dir.mkdir()
    models_root = tmp_path / "models"; models_root.mkdir()
    transfer = tmp_path / "transfer"; transfer.mkdir()
    psk = generate_psk(); s_half, c_half = split_psk(psk)
    write_half(secrets_dir / "half.bin", s_half)
    write_half(secrets_dir / "client_half.bin", c_half)
    server_id = Identity.generate("test-server")
    client_id = Identity.generate("test-client")
    (secrets_dir / "identity.json").write_text(json.dumps(server_id.to_dict()))
    (secrets_dir / "client_peer.json").write_text(json.dumps(client_id.public().to_dict()))
    monkeypatch.setenv("REMOTE_SECRETS_DIR", str(secrets_dir))
    monkeypatch.setenv("REMOTE_MODELS_ROOT", str(models_root))
    monkeypatch.setenv("REMOTE_TRANSFER_DIR", str(transfer))
    monkeypatch.setenv("REMOTE_MODEL", "qwen-image")
    monkeypatch.setenv("REMOTE_SKIP_PIPELINE_LOAD", "1")
    monkeypatch.setenv("REMOTE_SKIP_LOCKDOWN", "1")
    monkeypatch.setenv("REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK", "1")

    for m in list(sys.modules):
        if m.startswith("remote_server"):
            sys.modules.pop(m)
    pytest.importorskip("torch")
    import torch
    from remote_server import app as app_mod

    class StubPipeline:
        def apply_loras(self, loras, weights): return []
        def clear_loras(self): pass
        def generate(self, payload, ws, mode):
            return torch.zeros(1, 16, 64, 64, dtype=torch.bfloat16), {
                "model": "qwen-image", "seed": payload["seed"],
                "steps": payload["steps"], "cfg": payload["cfg"],
                "height": payload["height"], "width": payload["width"],
                "scaling": 0.18215, "vae_required": "qwen_image_vae.safetensors",
            }
        def unload(self): pass

    with TestClient(app_mod.app) as client:
        app_mod._pipeline = StubPipeline()
        yield {"client": client, "app_mod": app_mod,
               "server_id": server_id, "client_id": client_id,
               "psk": psk, "transfer": transfer}


# ─── tripwire-trip blocks every protected endpoint ────────────────


def test_tripwire_trip_blocks_protected_routes(provisioned):
    app_mod = provisioned["app_mod"]
    from remote_server.tripwire import get_tripwire
    tw = get_tripwire()
    assert tw is not None

    # baseline: a status request works
    psk = provisioned["psk"]
    r = provisioned["client"].get("/v1/server_status",
                                   headers=_signed_headers(psk, b""))
    assert r.status_code == 200, r.text
    assert r.json()["compromised"] is False

    # trip the tripwire directly (simulates a detected docker exec)
    tw._trip("stranger_child", {"strangers": [9999]})
    assert tw.compromised

    # every protected route now 503s
    for method, path in [
        ("GET", "/v1/server_status"),
        ("GET", "/v1/model"),
        ("GET", "/v1/loras"),
        ("DELETE", "/v1/loras/" + "a" * 64),
    ]:
        r = provisioned["client"].request(
            method, path, headers=_signed_headers(psk, b""),
        )
        assert r.status_code == 503
        assert "compromised" in r.json()["detail"]

    # unauth routes still work — useful for the client to detect a sealed pod
    r = provisioned["client"].get("/healthz")
    assert r.status_code == 200


def test_tripwire_wipes_combined_psk(provisioned):
    app_mod = provisioned["app_mod"]
    from remote_server.tripwire import get_tripwire
    assert app_mod._combined_psk is not None
    get_tripwire()._trip("tracer", {"tracer_pid": 42})
    # the on_trip callback wiped it
    assert app_mod._combined_psk is None


# ─── RAM-only LoRA cache: nothing on disk ─────────────────────────


def test_no_lora_files_in_transfer_dir(provisioned):
    """Submit a generation that includes a LoRA — verify the transfer dir
    holds zero .safetensors files before and after."""
    import base64
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]
    transfer = provisioned["transfer"]

    def _files_in_transfer() -> list[Path]:
        return [p for p in transfer.rglob("*") if p.is_file()]

    assert _files_in_transfer() == []

    lora_bytes = _toy_safetensors_bytes()
    lora_sha = hashlib.sha256(lora_bytes).hexdigest()
    payload = {
        "_kind": "txt2img", "prompt": "x", "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
        "loras": [{
            "sha256": lora_sha, "weight": 1.0,
            "bytes_b64": base64.b64encode(lora_bytes).decode("ascii"),
        }],
    }
    body = _make_body(server_id, client_id, payload)
    r = provisioned["client"].post(
        "/v1/generate/txt2img", content=body, headers=_signed_headers(psk, body),
    )
    assert r.status_code == 200, r.text

    # After the request: no .safetensors anywhere in transfer dir
    files = _files_in_transfer()
    safetensors_files = [p for p in files if p.suffix == ".safetensors"]
    assert safetensors_files == [], f"expected no safetensors files, got {safetensors_files}"
    # transfer dir is also empty of any transient workspace subdirs
    # (TransferWorkspace.__exit__ cleans up immediately)
    workspace_subdirs = [p for p in transfer.iterdir() if p.is_dir()]
    assert workspace_subdirs == [], f"leftover workspace dirs: {workspace_subdirs}"

    # second request: cache HIT (no bytes), still no files
    payload2 = dict(payload, loras=[{"sha256": lora_sha, "weight": 0.5}], seed=1)
    body2 = _make_body(server_id, client_id, payload2)
    r = provisioned["client"].post(
        "/v1/generate/txt2img", content=body2, headers=_signed_headers(psk, body2),
    )
    assert r.status_code == 200, r.text
    safetensors_files = [p for p in transfer.rglob("*") if p.suffix == ".safetensors"]
    assert safetensors_files == []


def test_lora_cache_holds_state_dict(provisioned):
    """RAM cache exposes parsed state_dict, not a path."""
    import base64
    psk = provisioned["psk"]
    server_id = provisioned["server_id"]; client_id = provisioned["client_id"]

    lora_bytes = _toy_safetensors_bytes()
    sha = hashlib.sha256(lora_bytes).hexdigest()
    payload = {
        "_kind": "txt2img", "prompt": "x", "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 0,
        "loras": [{
            "sha256": sha, "weight": 1.0,
            "bytes_b64": base64.b64encode(lora_bytes).decode("ascii"),
        }],
    }
    body = _make_body(server_id, client_id, payload)
    r = provisioned["client"].post(
        "/v1/generate/txt2img", content=body, headers=_signed_headers(psk, body),
    )
    assert r.status_code == 200, r.text

    cache = provisioned["app_mod"]._lora_cache
    entry = cache.get(sha)
    assert entry is not None
    assert entry.state_dict is not None
    # state_dict is a real dict[str, torch.Tensor]
    import torch
    for k, v in entry.state_dict.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.Tensor)


# ─── helpers ────────────────────────────────────────────────────────


def _toy_safetensors_bytes() -> bytes:
    """Build a 2-tensor safetensors blob in pure Python so the test runs
    without writing anything to disk."""
    import struct, json as _json
    import torch
    tensors = {
        "lora_A": torch.zeros(4, 8, dtype=torch.float32),
        "lora_B": torch.ones(8, 4, dtype=torch.float32),
    }
    # serialise via safetensors.torch.save (returns bytes directly)
    from safetensors.torch import save
    return save(tensors)


# ─── lockdown: scrub_env zeroes targeted env vars ─────────────────


def test_scrub_env_clears_targeted_keys(monkeypatch):
    from remote_server import lockdown
    monkeypatch.setenv("REMOTE_SERVER_HALF_B64", "AAAAAAAAAAAA")
    monkeypatch.setenv("REMOTE_CLIENT_HALF_B64", "BBBBBBBBBB")
    monkeypatch.setenv("TS_AUTHKEY",             "tskey-auth-test")
    monkeypatch.setenv("UNRELATED_VAR",          "keepme")

    import os
    assert os.environ.get("REMOTE_SERVER_HALF_B64") == "AAAAAAAAAAAA"
    cleared = lockdown.scrub_env()
    assert "REMOTE_SERVER_HALF_B64" in cleared
    assert "REMOTE_CLIENT_HALF_B64" in cleared
    assert "TS_AUTHKEY" in cleared
    assert "UNRELATED_VAR" not in cleared
    assert "REMOTE_SERVER_HALF_B64" not in os.environ
    assert os.environ.get("UNRELATED_VAR") == "keepme"


def test_apply_lockdowns_returns_status_on_any_platform():
    from remote_server.lockdown import apply_lockdowns
    s = apply_lockdowns()
    # We only care that it returned a status object; success flags depend
    # on platform + CAP_SYS_PTRACE etc.
    assert hasattr(s, "set_dumpable")
    assert hasattr(s, "no_new_privs")
    assert hasattr(s, "env_scrubbed")
