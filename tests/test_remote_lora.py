"""Offline tests for the remote-generation LoRA client fixes: by-reference
specs, the local-path chokepoint, sha verification, and HF URL validation.
No pod or GPU required — these exercise pure validation/prep logic."""
import pytest

from server import remote_gen
from server import server_config


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(server_config, "_config", {**server_config._config, "lora_dirs": str(d)})
    remote_gen._lora_hash_cache.clear()
    return d


def _write_lora(d, name="x.safetensors", data=b"LORA-BYTES"):
    p = d / name
    p.write_bytes(data)
    return p


# ── by-reference (sha256-only) specs ────────────────────────────────

def test_sha256_only_reference_passes_through(lora_dir):
    specs = remote_gen._prepare_loras([{"sha256": "a" * 64, "weight": 0.8}])
    assert len(specs) == 1
    assert specs[0].sha256 == "a" * 64
    assert specs[0].bytes is None and specs[0].path is None  # nothing read locally


def test_bytes_spec_hashes_directly(lora_dir):
    from remote_server.crypto import sha256_hex
    specs = remote_gen._prepare_loras([{"bytes": b"abc", "weight": 1.0}])
    assert specs[0].sha256 == sha256_hex(b"abc")


def test_spec_needs_path_bytes_or_sha(lora_dir):
    with pytest.raises(ValueError, match="path.*bytes.*sha256"):
        remote_gen._prepare_loras([{"weight": 1.0}])


# ── local-path chokepoint ───────────────────────────────────────────

def test_valid_local_lora_is_hashed(lora_dir):
    from remote_server.crypto import sha256_hex
    p = _write_lora(lora_dir)
    specs = remote_gen._prepare_loras([{"path": str(p), "weight": 1.0}])
    assert specs[0].sha256 == sha256_hex(b"LORA-BYTES")
    assert specs[0].bytes is None  # deferred until the server actually 404s


def test_path_outside_lora_dirs_rejected(lora_dir, tmp_path):
    outside = tmp_path / "secret.safetensors"
    outside.write_bytes(b"x")
    with pytest.raises(PermissionError, match="outside the allowed"):
        remote_gen._prepare_loras([{"path": str(outside)}])


def test_non_lora_suffix_rejected(lora_dir):
    p = _write_lora(lora_dir, name="half.bin")
    with pytest.raises(ValueError, match="safetensors"):
        remote_gen._prepare_loras([{"path": str(p)}])


def test_caller_sha_mismatch_raises(lora_dir):
    p = _write_lora(lora_dir)
    with pytest.raises(ValueError, match="mismatch"):
        remote_gen._prepare_loras([{"path": str(p), "sha256": "b" * 64}])


def test_empty_lora_dirs_blocks_local_upload(lora_dir, monkeypatch):
    monkeypatch.setattr(server_config, "_config", {**server_config._config, "lora_dirs": ""})
    p = _write_lora(lora_dir)
    with pytest.raises(PermissionError, match="disabled"):
        remote_gen._prepare_loras([{"path": str(p)}])


# ── deferred byte loading on server cache-miss ──────────────────────

def test_payload_loads_bytes_only_when_requested(lora_dir):
    p = _write_lora(lora_dir)
    specs = remote_gen._prepare_loras([{"path": str(p), "weight": 1.0}])
    sha = specs[0].sha256
    # by-reference: no bytes
    ref = remote_gen._build_loras_payload(specs, include_bytes_for=set())
    assert "bytes_b64" not in ref[0]
    # server asks → bytes are read from disk now
    withb = remote_gen._build_loras_payload(specs, include_bytes_for={sha})
    assert "bytes_b64" in withb[0]


def test_payload_sha_only_cache_miss_errors(lora_dir):
    specs = remote_gen._prepare_loras([{"sha256": "c" * 64, "weight": 1.0}])
    with pytest.raises(RuntimeError, match="referenced by sha256 only"):
        remote_gen._build_loras_payload(specs, include_bytes_for={"c" * 64})


# ── HuggingFace URL validation (server side) ────────────────────────

@pytest.mark.parametrize("repo,filename", [
    ("../evil", "x.safetensors"),
    ("user/repo", "../../other/resolve/main/x.safetensors"),
    ("user/repo", "x.safetensors?token=leak"),
    ("user/repo", "a\\b.safetensors"),
    ("user", "x.safetensors"),                 # repo missing '/'
])
def test_hf_fetch_rejects_injection(repo, filename):
    from remote_server.lora_fetch import fetch_from_hf, LoRAFetchError
    with pytest.raises(LoRAFetchError):
        fetch_from_hf(cache=None, repo=repo, filename=filename)


def test_hf_safe_component_accepts_normal_paths():
    from remote_server.lora_fetch import _safe_url_component
    assert _safe_url_component("filename", "sub/dir/w.safetensors", allow_slash=True)
