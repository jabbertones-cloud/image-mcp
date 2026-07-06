"""Round-trip + tamper-detection for the safetensors latent blob."""
from __future__ import annotations

import pytest


def test_roundtrip_dtype_and_shape():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import torch
    from remote_server.latent_blob import pack_latent_blob, unpack_latent_blob

    lat = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16)
    md = {
        "model":        "qwen-image",
        "seed":         42,
        "steps":        20,
        "cfg":          4.0,
        "height":       1024, "width": 1024,
        "scaling":      0.18215,
        "vae_required": "qwen_image_vae.safetensors",
    }
    blob = pack_latent_blob(lat, md)
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 100

    t, md2 = unpack_latent_blob(blob)
    assert t.shape == lat.shape
    assert t.dtype == torch.bfloat16
    # value preserved bit-for-bit (safetensors is byte-deterministic)
    assert torch.equal(t, lat)
    assert md2["model"] == "qwen-image"
    assert md2["seed"]  == 42
    assert md2["scaling"] == 0.18215


def test_metadata_survives_str_coercion():
    """Even if a caller passes non-string values they should round-trip."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import torch
    from remote_server.latent_blob import pack_latent_blob, unpack_latent_blob

    lat = torch.zeros(1, 16, 8, 8, dtype=torch.float32)
    md = {"model": "x", "seed": 0, "steps": 1, "cfg": 1.0, "height": 64,
          "width": 64, "scaling": 1.0, "vae_required": "x.safetensors",
          "extra_int": 99, "extra_float": 3.14}
    out, md2 = unpack_latent_blob(pack_latent_blob(lat, md))
    assert md2["extra_int"] == 99
    assert md2["extra_float"] == 3.14


def test_corrupted_blob_raises():
    pytest.importorskip("safetensors")
    from remote_server.latent_blob import unpack_latent_blob
    with pytest.raises(Exception):
        unpack_latent_blob(b"not a safetensors header")
