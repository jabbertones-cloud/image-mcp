"""Pack a generated latent tensor as a self-describing safetensors blob.

The blob carries shape + dtype + the model name that produced it, so the
client can pick the right VAE to decode it. We use safetensors because:

* Byte-deterministic (good for hashing / replay defence).
* dtype + shape encoded in the header (the client doesn't need extra metadata).
* Already a tier-1 dep of every torch project — no new wheel for the client.

Format::

    {
      "latent":   <Tensor: (B, C, F_or_1, H/8, W/8) or (B, C, H/8, W/8)>,
      "metadata": {
        "model":         "qwen-image" | "qwen-image-edit-2511",
        "seed":          int,
        "steps":         int,
        "cfg":           float,
        "height":        int,    # pixel-space output height
        "width":         int,    # pixel-space output width
        "scaling":       float,  # latent scaling factor expected by the VAE
        "vae_required":  str,    # filename hint for the client
      }
    }

The metadata dict is stringified into the safetensors __metadata__ slot
because safetensors only accepts dict[str, str] there. We JSON-encode it.
"""
from __future__ import annotations

import json
from typing import Any


def pack_latent_blob(latent, metadata: dict[str, Any]) -> bytes:
    """latent is a torch.Tensor; returned bytes can be sealed and sent."""
    from safetensors.torch import save                # local import — heavy
    md_json = {k: str(v) if not isinstance(v, str) else v
               for k, v in metadata.items()}
    md_json["__json__"] = json.dumps(metadata)
    return save({"latent": latent.detach().contiguous().cpu()}, metadata=md_json)


def unpack_latent_blob(blob: bytes) -> tuple["torch.Tensor", dict[str, Any]]:
    """Client-side: parse the safetensors blob into (tensor, metadata)."""
    from safetensors import safe_open
    import io
    import torch       # noqa: F401   (loaded lazily on first call)

    # safetensors supports loading from bytes via a tmp file or from a
    # memoryview through the rust binding. Easiest portable path: a real
    # temp file in tmpfs.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        with safe_open(path, framework="pt", device="cpu") as st:
            md_raw = dict(st.metadata() or {})
            try:
                metadata = json.loads(md_raw.get("__json__", "{}"))
            except json.JSONDecodeError:
                metadata = {k: v for k, v in md_raw.items() if k != "__json__"}
            t = st.get_tensor("latent")
        return t, metadata
    finally:
        try: os.unlink(path)
        except OSError: pass
