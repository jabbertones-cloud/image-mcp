"""Bounded artifact metadata for handoff to ChatGPT or specialist editors."""
from __future__ import annotations
import hashlib
import mimetypes
from pathlib import Path
from typing import Any
from PIL import Image

from .path_safety import assert_within_root


def artifact_descriptor(path: str | Path) -> dict[str, Any]:
    p = assert_within_root(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    result: dict[str, Any] = {
        "path": str(p),
        "name": p.name,
        "mimeType": mime,
        "sizeBytes": p.stat().st_size,
        "sha256": h.hexdigest(),
    }
    try:
        with Image.open(p) as img:
            result["width"], result["height"] = img.size
    except Exception:
        pass
    return result


def photoshop_handoff(path: str | Path) -> dict[str, Any]:
    return {
        "kind": "photoshop_handoff",
        "artifact": artifact_descriptor(path),
        "contractVersion": 1,
    }
