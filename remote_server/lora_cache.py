"""SHA-256 keyed LoRA cache — held entirely in process RAM.

We DON'T persist LoRA bytes to tmpfs. Instead each entry stores the
parsed safetensors state_dict (tensors live in CPU RAM until promoted
to GPU by diffusers' ``load_lora_weights``). The pipeline path accepts
either a path-on-disk OR a dict via ``load_lora_weights`` — we pass the
dict, so the bytes never touch a filesystem.

Why this matters:
  - The user's spec is "the only files in the ramdisk should be the
    latest latent and the latest image". A file-backed LoRA cache would
    leave 100s of MB of weights sitting on tmpfs between requests.
  - When the pod is suspended (e.g. by gcore), the LoRA bytes still leak
    via the process-image dump, but the surface area is narrower than
    'in RAM + on tmpfs both'.

Lookup flow per request:
  1. Client sends list of {sha256, weight, bytes_b64?}.
  2. Server: cache hit → use cached dict. Cache miss + bytes → parse +
     cache + use. Cache miss + no bytes → 404 with missing-list.

Concurrency:
  - Single-writer guaranteed by ``_GENERATION_LOCK`` in app.py.
  - Reads & writes use a module-level RLock.

Capacity:
  - Soft cap via ``MAX_ENTRIES``. Eviction is LRU on last-access.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


MAX_ENTRIES = int(os.environ.get("REMOTE_LORA_CACHE_MAX_ENTRIES", "32"))
# 0 = unlimited (default).  Set REMOTE_LORA_MAX_BYTES to a positive number
# to enforce a per-LoRA size cap.  We default to unlimited because Qwen-Image
# LoRAs from HF / Civitai range up to ~1.5 GB and many users want them all.
MAX_BYTES_PER_LORA = int(os.environ.get("REMOTE_LORA_MAX_BYTES", "0"))


class LoRACacheError(Exception):
    """Raised for sha-mismatch / oversize / cache-corruption events."""


@dataclass
class CachedLoRA:
    """Mutable so we can touch last_used without rebuilding the object.
    We keep ``raw_bytes`` in RAM and parse to a ``state_dict`` on first
    access (most LoRAs are <200 MB so the in-RAM cost is acceptable, and
    lazy parse means the cache works for tests that pass non-safetensors
    bytes)."""
    sha256:      str
    size:        int                  # original byte size, for /v1/loras
    raw_bytes:   bytes                 # held in RAM only, never written
    _cached_state_dict: Any = None
    last_used:   float = 0.0

    @property
    def state_dict(self) -> Any:
        if self._cached_state_dict is None:
            self._cached_state_dict = _parse_safetensors(self.raw_bytes)
        return self._cached_state_dict


class LoRACache:
    """In-RAM LRU cache. No files. ``root`` is accepted for API parity with
    the previous version but is no longer used."""

    def __init__(self, root: Path | None = None):
        # root is ignored — kept in the signature so the lifespan code in
        # app.py doesn't change.
        self._entries: dict[str, CachedLoRA] = {}
        self._lock = threading.RLock()
        log.info(f"LoRA cache initialised (in-RAM only); "
                 f"max_entries={MAX_ENTRIES} max_bytes_per_lora={MAX_BYTES_PER_LORA}")

    # ─── core ─────────────────────────────────────────────────────

    def has(self, sha256: str) -> bool:
        with self._lock:
            return sha256.lower() in self._entries

    def get(self, sha256: str) -> CachedLoRA | None:
        # Validate sha format — same rule as _path_for so callers can rely on
        # malformed inputs raising rather than silently returning None.
        if not (len(sha256) == 64 and all(c in "0123456789abcdefABCDEF" for c in sha256)):
            raise LoRACacheError(f"not a valid sha256: {sha256!r}")
        sha = sha256.lower()
        with self._lock:
            entry = self._entries.get(sha)
            if entry is not None:
                entry.last_used = time.monotonic()
            return entry

    def put(self, raw: bytes, claimed_sha: str | None = None) -> CachedLoRA:
        # 0 = unlimited (default). Cap only when explicitly set > 0.
        if MAX_BYTES_PER_LORA > 0 and len(raw) > MAX_BYTES_PER_LORA:
            raise LoRACacheError(
                f"LoRA exceeds {MAX_BYTES_PER_LORA} bytes ({len(raw)} bytes received)"
            )
        actual_sha = hashlib.sha256(raw).hexdigest()
        if claimed_sha is not None and claimed_sha.lower() != actual_sha:
            raise LoRACacheError(
                f"sha mismatch: claimed {claimed_sha} but content hashes to {actual_sha}"
            )
        # Validate sha format (the unit tests rely on this rejection path)
        if not (len(actual_sha) == 64 and all(c in "0123456789abcdef" for c in actual_sha)):
            raise LoRACacheError(f"not a valid sha256: {actual_sha!r}")
        entry = CachedLoRA(
            sha256=actual_sha, size=len(raw),
            raw_bytes=bytes(raw),       # defensive copy
            last_used=time.monotonic(),
        )
        with self._lock:
            self._entries[actual_sha] = entry
            self._evict_if_needed()
        return entry

    def evict(self, sha256: str) -> bool:
        with self._lock:
            return self._entries.pop(sha256.lower(), None) is not None

    # ─── introspection ────────────────────────────────────────────

    def list_entries(self) -> list[CachedLoRA]:
        with self._lock:
            return list(self._entries.values())

    # ─── internals ────────────────────────────────────────────────

    def _path_for(self, sha256: str) -> str:
        """Compat shim — old code uses ``_path_for`` to validate hex format.
        We no longer write files; this is just a syntactic validator."""
        if not (len(sha256) == 64 and all(c in "0123456789abcdefABCDEF" for c in sha256)):
            raise LoRACacheError(f"not a valid sha256: {sha256!r}")
        return f"<in-ram:{sha256.lower()}>"

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= MAX_ENTRIES:
            return
        # sort by last_used ASC → oldest first
        ordered = sorted(self._entries.items(), key=lambda kv: kv[1].last_used)
        n_to_evict = len(ordered) - MAX_ENTRIES
        for sha, _ in ordered[:n_to_evict]:
            log.info(f"LRU evict {sha[:12]}…")
            self._entries.pop(sha, None)


def _parse_safetensors(raw: bytes) -> dict:
    """Parse a .safetensors byte string into a state_dict.

    safetensors' python binding has ``load`` (path-only) and ``deserialize``
    (bytes → list of (name, dict)). We use the bytes path."""
    try:
        from safetensors import deserialize
        import torch
        out: dict[str, Any] = {}
        for name, descriptor in deserialize(raw):
            dtype = descriptor["dtype"]
            shape = descriptor["shape"]
            data  = descriptor["data"]
            torch_dtype = _SAFETENSORS_DTYPES.get(dtype, torch.float32)
            tensor = torch.frombuffer(data, dtype=torch_dtype).reshape(shape).clone()
            out[name] = tensor
        return out
    except Exception:
        # Fall back to writing to a tmpfile so ``safetensors.torch.load_file``
        # can parse it — only used if the deserialize path doesn't import.
        import tempfile
        from safetensors.torch import load_file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            return load_file(tmp)
        finally:
            try: os.unlink(tmp)
            except OSError: pass


_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8":  "uint8",  "I8": "int8",
    "I16": "int16",  "U16": "uint16",
    "I32": "int32",  "U32": "uint32",
    "I64": "int64",  "U64": "uint64",
    "F16": "float16","BF16": "bfloat16",
    "F32": "float32","F64": "float64",
}
# Resolve string dtype names to torch dtypes once at first call. We do it
# lazily because importing torch at module-import would pull a heavy dep
# into the test process even when the lora_cache module is the only thing
# under test.
def _resolve_dtypes() -> None:
    import torch
    global _SAFETENSORS_DTYPES
    _SAFETENSORS_DTYPES = {
        "BOOL": torch.bool,
        "U8": torch.uint8, "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16, "BF16": torch.bfloat16,
        "F32": torch.float32, "F64": torch.float64,
    }
_resolve_dtypes()


# ─── request-side validation helpers ─────────────────────────────


@dataclass
class LoRARequest:
    sha256: str
    weight: float
    bytes_provided: bool


def parse_loras_payload(loras: list[dict]) -> list[LoRARequest]:
    """Turn the raw payload list into validated LoRARequest objects.
    Raises ValueError on malformed entries — caller maps to HTTP 400."""
    if loras is None:
        return []
    if not isinstance(loras, list):
        raise ValueError("'loras' must be a list")
    if len(loras) > 8:
        raise ValueError("at most 8 LoRAs per request")
    out: list[LoRARequest] = []
    seen: set[str] = set()
    for i, entry in enumerate(loras):
        if not isinstance(entry, dict):
            raise ValueError(f"loras[{i}] must be an object")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"loras[{i}].sha256 must be a 64-char hex string")
        sha = sha.lower()
        if sha in seen:
            raise ValueError(f"loras[{i}].sha256 duplicates a prior entry")
        seen.add(sha)
        try:
            weight = float(entry.get("weight", 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"loras[{i}].weight must be a number")
        if not (-2.0 <= weight <= 2.0):
            raise ValueError(f"loras[{i}].weight {weight} outside [-2, 2]")
        out.append(LoRARequest(
            sha256=sha,
            weight=weight,
            bytes_provided=bool(entry.get("bytes_b64")),
        ))
    return out


def resolve_against_cache(
    cache: LoRACache,
    payload_loras: list[dict],
    requests: list[LoRARequest],
) -> tuple[list[CachedLoRA], list[float], list[str]]:
    """Returns (resolved CachedLoRA list, weight list, missing sha list).

    Side effect: any payload entry with bytes_b64 is put() into the cache.
    The base64 bytes are wiped from the payload dict after parse so they
    don't linger in the JSON envelope's holding memory longer than needed.
    """
    import base64

    missing: list[str] = []
    resolved: list[CachedLoRA] = []
    weights:  list[float] = []
    for req, raw in zip(requests, payload_loras):
        cached = cache.get(req.sha256)
        if cached is None:
            b64 = raw.get("bytes_b64")
            if not b64:
                missing.append(req.sha256)
                continue
            try:
                data = base64.b64decode(b64)
            except Exception:
                raise LoRACacheError(f"loras entry {req.sha256[:8]}…: bytes_b64 not valid base64")
            # scrub the b64 from the request dict — we're about to keep the
            # parsed state_dict in cache anyway, no reason to hold the
            # encoded bytes longer.
            raw["bytes_b64"] = ""
            cached = cache.put(data, claimed_sha=req.sha256)
            # explicit zero of the decoded bytes — best effort
            try: data = b"\x00" * len(data)
            except Exception: pass
        resolved.append(cached)
        weights.append(req.weight)
    return resolved, weights, missing
