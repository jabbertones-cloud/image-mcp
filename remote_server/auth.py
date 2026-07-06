"""Split-PSK proof check + replay defence — same pattern as QwenCharLoRA.

Every protected endpoint depends on ``require_proof``. The body is read once
and cached (FastAPI's ``request.body()`` can only be awaited once safely);
downstream handlers receive ``ProofedRequest.body``.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from remote_server.crypto import verify_request_proof

log = logging.getLogger(__name__)


@dataclass
class ProofedRequest:
    request_id: str
    body: bytes


class ReplayCache:
    def __init__(self, window_seconds: float = 300.0, max_entries: int = 4096):
        self._window = window_seconds
        self._seen: deque[tuple[float, str]] = deque(maxlen=max_entries)
        self._set: set[str] = set()

    def check_and_add(self, request_id: str) -> bool:
        now = time.monotonic()
        while self._seen and (now - self._seen[0][0]) > self._window:
            _, rid = self._seen.popleft()
            self._set.discard(rid)
        if request_id in self._set:
            return False
        self._seen.append((now, request_id))
        self._set.add(request_id)
        return True


_replay_cache: ReplayCache | None = None


def _get_replay_cache() -> ReplayCache:
    global _replay_cache
    if _replay_cache is None:
        from remote_server.app import get_settings
        _replay_cache = ReplayCache(window_seconds=get_settings().replay_window_s)
    return _replay_cache


async def require_proof(request: Request) -> ProofedRequest:
    from remote_server.app import _check_not_compromised, get_combined_psk, get_settings
    # tripwire gate before we do any crypto work — saves CPU when the
    # session has already been marked unsafe.
    _check_not_compromised()
    settings = get_settings()
    combined = get_combined_psk()

    request_id = request.headers.get("X-RGEN-Request-Id", "").strip()
    proof      = request.headers.get("X-RGEN-Proof", "").strip()
    if not request_id or not proof:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-RGEN-Request-Id or X-RGEN-Proof",
        )

    content_length = int(request.headers.get("content-length", "0") or 0)
    if content_length > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"body exceeds {settings.max_upload_bytes} bytes",
        )

    body = await request.body()
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="body exceeded declared limit",
        )

    body_hash = hashlib.sha256(body).hexdigest()
    if not verify_request_proof(combined, request_id, body_hash, proof):
        log.warning(
            f"proof rejected for {request.method} {request.url.path} "
            f"(rid={request_id[:8]}…)"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid PSK proof",
        )

    if not _get_replay_cache().check_and_add(request_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="duplicate request_id (replay)",
        )

    return ProofedRequest(request_id=request_id, body=body)
