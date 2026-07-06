"""Server-side LoRA fetch from HuggingFace or Civitai.

The pod has the bandwidth; the client doesn't. Letting the server pull
LoRAs directly avoids the user's home upload being a bottleneck and keeps
sensitive auth tokens (HF / Civitai) in the pod's RAM rather than echoing
through every request.

Both fetchers stream into ``MAX_BYTES_PER_LORA``-bounded ``bytearray``
buffers; if the upstream advertises a larger Content-Length we abort
before reading a single byte of payload. The fetched bytes are then
``cache.put()``-ed and the buffer is zeroed.

Auth handling:
  * HF: optional ``REMOTE_HF_TOKEN`` env var, or per-call override.
  * Civitai: optional ``REMOTE_CIVITAI_API_KEY`` env var, or per-call
    override (required for some gated models).

Neither token is ever logged.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from remote_server.lora_cache import (
    CachedLoRA, LoRACache, LoRACacheError, MAX_BYTES_PER_LORA,
)

log = logging.getLogger(__name__)


class LoRAFetchError(Exception):
    pass


@dataclass
class FetchResult:
    sha256: str
    bytes_read: int
    source:    str             # "hf" | "civitai"
    upstream_url: str          # the actual URL we hit, for diagnostics


# ─── streaming utility ─────────────────────────────────────────────


def _stream_to_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 600.0,
    follow_redirects: bool = True,
) -> bytes:
    """Stream a download into a single bytearray, bounded by MAX_BYTES_PER_LORA.

    Returns the raw bytes (defensive copy) — the bytearray is zeroed before
    return to keep the original buffer out of the process's heap traces."""
    with httpx.Client(timeout=timeout_s, follow_redirects=follow_redirects) as cli:
        with cli.stream("GET", url, headers=headers) as resp:
            if resp.status_code != 200:
                # Try to surface a useful error message without exposing auth
                # via stack trace.
                detail = (resp.text[:200] if resp.headers.get("content-type", "").startswith("text") else "")
                raise LoRAFetchError(
                    f"upstream {resp.status_code} for {url}: {detail}"
                )
            # MAX_BYTES_PER_LORA == 0 disables the cap entirely (default).
            cap = MAX_BYTES_PER_LORA
            adv = resp.headers.get("content-length")
            if cap > 0 and adv is not None:
                try:
                    if int(adv) > cap:
                        raise LoRAFetchError(
                            f"advertised size {adv} exceeds cap {cap}"
                        )
                except ValueError:
                    pass     # invalid Content-Length, just stream & cap

            buf = bytearray()
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                buf.extend(chunk)
                if cap > 0 and len(buf) > cap:
                    raise LoRAFetchError(
                        f"streamed body exceeded cap {cap}"
                    )
            out = bytes(buf)
            # zero the bytearray (defence in depth: the heap copy gets cleared
            # before GC takes it).
            for i in range(len(buf)): buf[i] = 0
            return out


# ─── HuggingFace ────────────────────────────────────────────────────


_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_HF_REV_RE  = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _safe_url_component(name: str, value: str, *, allow_slash: bool) -> str:
    """Reject anything that could redirect the authenticated request off the
    intended huggingface.co/<repo>/resolve/<rev>/<file> path: URL metacharacters,
    backslashes, whitespace, and ``.`` / ``..`` path segments."""
    if any(c in value for c in ("?", "#", "\\", "@", " ", "\t", "\n", "\r")):
        raise LoRAFetchError(f"illegal characters in {name}: {value!r}")
    segments = value.split("/") if allow_slash else [value]
    if not allow_slash and "/" in value:
        raise LoRAFetchError(f"{name} may not contain '/': {value!r}")
    if any(seg in ("", ".", "..") for seg in segments):
        raise LoRAFetchError(f"path traversal or empty segment in {name}: {value!r}")
    return value


def fetch_from_hf(
    cache: LoRACache,
    *,
    repo:      str,
    filename:  str,
    revision:  str | None = None,
    token:     str | None = None,
) -> FetchResult:
    """Pull a single LoRA file from HuggingFace and put() it into the cache.

    Uses the resolve URL directly (avoids huggingface_hub dependency for
    a one-file pull). ``token`` overrides REMOTE_HF_TOKEN env."""
    if not repo or not _HF_REPO_RE.match(repo):
        raise LoRAFetchError(f"bad repo spec: {repo!r} (expected 'namespace/name')")
    if not filename:
        raise LoRAFetchError("filename is required")
    rev = revision or "main"
    if not _HF_REV_RE.match(rev):
        raise LoRAFetchError(f"bad revision: {rev!r}")
    # repo already regex-checked (single '/', safe charset); guard rev + filename.
    _safe_url_component("revision", rev, allow_slash=True)
    _safe_url_component("filename", filename, allow_slash=True)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{filename}"

    headers: dict[str, str] = {"User-Agent": "remote-gen-server"}
    auth = token or os.environ.get("REMOTE_HF_TOKEN")
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    raw = _stream_to_bytes(url, headers=headers)
    entry = cache.put(raw)
    try:
        # zero our local copy
        raw = b"\x00" * len(raw)
    except Exception: pass
    return FetchResult(
        sha256=entry.sha256, bytes_read=entry.size,
        source="hf", upstream_url=url,
    )


# ─── Civitai ────────────────────────────────────────────────────────


_CIVITAI_API   = "https://civitai.com/api/v1"
_CIVITAI_DL    = "https://civitai.com/api/download/models/{version_id}"
_CIVITAI_TOKEN = os.environ.get("REMOTE_CIVITAI_API_KEY", "")


def fetch_from_civitai(
    cache: LoRACache,
    *,
    model_id:   int | None = None,
    version_id: int | None = None,
    civitai_url: str | None = None,
    token:      str | None = None,
) -> FetchResult:
    """Pull a LoRA from Civitai. You can specify any one of:

      * ``version_id`` — fastest, direct download.
      * ``model_id``   — we look up the model's *latest* version.
      * ``civitai_url``— parsed for /models/<id> and ?modelVersionId=<id>.

    Civitai puts the file path behind a redirect; httpx follows by default.
    """
    if version_id is None and model_id is None and not civitai_url:
        raise LoRAFetchError("need one of: version_id, model_id, civitai_url")

    if civitai_url:
        m_id, v_id = _parse_civitai_url(civitai_url)
        version_id = version_id or v_id
        model_id   = model_id   or m_id

    auth = token or _CIVITAI_TOKEN
    headers: dict[str, str] = {"User-Agent": "remote-gen-server"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    if version_id is None:
        # Resolve the model's latest published version.
        if model_id is None:
            raise LoRAFetchError("no version_id and no model_id to look up")
        with httpx.Client(timeout=30.0) as cli:
            r = cli.get(f"{_CIVITAI_API}/models/{model_id}", headers=headers)
        if r.status_code != 200:
            raise LoRAFetchError(f"civitai model lookup {model_id} failed: HTTP {r.status_code}")
        try:
            payload = r.json()
            versions = payload.get("modelVersions") or []
            if not versions:
                raise LoRAFetchError(f"civitai model {model_id} has no versions")
            version_id = int(versions[0]["id"])
        except (KeyError, ValueError, TypeError) as e:
            raise LoRAFetchError(f"unexpected civitai model payload: {e!r}")

    url = _CIVITAI_DL.format(version_id=version_id)
    raw = _stream_to_bytes(url, headers=headers, timeout_s=900.0)
    entry = cache.put(raw)
    try:
        raw = b"\x00" * len(raw)
    except Exception: pass
    return FetchResult(
        sha256=entry.sha256, bytes_read=entry.size,
        source="civitai", upstream_url=url,
    )


def _parse_civitai_url(url: str) -> tuple[int | None, int | None]:
    """Return (model_id, version_id) parsed from a Civitai web URL.

    Accepted shapes:
      https://civitai.com/models/12345
      https://civitai.com/models/12345/some-slug
      https://civitai.com/models/12345?modelVersionId=67890
    """
    m_id: int | None = None
    v_id: int | None = None
    m = re.search(r"/models/(\d+)", url)
    if m: m_id = int(m.group(1))
    m = re.search(r"[?&]modelVersionId=(\d+)", url)
    if m: v_id = int(m.group(1))
    if not m_id and not v_id:
        raise LoRAFetchError(f"can't parse civitai URL: {url!r}")
    return m_id, v_id
