"""Server-side LoRA fetch tests — mock httpx so we don't touch the network."""
from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from remote_server.lora_cache import LoRACache
from remote_server.lora_fetch import (
    LoRAFetchError, _parse_civitai_url, fetch_from_civitai, fetch_from_hf,
)


def _import_fetch():
    """Defeats the provisioned-fixture module purge: returns the current
    lora_fetch module (re-importing if necessary) so monkeypatched
    module-level constants take effect."""
    import importlib, remote_server.lora_fetch as lf
    return importlib.reload(lf) if "remote_server.lora_fetch" in sys.modules else lf


# ─── helpers ─────────────────────────────────────────────────────────


@contextmanager
def _fake_stream(payload: bytes, *, status: int = 200,
                 content_length: str | None = None,
                 chunk_size: int = 64 * 1024):
    """httpx.Client(...) → cli.stream(...) context manager returning a
    response object with iter_bytes() yielding ``payload`` in chunks."""
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    if content_length is not None:
        response.headers["content-length"] = content_length
    response.text = ""
    response.iter_bytes.return_value = iter(
        payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)
    )

    @contextmanager
    def stream_cm(method, url, **kw):
        yield response

    client = MagicMock()
    client.stream = stream_cm
    client.__enter__.return_value = client
    client.__exit__.return_value = False

    with patch("remote_server.lora_fetch.httpx.Client", return_value=client):
        yield response


# ─── HuggingFace fetcher ─────────────────────────────────────────────


def test_fetch_from_hf_round_trip(tmp_path):
    cache = LoRACache(tmp_path)
    blob = b"FAKE-LORA-PAYLOAD-" + b"\x00" * 4096
    expected_sha = hashlib.sha256(blob).hexdigest()
    with _fake_stream(blob, content_length=str(len(blob))):
        res = fetch_from_hf(
            cache, repo="user/repo", filename="pytorch_lora_weights.safetensors",
        )
    assert res.sha256 == expected_sha
    assert res.bytes_read == len(blob)
    assert res.source == "hf"
    assert "user/repo" in res.upstream_url
    assert cache.has(expected_sha)


def test_fetch_from_hf_rejects_oversize_content_length(tmp_path, monkeypatch):
    lf = _import_fetch()
    monkeypatch.setattr(lf, "MAX_BYTES_PER_LORA", 1024)
    cache = LoRACache(tmp_path)
    with patch("remote_server.lora_fetch.httpx.Client") as cls:
        with _stream_via(cls, b"x", content_length="9999"):
            with pytest.raises(lf.LoRAFetchError):
                lf.fetch_from_hf(cache, repo="user/r", filename="x.safetensors")


def test_fetch_from_hf_rejects_oversize_streamed(tmp_path, monkeypatch):
    lf = _import_fetch()
    monkeypatch.setattr(lf, "MAX_BYTES_PER_LORA", 1024)
    cache = LoRACache(tmp_path)
    payload = b"x" * 4096
    with patch("remote_server.lora_fetch.httpx.Client") as cls:
        with _stream_via(cls, payload):
            with pytest.raises(lf.LoRAFetchError):
                lf.fetch_from_hf(cache, repo="user/r", filename="x.safetensors")


@contextmanager
def _stream_via(client_class_mock, payload: bytes, *,
                status: int = 200, content_length: str | None = None,
                chunk_size: int = 64 * 1024):
    """Variant of _fake_stream that wires through an already-obtained
    httpx.Client mock — needed when we monkeypatch the module first."""
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    if content_length is not None:
        response.headers["content-length"] = content_length
    response.text = ""
    response.iter_bytes.return_value = iter(
        payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)
    )

    @contextmanager
    def stream_cm(method, url, **kw):
        yield response

    client = MagicMock()
    client.stream = stream_cm
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client_class_mock.return_value = client
    try:
        yield response
    finally:
        pass


def test_fetch_from_hf_rejects_bad_status(tmp_path):
    cache = LoRACache(tmp_path)
    with _fake_stream(b"oops", status=404):
        with pytest.raises(LoRAFetchError) as exc:
            fetch_from_hf(cache, repo="u/r", filename="x.safetensors")
    assert "404" in str(exc.value)


def test_fetch_from_hf_requires_filename(tmp_path):
    cache = LoRACache(tmp_path)
    with pytest.raises(LoRAFetchError):
        fetch_from_hf(cache, repo="u/r", filename="")
    with pytest.raises(LoRAFetchError):
        fetch_from_hf(cache, repo="not-a-repo", filename="x.safetensors")


def test_fetch_from_hf_uses_revision_in_url(tmp_path):
    cache = LoRACache(tmp_path)
    with _fake_stream(b"\x00" * 64) as resp:
        res = fetch_from_hf(cache, repo="u/r", filename="x.safetensors",
                            revision="abc1234")
    assert "abc1234" in res.upstream_url


# ─── Civitai fetcher ─────────────────────────────────────────────────


def test_civitai_url_parser():
    assert _parse_civitai_url("https://civitai.com/models/12345") == (12345, None)
    assert _parse_civitai_url("https://civitai.com/models/12345/some-slug") == (12345, None)
    assert _parse_civitai_url(
        "https://civitai.com/models/12345?modelVersionId=67890"
    ) == (12345, 67890)
    with pytest.raises(LoRAFetchError):
        _parse_civitai_url("https://example.com/")


def test_fetch_from_civitai_with_explicit_version_id(tmp_path):
    cache = LoRACache(tmp_path)
    blob = b"CIVITAI-LORA-" + b"\x00" * 2048
    expected = hashlib.sha256(blob).hexdigest()
    with _fake_stream(blob, content_length=str(len(blob))):
        res = fetch_from_civitai(cache, version_id=42)
    assert res.sha256 == expected
    assert "42" in res.upstream_url
    assert res.source == "civitai"


def test_fetch_from_civitai_resolves_model_id_to_latest_version(tmp_path):
    """When the caller passes only model_id, the fetcher first hits the
    /api/v1/models/<id> endpoint, picks modelVersions[0].id, then streams."""
    cache = LoRACache(tmp_path)
    blob = b"FROM-MODEL-ID-" + b"\x00" * 512
    expected = hashlib.sha256(blob).hexdigest()

    # mock the lookup call (httpx.Client.get) and the download stream
    lookup_resp = MagicMock()
    lookup_resp.status_code = 200
    lookup_resp.json.return_value = {
        "modelVersions": [{"id": 9999}, {"id": 8888}],
    }
    lookup_client = MagicMock()
    lookup_client.get.return_value = lookup_resp
    lookup_client.__enter__.return_value = lookup_client
    lookup_client.__exit__.return_value = False

    # We want the FIRST httpx.Client call (lookup) to return lookup_client
    # and the SECOND call (stream) to return the streaming client.
    stream_response = MagicMock()
    stream_response.status_code = 200
    stream_response.headers = {"content-length": str(len(blob))}
    stream_response.iter_bytes.return_value = iter([blob])

    @contextmanager
    def stream_cm(method, url, **kw):
        yield stream_response

    stream_client = MagicMock()
    stream_client.stream = stream_cm
    stream_client.__enter__.return_value = stream_client
    stream_client.__exit__.return_value = False

    clients = iter([lookup_client, stream_client])

    def client_factory(*a, **kw):
        return next(clients)

    with patch("remote_server.lora_fetch.httpx.Client", side_effect=client_factory):
        res = fetch_from_civitai(cache, model_id=12345)
    assert res.sha256 == expected
    assert "9999" in res.upstream_url    # used the latest version


def test_fetch_from_civitai_needs_some_identifier(tmp_path):
    cache = LoRACache(tmp_path)
    with pytest.raises(LoRAFetchError):
        fetch_from_civitai(cache)


def test_fetch_from_civitai_propagates_url_parse_errors(tmp_path):
    cache = LoRACache(tmp_path)
    with pytest.raises(LoRAFetchError):
        fetch_from_civitai(cache, civitai_url="https://example.com/no-ids-here")
