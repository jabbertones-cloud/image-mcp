"""Offline unit tests for the default-deny auth middleware. Exercises the
middleware coroutine directly with fakes — the real app lifespan installs OS
lockdowns + a tripwire and loads a GPU pipeline, which must not run in tests."""
import asyncio
import types

import pytest

from remote_server import app as app_mod


class _FakeURL:
    def __init__(self, path):
        self.path = path


class _FakeRequest:
    def __init__(self, path, method="POST", headers=None):
        self.url = _FakeURL(path)
        self.method = method
        self.headers = headers or {}


async def _call_next_sentinel(request):
    return "PASSED_THROUGH"


def _run(path, headers=None, method="POST", tripwire=None, monkeypatch=None):
    monkeypatch.setattr(app_mod, "get_tripwire", lambda: tripwire)
    req = _FakeRequest(path, method=method, headers=headers)
    return asyncio.run(app_mod._default_deny(req, _call_next_sentinel))


PROOF = {"X-RGEN-Request-Id": "rid", "X-RGEN-Proof": "proof"}


def test_healthz_is_unauthenticated(monkeypatch):
    assert _run("/healthz", headers={}, method="GET", monkeypatch=monkeypatch) == "PASSED_THROUGH"


def test_pubkey_is_unauthenticated(monkeypatch):
    assert _run("/v1/pubkey", headers={}, method="GET", monkeypatch=monkeypatch) == "PASSED_THROUGH"


def test_protected_route_without_proof_is_401(monkeypatch):
    resp = _run("/v1/generate/txt2img", headers={}, monkeypatch=monkeypatch)
    assert resp.status_code == 401


def test_protected_route_with_proof_passes(monkeypatch):
    assert _run("/v1/loras", headers=PROOF, monkeypatch=monkeypatch) == "PASSED_THROUGH"


def test_unknown_new_route_is_default_denied(monkeypatch):
    # A hypothetical future route that forgot Depends(require_proof) is still
    # blocked by the middleware — the whole point of default-deny.
    resp = _run("/v1/some/new/debug/route", headers={}, monkeypatch=monkeypatch)
    assert resp.status_code == 401


def test_compromised_pod_blocks_even_with_proof(monkeypatch):
    tw = types.SimpleNamespace(compromised=True)
    resp = _run("/v1/generate/txt2img", headers=PROOF, tripwire=tw, monkeypatch=monkeypatch)
    assert resp.status_code == 503
    assert "compromised" in resp.body.decode().lower()


def test_options_preflight_allowed(monkeypatch):
    assert _run("/v1/loras", headers={}, method="OPTIONS", monkeypatch=monkeypatch) == "PASSED_THROUGH"
