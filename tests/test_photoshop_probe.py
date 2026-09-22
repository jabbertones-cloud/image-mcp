import sys
import types
from unittest.mock import patch
from server.photoshop_probe import probe_photoshop


def install_runtime(monkeypatch, broker, host):
    pkg = types.ModuleType("dcc_mcp_photoshop")
    pkg.__path__ = []
    runtime = types.ModuleType("dcc_mcp_photoshop.runtime_probe")
    runtime.probe_broker = lambda *_: broker
    runtime.probe_photoshop = lambda *_: host
    monkeypatch.setitem(sys.modules, "dcc_mcp_photoshop", pkg)
    monkeypatch.setitem(sys.modules, "dcc_mcp_photoshop.runtime_probe", runtime)


def test_broker_health_alone_never_means_ready(monkeypatch):
    install_runtime(monkeypatch, {"ok": True, "sessions": 0}, {"ok": True, "version": "27"})
    r = probe_photoshop()
    assert not r["ready"] and r["status"] == "bridge_session_missing"


def test_multiple_sessions_fail_closed(monkeypatch):
    install_runtime(monkeypatch, {"ok": True, "sessions": 2}, {"ok": True, "version": "27"})
    assert probe_photoshop()["status"] == "ambiguous_bridge_sessions"


def test_host_rpc_failure_is_not_ready(monkeypatch):
    install_runtime(monkeypatch, {"ok": True, "sessions": 1}, {"ok": False, "error_type": "host_rpc_failed"})
    assert probe_photoshop()["status"] == "host_rpc_failed"


def test_only_real_host_rpc_sets_photoshop_ready(monkeypatch):
    install_runtime(monkeypatch, {"ok": True, "sessions": 1}, {"ok": True, "version": "27.0"})
    r = probe_photoshop()
    assert r["ready"] is True
    assert r["status"] == "PHOTOSHOP_READY"


def test_runtime_missing_is_classified():
    with patch.dict(sys.modules, {"dcc_mcp_photoshop": None}):
        r = probe_photoshop()
    assert not r["ready"]
