"""Fail-closed Photoshop readiness using the DCC donor's real host-RPC contract."""
from __future__ import annotations
import os
from typing import Any


def probe_photoshop(endpoint: str | None = None, *, timeout: float = 2.0) -> dict[str, Any]:
    """PHOTOSHOP_READY means a real typed Photoshop host RPC succeeded.

    A broker HTTP response alone is deliberately insufficient.
    """
    broker_url = endpoint or os.environ.get("ADOBEPY_BROKER_URL", "http://127.0.0.1:47391")
    try:
        from dcc_mcp_photoshop.runtime_probe import probe_broker, probe_photoshop as probe_host
    except ImportError:
        return {"ready": False, "status": "runtime_unavailable"}

    broker = probe_broker(broker_url, timeout)
    if not broker.get("ok"):
        return {"ready": False, "status": broker.get("error_type", "broker_failed"), "broker": broker}
    if broker.get("sessions", 0) != 1:
        return {
            "ready": False,
            "status": "bridge_session_missing" if broker.get("sessions", 0) == 0 else "ambiguous_bridge_sessions",
            "broker": broker,
        }

    host = probe_host(broker_url, timeout)
    if not host.get("ok"):
        return {"ready": False, "status": host.get("error_type", "host_rpc_failed"), "broker": broker, "host": host}
    return {"ready": True, "status": "PHOTOSHOP_READY", "broker": broker, "host": host}
