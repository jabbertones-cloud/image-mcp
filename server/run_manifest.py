"""Durable, bounded operation manifests for MCP image work."""
from __future__ import annotations
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .path_safety import atomic_write

RunStatus = Literal["in_progress", "success", "partial", "error"]
NodeOutcome = Literal["success", "error", "skipped"]


@dataclass
class ErrorDetail:
    message: str
    code: str | None = None
    retryable: bool | None = None
    errorClass: str | None = None
    suggestion: str | None = None


@dataclass
class RunNode:
    id: str
    op: str
    outcome: NodeOutcome
    artifactPath: str | None = None
    startedAtMs: int | None = None
    endedAtMs: int | None = None
    durationMs: int | None = None
    error: ErrorDetail | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    runId: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: RunStatus = "in_progress"
    startedAtMs: int = field(default_factory=lambda: int(time.time() * 1000))
    endedAtMs: int | None = None
    nodes: list[RunNode] = field(default_factory=list)
    finalOutput: dict[str, Any] | None = None
    error: ErrorDetail | None = None
    schemaVersion: int = 1

    def finish(self, status: RunStatus, *, final_output=None, error=None) -> None:
        self.status = status
        self.endedAtMs = int(time.time() * 1000)
        self.finalOutput = final_output
        self.error = error


def _jsonable(manifest: RunManifest) -> dict[str, Any]:
    return asdict(manifest)


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    payload = json.dumps(_jsonable(manifest), indent=2, sort_keys=True).encode()
    atomic_write(path, lambda tmp: tmp.write_bytes(payload))
