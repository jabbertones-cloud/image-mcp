"""Filesystem boundary and atomic-write helpers for ImageTools MCP.

Behavior follows the hardened path contract audited from image-gen-mcp:
canonicalize configured roots, reject traversal/symlink escapes, and permit a
future output only when its nearest existing ancestor resolves inside the root.
This module is an independent Python implementation.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


class PathRootViolation(ValueError):
    pass


def configured_root() -> Path | None:
    raw = os.environ.get("IMAGETOOLS_INPUT_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.exists():
        raise PathRootViolation(f"configured input root does not exist: {root}")
    return root.resolve(strict=True)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_within_root(path: str | os.PathLike[str], *, allow_missing: bool = False) -> Path:
    root = configured_root()
    candidate = Path(path).expanduser()
    if root is None:
        return candidate.resolve(strict=not allow_missing)

    absolute = candidate if candidate.is_absolute() else root / candidate
    if absolute.exists():
        resolved = absolute.resolve(strict=True)
        if not _inside(resolved, root):
            raise PathRootViolation(f"path escapes configured root: {path}")
        return resolved

    if not allow_missing:
        raise FileNotFoundError(f"no such file: {path}")

    # A future path is safe only when both its lexical location and the real
    # nearest existing ancestor remain under the canonical root. This rejects
    # existing symlink components that point outside the root.
    lexical = Path(os.path.abspath(os.path.normpath(str(absolute))))
    if not _inside(lexical, root):
        raise PathRootViolation(f"path escapes configured root: {path}")

    ancestor = lexical
    missing_parts: list[str] = []
    while not ancestor.exists():
        missing_parts.append(ancestor.name)
        parent = ancestor.parent
        if parent == ancestor:
            raise PathRootViolation(f"cannot resolve path under configured root: {path}")
        ancestor = parent
    real_ancestor = ancestor.resolve(strict=True)
    if not _inside(real_ancestor, root):
        raise PathRootViolation(f"path escapes configured root via symlink: {path}")
    resolved = real_ancestor.joinpath(*reversed(missing_parts))
    if not _inside(resolved, root):
        raise PathRootViolation(f"path escapes configured root: {path}")
    return resolved


def atomic_write(path: str | os.PathLike[str], writer: Callable[[Path], T]) -> T:
    """Write beside the destination, then atomically replace it.

    The writer receives a temporary path in the destination directory. A
    failed writer leaves the prior destination untouched and the temp file is
    removed best-effort.
    """
    dest = assert_within_root(path, allow_missing=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        result = writer(tmp)
        os.replace(tmp, dest)
        return result
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
