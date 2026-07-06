"""Per-request transfer workspace — tmpfs scratch for control images
and any other data-in-flight that has to hit a path.

Pattern:

    with TransferWorkspace(settings.transfer_dir) as ws:
        ctrl_path = ws.write_image_bytes(control_image_bytes)
        pipe(prompt=p, control_image=PIL.Image.open(ctrl_path))
        # ws.__exit__ scrubs + rm -rfs the directory; tmpfs makes this
        # equivalent to releasing pages.
"""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path
from types import TracebackType
from typing import Optional


class TransferWorkspace:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._dir: Path | None = None

    def __enter__(self) -> "TransferWorkspace":
        self._root.mkdir(parents=True, exist_ok=True)
        name = secrets.token_urlsafe(12)
        self._dir = self._root / name
        self._dir.mkdir(parents=True, exist_ok=False)
        try: self._dir.chmod(0o700)
        except OSError: pass
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if not self._dir:
            return
        # belt-and-brace: overwrite small files with random bytes before unlink.
        # tmpfs makes this strictly unnecessary, but it costs micro-seconds and
        # documents intent. Skip large files; tmpfs cleanup is enough there.
        try:
            for p in self._dir.rglob("*"):
                if p.is_file() and p.stat().st_size < (4 * 1024 * 1024):
                    try:
                        with open(p, "rb+") as f:
                            n = f.seek(0, 2); f.seek(0)
                            f.write(secrets.token_bytes(min(n, 1024 * 1024)))
                    except OSError:
                        pass
        except OSError:
            pass
        shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None

    @property
    def dir(self) -> Path:
        assert self._dir is not None, "use inside a `with` block"
        return self._dir

    def write_image_bytes(self, data: bytes, suffix: str = ".png") -> Path:
        """Write raw image bytes (any PIL-decodable format) and return path."""
        name = secrets.token_urlsafe(8) + suffix
        path = self.dir / name
        path.write_bytes(data)
        return path

    def write_bytes(self, data: bytes, name: str) -> Path:
        path = self.dir / name
        path.write_bytes(data)
        return path
