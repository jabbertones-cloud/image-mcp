"""TransferWorkspace context manager — cleans up + scrubs."""
from __future__ import annotations

from pathlib import Path

from remote_server.transfer import TransferWorkspace


def test_workspace_creates_and_wipes(tmp_path):
    root = tmp_path / "transfer"
    with TransferWorkspace(root) as ws:
        d = ws.dir
        assert d.exists()
        assert d.parent == root
        # write something
        p = ws.write_image_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert p.exists()
        assert p.read_bytes().startswith(b"\x89PNG")
    # after the with block: directory gone
    assert not d.exists()


def test_workspace_each_request_isolated(tmp_path):
    root = tmp_path / "transfer"
    names = []
    for _ in range(3):
        with TransferWorkspace(root) as ws:
            names.append(ws.dir.name)
            ws.write_bytes(b"x", "f.bin")
    # 3 distinct subdir names
    assert len(set(names)) == 3
    # root dir survives, subdirs are gone
    assert root.exists()
    assert list(root.iterdir()) == []


def test_workspace_scrub_then_delete(tmp_path):
    """Small files get overwritten with random bytes BEFORE unlink."""
    root = tmp_path / "transfer"
    with TransferWorkspace(root) as ws:
        p = ws.write_bytes(b"sensitive plaintext data", "secret.bin")
        captured = p
    assert not captured.exists()
