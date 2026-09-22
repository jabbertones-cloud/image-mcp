from pathlib import Path

import pytest

from server.path_safety import PathRootViolation, assert_within_root, atomic_write


def test_unset_root_allows_existing_path(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGETOOLS_INPUT_ROOT", raising=False)
    p = tmp_path / "a.txt"
    p.write_text("x")
    assert assert_within_root(p) == p.resolve()


def test_allows_file_inside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    p = tmp_path / "a.txt"
    p.write_text("x")
    assert assert_within_root(p) == p.resolve()


def test_rejects_traversal_outside_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(root))
    with pytest.raises(PathRootViolation):
        assert_within_root(root / ".." / "outside.txt")


def test_rejects_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "x.txt").write_text("x")
    (root / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(root))
    with pytest.raises(PathRootViolation):
        assert_within_root(root / "link" / "x.txt")


def test_allows_future_path_inside_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(root))
    future = root / "new" / "out.png"
    assert assert_within_root(future, allow_missing=True) == future


def test_root_symlink_is_canonicalized(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    p = real / "x.txt"
    p.write_text("x")
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(alias))
    assert assert_within_root(alias / "x.txt") == p.resolve()


def test_atomic_write_preserves_old_destination_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    dest = tmp_path / "out.txt"
    dest.write_text("old")

    def fail(tmp: Path):
        tmp.write_text("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write(dest, fail)
    assert dest.read_text() == "old"
    assert not list(tmp_path.glob(".out.txt.*.tmp"))


def test_atomic_write_replaces_destination(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    dest = tmp_path / "out.txt"
    dest.write_text("old")

    atomic_write(dest, lambda tmp: tmp.write_text("new"))
    assert dest.read_text() == "new"
