"""Round-trip saves across formats; smoke-test info + supported_formats."""
from pathlib import Path

import pytest
from PIL import Image

from server import io_formats


@pytest.mark.parametrize("ext,quality", [
    ("png", None),
    ("jpg", 80),
    ("bmp", None),
    ("webp", 80),
    ("tiff", None),
    ("gif", None),
])
def test_save_load_roundtrip(tmp_path: Path, ext: str, quality):
    img = Image.new("RGB", (40, 30), "orange")
    out = tmp_path / f"out.{ext}"
    info = io_formats.save_image(img, str(out), quality=quality)
    assert Path(info["path"]).exists()
    loaded = io_formats.load_image(str(out))
    assert loaded.size == (40, 30)


def test_jpeg_flattens_alpha(tmp_path: Path):
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    info = io_formats.save_image(img, str(tmp_path / "x.jpg"))
    assert info["mode"] == "RGB"  # alpha flattened


def test_image_info_returns_dims_and_format(tmp_path: Path):
    p = tmp_path / "x.png"
    Image.new("RGB", (12, 8), "blue").save(p)
    info = io_formats.image_info(str(p))
    assert info["width"] == 12 and info["height"] == 8
    assert info["format"] == "PNG"


def test_supported_formats_lists_pillow_extensions():
    info = io_formats.supported_formats()
    for must_have in ("png", "jpg", "bmp", "webp", "tiff", "gif"):
        assert must_have in info["load"], f"missing {must_have} in load list"
        assert must_have in info["save"], f"missing {must_have} in save list"


def test_save_rejects_svg_and_raw(tmp_path: Path):
    img = Image.new("RGB", (4, 4), "white")
    with pytest.raises(ValueError):
        io_formats.save_image(img, str(tmp_path / "x.svg"))
    with pytest.raises(ValueError):
        io_formats.save_image(img, str(tmp_path / "x.cr2"))


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        io_formats.load_image(str(tmp_path / "nope.png"))


def test_input_root_allows_file_inside_root(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    p = root / "inside.png"
    Image.new("RGB", (2, 2), "red").save(p)
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(root))
    assert io_formats.load_image(str(p)).size == (2, 2)


def test_input_root_rejects_traversal(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2), "red").save(outside)
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(root))
    with pytest.raises(io_formats.PathRootViolation):
        io_formats.load_image(str(root / ".." / "outside.png"))


def test_input_root_rejects_symlink_escape(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2), "red").save(outside)
    link = root / "link.png"
    link.symlink_to(outside)
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(root))
    with pytest.raises(io_formats.PathRootViolation):
        io_formats.load_image(str(link))


def test_output_root_rejects_escape(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("IMAGETOOLS_OUTPUT_ROOT", str(root))
    with pytest.raises(io_formats.PathRootViolation):
        io_formats.save_image(
            Image.new("RGB", (2, 2), "red"),
            str(tmp_path / "outside.png"),
        )


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path: Path):
    target = tmp_path / "atomic.png"
    io_formats.save_image(Image.new("RGB", (2, 2), "red"), str(target))
    assert target.exists()
    assert not (tmp_path / ".atomic.png.tmp").exists()
