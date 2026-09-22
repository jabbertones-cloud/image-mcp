from PIL import Image
from server.artifact import artifact_descriptor, photoshop_handoff


def test_image_artifact_descriptor_is_bounded_and_verifiable(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    p = tmp_path / "x.png"
    Image.new("RGB", (12, 7), "red").save(p)
    d = artifact_descriptor(p)
    assert d["mimeType"] == "image/png"
    assert (d["width"], d["height"]) == (12, 7)
    assert len(d["sha256"]) == 64
    assert "bytes" not in d and "base64" not in d


def test_photoshop_handoff_uses_artifact_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    p = tmp_path / "master.png"
    Image.new("RGBA", (4, 5)).save(p)
    h = photoshop_handoff(p)
    assert h["kind"] == "photoshop_handoff"
    assert h["artifact"]["path"] == str(p.resolve())
    assert h["contractVersion"] == 1
