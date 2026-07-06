"""Tests for adaptive preview encoding, inspect_region, canvases_overview,
the store-wide undo byte budget, and the generated-image placement helper."""
import io

import pytest
from PIL import Image

from server import canvas as canvas_mod
from server import image_tools_server as srv
from server.canvas import store


@pytest.fixture(autouse=True)
def clean_store():
    for cid in list(store.list_ids()):
        store.close(cid)
    yield
    for cid in list(store.list_ids()):
        store.close(cid)


def _noise_image(w=1024, h=1024):
    """Photographic-ish content: random noise defeats PNG compression."""
    import random

    img = Image.new("RGB", (w, h))
    img.putdata([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                 for _ in range(w * h)])
    return img


# ── preview encoding ────────────────────────────────────────────────

def test_encode_preview_small_graphics_stay_png():
    img = Image.new("RGB", (500, 500), (200, 30, 30))
    data, fmt, size = srv._encode_preview(img, max_size=512)
    assert fmt == "png"
    assert size == (500, 500)
    assert img.width * img.height <= srv._PREVIEW_PNG_PIXELS


def test_encode_preview_photo_goes_jpeg_under_budget():
    data, fmt, size = srv._encode_preview(_noise_image(), max_size=1024)
    assert fmt == "jpeg"
    assert len(data) <= srv._PREVIEW_BUDGET


def test_encode_preview_transparency_stays_png():
    img = _noise_image(700, 700).convert("RGBA")
    img.putalpha(Image.new("L", img.size, 128))
    data, fmt, size = srv._encode_preview(img, max_size=700)
    assert fmt == "png"
    assert Image.open(io.BytesIO(data)).mode == "RGBA"


def test_encode_preview_lossless_forces_png():
    data, fmt, size = srv._encode_preview(_noise_image(700, 700), max_size=700, lossless=True)
    assert fmt == "png"


def test_encode_preview_reports_true_downscaled_size():
    # A large opaque noise image forced lossless can't be JPEG'd, so it must be
    # shrunk to fit the budget — the reported size must reflect that shrink.
    data, fmt, (ow, oh) = srv._encode_preview(_noise_image(1600, 1600), lossless=True)
    assert fmt == "png"
    assert len(data) <= srv._PREVIEW_BUDGET
    assert max(ow, oh) < 1600, "oversized lossless preview must report its shrunk size"


# ── inspect_region ──────────────────────────────────────────────────

def test_inspect_region_native_resolution():
    img = Image.new("RGB", (2000, 1500), (10, 10, 10))
    img.paste(Image.new("RGB", (50, 50), (255, 0, 0)), (100, 200))
    cid = store.put_image(img)
    note, mcp_img = srv.inspect_region(cid, 90, 190, 200, 200)
    assert "1:1" in note
    out = Image.open(io.BytesIO(mcp_img.data))
    assert out.size == (200, 200)
    assert out.convert("RGB").getpixel((20, 20)) == (255, 0, 0)  # colour-exact (lossless)


def test_inspect_region_reports_real_scale_when_downscaled():
    # A detail-rich region bigger than the budget is downscaled; the note must
    # say so rather than claiming 1:1 (the old bug).
    cid = store.put_image(_noise_image(1600, 1600))
    note, mcp_img = srv.inspect_region(cid, 0, 0, 1600, 1600, max_size=1600)
    out = Image.open(io.BytesIO(mcp_img.data))
    if max(out.size) < 1600:
        assert "downscaled" in note and "at 1:1 pixel scale" not in note
    else:
        assert "at 1:1 pixel scale" in note


def test_inspect_region_clamps_and_rejects_outside():
    cid = store.put_image(Image.new("RGB", (100, 100)))
    note, mcp_img = srv.inspect_region(cid, 50, 50, 500, 500)
    assert Image.open(io.BytesIO(mcp_img.data)).size == (50, 50)
    with pytest.raises(ValueError, match="outside"):
        srv.inspect_region(cid, 200, 0, 10, 10)


def test_inspect_region_matches_full_compose():
    # Region compositing must equal cropping the full composite.
    img = _noise_image(300, 300)
    cid = store.put_image(img)
    region = store.compose_region(cid, (40, 60, 140, 160))
    full = store.compose(cid).crop((40, 60, 140, 160))
    assert region.tobytes() == full.tobytes()


# ── canvases_overview ───────────────────────────────────────────────

def test_canvases_overview_labels_all():
    ids = [store.put_image(Image.new("RGB", (64, 64), (i * 40, 0, 0))) for i in range(3)]
    res = srv.canvases_overview()
    assert "3 of 3" in res[0]
    assert Image.open(io.BytesIO(res[1].data)).width > 0 and ids


def test_canvases_overview_empty_store():
    res = srv.canvases_overview()
    assert "No canvases" in res[0]


# ── undo/redo history budget ────────────────────────────────────────

def test_undo_history_byte_budget(monkeypatch):
    monkeypatch.setattr(canvas_mod, "MAX_HISTORY_MB", 1)  # 1 MB global budget
    cid = store.put_image(Image.new("RGB", (400, 400)))   # ~640 KB per snapshot
    for _ in range(5):
        store.snapshot(cid)
    e = store.entry(cid)
    assert len(e.undo_stack) == 1, "budget should evict all but the newest snapshot"
    assert store.undo(cid) is True, "one undo must always remain possible"


def test_undo_budget_is_store_wide(monkeypatch):
    # Two canvases share one global budget; total undo bytes must stay bounded.
    monkeypatch.setattr(canvas_mod, "MAX_HISTORY_MB", 2)
    a = store.put_image(Image.new("RGB", (500, 500)))  # ~1 MB per snapshot
    b = store.put_image(Image.new("RGB", (500, 500)))
    for _ in range(4):
        store.snapshot(a)
        store.snapshot(b)
    total = sum(s.cost for cid in store.list_ids()
                for s in store.entry(cid).undo_stack)
    assert total <= 2 * 1024 * 1024
    # each canvas keeps at least one undo step
    assert len(store.entry(a).undo_stack) >= 1
    assert len(store.entry(b).undo_stack) >= 1


def test_redo_keeps_newest_state_under_pressure(monkeypatch):
    # Regression: byte-trimming the redo stack from index 0 dropped the newest
    # state, so a full redo-forward could never reach where you started.
    monkeypatch.setattr(canvas_mod, "MAX_HISTORY_MB", 1)
    cid = store.put_image(Image.new("RGB", (300, 300), (0, 0, 0)))
    colors = [(10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)]
    for c in colors:
        store.snapshot(cid)
        store.replace_active_image(cid, Image.new("RGBA", (300, 300), c + (255,)))
    newest = store.compose(cid).getpixel((0, 0))
    # undo all the way, then redo all the way back
    while store.undo(cid):
        pass
    while store.redo(cid):
        pass
    assert store.compose(cid).getpixel((0, 0)) == newest, \
        "redo must return to the newest state, not a budget-evicted stale one"


def test_undo_step_cap_still_applies(monkeypatch):
    monkeypatch.setattr(canvas_mod, "MAX_HISTORY_MB", 10_000)
    cid = store.put_image(Image.new("RGB", (8, 8)))
    for _ in range(canvas_mod.MAX_HISTORY + 10):
        store.snapshot(cid)
    assert len(store.entry(cid).undo_stack) == canvas_mod.MAX_HISTORY
