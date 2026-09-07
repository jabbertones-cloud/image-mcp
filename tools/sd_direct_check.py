"""Exercise server.sd directly (no MCP) at a sane SD1.5 size so output quality can be judged.

    <live venv python> tools\\sd_direct_check.py [size] [steps]

Writes txt2img / img2img (two strengths) / inpaint PNGs to %TEMP%\\imagetools_smoke\\direct_*.png
and prints per-call timings plus a crude "is this noise" score (mean local variance).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from server import sd  # noqa: E402

OUT = Path(os.environ.get("TEMP", ".")) / "imagetools_smoke"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "Lykon/dreamshaper-8"
INPAINT = "Lykon/dreamshaper-8-inpainting"


def noise_score(img: Image.Image) -> float:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    dx = np.abs(np.diff(a, axis=1)).mean()
    dy = np.abs(np.diff(a, axis=0)).mean()
    return round(float((dx + dy) / 2), 2)  # photos ~3-12; pure noise ~40+


def timed(label, fn):
    t0 = time.perf_counter()
    img = fn()
    dt = time.perf_counter() - t0
    p = OUT / f"direct_{label}.png"
    img.save(p)
    print(f"{label:18} {dt:6.1f}s  noise={noise_score(img):5.2f}  -> {p}", flush=True)
    return img


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    base = timed("txt2img", lambda: sd.sd_txt2img(
        "a lighthouse on a rocky coast at sunset, photorealistic", width=size, height=size, steps=steps, seed=7,
        model=MODEL))
    timed("img2img_s0.5", lambda: sd.sd_img2img(base, "the same lighthouse in a snowstorm", strength=0.5,
                                                 steps=steps, seed=7, model=MODEL))
    timed("img2img_s0.75", lambda: sd.sd_img2img(base, "the same lighthouse in a snowstorm", strength=0.75,
                                                  steps=steps, seed=7, model=MODEL))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=255)
    timed("inpaint", lambda: sd.sd_inpaint(base, mask, "a red hot air balloon", steps=steps, seed=7, model=INPAINT))
    print(sd.sd_status())


if __name__ == "__main__":
    main()
