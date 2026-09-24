#!/usr/bin/env python3
"""Inpaint only the ring/eye band so nebula energy of the arte escura remains."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path("/workspace/ophanim")
ASSET = ROOT / "assets"
MASK = ROOT / "masks"


def inpaint_plate(art_path: Path, hole_path: Path, out_path: Path, portrait: bool) -> None:
    art = np.array(Image.open(art_path).convert("RGB")).astype(np.float32)
    hole = np.array(Image.open(hole_path).convert("L")).astype(np.float32) / 255.0
    h, w = hole.shape
    # Stronger blur fill from surrounding nebula — keep red/orange character
    blurred = np.array(
        Image.open(art_path).convert("RGB").filter(ImageFilter.GaussianBlur(radius=36))
    ).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h * (0.463 if portrait else 0.415)
    # Sample fiery nebula just outside the hole
    ring = (hole < 0.12) & (
        (((xx - cx) / (w * 0.22)) ** 2 + ((yy - cy) / (h * 0.22)) ** 2) < 1.8
    )
    if ring.any():
        mean = art[ring].mean(axis=0)
    else:
        mean = np.array([90.0, 25.0, 18.0], dtype=np.float32)
    fill = blurred * 0.72 + mean.reshape(1, 1, 3) * 0.28
    # Soften hole so edges don't cut the hourglass glow
    hole_s = ndimage.gaussian_filter(np.clip(hole * 1.05, 0, 1), sigma=10)[..., None]
    out = art * (1 - hole_s) + fill * hole_s
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)
    print("wrote", out_path)


def main() -> None:
    inpaint_plate(
        ASSET / "arte_computador_1920x1080.png",
        MASK / "computador_center_hole.png",
        ASSET / "bg_computador.png",
        False,
    )
    inpaint_plate(
        ASSET / "arte_celular_1080x1920.png",
        MASK / "celular_center_hole.png",
        ASSET / "bg_celular.png",
        True,
    )


if __name__ == "__main__":
    main()
