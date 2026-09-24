#!/usr/bin/env python3
"""Inpaint / soft-clear the center rings so 3D layers can composite cleanly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path("/workspace/ophanim")
ASSET = ROOT / "assets"
MASK = ROOT / "masks"


def inpaint_plate(art_path: Path, hole_path: Path, out_path: Path) -> None:
    art = np.array(Image.open(art_path).convert("RGB")).astype(np.float32)
    hole = np.array(Image.open(hole_path).convert("L")).astype(np.float32) / 255.0
    # Strong blur of whole image as nebula fill
    blurred = np.array(
        Image.open(art_path).convert("RGB").filter(ImageFilter.GaussianBlur(radius=28))
    ).astype(np.float32)
    # Also sample mean color of near-center non-hole ring of nebula (red glow)
    h, w = hole.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    ring = (hole < 0.15) & ((((xx - cx) / (w * 0.2)) ** 2 + ((yy - cy) / (h * 0.2)) ** 2) < 1.5)
    if ring.any():
        mean = art[ring].mean(axis=0)
    else:
        mean = np.array([80.0, 20.0, 15.0])
    fill = blurred * 0.65 + mean.reshape(1, 1, 3) * 0.35
    # Expand hole a bit
    hole_s = ndimage.gaussian_filter(hole, sigma=8)
    hole_s = np.clip(hole_s * 1.15, 0, 1)[..., None]
    out = art * (1 - hole_s) + fill * hole_s
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)
    print("wrote", out_path)


def main() -> None:
    inpaint_plate(
        ASSET / "arte_computador_1920x1080.png",
        MASK / "computador_center_hole.png",
        ASSET / "bg_computador.png",
    )
    inpaint_plate(
        ASSET / "arte_celular_1080x1920.png",
        MASK / "celular_center_hole.png",
        ASSET / "bg_celular.png",
    )


if __name__ == "__main__":
    main()
