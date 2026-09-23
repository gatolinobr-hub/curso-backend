#!/usr/bin/env python3
"""Prepare upscaled arts, text masks, and textures for the Ophanim wallpaper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy import ndimage

ROOT = Path("/workspace/ophanim")
ASSET = ROOT / "assets"
MASK = ROOT / "masks"
TEX = ROOT / "textures"
MASK.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)


def make_text_mask(im: Image.Image, name: str, portrait: bool) -> dict:
    a = np.array(im).astype(np.float32)
    h, w = a.shape[:2]
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    white = (r > 200) & (g > 200) & (b > 200) & ((r + g + b) > 620)
    lines = (
        (r > 160)
        & (g > 160)
        & (b > 160)
        & (np.abs(r - g) < 25)
        & (np.abs(g - b) < 25)
        & ((r + g + b) > 500)
    )
    protect = ndimage.binary_dilation(white | lines, iterations=2)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    if portrait:
        cy = h * 0.463
        rx, ry = w * 0.42, h * 0.22
    else:
        cy = h * 0.415
        rx, ry = w * 0.30, h * 0.52
    anim = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    force = np.zeros((h, w), dtype=bool)
    if portrait:
        force[: int(h * 0.18), :] = True
        force[int(h * 0.72) :, :] = True
        force[:, : int(w * 0.12)] = True
        force[:, int(w * 0.88) :] = True
        force[int(h * 0.18) : int(h * 0.72), : int(w * 0.22)] = True
        force[int(h * 0.18) : int(h * 0.72), int(w * 0.78) :] = True
    else:
        force[: int(h * 0.14), :] = True
        force[int(h * 0.84) :, :] = True
        force[:, : int(w * 0.16)] = True
        force[:, int(w * 0.84) :] = True
        force[int(h * 0.14) : int(h * 0.84), : int(w * 0.26)] = True
        force[int(h * 0.14) : int(h * 0.84), int(w * 0.74) :] = True
    protect_final = force.copy()
    protect_final[anim] = protect[anim]
    protect_final = protect_final | (protect & ~anim)
    Image.fromarray((protect_final.astype(np.uint8) * 255)).save(
        MASK / f"{name}_text_protect.png"
    )
    allow = (~protect_final) & anim
    allow_f = ndimage.gaussian_filter(allow.astype(np.float32), sigma=4)
    allow_f = np.clip(allow_f / (allow_f.max() + 1e-8), 0, 1)
    Image.fromarray((allow_f * 255).astype(np.uint8)).save(
        MASK / f"{name}_anim_allow.png"
    )
    hole = ((xx - cx) / (rx * 0.95)) ** 2 + ((yy - cy) / (ry * 0.95)) ** 2 <= 1.0
    hole = hole & (~protect_final)
    hole_f = ndimage.gaussian_filter(hole.astype(np.float32), sigma=6)
    hole_f = np.clip(hole_f / (hole_f.max() + 1e-8), 0, 1)
    Image.fromarray((hole_f * 255).astype(np.uint8)).save(
        MASK / f"{name}_center_hole.png"
    )
    print(
        name,
        "protect%",
        round(protect_final.mean() * 100, 1),
        "anim%",
        round(allow.mean() * 100, 1),
        "cxcy",
        round(cx),
        round(cy),
        "rxry",
        round(rx),
        round(ry),
    )
    return {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "W": w, "H": h}


def make_ring_texture(path: Path, size: int = 1024, eyes_u: int = 48, eyes_v: int = 6) -> None:
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 40, (size, size), dtype=np.uint8)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:, :, 0] = 28 + noise
    base[:, :, 1] = 14 + noise // 2
    base[:, :, 2] = 10 + noise // 3
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    cell_w = size / eyes_u
    cell_h = size / eyes_v
    for j in range(eyes_v):
        for i in range(eyes_u):
            cx = (i + 0.5) * cell_w + (0 if j % 2 == 0 else cell_w * 0.25)
            cy = (j + 0.5) * cell_h
            for ox in (0, -size, size):
                x = cx + ox
                rw, rh = cell_w * 0.42, cell_h * 0.38
                draw.ellipse(
                    [x - rw, cy - rh, x + rw, cy + rh],
                    fill=(55, 28, 18),
                    outline=(90, 40, 25),
                )
                rw2, rh2 = rw * 0.78, rh * 0.78
                draw.ellipse(
                    [x - rw2, cy - rh2, x + rw2, cy + rh2], fill=(210, 190, 170)
                )
                ri = min(rw2, rh2) * 0.55
                dx = ((i * 7 + j * 13) % 5 - 2) * 0.6
                dy = ((i * 3 + j * 11) % 5 - 2) * 0.4
                draw.ellipse(
                    [x + dx - ri, cy + dy - ri, x + dx + ri, cy + dy + ri],
                    fill=(140, 45, 30),
                )
                rp = ri * 0.45
                draw.ellipse(
                    [x + dx - rp, cy + dy - rp, x + dx + rp, cy + dy + rp],
                    fill=(10, 5, 5),
                )
                draw.ellipse(
                    [
                        x + dx - rp * 0.3,
                        cy + dy - rp * 0.6,
                        x + dx + rp * 0.1,
                        cy + dy - rp * 0.2,
                    ],
                    fill=(255, 220, 180),
                )
    ImageEnhance.Color(img).enhance(1.15).save(path)
    print("texture", path)


def main() -> None:
    h = Image.open(ASSET / "arte_computador_1920x1080.png").convert("RGB")
    v = Image.open(ASSET / "arte_celular_1080x1920.png").convert("RGB")
    Image.open(ASSET / "arte_horizontal_src.jpg").convert("RGB").save(
        ASSET / "arte_original.jpeg", quality=95
    )
    meta = {
        "computador": make_text_mask(h, "computador", False),
        "celular": make_text_mask(v, "celular", True),
    }
    (ASSET / "layout_meta.json").write_text(json.dumps(meta, indent=2))

    ah = np.array(h)
    r, g, b = ah[:, :, 0].astype(float), ah[:, :, 1].astype(float), ah[:, :, 2].astype(
        float
    )
    cyan = (b > 140) & (g > 100) & (b > r + 30) & (g > r + 10)
    ys, xs = np.where(cyan)
    ecx, ecy = int(xs.mean()), int(ys.mean())
    print("eye center", ecx, ecy)
    s = 160
    eye = h.crop((ecx - s, ecy - s, ecx + s, ecy + s)).resize(
        (512, 512), Image.Resampling.LANCZOS
    )
    eye.save(TEX / "central_eye.png")
    yy, xx = np.mgrid[0:512, 0:512]
    Image.fromarray((((xx - 256) ** 2 + (yy - 256) ** 2) < 90**2).astype(np.uint8) * 255).save(
        TEX / "iris_mask.png"
    )
    Image.fromarray(
        (((xx - 256) ** 2 + (yy - 256) ** 2) < 200**2).astype(np.uint8) * 255
    ).save(TEX / "sclera_mask.png")
    make_ring_texture(TEX / "ring_eyes.png")

    iris = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(iris)
    d.ellipse([8, 8, 248, 248], fill=(40, 160, 200, 255))
    for rad in range(120, 20, -8):
        col = (30 + rad // 2, 140 + rad // 3, 190 + rad // 2, 255)
        d.ellipse([128 - rad, 128 - rad, 128 + rad, 128 + rad], outline=col)
    d.ellipse([128 - 28, 128 - 28, 128 + 28, 128 + 28], fill=(5, 8, 12, 255))
    d.ellipse([110, 100, 125, 115], fill=(220, 240, 255, 200))
    iris.save(TEX / "iris_cyan.png")
    print("prepare_assets done")


if __name__ == "__main__":
    main()
