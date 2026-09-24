#!/usr/bin/env python3
"""
Build art-faithful textures:
- central eye cutout from the poster (circular matte)
- ring albedo baked from the poster via polar unwrap of the ring band
- tighter center hole so the hourglass nebula survives
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

ROOT = Path("/workspace/ophanim")
ASSET = ROOT / "assets"
TEX = ROOT / "textures"
MASK = ROOT / "masks"
TEX.mkdir(parents=True, exist_ok=True)
MASK.mkdir(parents=True, exist_ok=True)


def find_eye_center(a: np.ndarray) -> tuple[int, int]:
    r, g, b = a[:, :, 0].astype(float), a[:, :, 1].astype(float), a[:, :, 2].astype(float)
    cyan = (b > 140) & (g > 100) & (b > r + 30) & (g > r + 10)
    ys, xs = np.where(cyan)
    if len(xs) == 0:
        h, w = a.shape[:2]
        return w // 2, int(h * 0.42)
    return int(xs.mean()), int(ys.mean())


def polar_sample(a: np.ndarray, ecx: int, ecy: int, r0: float, r1: float, out_w: int, out_h: int) -> np.ndarray:
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    theta = xx / out_w * 2 * np.pi
    rad = r0 + (yy / max(out_h - 1, 1)) * (r1 - r0)
    src_x = ecx + rad * np.cos(theta)
    src_y = ecy + rad * np.sin(theta)
    x0 = np.clip(np.floor(src_x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(src_y).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (src_x - np.floor(src_x))[..., None]
    wy = (src_y - np.floor(src_y))[..., None]
    c = a.astype(np.float32)
    out = (
        c[y0, x0] * (1 - wx) * (1 - wy)
        + c[y0, x1] * wx * (1 - wy)
        + c[y1, x0] * (1 - wx) * wy
        + c[y1, x1] * wx * wy
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def build_for(art_path: Path, name: str, portrait: bool) -> dict:
    art = Image.open(art_path).convert("RGB")
    a = np.array(art)
    h, w = a.shape[:2]
    ecx, ecy = find_eye_center(a)
    print(name, "eye", ecx, ecy, "size", w, h)

    # --- eye cutout (preserve photoreal cyan eye) ---
    eye_r = int(min(w, h) * (0.095 if not portrait else 0.10))
    eye = art.crop((ecx - eye_r, ecy - eye_r, ecx + eye_r, ecy + eye_r)).resize(
        (512, 512), Image.Resampling.LANCZOS
    )
    ea = np.array(eye.convert("RGBA"))
    yy, xx = np.mgrid[0:512, 0:512]
    rad = np.sqrt((xx - 256.0) ** 2 + (yy - 256.0) ** 2)
    # Keep eyeball, soft edge — exclude outer ring fragments
    alpha = (np.clip((228 - rad) / 16, 0, 1) * 255).astype(np.uint8)
    ea[:, :, 3] = alpha
    Image.fromarray(ea).save(TEX / f"central_eye_{name}.png")
    if name == "computador":
        Image.fromarray(ea).save(TEX / "central_eye.png")

    # Iris crop for saccades
    iris = eye.crop((256 - 105, 256 - 105, 256 + 105, 256 + 105)).resize(
        (256, 256), Image.Resampling.LANCZOS
    )
    ia = np.array(iris.convert("RGBA"))
    yy, xx = np.mgrid[0:256, 0:256]
    rad = np.sqrt((xx - 128.0) ** 2 + (yy - 128.0) ** 2)
    ia[:, :, 3] = (np.clip((112 - rad) / 8, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(ia).save(TEX / f"iris_{name}.png")
    if name == "computador":
        Image.fromarray(ia).save(TEX / "iris_cyan.png")

    # --- ring albedo from polar unwrap of original ring band ---
    r0 = eye_r * 1.15
    r1 = eye_r * (3.6 if not portrait else 3.2)
    strip = polar_sample(a, ecx, ecy, r0, r1, 1536, 288)
    # Suppress nebula whites/cyans that leaked into the unwrap
    sr, sg, sb = strip[:, :, 0].astype(float), strip[:, :, 1].astype(float), strip[:, :, 2].astype(float)
    leak = (((sr + sg + sb) > 500) | ((sb > sr + 25) & (sg > sr))).astype(np.float32)
    leak = ndimage.gaussian_filter(leak, sigma=2.5)[..., None]
    copper = np.array([72, 28, 18], dtype=np.float32)
    strip_f = strip.astype(np.float32) * (1 - leak * 0.8) + copper * (leak * 0.8)
    ring_img = Image.fromarray(np.clip(strip_f, 0, 255).astype(np.uint8))
    ring_img = ImageEnhance.Contrast(ring_img).enhance(1.2)
    ring_img = ImageEnhance.Color(ring_img).enhance(1.15)
    ring_img.save(TEX / f"ring_eyes_{name}.png")
    if name == "computador":
        ring_img.save(TEX / "ring_eyes.png")

    # --- tighter masks: protect text, allow anim only on rings/eye ---
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = float(ecx), float(ecy)
    if portrait:
        rx, ry = w * 0.38, h * 0.20
    else:
        rx, ry = w * 0.26, h * 0.46
    anim = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    rch, gch, bch = a[:, :, 0].astype(float), a[:, :, 1].astype(float), a[:, :, 2].astype(float)
    white = (rch > 200) & (gch > 200) & (bch > 200) & ((rch + gch + bch) > 620)
    lines = (
        (rch > 160)
        & (gch > 160)
        & (bch > 160)
        & (np.abs(rch - gch) < 25)
        & (np.abs(gch - bch) < 25)
        & ((rch + gch + bch) > 500)
    )
    protect = ndimage.binary_dilation(white | lines, iterations=2)

    force = np.zeros((h, w), dtype=bool)
    if portrait:
        force[: int(h * 0.17), :] = True
        force[int(h * 0.74) :, :] = True
        force[:, : int(w * 0.11)] = True
        force[:, int(w * 0.89) :] = True
        force[int(h * 0.17) : int(h * 0.74), : int(w * 0.20)] = True
        force[int(h * 0.17) : int(h * 0.74), int(w * 0.80) :] = True
    else:
        force[: int(h * 0.13), :] = True
        force[int(h * 0.85) :, :] = True
        force[:, : int(w * 0.15)] = True
        force[:, int(w * 0.85) :] = True
        force[int(h * 0.13) : int(h * 0.85), : int(w * 0.24)] = True
        force[int(h * 0.13) : int(h * 0.85), int(w * 0.76) :] = True

    protect_final = force.copy()
    protect_final[anim] = protect[anim]
    protect_final = protect_final | (protect & ~anim)
    Image.fromarray((protect_final.astype(np.uint8) * 255)).save(MASK / f"{name}_text_protect.png")

    allow = (~protect_final) & anim
    allow_f = ndimage.gaussian_filter(allow.astype(np.float32), sigma=3.5)
    allow_f = np.clip(allow_f / (allow_f.max() + 1e-8), 0, 1)
    Image.fromarray((allow_f * 255).astype(np.uint8)).save(MASK / f"{name}_anim_allow.png")

    # Hole: remove original rings/eye so 3D can replace — keep outer nebula
    hole = ((xx - cx) / (rx * 0.88)) ** 2 + ((yy - cy) / (ry * 0.88)) ** 2 <= 1.0
    hole = hole & (~protect_final)
    hole_f = ndimage.gaussian_filter(hole.astype(np.float32), sigma=5)
    hole_f = np.clip(hole_f / (hole_f.max() + 1e-8), 0, 1)
    Image.fromarray((hole_f * 255).astype(np.uint8)).save(MASK / f"{name}_center_hole.png")

    # Inpaint plate: fill hole with nebula (keep fiery look)
    blurred = np.array(art.filter(ImageFilter.GaussianBlur(radius=40))).astype(np.float32)
    ring = (hole_f < 0.15) & (
        (((xx - cx) / (w * 0.2)) ** 2 + ((yy - cy) / (h * 0.2)) ** 2) < 1.6
    )
    mean = a[ring].mean(axis=0) if ring.any() else np.array([95, 28, 18], dtype=np.float32)
    fill = blurred * 0.7 + mean.reshape(1, 1, 3) * 0.3
    hs = hole_f[..., None]
    bg = a.astype(np.float32) * (1 - hs) + fill * hs
    Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save(ASSET / f"bg_{name}.png")

    meta = {"cx": float(ecx), "cy": float(ecy), "rx": float(rx), "ry": float(ry), "W": w, "H": h, "eye_r": eye_r}
    print(name, "masks done")
    return meta


def main() -> None:
    meta = {
        "computador": build_for(ASSET / "arte_computador_1920x1080.png", "computador", False),
        "celular": build_for(ASSET / "arte_celular_1080x1920.png", "celular", True),
    }
    (ASSET / "layout_meta.json").write_text(json.dumps(meta, indent=2))
    print("ok")


if __name__ == "__main__":
    main()
