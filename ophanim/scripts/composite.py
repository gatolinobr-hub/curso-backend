#!/usr/bin/env python3
"""
Faithful composite:
  - Start from the original poster every frame (identity preserved)
  - Animate the REAL eye with 2D iris shift + blink on original pixels
  - Soft-clear only the original ring band, then overlay 3D rings
  - Restore text/diagrams from the original
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path("/workspace/ophanim")
DURATION = 10.0
FPS = 24
N = int(DURATION * FPS)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.float32)


def load_l(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0


def blink_amount(t: float) -> float:
    def pulse(c: float, w: float = 0.028) -> float:
        d = min(abs(t - c), abs(t - c + 1), abs(t - c - 1))
        return math.exp(-((d / w) ** 2))

    return min(1.0, pulse(0.28) + pulse(0.72))


def iris_offset(t: float) -> tuple[float, float]:
    # Zero at t=0 and t=1 for identity / seamless loop
    x = 0.045 * math.sin(2 * math.pi * t) + 0.02 * math.sin(4 * math.pi * t)
    y = 0.03 * math.sin(2 * math.pi * t) + 0.015 * math.sin(6 * math.pi * t)
    return x, y


def animate_eye(art: np.ndarray, ecx: int, ecy: int, eye_r: int, t: float) -> np.ndarray:
    """Shift cyan iris inside the real eyeball; soft blink over original pixels."""
    out = art.copy()
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - ecx) ** 2 + (yy - ecy) ** 2)
    eyeball = dist <= eye_r * 0.95
    r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    cyan = eyeball & (b > 120) & (g > 90) & (b > r + 20)
    if not cyan.any():
        cyan = eyeball & (dist < eye_r * 0.45)

    # Iris saccade via remap of cyan neighborhood
    ox, oy = iris_offset(t)
    dx = ox * eye_r
    dy = oy * eye_r
    # Source coordinates for iris pixels
    src_x = np.clip((xx - dx).astype(np.int32), 0, w - 1)
    src_y = np.clip((yy - dy).astype(np.int32), 0, h - 1)
    shifted = art[src_y, src_x]
    # Soft iris mask
    iris_soft = ndimage.gaussian_filter(cyan.astype(np.float32), sigma=1.2)
    iris_soft = np.clip(iris_soft / (iris_soft.max() + 1e-8), 0, 1)[..., None]
    out = out * (1 - iris_soft) + shifted * iris_soft

    # Blink: dark lids sweeping from top/bottom over eyeball
    bamt = blink_amount(t)
    if bamt > 0.02:
        # lid color from surrounding dark flesh near eye
        sample = art[
            max(0, ecy - eye_r) : max(1, ecy - int(eye_r * 0.55)),
            max(0, ecx - 8) : min(w, ecx + 8),
        ]
        if sample.size:
            lid_col = sample.mean(axis=(0, 1))
        else:
            lid_col = np.array([25.0, 10.0, 8.0], dtype=np.float32)
        # Upper/lower coverage fraction of eyeball
        # y normalized inside eyeball: -1..1
        yn = (yy - ecy) / max(eye_r, 1)
        # open lids leave center; closed covers |yn| from edges inward
        cover = np.clip((bamt - (0.55 - np.abs(yn))) / 0.35, 0, 1)
        cover = cover * eyeball.astype(np.float32)
        cover = ndimage.gaussian_filter(cover, sigma=1.5)[..., None]
        out = out * (1 - cover) + lid_col.reshape(1, 1, 3) * cover

    return out


def color_match_overlay(
    overlay_rgb: np.ndarray, overlay_a: np.ndarray, target: np.ndarray, ring_mask: np.ndarray
) -> np.ndarray:
    """Match mean color of overlay to original ring band for better blend."""
    m = (overlay_a > 0.15) & (ring_mask > 0.2)
    if m.sum() < 100:
        return overlay_rgb
    src = overlay_rgb[m]
    dst = target[m]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_std = src.std(axis=0) + 1e-3
    dst_std = dst.std(axis=0) + 1e-3
    matched = (overlay_rgb - src_mean) * (dst_std / src_std) + dst_mean
    return np.clip(matched, 0, 255)


def build_ring_clear_mask(art: np.ndarray, ecx: int, ecy: int, eye_r: int, allow: np.ndarray) -> np.ndarray:
    """Mask covering original rings but NOT the central eyeball (keep real eye)."""
    h, w = art.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - ecx) ** 2 + (yy - ecy) ** 2)
    # Annulus around eye where rings live
    annulus = (dist > eye_r * 0.85) & (dist < eye_r * 3.8)
    r, g, b = art[:, :, 0], art[:, :, 1], art[:, :, 2]
    # Dark/reddish ring material (exclude bright nebula & white text)
    dark_ring = annulus & (r + g + b < 480) & (r > 25)
    m = (dark_ring.astype(np.float32) * allow)
    m = ndimage.binary_dilation(m > 0.15, iterations=2).astype(np.float32)
    m = ndimage.gaussian_filter(m, sigma=3.5)
    # Never clear the eyeball core
    m = m * (dist > eye_r * 0.75).astype(np.float32)
    return np.clip(m / (m.max() + 1e-8), 0, 1)


def soft_inpaint(art: np.ndarray, clear: np.ndarray) -> np.ndarray:
    blurred = np.array(
        Image.fromarray(art.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=28))
    ).astype(np.float32)
    # Prefer local nebula reds
    mean = art[clear < 0.1].mean(axis=0) if (clear < 0.1).any() else np.array([80, 25, 18.0])
    fill = blurred * 0.75 + mean.reshape(1, 1, 3) * 0.25
    c = clear[..., None]
    return art * (1 - c) + fill * c


def composite_frame(
    art: np.ndarray,
    overlay_rgba: np.ndarray,
    rest_rgba: np.ndarray,
    allow: np.ndarray,
    protect: np.ndarray,
    clear: np.ndarray,
    bg: np.ndarray,
    ecx: int,
    ecy: int,
    eye_r: int,
    t: float,
) -> np.ndarray:
    """
    Faithful composite using rest-delta for rings:
      out = original_with_eye_anim + (curr_3d - rest_3d)
    At t=0, delta is 0 → poster identity is preserved.
    """
    base = animate_eye(art, ecx, ecy, eye_r, t)

    h, w = art.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - ecx) ** 2 + (yy - ecy) ** 2)
    eye_protect = np.clip((eye_r * 1.08 - dist) / (eye_r * 0.28 + 1e-6), 0, 1)

    def prep(ov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        o = ov.astype(np.float32)
        a = np.clip((o[:, :, 3] / 255.0) * allow, 0, 1) * (1.0 - eye_protect)
        rgb = color_match_overlay(o[:, :, :3], a, art, clear)
        return rgb, a

    curr_rgb, curr_a = prep(overlay_rgba)
    rest_rgb, rest_a = prep(rest_rgba)

    # Premultiplied delta — identity at rest
    delta = curr_rgb * curr_a[..., None] - rest_rgb * rest_a[..., None]

    # Where rings leave, gently reveal inpainted nebula instead of subtracting CGI color from art
    leave = np.clip(rest_a - curr_a, 0, 1)[..., None]
    arrive = np.clip(curr_a - rest_a, 0, 1)[..., None]
    overlap = np.minimum(curr_a, rest_a)[..., None]

    out = base.copy()
    # Arriving ring pixels: show 3D ring
    out = out * (1 - arrive) + curr_rgb * arrive
    # Overlap (ring still covering): blend toward current 3D
    out = out * (1 - overlap * 0.85) + curr_rgb * (overlap * 0.85)
    # Leaving: soft reveal bg / keep art mix
    out = out * (1 - leave * 0.7) + bg * (leave * 0.7)

    # Near t=0/1, fade any residual toward the untouched poster+eye
    # (keeps loop ends identical to the drawing)
    edge = min(t, 1.0 - t)
    fade = float(np.clip(edge / 0.06, 0, 1))  # first/last ~0.6s soft
    out = base * (1 - fade) + out * fade

    # Force-protect eyeball & text from any residual
    ep = eye_protect[..., None]
    out = out * (1 - ep) + base * ep
    p = protect[..., None]
    out = out * (1 - p) + art * p
    return np.clip(out, 0, 255).astype(np.uint8)


def run_variant(variant: str, preview: bool = False) -> None:
    meta = json.loads((ROOT / "assets/layout_meta.json").read_text())[variant]
    ecx, ecy = int(meta["cx"]), int(meta["cy"])
    eye_r = int(meta.get("eye_r", 90))

    if variant == "computador":
        art_p = ROOT / "assets/arte_computador_1920x1080.png"
        frames_dir = ROOT / "frames/computador"
        out_mp4 = ROOT / "final/wallpaper_computador_1920x1080.mp4"
        preview_dir = ROOT / "final/preview/computador"
        allow_p = ROOT / "masks/computador_anim_allow.png"
        protect_p = ROOT / "masks/computador_text_protect.png"
    else:
        art_p = ROOT / "assets/arte_celular_1080x1920.png"
        frames_dir = ROOT / "frames/celular"
        out_mp4 = ROOT / "final/wallpaper_celular_1080x1920.mp4"
        preview_dir = ROOT / "final/preview/celular"
        allow_p = ROOT / "masks/celular_anim_allow.png"
        protect_p = ROOT / "masks/celular_text_protect.png"

    art = load_rgb(art_p)
    allow = load_l(allow_p)
    protect = load_l(protect_p)
    clear = build_ring_clear_mask(art, ecx, ecy, eye_r, allow)
    bg = soft_inpaint(art, clear)
    Image.fromarray(bg.astype(np.uint8)).save(ROOT / "assets" / f"bg_{variant}_rings.png")
    Image.fromarray((clear * 255).astype(np.uint8)).save(
        ROOT / "masks" / f"{variant}_ring_clear.png"
    )

    h, w = art.shape[:2]
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_frames = ROOT / "frames" / f"{variant}_comp"
    out_frames.mkdir(parents=True, exist_ok=True)

    if preview:
        indices = {
            0: "preview_0.png",
            25: "preview_25.png",
            50: "preview_50.png",
            75: "preview_75.png",
        }
        rest = Image.open(frames_dir / "preview_0.png").convert("RGBA")
        if rest.size != (w, h):
            rest = rest.resize((w, h), Image.Resampling.LANCZOS)
        rest_a = np.array(rest)
        for pct, name in indices.items():
            t = pct / 100.0
            ov = Image.open(frames_dir / name).convert("RGBA")
            if ov.size != (w, h):
                ov = ov.resize((w, h), Image.Resampling.LANCZOS)
            comp = composite_frame(
                art, np.array(ov), rest_a, allow, protect, clear, bg, ecx, ecy, eye_r, t
            )
            Image.fromarray(comp).save(preview_dir / f"frame_{pct}pct.png")
            print("preview", variant, pct)
        return

    rest = Image.open(frames_dir / "frame_0000.png").convert("RGBA")
    if rest.size != (w, h):
        rest = rest.resize((w, h), Image.Resampling.LANCZOS)
    rest_a = np.array(rest)

    for i in range(N):
        t = i / N
        ov = Image.open(frames_dir / f"frame_{i:04d}.png").convert("RGBA")
        if ov.size != (w, h):
            ov = ov.resize((w, h), Image.Resampling.LANCZOS)
        comp = composite_frame(
            art, np.array(ov), rest_a, allow, protect, clear, bg, ecx, ecy, eye_r, t
        )
        Image.fromarray(comp).save(out_frames / f"frame_{i:04d}.png")
        if i % 30 == 0:
            print(f"comp {variant} {i}/{N}")

    for pct in (0, 25, 50, 75):
        i = int(round(pct / 100 * (N - 1)))
        Image.open(out_frames / f"frame_{i:04d}.png").save(
            preview_dir / f"frame_{pct}pct.png"
        )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "24",
            "-i",
            str(out_frames / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-movflags",
            "+faststart",
            "-an",
            str(out_mp4),
        ]
    )
    print("wrote", out_mp4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["computador", "celular", "both"])
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    variants = ["computador", "celular"] if args.variant == "both" else [args.variant]
    for v in variants:
        run_variant(v, preview=args.preview)


if __name__ == "__main__":
    main()
