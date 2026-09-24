#!/usr/bin/env python3
"""Composite transparent 3D frames onto artwork with text protection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/ophanim")


def composite_frame(
    art: np.ndarray,
    bg: np.ndarray,
    overlay_rgba: np.ndarray,
    allow: np.ndarray,
    protect: np.ndarray,
    hole: np.ndarray,
) -> np.ndarray:
    """art/bg HxWx3; overlay HxWx4; masks HxW float 0..1"""
    a = art.astype(np.float32)
    b = bg.astype(np.float32)
    o = overlay_rgba.astype(np.float32)
    # Clear original rings/eye strongly in hole, keep outer art
    h = np.clip(hole * 1.1, 0, 1)[..., None]
    base = a * (1.0 - h) + b * h
    # Overlay alpha only where animation is allowed
    oa = (o[:, :, 3:4] / 255.0) * allow[..., None]
    # Premultiplied-style over
    out = base * (1.0 - oa) + o[:, :, :3] * oa
    # Restore protected text/diagrams from original art
    p = protect[..., None]
    out = out * (1.0 - p) + a * p
    return np.clip(out, 0, 255).astype(np.uint8)


def run_variant(variant: str, preview: bool = False) -> None:
    if variant == "computador":
        art_p = ROOT / "assets/arte_computador_1920x1080.png"
        bg_p = ROOT / "assets/bg_computador.png"
        allow_p = ROOT / "masks/computador_anim_allow.png"
        protect_p = ROOT / "masks/computador_text_protect.png"
        hole_p = ROOT / "masks/computador_center_hole.png"
        frames_dir = ROOT / "frames/computador"
        out_mp4 = ROOT / "final/wallpaper_computador_1920x1080.mp4"
        preview_dir = ROOT / "preview/computador"
    else:
        art_p = ROOT / "assets/arte_celular_1080x1920.png"
        bg_p = ROOT / "assets/bg_celular.png"
        allow_p = ROOT / "masks/celular_anim_allow.png"
        protect_p = ROOT / "masks/celular_text_protect.png"
        hole_p = ROOT / "masks/celular_center_hole.png"
        frames_dir = ROOT / "frames/celular"
        out_mp4 = ROOT / "final/wallpaper_celular_1080x1920.mp4"
        preview_dir = ROOT / "preview/celular"

    art = np.array(Image.open(art_p).convert("RGB"))
    bg = np.array(Image.open(bg_p).convert("RGB"))
    allow = np.array(Image.open(allow_p).convert("L")).astype(np.float32) / 255.0
    protect = np.array(Image.open(protect_p).convert("L")).astype(np.float32) / 255.0
    hole = np.array(Image.open(hole_p).convert("L")).astype(np.float32) / 255.0

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
        for pct, name in indices.items():
            ov = Image.open(frames_dir / name).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
            comp = composite_frame(art, bg, np.array(ov), allow, protect, hole)
            Image.fromarray(comp).save(preview_dir / f"frame_{pct:02d}.png")
            print("preview", variant, pct)
        return

    # Full sequence
    n = 240
    for i in range(n):
        fp = frames_dir / f"frame_{i:04d}.png"
        ov = Image.open(fp).convert("RGBA")
        if ov.size != (w, h):
            ov = ov.resize((w, h), Image.Resampling.LANCZOS)
        comp = composite_frame(art, bg, np.array(ov), allow, protect, hole)
        Image.fromarray(comp).save(out_frames / f"frame_{i:04d}.png")
        if i % 30 == 0:
            print(f"comp {variant} {i}/{n}")

    # Also write preview stills from composed sequence
    for pct in (0, 25, 50, 75):
        i = int(round(pct / 100 * (n - 1)))
        Image.open(out_frames / f"frame_{i:04d}.png").save(
            preview_dir / f"frame_{pct:02d}.png"
        )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    cmd = [
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
        "18",
        "-movflags",
        "+faststart",
        "-an",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)
    print("wrote", out_mp4, out_mp4.stat().st_size)


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
