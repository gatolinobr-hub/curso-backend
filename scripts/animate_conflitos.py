#!/usr/bin/env python3
"""Animate CONFLITOS album cover into a seamless 8s cinematic loop."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

SRC = Path("/home/ubuntu/.cursor/projects/workspace/assets/01a0ad12-5bbf-77d3-9b8f-fa0e10095223.jpg")
OUT_DIR = Path("/workspace/artifacts")
FRAMES_DIR = Path("/tmp/conflitos_frames")
ARTIFACTS = Path("/opt/cursor/artifacts")

DURATION = 8.0
FPS = 24
N_FRAMES = int(DURATION * FPS)  # 192
OUT_SIZE = 1024  # square delivery

# Clock geometry (tuned to twin-bell face center on source 1254px)
CLOCK_CX = 985.0
CLOCK_CY = 855.0
CLOCK_R = 175.0

# Neon sign ROI on source
NEON_Y0, NEON_Y1 = 240, 530
NEON_X0, NEON_X1 = 430, 810


def seamless_noise(n: int, seed: int, octaves: int = 4) -> np.ndarray:
    """Periodic 1D noise in [0, 1] for seamless looping."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float64)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        freq = 1 + o
        phase = rng.uniform(0, 2 * np.pi)
        # mix sin/cos with random weights for irregular feel
        w1, w2 = rng.normal(0, 1), rng.normal(0, 1)
        sig += amp * (w1 * np.sin(freq * t + phase) + w2 * np.cos(freq * t + phase * 0.7))
        total += amp * abs(w1) + amp * abs(w2)
        amp *= 0.55
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    return sig.astype(np.float32)


def build_flicker(n: int) -> np.ndarray:
    """Irregular neon electrical flicker, mostly on, occasional dips."""
    base = seamless_noise(n, seed=11, octaves=5)
    micro = seamless_noise(n, seed=22, octaves=6)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # sharper, more noticeable brownout dips (still periodic / seamless)
    dips = 1.0 - 0.72 * (np.sin(3 * t + 0.4) ** 28)
    dips -= 0.55 * (np.sin(7 * t + 2.1) ** 50)
    dips -= 0.40 * (np.sin(11 * t + 5.0) ** 70)
    dips -= 0.30 * (np.sin(5 * t + 3.3) ** 90)
    flick = 0.92 + 0.22 * base + 0.10 * (micro - 0.5) * 2
    flick *= np.clip(dips, 0.22, 1.0)
    return np.clip(flick, 0.22, 1.22).astype(np.float32)


def build_masks(h: int, w: int, img: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    yy, xx = np.mgrid[0:h, 0:w]

    neon_roi = np.zeros((h, w), dtype=bool)
    neon_roi[NEON_Y0:NEON_Y1, NEON_X0:NEON_X1] = True

    # Bright neon tubes + soft glow (do not remask letters separately — intensity only)
    neon_core = neon_roi & (g > 130) & (g > r + 12) & (g > b - 5)
    neon_glow = neon_roi & (g > 70) & (g > r + 5) & ((g - r) > 0)
    neon_glow = neon_glow | ndimage.binary_dilation(neon_core, iterations=6)
    neon_w = ndimage.gaussian_filter(neon_glow.astype(np.float32), sigma=3.5)
    neon_w = neon_w / (neon_w.max() + 1e-8)
    # slightly stronger on core
    core_w = ndimage.gaussian_filter(neon_core.astype(np.float32), sigma=1.2)
    core_w = core_w / (core_w.max() + 1e-8)
    neon_w = np.clip(neon_w * 0.65 + core_w * 0.55, 0, 1)

    # Floor reflections: lower half, green-tinted wet patches
    floor = yy > int(h * 0.52)
    refl = floor & (g > r + 4) & (g > 28) & (g < 170) & (b > 20)
    # exclude clock face interior
    dist_clock = np.sqrt((xx - CLOCK_CX) ** 2 + (yy - CLOCK_CY) ** 2)
    refl = refl & (dist_clock > CLOCK_R * 0.92)
    refl_w = ndimage.gaussian_filter(refl.astype(np.float32), sigma=4.0)
    refl_w = refl_w / (refl_w.max() + 1e-8)

    # Cable-ish dark linear structures on floor / right wall for subtle warp
    dark = (img.mean(2) < 45) & (yy > int(h * 0.35))
    cables = dark & ~((dist_clock < CLOCK_R * 1.05))
    cables = ndimage.binary_opening(cables, iterations=1)
    cable_w = ndimage.gaussian_filter(cables.astype(np.float32), sigma=5.0)
    cable_w = cable_w / (cable_w.max() + 1e-8)

    # Soft ambient light influence near neon
    ambient = ndimage.gaussian_filter(neon_w, sigma=28.0)
    ambient = ambient / (ambient.max() + 1e-8)

    # Film border (dark frame) — keep stable weight for overlays
    border = (xx < 48) | (xx > w - 48) | (yy < 48) | (yy > h - 48)
    border_w = border.astype(np.float32)

    # Clock face disk for second hand
    clock_face = (dist_clock < CLOCK_R * 0.78).astype(np.float32)
    clock_face = ndimage.gaussian_filter(clock_face, sigma=1.5)

    return {
        "neon": neon_w.astype(np.float32),
        "refl": refl_w.astype(np.float32),
        "cable": cable_w.astype(np.float32),
        "ambient": ambient.astype(np.float32),
        "border": border_w,
        "clock_face": clock_face.astype(np.float32),
        "dist_clock": dist_clock.astype(np.float32),
    }


def make_dust(n_particles: int, n_frames: int, h: int, w: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    # base positions; motion is periodic so loop closes
    x0 = rng.uniform(40, w - 40, n_particles)
    y0 = rng.uniform(40, h - 40, n_particles)
    size = rng.uniform(0.6, 2.4, n_particles)
    bright = rng.uniform(0.25, 0.85, n_particles)
    depth = rng.uniform(0.3, 1.0, n_particles)  # closer = larger parallax
    phase_x = rng.uniform(0, 2 * np.pi, n_particles)
    phase_y = rng.uniform(0, 2 * np.pi, n_particles)
    amp_x = rng.uniform(4, 28, n_particles) * depth
    amp_y = rng.uniform(6, 36, n_particles) * depth
    # some drift cycles at 1 period / loop
    return {
        "x0": x0,
        "y0": y0,
        "size": size,
        "bright": bright,
        "depth": depth,
        "phase_x": phase_x,
        "phase_y": phase_y,
        "amp_x": amp_x,
        "amp_y": amp_y,
    }


def render_dust_layer(dust, frame_i: int, n_frames: int, h: int, w: int) -> np.ndarray:
    layer = np.zeros((h, w), dtype=np.float32)
    t = 2 * math.pi * frame_i / n_frames
    xs = dust["x0"] + dust["amp_x"] * np.sin(t + dust["phase_x"])
    ys = dust["y0"] + dust["amp_y"] * np.cos(t * 0.5 + dust["phase_y"]) + dust["amp_y"] * 0.35 * np.sin(t)
    # occasional near-lens particles: boost a few based on depth
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
            continue
        s = dust["size"][i] * (0.7 + 0.6 * dust["depth"][i])
        b = dust["bright"][i]
        # soft gaussian splat
        r = max(1, int(s * 2))
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        fall = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * (s * 0.7) ** 2 + 1e-6)))
        layer[y0:y1, x0:x1] += (fall * b).astype(np.float32)
    return np.clip(layer, 0, 1)


def camera_transform(img: np.ndarray, frame_i: int, n_frames: int) -> np.ndarray:
    """Slow push-in + subtle handheld micro-movement; periodic for seamless loop."""
    h, w = img.shape[:2]
    t = 2 * math.pi * frame_i / n_frames
    # push-in then ease back (seamless): 1.0 -> ~1.06 -> 1.0
    zoom = 1.0 + 0.06 * (0.5 - 0.5 * math.cos(t))
    # bias push slightly toward neon/clock composition center
    target_cx = w * 0.52
    target_cy = h * 0.48
    # handheld micro sway
    shake_x = 2.2 * math.sin(t * 2 + 0.3) + 1.1 * math.sin(t * 5 + 1.7)
    shake_y = 1.8 * math.cos(t * 2 + 0.9) + 0.9 * math.sin(t * 4 + 0.2)

    # crop window size
    cw = w / zoom
    ch = h / zoom
    # center drifts toward target as we push in
    pull = (zoom - 1.0) / 0.06  # 0..1
    cx = (w * 0.5) * (1 - 0.35 * pull) + target_cx * (0.35 * pull) + shake_x
    cy = (h * 0.5) * (1 - 0.25 * pull) + target_cy * (0.25 * pull) + shake_y
    x0 = cx - cw / 2
    y0 = cy - ch / 2

    # map output pixels to source
    out_y, out_x = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = x0 + out_x * (cw / w)
    src_y = y0 + out_y * (ch / h)

    channels = []
    for c in range(3):
        channels.append(
            ndimage.map_coordinates(img[:, :, c], [src_y, src_x], order=1, mode="nearest")
        )
    return np.stack(channels, axis=-1)


def apply_cable_warp(img: np.ndarray, cable_w: np.ndarray, frame_i: int, n_frames: int) -> np.ndarray:
    """Extremely subtle cable displacement."""
    h, w = img.shape[:2]
    t = 2 * math.pi * frame_i / n_frames
    # displacement fields ~0.3–0.8 px
    dx = (0.55 * math.sin(t + 0.4) + 0.25 * math.sin(2 * t + 1.2)) * cable_w
    dy = (0.35 * math.cos(t + 0.8) + 0.2 * math.sin(3 * t)) * cable_w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = xx - dx
    src_y = yy - dy
    channels = []
    for c in range(3):
        channels.append(
            ndimage.map_coordinates(img[:, :, c], [src_y, src_x], order=1, mode="nearest")
        )
    return np.stack(channels, axis=-1)


def apply_neon_and_light(
    img: np.ndarray,
    masks: dict[str, np.ndarray],
    flick: float,
    room_flick: float,
) -> np.ndarray:
    out = img.copy()
    neon = masks["neon"][..., None]
    refl = masks["refl"][..., None]
    ambient = masks["ambient"][..., None]

    # Neon intensity: boost green/cyan when on, dim when flicker dips
    # Preserve hue of letters — scale toward mint glow without morphing structure
    neon_gain = 0.42 + 0.58 * flick  # relative
    mint = np.array([0.70, 1.22, 1.08], dtype=np.float32)
    boost = (neon_gain - 1.0)  # negative when dim
    out += neon * boost * 95.0 * mint
    # slight bloom on bright flicker
    if flick > 1.0:
        out += neon * (flick - 1.0) * 70.0 * mint

    # Floor reflections pulse with neon
    refl_gain = (flick - 1.0) * 48.0
    out += refl * refl_gain * np.array([0.55, 1.0, 0.85], dtype=np.float32)

    # Unstable room electrical — global subtle linked to room_flick
    room = 0.97 + 0.06 * room_flick
    out *= room
    # ambient spill near neon
    out += ambient * (flick - 0.9) * 12.0 * np.array([0.4, 0.85, 0.7], dtype=np.float32)

    return np.clip(out, 0, 255)


def draw_second_hand(img: np.ndarray, masks: dict, frame_i: int, n_frames: int) -> np.ndarray:
    """Rotate a thin second hand once per loop; keep other hands untouched."""
    h, w = img.shape[:2]
    # Full rotation over loop for seamless return
    angle = -2 * math.pi * frame_i / n_frames - math.pi / 2  # start at 12
    length = CLOCK_R * 0.70
    cx, cy = CLOCK_CX, CLOCK_CY
    x2 = cx + length * math.cos(angle)
    y2 = cy + length * math.sin(angle)
    # counterweight stub opposite tip (classic analog second hand)
    x0 = cx - length * 0.18 * math.cos(angle)
    y0 = cy - length * 0.18 * math.sin(angle)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # dark hand with thin lighter edge for readability on dirty face
    draw.line([(x0, y0), (x2, y2)], fill=(8, 14, 12, 255), width=3)
    draw.line([(x0, y0), (x2, y2)], fill=(32, 42, 36, 220), width=1)
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(10, 16, 14, 255))
    draw.ellipse([x2 - 2.5, y2 - 2.5, x2 + 2.5, y2 + 2.5], fill=(14, 20, 18, 240))

    ov = np.array(overlay).astype(np.float32)
    # Restrict to clock disk but keep strong opacity
    face = (masks["dist_clock"] < CLOCK_R * 0.82).astype(np.float32)
    face = ndimage.gaussian_filter(face, sigma=0.8)
    alpha = (ov[:, :, 3:4] / 255.0) * face[..., None]
    rgb = ov[:, :, :3]
    out = img * (1 - alpha) + rgb * alpha
    return out


def apply_analog_interference(
    img: np.ndarray, frame_i: int, n_frames: int, rng: np.random.Generator
) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    t = 2 * math.pi * frame_i / n_frames

    # Subtle rolling luma instability
    row_wave = (1.0 + 0.012 * np.sin(np.linspace(0, 8 * math.pi, h) + t * 2)).astype(np.float32)
    out *= row_wave[:, None, None]

    # Occasional horizontal distortion bands (periodic triggers)
    band_trigger = (math.sin(t * 3 + 0.5) > 0.92) or (math.sin(t * 5 + 2.0) > 0.95)
    if band_trigger:
        # 1–3 bands
        n_bands = 1 + int(abs(math.sin(t * 7)) * 2)
        for b in range(n_bands):
            by = int((0.15 + 0.7 * abs(math.sin(t * (3 + b) + b))) * h)
            bh = 2 + int(4 * abs(math.cos(t + b)))
            shift = int(6 * math.sin(t * 11 + b * 2))
            y0, y1 = max(0, by), min(h, by + bh)
            if y1 > y0:
                out[y0:y1] = np.roll(out[y0:y1], shift, axis=1)

    # Very subtle chroma bleed / VHS RGB misalign on alternate frames
    if frame_i % 17 == 0 or math.sin(t * 4) > 0.97:
        shift = 1 if math.sin(t) > 0 else -1
        r = np.roll(out[:, :, 0], shift, axis=1)
        b = np.roll(out[:, :, 2], -shift, axis=1)
        out[:, :, 0] = out[:, :, 0] * 0.7 + r * 0.3
        out[:, :, 2] = out[:, :, 2] * 0.7 + b * 0.3

    # Soft scanlines
    scan = np.ones((h, 1, 1), dtype=np.float32)
    scan[::2] = 0.985
    out *= scan

    # Film grain
    grain = rng.normal(0, 4.5, out.shape).astype(np.float32)
    # slightly coarser chroma grain
    grain = ndimage.gaussian_filter(grain, sigma=(0.4, 0.4, 0))
    out += grain

    # Occasional fine scratches (vertical-ish)
    if math.sin(t * 2 + 1.1) > 0.85:
        sx = int((0.2 + 0.6 * abs(math.sin(t * 3))) * w)
        scratch = np.zeros((h, w), dtype=np.float32)
        for dx in range(-1, 2):
            x = np.clip(sx + dx, 0, w - 1)
            scratch[:, x] = 0.15 * (1.2 - abs(dx))
        # jitter scratch along y
        jitter = (2 * np.sin(np.linspace(0, 20 * math.pi, h) + t)).astype(int)
        for y in range(h):
            x = np.clip(sx + jitter[y], 0, w - 1)
            scratch[y, x] = max(scratch[y, x], 0.22)
        out += scratch[:, :, None] * 40.0

    # Specks / dust hits on film
    if frame_i % 23 == 5:
        for _ in range(rng.integers(2, 6)):
            px = int(rng.integers(60, w - 60))
            py = int(rng.integers(60, h - 60))
            out[py : py + 2, px : px + 2] += rng.uniform(20, 50)

    return np.clip(out, 0, 255)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if FRAMES_DIR.exists():
        for p in FRAMES_DIR.glob("*.png"):
            p.unlink()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    src_img = Image.open(SRC).convert("RGB")
    src = np.array(src_img).astype(np.float32)
    h, w = src.shape[:2]
    assert h == w, "source must be square"

    print(f"Source {w}x{h}, rendering {N_FRAMES} frames @ {FPS}fps")
    masks = build_masks(h, w, src)
    flick_curve = build_flicker(N_FRAMES)
    room_curve = seamless_noise(N_FRAMES, seed=99, octaves=3)
    dust = make_dust(90, N_FRAMES, h, w, seed=7)

    # Pre-warp base isn't needed; per-frame from original for stability
    rng = np.random.default_rng(2026)

    for i in range(N_FRAMES):
        frame = src.copy()
        frame = apply_cable_warp(frame, masks["cable"], i, N_FRAMES)
        frame = apply_neon_and_light(frame, masks, float(flick_curve[i]), float(room_curve[i]))
        frame = draw_second_hand(frame, masks, i, N_FRAMES)

        # Dust before camera so push-in parallax affects floating particles slightly
        dust_layer = render_dust_layer(dust, i, N_FRAMES, h, w)
        # near-lens: a few brighter when deep
        frame = frame + dust_layer[..., None] * 55.0 * np.array([0.85, 1.0, 0.9])

        frame = camera_transform(frame, i, N_FRAMES)
        frame = apply_analog_interference(frame, i, N_FRAMES, rng)

        # Mild cinematic teal grade preserve
        frame = np.clip(frame, 0, 255)

        # Resize to delivery size
        im = Image.fromarray(frame.astype(np.uint8))
        if OUT_SIZE != w:
            im = im.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
        im.save(FRAMES_DIR / f"frame_{i:04d}.png", optimize=False)

        if i % 24 == 0 or i == N_FRAMES - 1:
            print(f"  frame {i+1}/{N_FRAMES} (flicker={flick_curve[i]:.2f})")

    mp4_path = OUT_DIR / "conflitos_album_cover_loop.mp4"
    web_path = OUT_DIR / "conflitos_album_cover_loop_preview.mp4"
    gif_path = OUT_DIR / "conflitos_album_cover_loop.gif"

    # High quality seamless loop mp4
    os.system(
        f'ffmpeg -y -framerate {FPS} -i {FRAMES_DIR}/frame_%04d.png '
        f'-c:v libx264 -pix_fmt yuv420p -crf 17 -preset slow '
        f'-movflags +faststart -vf "scale={OUT_SIZE}:{OUT_SIZE}" '
        f'{mp4_path}'
    )
    # Copy to artifacts
    os.system(f"cp {mp4_path} {ARTIFACTS}/conflitos_album_cover_loop.mp4")

    # Lighter preview
    os.system(
        f'ffmpeg -y -i {mp4_path} -c:v libx264 -crf 23 -preset medium '
        f'-movflags +faststart {web_path}'
    )
    os.system(f"cp {web_path} {ARTIFACTS}/conflitos_album_cover_loop_preview.mp4")

    # Short GIF for quick preview (half res, lower fps)
    os.system(
        f'ffmpeg -y -i {mp4_path} -vf "fps=12,scale=512:512:flags=lanczos" '
        f'-loop 0 {gif_path}'
    )
    os.system(f"cp {gif_path} {ARTIFACTS}/conflitos_album_cover_loop.gif")

    # Save first/mid/last stills for verification
    for idx, name in [(0, "frame_start"), (N_FRAMES // 2, "frame_mid"), (N_FRAMES - 1, "frame_end")]:
        p = FRAMES_DIR / f"frame_{idx:04d}.png"
        outp = OUT_DIR / f"{name}.png"
        Image.open(p).save(outp)
        os.system(f"cp {outp} {ARTIFACTS}/{name}.png")

    print("Done:", mp4_path)
    print("Size:", mp4_path.stat().st_size if mp4_path.exists() else "missing")


if __name__ == "__main__":
    main()
