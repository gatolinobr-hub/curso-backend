#!/usr/bin/env python3
"""Animate CONFLITOS album cover into a seamless 8s cinematic loop."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

SRC = Path("/workspace/assets/conflitos_album_cover.jpg")
OUT_DIR = Path("/workspace/artifacts")
FRAMES_DIR = Path("/tmp/conflitos_frames")
ARTIFACTS = Path("/opt/cursor/artifacts")

DURATION = 8.0
FPS = 24
N_FRAMES = int(DURATION * FPS)  # 192
OUT_SIZE = 1024  # square delivery

# Clock geometry — hand pivot + face radius on source 1254px
CLOCK_CX = 968.0
CLOCK_CY = 801.0
CLOCK_R = 168.0

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
    """Irregular neon electrical flicker — clear dips + micro buzz, seamless."""
    base = seamless_noise(n, seed=11, octaves=5)
    micro = seamless_noise(n, seed=22, octaves=7)
    buzz = seamless_noise(n, seed=33, octaves=3)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Brownout dips (periodic so the loop closes)
    dips = 1.0 - 0.78 * (np.sin(2 * t + 0.6) ** 22)
    dips -= 0.62 * (np.sin(5 * t + 1.4) ** 36)
    dips -= 0.48 * (np.sin(9 * t + 2.8) ** 55)
    dips -= 0.35 * (np.sin(13 * t + 4.1) ** 70)
    dips -= 0.28 * (np.sin(3 * t + 5.2) ** 85)
    # Mostly-on baseline with audible electrical instability
    flick = 0.88 + 0.26 * base + 0.10 * (micro - 0.5) * 2 + 0.05 * (buzz - 0.5)
    flick *= np.clip(dips, 0.18, 1.0)
    flick = np.clip(flick, 0.20, 1.28).astype(np.float32)
    # Circular soft blur so sharp dips don't create a seam at the loop point
    kern = np.array([0.08, 0.18, 0.48, 0.18, 0.08], dtype=np.float32)
    pad = len(kern) // 2
    ext = np.concatenate([flick[-pad:], flick, flick[:pad]])
    flick = np.convolve(ext, kern, mode="valid")
    flick = np.clip(flick, 0.20, 1.28).astype(np.float32)
    # Explicit end→start blend so the loop has no lighting pop
    bridge = 20
    for i in range(bridge):
        t = (i + 1) / bridge
        idx = n - bridge + i
        flick[idx] = (1.0 - t) * flick[idx] + t * flick[0]
    return flick.astype(np.float32)


def build_masks(h: int, w: int, img: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    yy, xx = np.mgrid[0:h, 0:w]
    lum = img.mean(2)

    neon_roi = np.zeros((h, w), dtype=bool)
    neon_roi[NEON_Y0:NEON_Y1, NEON_X0:NEON_X1] = True

    # Bright neon tubes + soft glow (intensity only — never remask letter shapes)
    neon_core = neon_roi & (g > 120) & (g > r + 10) & (g > b - 8)
    neon_glow = neon_roi & (g > 65) & (g > r + 4)
    neon_glow = neon_glow | ndimage.binary_dilation(neon_core, iterations=8)
    neon_w = ndimage.gaussian_filter(neon_glow.astype(np.float32), sigma=2.8)
    neon_w = neon_w / (neon_w.max() + 1e-8)
    core_w = ndimage.gaussian_filter(neon_core.astype(np.float32), sigma=1.0)
    core_w = core_w / (core_w.max() + 1e-8)
    neon_w = np.clip(neon_w * 0.55 + core_w * 0.70, 0, 1)

    # Soft bloom halo around the sign (wall wash)
    bloom = ndimage.gaussian_filter(neon_w, sigma=14.0)
    bloom = bloom / (bloom.max() + 1e-8)

    dist_clock = np.sqrt((xx - CLOCK_CX) ** 2 + (yy - CLOCK_CY) ** 2)

    # Floor reflections: wet green-tinted patches + cable highlights
    floor = yy > int(h * 0.50)
    refl = floor & (g > r + 3) & (g > 22) & (g < 180)
    refl = refl & (dist_clock > CLOCK_R * 0.95)
    # also catch brighter wet sheen regardless of green bias
    sheen = floor & (lum > 35) & (lum < 140) & (g >= r - 5) & (dist_clock > CLOCK_R * 0.95)
    refl_w = ndimage.gaussian_filter((refl | sheen).astype(np.float32), sigma=3.5)
    refl_w = refl_w / (refl_w.max() + 1e-8)

    # Cable-ish dark structures for subtle warp
    dark = (lum < 45) & (yy > int(h * 0.35))
    cables = dark & ~(dist_clock < CLOCK_R * 1.05)
    cables = ndimage.binary_opening(cables, iterations=1)
    cable_w = ndimage.gaussian_filter(cables.astype(np.float32), sigma=5.0)
    cable_w = cable_w / (cable_w.max() + 1e-8)

    # Room ambient falloff from neon (walls + upper scene)
    ambient = ndimage.gaussian_filter(neon_w, sigma=36.0)
    ambient = ambient / (ambient.max() + 1e-8)

    # Clock glass / metal rim catch neon spill
    clock_glass = ((dist_clock < CLOCK_R * 0.92) & (dist_clock > CLOCK_R * 0.15)).astype(np.float32)
    clock_glass = ndimage.gaussian_filter(clock_glass, sigma=2.0)
    clock_glass = clock_glass / (clock_glass.max() + 1e-8)
    # brighter on upper-left of face (existing key light side)
    clock_lit = clock_glass * np.clip(1.15 - 0.55 * ((xx - CLOCK_CX) / CLOCK_R), 0.35, 1.2)
    clock_lit = clock_lit * np.clip(1.05 - 0.35 * ((yy - CLOCK_CY) / CLOCK_R), 0.45, 1.15)

    clock_face = (dist_clock < CLOCK_R * 0.90).astype(np.float32)
    clock_face = ndimage.gaussian_filter(clock_face, sigma=0.9)

    border = ((xx < 48) | (xx > w - 48) | (yy < 48) | (yy > h - 48)).astype(np.float32)

    return {
        "neon": neon_w.astype(np.float32),
        "bloom": bloom.astype(np.float32),
        "refl": refl_w.astype(np.float32),
        "cable": cable_w.astype(np.float32),
        "ambient": ambient.astype(np.float32),
        "border": border,
        "clock_face": clock_face.astype(np.float32),
        "clock_lit": clock_lit.astype(np.float32),
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
    """Modulate neon + room light. Flicker dims/brightens without reshaping letters."""
    out = img.astype(np.float32)
    neon = masks["neon"][..., None]
    bloom = masks["bloom"][..., None]
    refl = masks["refl"][..., None]
    ambient = masks["ambient"][..., None]
    clock_lit = masks["clock_lit"][..., None]
    mint = np.array([0.68, 1.20, 1.06], dtype=np.float32)

    # Multiplicative neon tube response (clear on/off feel)
    neon_mul = 0.28 + 0.72 * flick
    out = out * (1.0 - neon) + out * neon * neon_mul

    # Additive mint bloom / wall wash when bright; sink when dim
    bloom_amt = (flick - 0.85) * 55.0
    out += bloom * bloom_amt * mint

    if flick > 1.0:
        out += neon * (flick - 1.0) * 90.0 * mint
    elif flick < 0.55:
        out -= neon * (0.55 - flick) * 70.0

    # Wet floor + cable reflections track neon
    refl_amt = (flick - 0.75) * 52.0
    out += refl * refl_amt * np.array([0.50, 1.0, 0.82], dtype=np.float32)

    # Room electrical instability + neon ambient spill on walls
    room = 0.90 + 0.10 * flick + 0.05 * (room_flick - 0.5)
    out *= room
    out += ambient * (flick - 0.80) * 22.0 * np.array([0.35, 0.90, 0.70], dtype=np.float32)

    # Neon spill on clock glass / rim
    out += clock_lit * (flick - 0.80) * 18.0 * np.array([0.45, 0.95, 0.75], dtype=np.float32)

    return np.clip(out, 0, 255)


def draw_second_hand(img: np.ndarray, masks: dict, frame_i: int, n_frames: int) -> np.ndarray:
    """Sweeping second hand around the true hand pivot; seamless full rotation."""
    h, w = img.shape[:2]
    angle = -math.pi / 2 - 2 * math.pi * frame_i / n_frames
    length = CLOCK_R * 0.78
    stub = CLOCK_R * 0.16
    cx, cy = CLOCK_CX, CLOCK_CY
    x_tip = cx + length * math.cos(angle)
    y_tip = cy + length * math.sin(angle)
    x_stub = cx - stub * math.cos(angle)
    y_stub = cy - stub * math.sin(angle)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    shadow_off = 1.5
    draw.line(
        [(x_stub + shadow_off, y_stub + shadow_off), (x_tip + shadow_off, y_tip + shadow_off)],
        fill=(0, 0, 0, 90),
        width=4,
    )
    draw.line([(x_stub, y_stub), (cx, cy)], fill=(6, 8, 7, 255), width=3)
    draw.line([(cx, cy), (x_tip, y_tip)], fill=(4, 6, 5, 255), width=2)
    mid_x = cx + (length * 0.55) * math.cos(angle)
    mid_y = cy + (length * 0.55) * math.sin(angle)
    draw.line([(cx, cy), (mid_x, mid_y)], fill=(40, 48, 42, 160), width=1)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(8, 10, 9, 255))
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(28, 32, 30, 255))
    draw.ellipse([x_tip - 2, y_tip - 2, x_tip + 2, y_tip + 2], fill=(10, 12, 11, 255))
    draw.ellipse([x_stub - 3, y_stub - 3, x_stub + 3, y_stub + 3], fill=(8, 10, 9, 240))

    ov = np.array(overlay).astype(np.float32)
    face = (masks["dist_clock"] < CLOCK_R * 0.92).astype(np.float32)
    face = ndimage.gaussian_filter(face, sigma=0.6)
    alpha = (ov[:, :, 3:4] / 255.0) * face[..., None]
    return img * (1.0 - alpha) + ov[:, :, :3] * alpha


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
        # Draw hand AFTER lighting so it stays readable and catches glass spill below
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
