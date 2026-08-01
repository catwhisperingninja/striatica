#!/usr/bin/env python3
"""Striation banner v3 — bigger tagline, light gray option, layer exports.

Changes from v2:
- Tagline font size increased from 25 → 38
- Tagline color: light warm gray
- Exports separate layers as transparent PNGs for Photoshop compositing
"""

import math
import hashlib
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H = 3200, 1200
BG = (6, 6, 14)
FONTS = "/sessions/zen-admiring-shannon/mnt/.skills/skills/canvas-design/canvas-fonts"
OUT_DIR = "/sessions/zen-admiring-shannon/mnt/striatica/banner-layers"

UI_COLORS = [
    (20, 60, 140, 3.0),
    (30, 90, 180, 2.5),
    (40, 160, 200, 2.0),
    (60, 200, 200, 1.2),
    (80, 180, 120, 0.8),
    (160, 60, 180, 1.5),
    (200, 50, 160, 1.8),
    (220, 80, 120, 1.0),
    (230, 160, 60, 0.6),
    (240, 240, 240, 0.3),
]

_COLOR_TABLE = []
for r, g, b, w in UI_COLORS:
    _COLOR_TABLE.extend([(r, g, b)] * int(w * 10))


def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def dh(seed, i):
    return int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def pick_color(seed, i, brightness=1.0):
    idx = int(dh(seed, i) * len(_COLOR_TABLE)) % len(_COLOR_TABLE)
    base = _COLOR_TABLE[idx]
    bv = 0.7 + dh(seed + 99, i) * 0.6
    bv *= brightness
    return tuple(max(0, min(255, int(c * bv))) for c in base)


def h_line(draw, y, x0, x1, color, w=1):
    draw.line([(x0, int(y)), (x1, int(y))], fill=color, width=w)


def h_line_rgba(draw, y, x0, x1, color_rgba, w=1):
    draw.line([(x0, int(y)), (x1, int(y))], fill=color_rgba, width=w)


def draw_striations(draw, region, seed, density, brightness=0.3, rgba=False):
    x0, y0, x1, y1 = region
    span = y1 - y0
    n = int(span * density)
    for i in range(n):
        t = i / max(n - 1, 1)
        y = y0 + t * span
        c = pick_color(seed, i, brightness)
        if rgba:
            # Compute alpha from brightness
            a = min(255, int(brightness * 255 * 1.5))
            c = c + (a,)
        w = 1 if dh(seed + 500, i) > 0.3 else 2
        if dh(seed + 1000, i) > 0.90:
            gf = 0.2 + dh(seed + 2000, i) * 0.4
            gw = 0.02 + dh(seed + 3000, i) * 0.05
            gs = x0 + gf * (x1 - x0)
            ge = gs + gw * (x1 - x0)
            if rgba:
                h_line_rgba(draw, y, x0, gs, c, w)
                h_line_rgba(draw, y, ge, x1, c, w)
            else:
                h_line(draw, y, x0, gs, c, w)
                h_line(draw, y, ge, x1, c, w)
        else:
            if rgba:
                h_line_rgba(draw, y, x0, x1, c, w)
            else:
                h_line(draw, y, x0, x1, c, w)


def draw_ghosts(draw, seed, count=200):
    for i in range(count):
        y = dh(seed, i) * H
        a = int(dh(seed + 100, i) * 8) + 2
        c = (6 + a, 8 + a, 16 + a * 2)
        xs = dh(seed + 200, i) * W * 0.25
        xe = W - dh(seed + 300, i) * W * 0.25
        h_line(draw, y, xs, xe, c)


def draw_column(draw, cx, cy, cw, ch, level, seed, rgba=False):
    densities = [0.3, 0.55, 0.85, 1.3]
    brightness_levels = [0.25, 0.4, 0.6, 0.85]
    d = densities[min(level, 3)]
    br = brightness_levels[min(level, 3)]

    x0 = cx - cw / 2
    y0 = cy - ch / 2
    n = int(ch * d * 2.0)

    for i in range(n):
        t = i / max(n - 1, 1)
        ly = y0 + t * ch
        if ly < y0 or ly > y0 + ch:
            continue

        c = pick_color(seed, i, br)
        if rgba:
            a = min(255, int(br * 255 * 1.2))
            c = c + (a,)
        w = 1 if dh(seed + 500, i) > 0.25 else 2

        inset_l = dh(seed + 600, i) * cw * 0.08
        inset_r = dh(seed + 700, i) * cw * 0.08
        if rgba:
            h_line_rgba(draw, ly, x0 + inset_l, x0 + cw - inset_r, c, w)
        else:
            h_line(draw, ly, x0 + inset_l, x0 + cw - inset_r, c, w)


def draw_clearing(draw, cx, cy, half_w, half_h, feather=50):
    for f in range(feather, 0, -1):
        progress = 1.0 - (f / feather)
        alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
        r = int(BG[0] + (6 - BG[0]) * alpha)
        g = int(BG[1] + (6 - BG[1]) * alpha)
        b = int(BG[2] + (14 - BG[2]) * alpha + alpha * 3)
        draw.rectangle(
            [cx - half_w - f, cy - half_h - f,
             cx + half_w + f, cy + half_h + f],
            fill=(r, g, b),
        )
    draw.rectangle(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        fill=(7, 7, 16),
    )


def draw_glow_spots(img, seed, count=14):
    glow_layer = Image.new("RGB", (W, H), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    bcy = H * 0.5
    bh = H * 0.30

    for i in range(count):
        x = W * 0.08 + dh(seed, i) * W * 0.84
        y = bcy + (dh(seed + 10, i) - 0.5) * bh * 0.8
        if abs(x - W / 2) < W * 0.12 and abs(y - bcy) < H * 0.06:
            continue
        c = pick_color(seed + 50, i, brightness=1.0)
        r = 8 + int(dh(seed + 20, i) * 12)
        glow_draw.ellipse([x - r, y - r, x + r, y + r], fill=c)

    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=25))
    return ImageChops.add(img, glow_layer)


# ── Column sequence (shared) ────────────────────────────────────────────
SEQ = [
    (0, 0.8), (1, 0.8), (2, 0.8),
    (-1, 1.2), (-1, 0.6),
    (3, 0.9), (3, 0.85), (3, 0.9),
    (3, 0.85), (3, 0.9), (3, 0.85),
    (3, 0.9), (3, 0.85), (3, 0.9),
    (-1, 0.6), (-1, 1.2),
    (2, 0.7), (2, 0.7), (2, 0.7),
    (2, 0.7), (2, 0.7),
    (-1, 0.6), (-1, 1.2),
    (2, 0.8), (1, 0.8), (0, 0.8),
]
TOTAL_REL = sum(w for _, w in SEQ)
MARGIN_X = W * 0.10
USABLE = W - 2 * MARGIN_X
BCY = H * 0.5
BH = H * 0.30
COL_H = H * 0.32


def load_fonts():
    try:
        return {
            'title': ImageFont.truetype(f"{FONTS}/JetBrainsMono-Bold.ttf", 62),
            'tagline': ImageFont.truetype(f"{FONTS}/Jura-Medium.ttf", 38),
            'label': ImageFont.truetype(f"{FONTS}/JetBrainsMono-Regular.ttf", 14),
            'micro': ImageFont.truetype(f"{FONTS}/JetBrainsMono-Regular.ttf", 11),
        }
    except Exception:
        f = ImageFont.load_default()
        return {'title': f, 'tagline': f, 'label': f, 'micro': f}


def compute_text_positions(fonts):
    """Compute all text positions once, reuse for layers."""
    tmp = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(tmp)

    title = "s t r i a t i c a"
    bb = d.textbbox((0, 0), title, font=fonts['title'])
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (W - tw) / 2
    ty = BCY - th / 2 - 18  # slight upward shift to make room for bigger tagline

    tagline = "geometric atlas for machine intelligence"
    bb2 = d.textbbox((0, 0), tagline, font=fonts['tagline'])
    tw2 = bb2[2] - bb2[0]
    tag_y = ty + th + 22

    clear_cx = W / 2
    clear_cy = (ty + tag_y + 30) / 2
    clear_hw = max(tw, tw2) / 2 + 55
    clear_hh = (tag_y + 30 - ty + 16) / 2 + 15

    return {
        'title': title, 'tx': tx, 'ty': ty, 'tw': tw, 'th': th,
        'tagline': tagline, 'tag_x': (W - tw2) / 2, 'tag_y': tag_y,
        'clear_cx': clear_cx, 'clear_cy': clear_cy,
        'clear_hw': clear_hw, 'clear_hh': clear_hh,
    }


def draw_columns(draw, rgba=False):
    cursor = MARGIN_X
    for i, (level, rel_w) in enumerate(SEQ):
        actual_w = (rel_w / TOTAL_REL) * USABLE
        cx = cursor + actual_w / 2
        if level >= 0:
            draw_column(draw, cx, BCY, actual_w * 0.82, COL_H, level,
                        seed=700 + i * 13, rgba=rgba)
        cursor += actual_w


def draw_cartographic(draw):
    rc = (24, 32, 68)
    ry_t = int(BCY - BH / 2 - 35)
    ry_b = int(BCY + BH / 2 + 35)

    h_line(draw, ry_t, W * 0.06, W * 0.94, rc)
    h_line(draw, ry_b, W * 0.06, W * 0.94, rc)

    for i in range(25):
        x = W * 0.06 + i * (W * 0.88 / 24)
        th = 7 if i % 6 == 0 else 3
        tc = (36, 42, 90) if i % 6 == 0 else rc
        draw.line([(x, ry_t - th), (x, ry_t)], fill=tc, width=1)
        draw.line([(x, ry_b), (x, ry_b + th)], fill=tc, width=1)

    for xp, c in [(0.10, (14, 18, 42)), (0.90, (14, 18, 42)),
                  (0.33, (18, 22, 50)), (0.67, (18, 22, 50))]:
        x = int(xp * W)
        draw.line([(x, H * 0.1), (x, H * 0.9)], fill=c, width=1)


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE (full banner)
# ═══════════════════════════════════════════════════════════════════════

def build_composite():
    fonts = load_fonts()
    tp = compute_text_positions(fonts)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Layer 1: Ghosts
    draw_ghosts(draw, seed=42, count=250)

    # Layer 2: Background strata
    for i, yp in enumerate([0.12, 0.28, 0.5, 0.72, 0.88]):
        y = yp * H
        draw_striations(draw, (0, y - H * 0.035, W, y + H * 0.035),
                        seed=200 + i * 7, density=0.3, brightness=0.15)

    # Layer 3: Center band
    draw_striations(draw, (0, BCY - BH / 2, W, BCY + BH / 2),
                    seed=500, density=0.5, brightness=0.35)

    # Layer 4: Columns
    draw_columns(draw)

    # Layer 5: Cartographic
    draw_cartographic(draw)

    # Layer 6: Clearing
    draw_clearing(draw, tp['clear_cx'], tp['clear_cy'],
                  tp['clear_hw'], tp['clear_hh'], feather=55)

    # Ghost striations inside clearing
    for i in range(25):
        ly = tp['clear_cy'] - tp['clear_hh'] + dh(888, i) * tp['clear_hh'] * 2
        c = (10, 12, 26)
        xs = tp['clear_cx'] - tp['clear_hw'] + dh(889, i) * tp['clear_hw'] * 0.3
        xe = tp['clear_cx'] + tp['clear_hw'] - dh(890, i) * tp['clear_hw'] * 0.3
        h_line(draw, ly, xs, xe, c)

    # Layer 7: Title glow
    glow_colors = [
        (5, (15, 40, 80)),
        (4, (20, 60, 110)),
        (3, (30, 85, 140)),
        (2, (40, 110, 160)),
        (1, (55, 140, 180)),
    ]
    for spread, gc in glow_colors:
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                if dx * dx + dy * dy <= spread * spread:
                    draw.text((tp['tx'] + dx, tp['ty'] + dy),
                              tp['title'], fill=gc, font=fonts['title'])

    # Title crisp
    draw.text((tp['tx'], tp['ty']), tp['title'],
              fill=(220, 235, 248), font=fonts['title'])

    # Tagline — light warm gray, larger
    draw.text((tp['tag_x'], tp['tag_y']), tp['tagline'],
              fill=(170, 175, 185), font=fonts['tagline'])

    # Specimen labels
    lc = (35, 45, 95)
    mc = (25, 32, 70)
    draw.text((W * 0.04, H * 0.04), "v0.1.0", fill=lc, font=fonts['label'])
    draw.text((W * 0.87, H * 0.04), "gpt2-small · L6", fill=lc, font=fonts['label'])
    draw.text((W * 0.04, H * 0.94), "░▒▓  density register  ▓▒░",
              fill=mc, font=fonts['micro'])
    draw.text((W * 0.79, H * 0.94), "24,576 features · PCA → UMAP → 3D",
              fill=mc, font=fonts['micro'])
    cli = "░▒▓  s t r i a t i c a  ≡≡≡≡≡  ▓▒░"
    bb3 = draw.textbbox((0, 0), cli, font=fonts['micro'])
    draw.text(((W - bb3[2] + bb3[0]) / 2, H * 0.94), cli,
              fill=(16, 20, 40), font=fonts['micro'])

    # Glow spots
    img = draw_glow_spots(img, seed=314, count=14)

    return img


# ═══════════════════════════════════════════════════════════════════════
# LAYER EXPORTS (transparent PNGs for Photoshop)
# ═══════════════════════════════════════════════════════════════════════

def export_layer_background():
    """Layer: solid background + ghosts + background strata."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_ghosts(draw, seed=42, count=250)
    for i, yp in enumerate([0.12, 0.28, 0.5, 0.72, 0.88]):
        y = yp * H
        draw_striations(draw, (0, y - H * 0.035, W, y + H * 0.035),
                        seed=200 + i * 7, density=0.3, brightness=0.15)
    return img


def export_layer_center_band():
    """Layer: center striation band (transparent)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_striations(draw, (0, BCY - BH / 2, W, BCY + BH / 2),
                    seed=500, density=0.5, brightness=0.35, rgba=True)
    return img


def export_layer_columns():
    """Layer: density columns (transparent)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_columns(draw, rgba=True)
    return img


def export_layer_cartographic():
    """Layer: cartographic references + verticals (transparent)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Use RGBA colors
    rc = (24, 32, 68, 255)
    ry_t = int(BCY - BH / 2 - 35)
    ry_b = int(BCY + BH / 2 + 35)
    h_line_rgba(draw, ry_t, W * 0.06, W * 0.94, rc)
    h_line_rgba(draw, ry_b, W * 0.06, W * 0.94, rc)
    for i in range(25):
        x = W * 0.06 + i * (W * 0.88 / 24)
        th = 7 if i % 6 == 0 else 3
        tc = (36, 42, 90, 255) if i % 6 == 0 else rc
        draw.line([(x, ry_t - th), (x, ry_t)], fill=tc, width=1)
        draw.line([(x, ry_b), (x, ry_b + th)], fill=tc, width=1)
    for xp in [0.10, 0.90, 0.33, 0.67]:
        x = int(xp * W)
        c = (18, 22, 50, 200)
        draw.line([(x, H * 0.1), (x, H * 0.9)], fill=c, width=1)
    return img


def export_layer_title():
    """Layer: title text + glow (transparent)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fonts = load_fonts()
    tp = compute_text_positions(fonts)

    glow_colors = [
        (5, (15, 40, 80, 120)),
        (4, (20, 60, 110, 140)),
        (3, (30, 85, 140, 170)),
        (2, (40, 110, 160, 200)),
        (1, (55, 140, 180, 230)),
    ]
    for spread, gc in glow_colors:
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                if dx * dx + dy * dy <= spread * spread:
                    draw.text((tp['tx'] + dx, tp['ty'] + dy),
                              tp['title'], fill=gc, font=fonts['title'])

    draw.text((tp['tx'], tp['ty']), tp['title'],
              fill=(220, 235, 248, 255), font=fonts['title'])
    return img


def export_layer_tagline():
    """Layer: tagline text (transparent)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fonts = load_fonts()
    tp = compute_text_positions(fonts)
    draw.text((tp['tag_x'], tp['tag_y']), tp['tagline'],
              fill=(170, 175, 185, 255), font=fonts['tagline'])
    return img


def export_layer_specimen():
    """Layer: corner labels + CLI reference (transparent)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fonts = load_fonts()
    lc = (35, 45, 95, 255)
    mc = (25, 32, 70, 255)
    draw.text((W * 0.04, H * 0.04), "v0.1.0", fill=lc, font=fonts['label'])
    draw.text((W * 0.87, H * 0.04), "gpt2-small · L6", fill=lc, font=fonts['label'])
    draw.text((W * 0.04, H * 0.94), "░▒▓  density register  ▓▒░",
              fill=mc, font=fonts['micro'])
    draw.text((W * 0.79, H * 0.94), "24,576 features · PCA → UMAP → 3D",
              fill=mc, font=fonts['micro'])
    cli = "░▒▓  s t r i a t i c a  ≡≡≡≡≡  ▓▒░"
    bb3 = draw.textbbox((0, 0), cli, font=fonts['micro'])
    draw.text(((W - bb3[2] + bb3[0]) / 2, H * 0.94), cli,
              fill=(16, 20, 40, 100), font=fonts['micro'])
    return img


def export_layer_glow():
    """Layer: glow spots (RGB, for Screen blend in PS)."""
    glow_layer = Image.new("RGB", (W, H), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    for i in range(14):
        x = W * 0.08 + dh(314, i) * W * 0.84
        y = BCY + (dh(324, i) - 0.5) * BH * 0.8
        if abs(x - W / 2) < W * 0.12 and abs(y - BCY) < H * 0.06:
            continue
        c = pick_color(364, i, brightness=1.0)
        r = 8 + int(dh(334, i) * 12)
        glow_draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=25))
    return glow_layer


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Composite ───────────────────────────────────────────────────────
    print("Building composite...")
    composite = build_composite()
    composite_path = "/sessions/zen-admiring-shannon/mnt/striatica/striatica-banner.png"
    composite.save(composite_path, "PNG", optimize=True)
    print(f"  → {composite_path}")

    # ── Layer exports ───────────────────────────────────────────────────
    print("Exporting layers...")

    layers = [
        ("01-background.png", export_layer_background),
        ("02-center-band.png", export_layer_center_band),
        ("03-columns.png", export_layer_columns),
        ("04-cartographic.png", export_layer_cartographic),
        ("05-title-glow.png", export_layer_title),
        ("06-tagline.png", export_layer_tagline),
        ("07-specimen-labels.png", export_layer_specimen),
        ("08-glow-spots.png", export_layer_glow),
    ]

    for name, fn in layers:
        layer = fn()
        path = os.path.join(OUT_DIR, name)
        layer.save(path, "PNG")
        mode = layer.mode
        print(f"  → {name} ({mode})")

    print(f"\nDone. {len(layers)} layers in {OUT_DIR}/")
    print("Photoshop stacking order: 01 (bottom) → 08 (top)")
    print("08-glow-spots is RGB black — use Screen blend mode")


if __name__ == "__main__":
    main()
