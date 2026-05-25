#!/usr/bin/env python3
"""QA sheet: stacked distortions + single-distortion isolation."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from pathlib import Path

from generate.fonts import load_fonts
from generate.renderer import render_captcha, IMG_WIDTH, IMG_HEIGHT, SLOT_WIDTH
from generate.distortions import DISTORTIONS

FONT_DIR = Path("data/fonts")
CHARSET = "abcdefghijklmnopqrstuvwxyz"
N_CHARS = 5
DISTORTION_KEYS = list(DISTORTIONS.keys()) + ["char_jitter"]


def render_with_jitter(text, font, rng, jitter_px=4):
    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate(text):
        slot_x = i * SLOT_WIDTH
        bbox = draw.textbbox((0, 0), ch, font=font)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        x = slot_x + (SLOT_WIDTH - char_w) // 2 - bbox[0]
        y = (IMG_HEIGHT - char_h) // 2 - bbox[1]
        y += int(rng.integers(-jitter_px, jitter_px + 1))
        draw.text((x, y), ch, fill=0, font=font)
    return img


def make_sample(seed, fonts, active_distortions):
    font_names = sorted(fonts.keys())
    rng = np.random.default_rng(seed)
    text = "".join(rng.choice(list(CHARSET), size=N_CHARS))
    font_name = font_names[rng.integers(0, len(font_names))]
    font = fonts[font_name]

    if "char_jitter" in active_distortions:
        base = render_with_jitter(text, font, rng)
    else:
        base = render_captcha(text, font)

    arr = np.array(base)
    for key in DISTORTIONS:
        if key in active_distortions:
            arr = DISTORTIONS[key](arr, np.random.default_rng(seed))

    return arr, text, font_name


def make_qa():
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)
    font_names = sorted(fonts.keys())

    n_stacked = 10
    pixel_distortions = list(DISTORTIONS.keys())
    n_isolated = len(DISTORTION_KEYS) + 1  # +1 for clean baseline

    total_rows = n_stacked + 1 + n_isolated  # +1 for section divider row
    fig, axes = plt.subplots(total_rows, 1, figsize=(8, total_rows * 0.9),
                             gridspec_kw={"hspace": 0.6})

    # ── Section 1: random stacked ──────────────────────────────────────────
    for i in range(n_stacked):
        rng = np.random.default_rng(1000 + i)
        active = {k for k in DISTORTION_KEYS if rng.random() < 0.5}
        arr, text, font_name = make_sample(1000 + i, fonts, active)

        ax = axes[i]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.axis("off")
        distortion_str = ", ".join(sorted(active)) if active else "none"
        ax.set_title(f'text: "{text}" | distortions: {distortion_str}',
                     fontsize=7, pad=2, loc="left")

    # ── Divider ─────────────────────────────────────────────────────────────
    div_ax = axes[n_stacked]
    div_ax.axis("off")
    div_ax.text(0.5, 0.5, "SECTION 2: each distortion in isolation",
                ha="center", va="center", fontsize=9, fontweight="bold",
                transform=div_ax.transAxes)

    # ── Section 2: isolation ─────────────────────────────────────────────────
    isolation_seed = 2000
    isolation_items = [("clean (no distortion)", set())] + \
                      [(k, {k}) for k in DISTORTION_KEYS]

    for j, (label, active) in enumerate(isolation_items):
        arr, text, font_name = make_sample(isolation_seed, fonts, active)
        ax = axes[n_stacked + 1 + j]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.axis("off")
        ax.set_title(f'text: "{text}" | distortion: {label}',
                     fontsize=7, pad=2, loc="left")
    out = Path("qa_stacked.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Unlabelled version ───────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(n_stacked, 1, figsize=(4, n_stacked * 0.85),
                               gridspec_kw={"hspace": 0.05})
    for i in range(n_stacked):
        rng = np.random.default_rng(1000 + i)
        active = {k for k in DISTORTION_KEYS if rng.random() < 0.5}
        arr, _, _ = make_sample(1000 + i, fonts, active)
        axes2[i].imshow(arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes2[i].axis("off")

    plt.subplots_adjust(hspace=0.05)
    out2 = Path("qa_stacked_unlabelled.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    make_qa()
