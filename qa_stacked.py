#!/usr/bin/env python3
"""Quick QA: 10 CAPTCHAs with randomly stacked distortions, labelled."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from pathlib import Path

from generate.fonts import load_fonts
from generate.renderer import IMG_WIDTH, IMG_HEIGHT, SLOT_WIDTH
from generate.distortions import DISTORTIONS

FONT_DIR = Path("data/fonts")
CHARSET = "abcdefghijklmnopqrstuvwxyz"
N_CHARS = 5
N_SAMPLES = 10
DISTORTION_KEYS = list(DISTORTIONS.keys()) + ["char_jitter"]


def render_with_jitter(text, font, rng, jitter_px=5):
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


def apply_all(arr, active_distortions, rng):
    # apply pixel-level distortions in fixed order
    for key in list(DISTORTIONS.keys()):
        if key in active_distortions:
            arr = DISTORTIONS[key](arr, rng)
    return arr


def make_stacked_qa():
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)
    font_names = sorted(fonts.keys())

    fig, axes = plt.subplots(N_SAMPLES, 1, figsize=(10, N_SAMPLES * 1.4))
    fig.suptitle("Stacked distortions QA — each distortion independently random", fontsize=11)

    for i in range(N_SAMPLES):
        rng = np.random.default_rng(1000 + i)

        text = "".join(rng.choice(list(CHARSET), size=N_CHARS))
        font_name = font_names[rng.integers(0, len(font_names))]
        font = fonts[font_name]

        active = {k for k in DISTORTION_KEYS if rng.random() < 0.5}

        if "char_jitter" in active:
            base = render_with_jitter(text, font, rng)
        else:
            from generate.renderer import render_captcha
            base = render_captcha(text, font)

        arr = np.array(base)
        pixel_distortions = {k for k in active if k in DISTORTIONS}
        arr = apply_all(arr, pixel_distortions, rng)

        ax = axes[i]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.axis("off")

        label_parts = [f'"{text}" | {font_name}']
        if active:
            label_parts.append("distortions: " + ", ".join(sorted(active)))
        else:
            label_parts.append("distortions: none")
        ax.set_title(" | ".join(label_parts), fontsize=7, pad=2, loc="left")

    plt.tight_layout()
    out = Path("qa_stacked.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_stacked_qa()
