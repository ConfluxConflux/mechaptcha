#!/usr/bin/env python3
"""QA sheet: stacked distortions + single-distortion isolation."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from pathlib import Path

from training.fonts import load_fonts
from training.renderer import render_captcha, IMG_WIDTH, IMG_HEIGHT, SLOT_WIDTH
from training.distortions import DISTORTIONS, ALL_DISTORTION_KEYS, sample_distortions

FONT_DIR = Path("data/fonts")
CHARSET = "abcdefghijklmnopqrstuvwxyz"
N_CHARS = 5
DISTORTION_KEYS = ALL_DISTORTION_KEYS


def _sample_active(rng) -> set:
    return {k for k, v in sample_distortions(rng).items() if v}


def make_sample(seed, fonts, active_distortions):
    font_names = sorted(fonts.keys())
    rng = np.random.default_rng(seed)
    text = "".join(rng.choice(list(CHARSET), size=N_CHARS))
    font_name = font_names[rng.integers(0, len(font_names))]
    font = fonts[font_name]

    x_jitter = 9 if "spacing_jitter" in active_distortions else 0
    y_jitter = 4 if "char_jitter" in active_distortions else 0
    needs_jitter = x_jitter > 0 or y_jitter > 0
    base = render_captcha(text, font,
                          rng=rng if needs_jitter else None,
                          x_jitter_px=x_jitter,
                          y_jitter_px=y_jitter)

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
    fig, axes = plt.subplots(total_rows, 1, figsize=(3.5, total_rows * 1.4),
                             gridspec_kw={"hspace": 0.9})

    # ── Section 1: random stacked ──────────────────────────────────────────
    for i in range(n_stacked):
        rng = np.random.default_rng(1000 + i)
        active = _sample_active(rng)
        arr, text, font_name = make_sample(1000 + i, fonts, active)

        ax = axes[i]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        distortion_str = ", ".join(sorted(active)) if active else "none"
        ax.set_title(f'text: "{text}" | perturbations: {distortion_str}',
                     fontsize=7, pad=2, loc="left")

    # ── Divider ─────────────────────────────────────────────────────────────
    div_ax = axes[n_stacked]
    div_ax.axis("off")
    div_ax.text(0.5, 0.5, "SECTION 2: each perturbation in isolation",
                ha="center", va="center", fontsize=9, fontweight="bold",
                transform=div_ax.transAxes)

    # ── Section 2: isolation ─────────────────────────────────────────────────
    isolation_seed = 2000
    isolation_items = [("clean (no perturbation)", set())] + \
                      [(k, {k}) for k in DISTORTION_KEYS]

    for j, (label, active) in enumerate(isolation_items):
        arr, text, font_name = make_sample(isolation_seed, fonts, active)
        ax = axes[n_stacked + 1 + j]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(f'text: "{text}" | perturbation: {label}',
                     fontsize=7, pad=2, loc="left")
    out = Path("qa_stacked.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Unlabelled version ───────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(n_stacked, 1, figsize=(3.5, n_stacked * 0.9),
                               gridspec_kw={"hspace": 0.05})
    for i in range(n_stacked):
        rng = np.random.default_rng(1000 + i)
        active = _sample_active(rng)
        arr, _, _ = make_sample(1000 + i, fonts, active)
        axes2[i].imshow(arr, cmap="gray", vmin=0, vmax=255)
        axes2[i].axis("off")

    plt.subplots_adjust(hspace=0.05)
    out2 = Path("qa_stacked_unlabelled.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    make_qa()
