#!/usr/bin/env python3
"""Paper-quality figure: perturbation catalogue (left) + random dataset samples (right)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from training.fonts import load_fonts
from training.renderer import render_captcha
from training.distortions import DISTORTIONS, sample_distortions

FONT_DIR = Path("../data/fonts")
CHARSET = "abcdefghijklmnopqrstuvwxyz"
DEMO_WORD = "noise"        # shown for every catalogue row; change as desired
CATALOGUE_FONT = "lato"   # clean sans-serif; falls back to first available font

_LABEL = {
    "clean":          "clean",
    "easy_line":      "horiz. line",
    "hard_line":      "angled line",
    "wavy_line":      "wavy line",
    "two_lines":      "two lines",
    "blur":           "blur",
    "dots":           "dots",
    "salt_pepper":    "salt & pepper",
    "wave":           "wave",
    "rotation":       "rotation",
    "italic":         "italic",
    "bold":           "bold",
    "char_jitter":    "char jitter",
    "spacing_jitter": "spacing jitter",
}

# Display order — grouped by category
_ORDER = [
    "clean",
    "easy_line", "hard_line", "wavy_line", "two_lines",
    "blur", "dots", "salt_pepper",
    "wave", "rotation",
    "italic", "bold",
    "char_jitter", "spacing_jitter",
]

# Pastel background per group for subtle visual grouping
_BG = {
    "clean":          "#ffffff",
    "easy_line":      "#eef2ff",
    "hard_line":      "#eef2ff",
    "wavy_line":      "#eef2ff",
    "two_lines":      "#eef2ff",
    "blur":           "#fff5ec",
    "dots":           "#fff5ec",
    "salt_pepper":    "#fff5ec",
    "wave":           "#edfff5",
    "rotation":       "#edfff5",
    "italic":         "#fffbec",
    "bold":           "#fffbec",
    "char_jitter":    "#f8f0ff",
    "spacing_jitter": "#f8f0ff",
}

# Accent colour for the left-side category indicator dot
_ACCENT = {
    "clean":          "#aaaaaa",
    "easy_line":      "#7b8ee8",
    "hard_line":      "#7b8ee8",
    "wavy_line":      "#7b8ee8",
    "two_lines":      "#7b8ee8",
    "blur":           "#e88a50",
    "dots":           "#e88a50",
    "salt_pepper":    "#e88a50",
    "wave":           "#4db87a",
    "rotation":       "#4db87a",
    "italic":         "#c9a830",
    "bold":           "#c9a830",
    "char_jitter":    "#9c6dd4",
    "spacing_jitter": "#9c6dd4",
}


def _render_single(key: str, fonts: dict) -> np.ndarray:
    """Render DEMO_WORD with exactly one perturbation applied."""
    font_names = sorted(fonts.keys())
    font_key = CATALOGUE_FONT if CATALOGUE_FONT in fonts else font_names[0]
    font = fonts[font_key]
    rng = np.random.default_rng(42)
    x_jitter = 9 if key == "spacing_jitter" else 0
    y_jitter = 4 if key == "char_jitter" else 0
    base = render_captcha(DEMO_WORD, font,
                          rng=rng if (x_jitter or y_jitter) else None,
                          x_jitter_px=x_jitter,
                          y_jitter_px=y_jitter)
    arr = np.array(base)
    if key in DISTORTIONS:
        arr = DISTORTIONS[key](arr, np.random.default_rng(42))
    return arr


def _render_random(seed: int, fonts: dict) -> np.ndarray:
    """Render a random dataset sample."""
    font_names = sorted(fonts.keys())
    rng = np.random.default_rng(seed)
    text = "".join(rng.choice(list(CHARSET), size=5))
    font = fonts[font_names[rng.integers(0, len(font_names))]]
    active = {k for k, v in sample_distortions(rng).items() if v}
    x_jitter = 9 if "spacing_jitter" in active else 0
    y_jitter = 4 if "char_jitter" in active else 0
    base = render_captcha(text, font,
                          rng=rng if (x_jitter or y_jitter) else None,
                          x_jitter_px=x_jitter,
                          y_jitter_px=y_jitter)
    arr = np.array(base)
    for key in DISTORTIONS:
        if key in active:
            arr = DISTORTIONS[key](arr, np.random.default_rng(seed))
    return arr


def make_paper_figure():
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)

    n_cat = len(_ORDER)   # 14 rows in catalogue
    n_rcols = 3
    n_rrows = 10          # 30 random samples; 10×3 gives ~correct aspect in right panel

    fig = plt.figure(figsize=(7.0, 4.8), facecolor="white")

    # ── Outer layout: [left panel | right panel] ───────────────────────────────
    # left=0.155 leaves room for the text labels that sit outside the axes
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 0.70],
        left=0.155, right=0.99,
        top=0.91, bottom=0.01,
        wspace=0.07,
    )
    gs_left  = gridspec.GridSpecFromSubplotSpec(n_cat, 1,          subplot_spec=gs[0], hspace=0.12)
    gs_right = gridspec.GridSpecFromSubplotSpec(n_rrows, n_rcols,  subplot_spec=gs[1], hspace=0.04, wspace=0.04)

    # ── Left panel: one row per perturbation ────────────────────────────────────
    for i, key in enumerate(_ORDER):
        ax = fig.add_subplot(gs_left[i])
        ax.set_facecolor(_BG.get(key, "white"))
        ax.imshow(_render_single(key, fonts), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
            sp.set_color("#ccc")

        # Coloured category dot  ·  label text
        ax.plot(-0.015, 0.5, "o", ms=3.5,
                color=_ACCENT.get(key, "#aaa"),
                transform=ax.transAxes, clip_on=False)
        ax.text(-0.032, 0.5, _LABEL[key],
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=6.5, color="#222",
                fontstyle="italic" if key == "clean" else "normal")

    # ── Right panel: random sample grid ────────────────────────────────────────
    for j in range(n_rrows * n_rcols):
        row, col = divmod(j, n_rcols)
        ax = fig.add_subplot(gs_right[row, col])
        ax.imshow(_render_random(5000 + j, fonts), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.axis("off")

    # ── Panel headers ───────────────────────────────────────────────────────────
    # Positions approximated from gridspec geometry (left=0.155, right=0.99,
    # width_ratios=[1.0, 0.70], wspace=0.07).
    # Left panel images span roughly [0.155, 0.615]; right spans [0.660, 0.990].
    fig.text(0.385, 0.945, "Individual perturbations",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")
    fig.text(0.825, 0.945, "Dataset samples",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")

    # Thin vertical separator between panels
    line_x = 0.638
    fig.add_artist(plt.Line2D([line_x, line_x], [0.01, 0.93],
                               transform=fig.transFigure,
                               color="#ccc", linewidth=0.6, zorder=10))

    out = Path("paper_figure.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    out_svg = Path("paper_figure.svg")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}  |  {out_svg}")


if __name__ == "__main__":
    make_paper_figure()
