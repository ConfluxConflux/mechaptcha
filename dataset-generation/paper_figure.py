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
DEMO_WORD = "chars"        # shown for every catalogue cell
CATALOGUE_FONT = "lato"   # falls back to first available font

# Right-panel: show at most this many perturbation names to avoid overflow
_MAX_PERTURBS_SHOWN = 3

_LABEL = {
    "clean":          "clean",
    "easy_line":      "horiz. line",
    "hard_line":      "angled line",
    "wavy_line":      "wavy line",
    "two_lines":      "two lines",
    "blur":           "blur",
    "dots":           "dots",
    "salt_pepper":    "salt-and-pepper noise",
    "wave":           "wave",
    "rotation":       "rotation",
    "italic":         "italic",
    "bold":           "bold",
    "char_jitter":    "char jitter",
    "spacing_jitter": "spacing jitter",
}

# Display order for the single-column catalogue
_ORDER = [
    "clean",
    "easy_line", "hard_line", "wavy_line", "two_lines",
    "blur", "dots", "salt_pepper",
    "wave", "rotation",
    "italic", "bold",
    "char_jitter", "spacing_jitter",
]

# Border colour by category (matches chart palette)
_CAT_COLOR = {
    "clean":          "#888888",
    "easy_line":      "#C0392B",
    "hard_line":      "#C0392B",
    "wavy_line":      "#C0392B",
    "two_lines":      "#C0392B",
    "blur":           "#1f77b4",
    "dots":           "#1f77b4",
    "salt_pepper":    "#1f77b4",
    "wave":           "#7b2fa3",
    "rotation":       "#7b2fa3",
    "italic":         "#2ca02c",
    "bold":           "#2ca02c",
    "char_jitter":    "#7f7f7f",
    "spacing_jitter": "#7f7f7f",
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


def _render_random(seed: int, fonts: dict) -> tuple[np.ndarray, str, set]:
    """Render a random dataset sample; returns (image, text, active_perturbations)."""
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
    return arr, text, active


def make_paper_figure():
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)

    n_cat  = len(_ORDER)
    n_rrows = 14

    fig = plt.figure(figsize=(8.0, 7.5), facecolor="white")

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        left=0.01, right=0.99,
        top=0.94, bottom=0.01,
        wspace=0.10,
    )
    gs_cat = gridspec.GridSpecFromSubplotSpec(
        n_cat, 1, subplot_spec=gs[0],
        hspace=0.55,
    )
    gs_smp = gridspec.GridSpecFromSubplotSpec(
        n_rrows, 1, subplot_spec=gs[1],
        hspace=0.55,
    )

    # ── Left panel: perturbation catalogue ───────────────────────────────────
    for i, key in enumerate(_ORDER):
        ax = fig.add_subplot(gs_cat[i])
        ax.set_facecolor("white")
        ax.imshow(_render_single(key, fonts), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        color = _CAT_COLOR[key]
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
            sp.set_color(color)
        ax.set_title(_LABEL[key], fontsize=8, pad=3, loc="left",
                     color=color, fontweight="bold")

    # ── Right panel: random dataset samples ──────────────────────────────────
    for j in range(n_rrows):
        ax = fig.add_subplot(gs_smp[j])
        arr, text, active = _render_random(5000 + j, fonts)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("#aaaaaa")
        label = ", ".join(_LABEL.get(k, k) for k in sorted(active)) if active else "none"
        ax.set_title(label, fontsize=7, pad=3, loc="left", color="#555555")

    # ── Panel headers ─────────────────────────────────────────────────────────
    fig.text(0.26, 0.965, "Perturbation catalogue",
             ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")
    fig.text(0.74, 0.965, "Dataset samples",
             ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")

    fig.add_artist(plt.Line2D([0.515, 0.515], [0.01, 0.955],
                               transform=fig.transFigure,
                               color="#dddddd", linewidth=1.0, zorder=10))

    out = Path("paper_figure.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    out_svg = Path("paper_figure.svg")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}  |  {out_svg}")


if __name__ == "__main__":
    make_paper_figure()
