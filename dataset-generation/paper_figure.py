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
    "salt_pepper":    "salt & pepper",
    "wave":           "wave",
    "rotation":       "rotation",
    "italic":         "italic",
    "bold":           "bold",
    "char_jitter":    "char jitter",
    "spacing_jitter": "spacing jitter",
}

# 3-column × 5-row arrangement (14 items + 1 empty cell at end)
_ORDER = [
    "clean",      "easy_line",      "hard_line",
    "wavy_line",  "two_lines",      "blur",
    "dots",       "salt_pepper",    "wave",
    "rotation",   "italic",         "bold",
    "char_jitter","spacing_jitter",  None,         # None = empty cell
]

# Subtle pastel group backgrounds
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

    n_cat_cols = 3
    n_cat_rows = 5   # 3×5 = 15 cells, 14 perturbations + 1 empty
    n_rcols    = 3
    n_rrows    = 5   # 15 random samples

    # hspace ≈ 0.50 gives each row enough gap for the set_title label while keeping
    # cells at roughly the correct 2.5:1 aspect ratio for the 160×64 CAPTCHA images.
    HSPACE = 0.50
    WSPACE = 0.04

    fig = plt.figure(figsize=(7.0, 3.9), facecolor="white")

    # ── Outer layout: [catalogue | samples], equal-width panels ───────────────
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        left=0.01, right=0.99,
        top=0.91, bottom=0.01,
        wspace=0.06,
    )
    gs_cat = gridspec.GridSpecFromSubplotSpec(
        n_cat_rows, n_cat_cols, subplot_spec=gs[0],
        hspace=HSPACE, wspace=WSPACE,
    )
    gs_smp = gridspec.GridSpecFromSubplotSpec(
        n_rrows, n_rcols, subplot_spec=gs[1],
        hspace=HSPACE, wspace=WSPACE,
    )

    # ── Left panel: perturbation catalogue ────────────────────────────────────
    for idx, key in enumerate(_ORDER):
        row, col = divmod(idx, n_cat_cols)
        ax = fig.add_subplot(gs_cat[row, col])

        if key is None:          # empty cell
            ax.axis("off")
            continue

        ax.set_facecolor(_BG.get(key, "white"))
        ax.imshow(_render_single(key, fonts), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
            sp.set_color("#ccc")

        label = _LABEL[key]
        ax.set_title(
            f'perturbation: {label}',
            fontsize=7, pad=2, loc="left",
            fontstyle="italic" if key == "clean" else "normal",
        )

    # ── Right panel: random dataset samples ───────────────────────────────────
    for j in range(n_rrows * n_rcols):
        row, col = divmod(j, n_rcols)
        ax = fig.add_subplot(gs_smp[row, col])
        arr, text, active = _render_random(5000 + j, fonts)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
            sp.set_color("#ccc")

        distortion_str = ", ".join(sorted(active)) if active else "none"
        full = f'text: "{text}" | perturbations: {distortion_str}'
        short = full if len(full) <= _TITLE_MAX else full[:_TITLE_MAX - 1] + "…"
        ax.set_title(short, fontsize=7, pad=2, loc="left")

    # ── Panel headers ──────────────────────────────────────────────────────────
    fig.text(0.255, 0.945, "Individual perturbations",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")
    fig.text(0.745, 0.945, "Dataset samples",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")

    # Thin vertical separator
    fig.add_artist(plt.Line2D([0.505, 0.505], [0.01, 0.92],
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
