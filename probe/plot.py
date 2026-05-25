"""Visualisations of probe accuracy: heatmap and layer-by-layer line chart."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from probe.config import ALL_LAYERS
from probe.results import AllResults

# Experiments to highlight as controls (plotted at bottom / dashed lines)
_CONTROL_SUFFIXES = ("dumb_control", "variation_control")

# Grouping and colours for the line chart legend
_CATEGORIES: dict[str, list[str]] = {
    "Pixel-level noise": ["blur", "dots", "salt_pepper"],
    "Geometric":         ["rotation", "wave", "wavy_line", "easy_line", "hard_line", "two_lines"],
    "Font style":        ["bold", "italic"],
    "Controls":          ["dumb_control", "variation_control"],
}
_PALETTES: dict[str, list[str]] = {
    "Pixel-level noise": ["#1f77b4", "#aec7e8", "#4a90d9"],
    "Geometric":         ["#d62728", "#ff9896", "#e07070", "#9467bd", "#c5b0d5", "#8c5294"],
    "Font style":        ["#2ca02c", "#98df8a"],
    "Controls":          ["#7f7f7f", "#bdbdbd"],
}


def plot_heatmap(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    title: str = "Linear probe accuracy (test set)",
) -> None:
    """Save a heatmap of test accuracy for each experiment × layer.

    Cells are coloured from white (50% = chance) to dark blue (100%).
    Experiments that include 'control' in their name are grouped at the bottom.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def sort_key(name: str) -> tuple[int, str]:
        return (1, name) if any(s in name for s in _CONTROL_SUFFIXES) else (0, name)

    exp_names = sorted(results, key=sort_key)
    n_exp = len(exp_names)
    n_layers = len(layers)

    # Build matrix: rows=experiments, cols=layers; value=test_acc or NaN
    matrix = np.full((n_exp, n_layers), np.nan)
    for i, exp in enumerate(exp_names):
        for j, layer in enumerate(layers):
            if layer in results[exp]:
                matrix[i, j] = results[exp][layer].test_acc

    # Diverging colormap: 0.5=white, 1.0=dark blue
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)

    fig_h = max(4, 0.5 * n_exp + 1.5)
    fig_w = max(6, 1.6 * n_layers)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    # Annotate cells
    for i in range(n_exp):
        for j in range(n_layers):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.80 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=7, color=text_color)

    layer_labels = [l.replace("conv_block_", "cb") for l in layers]
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_exp))
    ax.set_yticklabels(exp_names, fontsize=8)

    # Draw a separator line before control experiments
    n_non_controls = sum(1 for e in exp_names if not any(s in e for s in _CONTROL_SUFFIXES))
    if 0 < n_non_controls < n_exp:
        ax.axhline(n_non_controls - 0.5, color="black", linewidth=1.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("test accuracy", fontsize=8)
    cbar.ax.axhline(0.5, color="red", linewidth=1, linestyle="--")

    ax.set_title(title, fontsize=10, pad=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
