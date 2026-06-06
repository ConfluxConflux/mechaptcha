"""Generate chart_lines_overlay.png: linear (solid) + MLP (dashed) superimposed."""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties

from probe.results import load_results
from probe.config import ALL_LAYERS

OUT = Path(__file__).parent / "chart_lines_overlay.png"

LINEAR_PATH = Path(__file__).parent / "results.json"
MLP_PATH    = Path(__file__).parent / "full_mlp" / "results.json"

_CONTROL_SUFFIXES = ("dumb_control", "variation_control")

_CATEGORIES: dict[str, list[str]] = {
    "Pixel-level noise": ["blur", "dots", "salt_pepper"],
    "Line artifact":     ["wavy_line", "easy_line", "hard_line", "two_lines"],
    "Geometric warp":    ["rotation", "wave"],
    "Font style":        ["bold", "italic"],
    "Controls":          ["dumb_control", "variation_control"],
}
_PALETTES: dict[str, list[str]] = {
    "Pixel-level noise": ["#1f77b4", "#aec7e8", "#4a90d9"],
    "Line artifact":     ["#8B1A1A", "#C0392B", "#E05555", "#F4AAAA"],
    "Geometric warp":    ["#7b2fa3", "#c5b0d5"],
    "Font style":        ["#2ca02c", "#98df8a"],
    "Controls":          ["#7f7f7f", "#bdbdbd"],
}
_DISPLAY: dict[str, str] = {
    "wave":              "letter wave",
    "easy_line":         "horizontal line",
    "hard_line":         "angled line",
    "salt_pepper":       "salt-and-pepper noise",
    "dumb_control":      "same-data control",
    "variation_control": "same-distribution control",
}


def _display(name: str) -> str:
    return _DISPLAY.get(name, name).replace("_", " ")


def _acc(results, exp, layer):
    try:
        v = results[exp][layer]
        return v.test_acc if hasattr(v, "test_acc") else v["test_acc"]
    except (KeyError, TypeError):
        return float("nan")


def main() -> None:
    lin = load_results(LINEAR_PATH)
    mlp = load_results(MLP_PATH)

    # Layers present in at least one of the two result sets (excluding bookends)
    all_layers = [
        l for l in ALL_LAYERS
        if l not in ("input", "logits")
        and (
            any(not np.isnan(_acc(lin, e, l)) for e in lin)
            or any(not np.isnan(_acc(mlp, e, l)) for e in mlp)
        )
    ]
    layer_labels = [l.replace("conv_block_", "cb") for l in all_layers]
    x = list(range(len(all_layers)))

    bold_fp   = FontProperties(weight="bold")
    normal_fp = FontProperties()

    fig, ax = plt.subplots(figsize=(9, 5))

    legend_handles: list = []

    for cat, exps in _CATEGORIES.items():
        colors = _PALETTES[cat]
        is_control = cat == "Controls"
        legend_handles.append(
            mlines.Line2D([], [], color="none", label=cat)
        )
        for i, exp in enumerate(exps):
            color = colors[i % len(colors)]
            base_ls = "--" if is_control else "-"

            # Linear (solid / base style)
            if exp in lin:
                ys_lin = [_acc(lin, exp, l) for l in all_layers]
                ax.plot(x, ys_lin,
                        marker="o", markersize=3.8, linewidth=1.6,
                        linestyle=base_ls, color=color, alpha=0.90, zorder=3)

            # MLP (dashed, slightly thinner, hollow marker)
            if exp in mlp:
                ys_mlp = [_acc(mlp, exp, l) for l in all_layers]
                mlp_ls = ":" if is_control else "--"
                ax.plot(x, ys_mlp,
                        marker="s", markersize=3.0, linewidth=1.2,
                        linestyle=mlp_ls, color=color, alpha=0.65,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=0.9, zorder=2)

            if exp in lin or exp in mlp:
                legend_handles.append(
                    mlines.Line2D([], [], color=color,
                                  linestyle=base_ls, marker="o",
                                  markersize=4, label=_display(exp))
                )

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Layer")
    ax.set_title("Linear probe (solid •) vs MLP probe (dashed ▫) — accuracy by layer")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Probe-type legend entries at top of legend
    style_handles = [
        mlines.Line2D([], [], color="#333", linestyle="-",  marker="o",
                      markersize=4, label="Linear probe"),
        mlines.Line2D([], [], color="#333", linestyle="--", marker="s",
                      markersize=3, markerfacecolor="white",
                      markeredgecolor="#333", label="MLP probe"),
    ]

    leg = ax.legend(
        handles=style_handles + [mlines.Line2D([], [], color="none", label="")] + legend_handles,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, frameon=True, fontsize=8, handlelength=2.2,
    )
    for text, handle in zip(leg.get_texts(), style_handles + [None] + legend_handles):
        if handle is None or handle.get_color() == "none":
            text.set_fontproperties(bold_fp)
            text.set_color("#333333")
        else:
            text.set_fontproperties(normal_fp)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
