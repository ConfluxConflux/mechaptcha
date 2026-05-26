"""Generate comparison charts: linear vs MLP probes, and full layer range (input→logits).

Usage (run after both probe runs complete):
    uv run python -m probe.plot_compare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _acc(d: dict, exp: str, layer: str) -> float:
    try:
        v = d[exp][layer]
        return v["test_acc"] if isinstance(v, dict) else v.test_acc
    except (KeyError, AttributeError):
        return float("nan")


# ── Linear vs MLP comparison ──────────────────────────────────────────────────

def plot_linear_vs_mlp(
    linear_path: Path,
    mlp_path: Path,
    output_path: Path,
) -> None:
    """Side-by-side heatmaps: linear (left) vs MLP (right)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lin = _load(linear_path)
    mlp = _load(mlp_path)

    LAYERS = ["conv_block_0", "conv_block_1", "conv_block_2", "conv_block_3", "pool", "embedding"]
    SHORT  = ["cb0", "cb1", "cb2", "cb3", "pool", "embed"]
    ORDER  = ["salt_pepper", "blur", "dots", "bold", "italic",
              "easy_line", "hard_line", "two_lines", "wavy_line", "wave", "rotation",
              "dumb_control", "variation_control"]
    ORDER  = [e for e in ORDER if e in lin]

    n_exp, n_l = len(ORDER), len(LAYERS)
    cmap = LinearSegmentedColormap.from_list("wb", ["white", "#1565C0"])
    norm = plt.Normalize(0.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("Linear probe  vs  MLP probe  (test accuracy)", fontsize=13, fontweight="bold")

    for ax, data, title in [(axes[0], lin, "Linear (logistic regression)"),
                             (axes[1], mlp, "MLP (64→32 hidden)")]:
        mat = np.array([[_acc(data, exp, l) for l in LAYERS] for exp in ORDER])
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for i in range(n_exp):
            for j in range(n_l):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.87 else "black")
        ax.set_xticks(range(n_l))
        ax.set_xticklabels(SHORT, fontsize=9)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels([e.replace("_", " ") for e in ORDER], fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)

        # Separator line above controls
        n_dist = sum(1 for e in ORDER if "control" not in e)
        ax.axhline(n_dist - 0.5, color="black", lw=1.5, ls="--")

    # Draw delta annotations (MLP - linear) on right plot
    ax_mlp = axes[1]
    mat_lin = np.array([[_acc(lin, exp, l) for l in LAYERS] for exp in ORDER])
    mat_mlp = np.array([[_acc(mlp, exp, l) for l in LAYERS] for exp in ORDER])
    delta = mat_mlp - mat_lin
    for i in range(n_exp):
        for j in range(n_l):
            d = delta[i, j]
            if not np.isnan(d) and abs(d) >= 0.01:
                sign = "+" if d > 0 else ""
                color = "#006400" if d > 0.02 else ("#990000" if d < -0.02 else "#555")
                ax_mlp.text(j, i + 0.32, f"{sign}{d:.0%}", ha="center", va="center",
                            fontsize=5.5, color=color, style="italic")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.72])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("test acc", fontsize=8)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cb.ax.tick_params(labelsize=7)

    fig.text(0.5, 0.01, "Italic small numbers on right = MLP − linear delta", ha="center",
             fontsize=8, color="#555")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 0.91, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Full layer range (input → logits) ─────────────────────────────────────────

def plot_full_layers(
    results_path: Path,
    output_path: Path,
) -> None:
    """Line chart spanning input → logits, showing the full information flow."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.font_manager import FontProperties

    data = _load(results_path)

    ALL_LAYERS = ["input", "conv_block_0", "conv_block_1", "conv_block_2",
                  "conv_block_3", "pool", "embedding", "logits"]
    LABELS = ["input\n(pixels)", "cb0", "cb1", "cb2", "cb3", "pool", "embed", "logits\n(output)"]

    CATEGORIES = {
        "Pixel-level noise": (["blur", "dots", "salt_pepper"],
                              ["#1f77b4", "#aec7e8", "#4a90d9"]),
        "Geometric":         (["rotation", "wave", "wavy_line", "easy_line", "hard_line", "two_lines"],
                              ["#d62728", "#ff9896", "#e07070", "#9467bd", "#c5b0d5", "#8c5294"]),
        "Font style":        (["bold", "italic"],
                              ["#2ca02c", "#98df8a"]),
        "Controls":          (["dumb_control", "variation_control"],
                              ["#7f7f7f", "#bdbdbd"]),
    }

    x = list(range(len(ALL_LAYERS)))
    bold_fp, normal_fp = FontProperties(weight="bold"), FontProperties()

    fig, ax = plt.subplots(figsize=(11, 5))
    legend_handles: list = []

    for cat, (exps, colors) in CATEGORIES.items():
        legend_handles.append(mlines.Line2D([], [], color="none", label=cat))
        for i, exp in enumerate(exps):
            if exp not in data:
                continue
            vals = [_acc(data, exp, l) for l in ALL_LAYERS]
            color = colors[i % len(colors)]
            is_ctrl = cat == "Controls"
            ax.plot(x, vals, marker="o", markersize=4, linewidth=1.5,
                    linestyle="--" if is_ctrl else "-", color=color, alpha=0.85)
            legend_handles.append(mlines.Line2D(
                [], [], color=color, linestyle="--" if is_ctrl else "-",
                marker="o", markersize=4, label=exp.replace("_", " ")))

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Linear probe accuracy: full layer range  (raw pixels → model output)")
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Shade the two "bookend" columns
    ax.axvspan(-0.4, 0.4, alpha=0.07, color="orange", label="_nolegend_")
    ax.axvspan(len(ALL_LAYERS) - 1.4, len(ALL_LAYERS) - 0.6, alpha=0.07,
               color="green", label="_nolegend_")
    ax.text(0, 1.03, "raw pixels", ha="center", va="bottom", fontsize=7, color="#a05000",
            transform=ax.get_xaxis_transform())
    ax.text(len(ALL_LAYERS) - 1, 1.03, "model output", ha="center", va="bottom",
            fontsize=7, color="#005000", transform=ax.get_xaxis_transform())

    leg = ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                    borderaxespad=0, frameon=True, fontsize=8, handlelength=2)
    for text, handle in zip(leg.get_texts(), legend_handles):
        if handle.get_color() == "none":
            text.set_fontproperties(bold_fp)
            text.set_color("#333333")
        else:
            text.set_fontproperties(normal_fp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    ROOT = Path("probe_results")

    if (ROOT / "mlp" / "results.json").exists():
        print("Generating linear vs MLP comparison...")
        plot_linear_vs_mlp(
            ROOT / "results.json",
            ROOT / "mlp" / "results.json",
            ROOT / "chart_linear_vs_mlp.png",
        )

    if (ROOT / "full" / "results.json").exists():
        print("Generating full layer range chart...")
        plot_full_layers(
            ROOT / "full" / "results.json",
            ROOT / "chart_full_layers.png",
        )
