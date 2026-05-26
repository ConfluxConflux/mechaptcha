"""Comparison charts: linear vs MLP probes, and full layer range (input→logits).

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
        return v["test_acc"] if isinstance(v, dict) else float("nan")
    except (KeyError, TypeError):
        return float("nan")


_HOOK_LAYERS = ["conv_block_0", "conv_block_1", "conv_block_2", "conv_block_3", "pool", "embedding"]
_HOOK_SHORT   = ["cb0", "cb1", "cb2", "cb3", "pool", "embed"]
_FULL_LAYERS  = ["input"] + _HOOK_LAYERS + ["logits"]
_FULL_LABELS  = ["input\n(pixels)", "cb0", "cb1", "cb2", "cb3", "pool", "embed", "logits\n(output)"]

_ORDER = ["salt_pepper", "blur", "dots", "bold", "italic",
          "easy_line", "hard_line", "two_lines", "wavy_line", "wave", "rotation",
          "dumb_control", "variation_control"]

_DISPLAY_NAMES: dict[str, str] = {
    "wave":      "letter wave",
    "easy_line": "horizontal line",
    "hard_line": "angled line",
}


def _display(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name).replace("_", " ")


_CATEGORIES = {
    "Pixel-level noise": (["blur", "dots", "salt_pepper"],          ["#1f77b4", "#aec7e8", "#4a90d9"]),
    "Geometric":         (["rotation", "wave", "wavy_line",
                           "easy_line", "hard_line", "two_lines"],   ["#d62728", "#ff9896", "#e07070",
                                                                       "#9467bd", "#c5b0d5", "#8c5294"]),
    "Font style":        (["bold", "italic"],                        ["#2ca02c", "#98df8a"]),
    "Controls":          (["dumb_control", "variation_control"],     ["#7f7f7f", "#bdbdbd"]),
}


# ── Linear vs MLP side-by-side heatmap ───────────────────────────────────────

def plot_linear_vs_mlp(linear_path: Path, mlp_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lin = _load(linear_path)
    mlp = _load(mlp_path)
    order = [e for e in _ORDER if e in lin]
    n_exp, n_l = len(order), len(_HOOK_LAYERS)

    cmap = LinearSegmentedColormap.from_list("wb", ["white", "#1565C0"])
    norm = plt.Normalize(0.5, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("Linear probe  vs  MLP probe  (test accuracy)", fontsize=13, fontweight="bold")

    for ax, data, title in [(axes[0], lin, "Linear (logistic regression)"),
                             (axes[1], mlp, "MLP (64→32 hidden)")]:
        mat = np.array([[_acc(data, exp, l) for l in _HOOK_LAYERS] for exp in order])
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for i in range(n_exp):
            for j in range(n_l):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.87 else "black")
        ax.set_xticks(range(n_l))
        ax.set_xticklabels(_HOOK_SHORT, fontsize=9)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels([_display(e) for e in order], fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        n_dist = sum(1 for e in order if "control" not in e)
        ax.axhline(n_dist - 0.5, color="black", lw=1.5, ls="--")

    # Delta annotations on MLP panel
    mat_lin = np.array([[_acc(lin, exp, l) for l in _HOOK_LAYERS] for exp in order])
    mat_mlp = np.array([[_acc(mlp, exp, l) for l in _HOOK_LAYERS] for exp in order])
    delta = mat_mlp - mat_lin
    for i in range(n_exp):
        for j in range(n_l):
            d = delta[i, j]
            if not np.isnan(d) and abs(d) >= 0.01:
                sign = "+" if d > 0 else ""
                col = "#006400" if d > 0.02 else ("#990000" if d < -0.02 else "#555")
                axes[1].text(j, i + 0.33, f"{sign}{d:.0%}", ha="center", va="center",
                             fontsize=5.5, color=col, style="italic")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.72])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("test acc", fontsize=8)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.01, "Small italic numbers on MLP panel = delta vs linear", ha="center",
             fontsize=8, color="#555")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Full layer range: linear + MLP overlaid ───────────────────────────────────

def plot_full_layers(
    linear_path: Path,
    output_path: Path,
    mlp_path: Path | None = None,
) -> None:
    """Line chart spanning input → logits. If mlp_path provided, overlays MLP as dashed."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.font_manager import FontProperties

    lin = _load(linear_path)
    mlp = _load(mlp_path) if mlp_path and mlp_path.exists() else None

    # Only include layers that actually appear in the data
    avail_layers = [l for l in _FULL_LAYERS
                    if any(not np.isnan(_acc(lin, e, l)) for e in lin)]
    avail_labels = [_FULL_LABELS[_FULL_LAYERS.index(l)] for l in avail_layers]
    x = list(range(len(avail_layers)))

    bold_fp = FontProperties(weight="bold")
    normal_fp = FontProperties()

    fig, ax = plt.subplots(figsize=(11, 5))
    legend_handles: list = []

    for cat, (exps, colors) in _CATEGORIES.items():
        legend_handles.append(mlines.Line2D([], [], color="none", label=cat))
        for i, exp in enumerate(exps):
            if exp not in lin:
                continue
            color = colors[i % len(colors)]
            is_ctrl = cat == "Controls"

            lin_vals = [_acc(lin, exp, l) for l in avail_layers]
            ax.plot(x, lin_vals, marker="o", markersize=4, linewidth=1.8,
                    linestyle="--" if is_ctrl else "-", color=color, alpha=0.85)

            if mlp:
                mlp_vals = [_acc(mlp, exp, l) for l in avail_layers]
                ax.plot(x, mlp_vals, marker="s", markersize=3, linewidth=1.1,
                        linestyle=":", color=color, alpha=0.55)

            legend_handles.append(mlines.Line2D(
                [], [], color=color,
                linestyle="--" if is_ctrl else "-",
                marker="o", markersize=4,
                label=exp.replace("_", " ")))

    # Legend entries for line styles
    if mlp:
        legend_handles.append(mlines.Line2D([], [], color="none", label="─── = linear"))
        legend_handles.append(mlines.Line2D([], [], color="#555", ls="-",  lw=1.8, label="solid = linear"))
        legend_handles.append(mlines.Line2D([], [], color="#555", ls=":",  lw=1.1, label="dotted = MLP"))

    ax.set_xticks(x)
    ax.set_xticklabels(avail_labels, fontsize=8)
    ax.set_ylabel("Test accuracy")
    suffix = " (linear = solid, MLP = dotted)" if mlp else ""
    ax.set_title(f"Probe accuracy across full layer range: raw pixels → model output{suffix}")
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Shade bookend columns
    if "input" in avail_layers:
        ix = avail_layers.index("input")
        ax.axvspan(ix - 0.4, ix + 0.4, alpha=0.07, color="orange")
        ax.text(ix, 1.035, "raw\npixels", ha="center", va="bottom", fontsize=6.5,
                color="#a05000", transform=ax.get_xaxis_transform())
    if "logits" in avail_layers:
        lx = avail_layers.index("logits")
        ax.axvspan(lx - 0.4, lx + 0.4, alpha=0.07, color="green")
        ax.text(lx, 1.035, "model\noutput", ha="center", va="bottom", fontsize=6.5,
                color="#005000", transform=ax.get_xaxis_transform())

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

    mlp_hook = ROOT / "mlp" / "results.json"
    lin_full  = ROOT / "full" / "results.json"
    mlp_full  = ROOT / "full_mlp" / "results.json"

    if mlp_hook.exists():
        print("Generating linear vs MLP comparison (hook layers)...")
        plot_linear_vs_mlp(
            ROOT / "results.json",
            mlp_hook,
            ROOT / "chart_linear_vs_mlp.png",
        )

    if lin_full.exists():
        print("Generating full layer range chart (linear)...")
        plot_full_layers(
            lin_full,
            ROOT / "chart_full_layers.png",
            mlp_path=mlp_full if mlp_full.exists() else None,
        )
