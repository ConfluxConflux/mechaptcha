"""Visualisations of probe accuracy: heatmap, line chart, architecture diagram, PCA scatter."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from probe.config import ALL_LAYERS
from probe.results import AllResults

_CONTROL_SUFFIXES = ("dumb_control", "variation_control")

_DISPLAY_NAMES: dict[str, str] = {
    "wave":      "letter wave",
    "easy_line": "horizontal line",
    "hard_line": "angled line",
}


def _display(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name).replace("_", " ")


def _get_acc(results: AllResults, exp: str, layer: str) -> float:
    try:
        v = results[exp][layer]
        return v.test_acc if hasattr(v, "test_acc") else v["test_acc"]
    except (KeyError, TypeError):
        return float("nan")

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

_ARCH_LAYER_LABELS = {
    "conv_block_0": "ConvBlock 0\n64 ch",
    "conv_block_1": "ConvBlock 1\n128 ch",
    "conv_block_2": "ConvBlock 2\n256 ch",
    "conv_block_3": "ConvBlock 3\n384 ch",
    "pool":         "AvgPool\n384×4×10",
    "embedding":    "Embedding\n512",
}
_ARCH_LAYER_HEIGHTS = {
    "conv_block_0": 0.52,
    "conv_block_1": 0.60,
    "conv_block_2": 0.70,
    "conv_block_3": 0.78,
    "pool":         0.65,
    "embedding":    0.85,
}
_ARCH_ORDER = [
    "salt_pepper", "blur", "dots", "bold", "italic",
    "easy_line", "hard_line", "two_lines", "wavy_line", "wave", "rotation",
]

_PCA_EXPS = [
    ("salt_pepper", "Salt & pepper  (99% → 60%)"),
    ("italic",      "Italic font  (builds: 92% → 99%)"),
    ("rotation",    "Rotation  (stays weak: 76–83%)"),
]
_PCA_LAYERS = [
    ("conv_block_0", "cb0 — first conv"),
    ("conv_block_2", "cb2 — mid conv"),
    ("embedding",    "Embedding — final repr"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _acc_color(acc: float):
    import matplotlib.pyplot as plt
    t = max(0.0, min(1.0, (acc - 0.5) / 0.5))
    return plt.cm.Blues(0.15 + 0.80 * t)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    title: str = "Linear probe accuracy (test set)",
) -> None:
    """Heatmap of test accuracy for each experiment × layer."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def sort_key(name: str) -> tuple[int, str]:
        return (1, name) if any(s in name for s in _CONTROL_SUFFIXES) else (0, name)

    exp_names = sorted(results, key=sort_key)
    n_exp, n_layers = len(exp_names), len(layers)

    matrix = np.full((n_exp, n_layers), np.nan)
    for i, exp in enumerate(exp_names):
        for j, layer in enumerate(layers):
            if layer in results[exp]:
                matrix[i, j] = results[exp][layer].test_acc

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_layers), max(4, 0.5 * n_exp + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    for i in range(n_exp):
        for j in range(n_layers):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.80 else "black")

    layer_labels = [l.replace("conv_block_", "cb") for l in layers]
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_exp))
    ax.set_yticklabels(exp_names, fontsize=8)

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


# ── Line chart ────────────────────────────────────────────────────────────────

def plot_lines(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    title: str = "Linear probe accuracy by layer",
) -> None:
    """Line chart of test accuracy vs. layer, grouped legend on the right."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.font_manager import FontProperties

    layer_list = [l for l in layers if l not in ("input", "logits")]
    layer_labels = [l.replace("conv_block_", "cb") for l in layer_list]
    x = list(range(len(layer_list)))
    bold_fp, normal_fp = FontProperties(weight="bold"), FontProperties()

    fig, ax = plt.subplots(figsize=(9, 5))
    legend_handles: list = []

    for cat, exps in _CATEGORIES.items():
        colors = _PALETTES[cat]
        legend_handles.append(mlines.Line2D([], [], color="none", label=cat))
        for i, exp in enumerate(exps):
            if exp not in results:
                continue
            vals = [_get_acc(results, exp, l) for l in layer_list]
            is_control = cat == "Controls"
            color = colors[i % len(colors)]
            ax.plot(x, vals, marker="o", markersize=4, linewidth=1.5,
                    linestyle="--" if is_control else "-", color=color, alpha=0.85)
            legend_handles.append(mlines.Line2D(
                [], [], color=color, linestyle="--" if is_control else "-",
                marker="o", markersize=4, label=_display(exp),
            ))

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Layer")
    ax.set_title(title)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

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


# ── Architecture diagram ──────────────────────────────────────────────────────

def plot_arch(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
) -> None:
    """Architecture diagram (top) + per-experiment heatmap (bottom).

    Top panel: CNN pipeline with colour-coded boxes (avg probe accuracy) and
    probe tap-point circles at each layer.
    Bottom panel: compact heatmap aligned under the same columns.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap

    plot_layers = [l for l in layers if l in _ARCH_LAYER_LABELS]
    if not plot_layers:
        return

    distortions = [e for e in results if "control" not in e]
    avg_acc = {
        l: np.mean([_get_acc(results, e, l) for e in distortions if l in results[e]])
        for l in plot_layers
    }

    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor("#fafafa")

    ax = fig.add_axes([0.01, 0.54, 0.96, 0.42])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    ax.set_title("CaptchaCNN architecture with linear probe tap-points",
                 fontsize=13, pad=6, fontweight="bold")

    img_x, cy = 0.12, 1.825
    ax.add_patch(FancyBboxPatch((img_x, cy - 0.275), 1.0, 0.55,
        boxstyle="round,pad=0.05", facecolor="white", edgecolor="#aaa", linewidth=1.5))
    for ci, ch in enumerate("ABCDE"):
        ax.text(img_x + 0.12 + ci * 0.18, cy, ch,
                ha="center", va="center", fontsize=9, color="#111",
                fontfamily="monospace", fontweight="bold")
    ax.text(img_x + 0.5, cy - 0.42, "Input\n1 × 64 × 160",
            ha="center", va="center", fontsize=7, color="#555")

    out_x = 8.88
    ax.add_patch(FancyBboxPatch((out_x, cy - 0.275), 1.0, 0.55,
        boxstyle="round,pad=0.05", facecolor="#f0fff0", edgecolor="#6a9", linewidth=1.5))
    for ci in range(5):
        ax.text(out_x + 0.13 + ci * 0.18, cy, "A",
                ha="center", va="center", fontsize=9, color="#2a7",
                fontfamily="monospace", fontweight="bold")
    ax.text(out_x + 0.5, cy - 0.42, "5 heads\n5 × 26 logits",
            ha="center", va="center", fontsize=7, color="#555")

    n = len(plot_layers)
    box_w = 0.85
    box_pitch = (out_x - 0.1 - 1.30) / n
    xs = [1.30 + i * box_pitch + (box_pitch - box_w) / 2 for i in range(n)]

    for i, layer in enumerate(plot_layers):
        x = xs[i]
        h = _ARCH_LAYER_HEIGHTS.get(layer, 0.65)
        col = _acc_color(avg_acc[layer])
        ax.add_patch(FancyBboxPatch((x, cy - h / 2), box_w, h,
            boxstyle="round,pad=0.04", facecolor=col, edgecolor="#444",
            linewidth=1.4, zorder=3))
        tc = "white" if avg_acc[layer] > 0.84 else "#111"
        ax.text(x + box_w / 2, cy, _ARCH_LAYER_LABELS[layer],
                ha="center", va="center", fontsize=6.5, color=tc, zorder=4, linespacing=1.3)

    def _arrow(x0: float, x1: float) -> None:
        ax.annotate("", xy=(x1, cy), xytext=(x0, cy),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.3), zorder=2)

    _arrow(img_x + 1.0, xs[0])
    for i in range(n - 1):
        _arrow(xs[i] + box_w, xs[i + 1])
    _arrow(xs[-1] + box_w, out_x)

    probe_y_line = cy - max(_ARCH_LAYER_HEIGHTS.get(l, 0.65) for l in plot_layers) / 2 - 0.08
    probe_y_dot  = probe_y_line - 0.55
    for i, layer in enumerate(plot_layers):
        tx = xs[i] + box_w / 2
        ax.plot([tx, tx], [probe_y_line, probe_y_dot + 0.11],
                color="#cc3300", lw=1.3, linestyle="--", zorder=1)
        ax.add_patch(plt.Circle((tx, probe_y_dot), 0.115, color="#cc3300", zorder=5))
        ax.text(tx, probe_y_dot, f"{avg_acc[layer]:.0%}",
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", zorder=6)
        short = layer.replace("conv_block_", "cb")
        ax.text(tx, probe_y_dot - 0.21, f"probe ({short})",
                ha="center", va="top", fontsize=5.8, color="#cc3300")

    ax.text(8.2, 0.35,
            "Box color = avg probe accuracy (distortions only)\n"
            "Red = logistic regression: can it tell batch A from batch B?",
            ha="center", va="center", fontsize=7.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#ddd", alpha=0.95))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0.5, 1.0))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.955, 0.56, 0.012, 0.37])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("avg probe acc", fontsize=7)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cb.ax.tick_params(labelsize=6)

    order = [e for e in _ARCH_ORDER if e in results]
    matrix = np.array([[results[exp][l]["test_acc"] for l in plot_layers] for exp in order])
    cmap2 = LinearSegmentedColormap.from_list("wb", ["white", "#1565C0"])
    ax_h = fig.add_axes([0.06, 0.03, 0.87, 0.44])
    im = ax_h.imshow(matrix, cmap=cmap2, norm=plt.Normalize(0.5, 1.0), aspect="auto")
    for i in range(len(order)):
        for j in range(len(plot_layers)):
            v = matrix[i, j]
            ax_h.text(j, i, f"{v:.0%}", ha="center", va="center",
                      fontsize=8, color="white" if v > 0.87 else "black")

    short_labels = [l.replace("conv_block_", "cb") for l in plot_layers]
    ax_h.set_xticks(range(len(plot_layers)))
    ax_h.set_xticklabels(short_labels, fontsize=9)
    ax_h.set_yticks(range(len(order)))
    ax_h.set_yticklabels([_display(e) for e in order], fontsize=9)
    ax_h.set_xlabel("Layer  (left → right = deeper)", fontsize=9)
    ax_h.set_title(
        "Each cell = one trained logistic regression  |  color = test accuracy  (50% = chance)",
        fontsize=9, pad=5)

    cbar2_ax = fig.add_axes([0.945, 0.03, 0.012, 0.44])
    cb2 = fig.colorbar(im, cax=cbar2_ax)
    cb2.set_label("test acc", fontsize=7)
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cb2.ax.tick_params(labelsize=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── PCA scatter ───────────────────────────────────────────────────────────────

def plot_pca(
    activations_dir: Path,
    results: AllResults,
    output_path: Path,
    n_samples: int = 600,
) -> None:
    """3×3 PCA scatter grid showing batch A vs B cluster separation.

    Rows = selected experiments, columns = selected layers.
    Shading shows the 2D logistic decision boundary (fit in PCA space for viz only).
    Requires scikit-learn.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle(
        "Activation space (PCA → 2D)    Blue = batch B (clean)    Red = batch A (distorted)\n"
        "Shading = 2D logistic decision boundary  |  'Probe acc' = original high-dim result",
        fontsize=10,
    )

    rng = np.random.default_rng(42)

    for row, (exp, exp_label) in enumerate(_PCA_EXPS):
        exp_dir = activations_dir / exp
        for col, (layer, layer_label) in enumerate(_PCA_LAYERS):
            ax = axes[row][col]
            a = np.load(exp_dir / f"test_batch_a_{layer}.npy")
            b = np.load(exp_dir / f"test_batch_b_{layer}.npy")

            idx = rng.choice(len(a), size=min(n_samples, len(a)), replace=False)
            a_s, b_s = a[idx], b[idx]
            X = np.vstack([a_s, b_s])
            y = np.array([1] * len(a_s) + [0] * len(b_s))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=0)
            X2 = pca.fit_transform(X_scaled)
            var = pca.explained_variance_ratio_

            lr = LogisticRegression(C=1.0, max_iter=500, random_state=0).fit(X2, y)
            score_2d = lr.score(X2, y)

            pad = 0.6
            x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
            y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                                 np.linspace(y_min, y_max, 150))
            Z = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                        colors=["#c8dff8", "#fcd5ce"], alpha=0.30, zorder=0)
            ax.contour(xx, yy, Z, levels=[0.5],
                       colors=["#444"], linewidths=1.2, linestyles="--", zorder=1)
            ax.scatter(X2[y == 0, 0], X2[y == 0, 1], s=9, alpha=0.45,
                       color="#3572b0", label="Batch B (clean)", linewidths=0, zorder=3)
            ax.scatter(X2[y == 1, 0], X2[y == 1, 1], s=9, alpha=0.45,
                       color="#e05c3a", label="Batch A (distorted)", linewidths=0, zorder=3)

            real_acc = results.get(exp, {}).get(layer, {})
            if hasattr(real_acc, "test_acc"):
                real_acc = real_acc.test_acc
            elif isinstance(real_acc, dict):
                real_acc = real_acc.get("test_acc", float("nan"))
            else:
                real_acc = float("nan")

            title_col = "#006400" if real_acc >= 0.9 else ("#cc6600" if real_acc >= 0.7 else "#990000")
            ax.set_title(f"{layer_label}\nProbe acc: {real_acc:.1%}   (2D: {score_2d:.1%})",
                         fontsize=8, pad=3, color=title_col)
            ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=6.5)
            ax.tick_params(labelsize=6)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right", markerscale=1.8,
                          framealpha=0.85, handlelength=1, borderpad=0.5)

        fig.text(0.005,
                 axes[row][0].get_position().y0 + axes[row][0].get_position().height / 2,
                 exp_label, ha="left", va="center", fontsize=9, fontweight="bold", rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
