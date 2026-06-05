"""Visualisations of probe accuracy: heatmap, line chart, architecture diagram, PCA scatter."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from probe.config import ALL_LAYERS
from probe.results import AllResults


def _save(fig, path: Path, pgf: bool) -> None:
    """Save fig to path (PNG/PDF/etc.) and, when pgf=True, also as a same-named .pgf file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if pgf:
        fig.savefig(path.with_suffix(".pgf"), bbox_inches="tight")

_CONTROL_SUFFIXES = ("dumb_control", "variation_control")

_DISPLAY_NAMES: dict[str, str] = {
    "wave":             "letter wave",
    "easy_line":        "horizontal line",
    "hard_line":        "angled line",
    "dumb_control":     "same-data control",
    "variation_control": "same-distribution control",
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
    "wavy_line", "easy_line", "hard_line", "two_lines", "rotation", "wave",
]

_PCA_EXPS = [
    ("salt_pepper", "salt-and-pepper noise  (99% → 60%)"),
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
    pgf: bool = False,
    ax=None,
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
    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_layers), max(4, 0.5 * n_exp + 1.5)))
    fig = ax.figure
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
    if _standalone:
        fig.tight_layout()
        _save(fig, output_path, pgf)
        plt.close(fig)


# ── Line chart ────────────────────────────────────────────────────────────────

def plot_lines(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    title: str = "Linear probe accuracy by layer",
    pgf: bool = False,
    ax=None,
) -> None:
    """Line chart of test accuracy vs. layer, grouped legend on the right."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.font_manager import FontProperties

    # Only include layers that have at least one experiment with real data
    layer_list = [
        l for l in layers
        if l not in ("input", "logits")
        and any(not np.isnan(_get_acc(results, e, l)) for e in results)
    ]
    layer_labels = [l.replace("conv_block_", "cb") for l in layer_list]
    x = list(range(len(layer_list)))
    bold_fp, normal_fp = FontProperties(weight="bold"), FontProperties()

    _standalone = ax is None
    if _standalone:
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

    if _standalone:
        _save(ax.figure, output_path, pgf)
        plt.close(ax.figure)


# ── Architecture diagram ──────────────────────────────────────────────────────

def plot_arch(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    pgf: bool = False,
) -> None:
    """Architecture diagram (top) + per-experiment heatmap (bottom).

    Top panel: CNN pipeline with probe tap-points at every probed layer,
    including input (raw pixels) and logits if present in results.
    Bottom panel: heatmap aligned under the same columns.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap

    # Middle CNN layers (have architecture boxes)
    cnn_layers = [l for l in layers if l in _ARCH_LAYER_LABELS]
    if not cnn_layers:
        return

    # Bookend layers probed outside the CNN body
    has_input  = "input"  in layers and any(not np.isnan(_get_acc(results, e, "input"))  for e in results)
    has_logits = "logits" in layers and any(not np.isnan(_get_acc(results, e, "logits")) for e in results)

    # All layers for heatmap columns (left to right)
    all_probe_layers = (["input"] if has_input else []) + cnn_layers + (["logits"] if has_logits else [])

    distortions = [e for e in results if "control" not in e]
    avg_acc = {
        l: np.mean([v for e in distortions if not np.isnan(v := _get_acc(results, e, l))])
        for l in all_probe_layers
    }

    fig = plt.figure(figsize=(19, 10))
    fig.patch.set_facecolor("#fafafa")

    ax = fig.add_axes([0.01, 0.54, 0.96, 0.42])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    ax.set_title("CaptchaCNN architecture with linear probe tap-points",
                 fontsize=13, pad=6, fontweight="bold")

    img_x, cy = 0.12, 1.825

    # ── Input box ─────────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((img_x, cy - 0.275), 1.0, 0.55,
        boxstyle="round,pad=0.05", facecolor="white", edgecolor="#aaa", linewidth=1.5))
    for ci, ch in enumerate("ABCDE"):
        ax.text(img_x + 0.12 + ci * 0.18, cy, ch,
                ha="center", va="center", fontsize=9, color="#111",
                fontfamily="monospace", fontweight="bold")
    ax.text(img_x + 0.5, cy - 0.42, "Input\n1 × 64 × 160",
            ha="center", va="center", fontsize=7, color="#555")

    # ── Output heads box ──────────────────────────────────────────────────────
    out_x = 8.88
    ax.add_patch(FancyBboxPatch((out_x, cy - 0.275), 1.0, 0.55,
        boxstyle="round,pad=0.05", facecolor="#f0fff0", edgecolor="#6a9", linewidth=1.5))
    for ci in range(5):
        ax.text(out_x + 0.13 + ci * 0.18, cy, "A",
                ha="center", va="center", fontsize=9, color="#2a7",
                fontfamily="monospace", fontweight="bold")
    ax.text(out_x + 0.5, cy - 0.42, "5 heads\n5 × 26 logits",
            ha="center", va="center", fontsize=7, color="#555")

    # ── CNN layer boxes ────────────────────────────────────────────────────────
    n = len(cnn_layers)
    box_w = 0.85
    box_pitch = (out_x - 0.1 - 1.30) / n
    xs = [1.30 + i * box_pitch + (box_pitch - box_w) / 2 for i in range(n)]

    for i, layer in enumerate(cnn_layers):
        x = xs[i]
        h = _ARCH_LAYER_HEIGHTS.get(layer, 0.65)
        col = _acc_color(avg_acc[layer])
        ax.add_patch(FancyBboxPatch((x, cy - h / 2), box_w, h,
            boxstyle="round,pad=0.04", facecolor=col, edgecolor="#444",
            linewidth=1.4, zorder=3))
        tc = "white" if avg_acc[layer] > 0.84 else "#111"
        ax.text(x + box_w / 2, cy, _ARCH_LAYER_LABELS[layer],
                ha="center", va="center", fontsize=6.5, color=tc, zorder=4, linespacing=1.3)

    # ── Arrows ────────────────────────────────────────────────────────────────
    def _arrow(x0: float, x1: float) -> None:
        ax.annotate("", xy=(x1, cy), xytext=(x0, cy),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.3), zorder=2)

    _arrow(img_x + 1.0, xs[0])
    for i in range(n - 1):
        _arrow(xs[i] + box_w, xs[i + 1])
    _arrow(xs[-1] + box_w, out_x)

    # ── Probe tap-points ──────────────────────────────────────────────────────
    probe_y_line = cy - max(_ARCH_LAYER_HEIGHTS.get(l, 0.65) for l in cnn_layers) / 2 - 0.08
    probe_y_dot  = probe_y_line - 0.55

    def _probe_dot(tx: float, layer: str, label: str, color: str = "#cc3300") -> None:
        ax.plot([tx, tx], [probe_y_line, probe_y_dot + 0.11],
                color=color, lw=1.3, linestyle="--", zorder=1)
        ax.add_patch(plt.Circle((tx, probe_y_dot), 0.115, color=color, zorder=5))
        ax.text(tx, probe_y_dot, f"{avg_acc[layer]:.0%}",
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", zorder=6)
        ax.text(tx, probe_y_dot - 0.21, label,
                ha="center", va="top", fontsize=5.8, color=color)

    # Bookend probes (different colour so they stand out)
    INPUT_COL  = "#805500"   # warm brown — "before the network"
    LOGITS_COL = "#1a6b1a"   # dark green — "after the network"

    if has_input:
        _probe_dot(img_x + 0.5, "input", "probe\n(pixels)", color=INPUT_COL)

    for i, layer in enumerate(cnn_layers):
        _probe_dot(xs[i] + box_w / 2, layer,
                   f"probe ({layer.replace('conv_block_', 'cb')})")

    if has_logits:
        _probe_dot(out_x + 0.5, "logits", "probe\n(logits)", color=LOGITS_COL)

    # ── Legend note ───────────────────────────────────────────────────────────
    ax.text(4.8, 0.32,
            "Box color = avg probe accuracy (perturbations only)  ·  "
            "Circles = logistic regression: can it separate batch A from batch B?\n"
            "Brown = raw pixel probe  ·  Red = intermediate layer probes  ·  Green = logit probe",
            ha="center", va="center", fontsize=7.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#ddd", alpha=0.95))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0.5, 1.0))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.955, 0.56, 0.010, 0.37])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("avg probe acc", fontsize=7)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cb.ax.tick_params(labelsize=6)

    # ── Heatmap (all probed layers) ────────────────────────────────────────────
    order = [e for e in _ARCH_ORDER if e in results]
    matrix = np.array([[_get_acc(results, exp, l) for l in all_probe_layers] for exp in order])
    cmap2 = LinearSegmentedColormap.from_list("wb", ["white", "#1565C0"])
    ax_h = fig.add_axes([0.06, 0.03, 0.87, 0.44])
    im = ax_h.imshow(matrix, cmap=cmap2, norm=plt.Normalize(0.5, 1.0), aspect="auto")
    for i in range(len(order)):
        for j in range(len(all_probe_layers)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax_h.text(j, i, f"{v:.0%}", ha="center", va="center",
                          fontsize=8, color="white" if v > 0.87 else "black")

    def _col_label(l: str) -> str:
        if l == "input":   return "input\n(pixels)"
        if l == "logits":  return "logits\n(output)"
        return l.replace("conv_block_", "cb")

    col_labels = [_col_label(l) for l in all_probe_layers]
    ax_h.set_xticks(range(len(all_probe_layers)))
    ax_h.set_xticklabels(col_labels, fontsize=8.5)
    ax_h.set_yticks(range(len(order)))
    ax_h.set_yticklabels([_display(e) for e in order], fontsize=9)
    ax_h.set_xlabel("Layer  (left → right = deeper into network)", fontsize=9)
    ax_h.set_title(
        "Each cell = one trained logistic regression  |  color = test accuracy  (50% = chance)",
        fontsize=9, pad=5)

    # Shade the bookend columns
    if has_input:
        ax_h.axvspan(-0.5, 0.5, alpha=0.06, color="orange")
    if has_logits:
        ax_h.axvspan(len(all_probe_layers) - 1.5, len(all_probe_layers) - 0.5, alpha=0.06, color="green")

    cbar2_ax = fig.add_axes([0.945, 0.03, 0.010, 0.44])
    cb2 = fig.colorbar(im, cax=cbar2_ax)
    cb2.set_label("test acc", fontsize=7)
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cb2.ax.tick_params(labelsize=6)

    _save(fig, output_path, pgf)
    plt.close(fig)


# ── Full-layer line chart (includes input + logits) ──────────────────────────

def plot_full_layers(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    title: str = "Probe accuracy across full layer range: raw pixels → model output",
    pgf: bool = False,
    ax=None,
) -> None:
    """Line chart including the input (raw-pixel) and logits bookend layers."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.font_manager import FontProperties

    layer_list = [l for l in layers if any(not np.isnan(_get_acc(results, e, l)) for e in results)]
    layer_labels = [l.replace("conv_block_", "cb") for l in layer_list]
    x = list(range(len(layer_list)))
    bold_fp, normal_fp = FontProperties(weight="bold"), FontProperties()

    # Background shading for bookend zones
    input_idx  = next((i for i, l in enumerate(layer_list) if l == "input"), None)
    logits_idx = next((i for i, l in enumerate(layer_list) if l == "logits"), None)

    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=(11, 5))

    if input_idx is not None:
        ax.axvspan(-0.5, input_idx + 0.5, color="#f5e6cc", alpha=0.5, zorder=0)
        ax.text(input_idx, 1.005, "raw\npixels", ha="center", va="bottom", fontsize=7, color="#8B6914")
    if logits_idx is not None:
        ax.axvspan(logits_idx - 0.5, len(layer_list) - 0.5, color="#d4edda", alpha=0.5, zorder=0)
        ax.text(logits_idx, 1.005, "model\noutput", ha="center", va="bottom", fontsize=7, color="#2d6a4f")

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
            ax.plot(x, vals, marker="o", markersize=3.5, linewidth=1.5,
                    linestyle="--" if is_control else "-", color=color, alpha=0.85)
            legend_handles.append(mlines.Line2D(
                [], [], color=color, linestyle="--" if is_control else "-",
                marker="o", markersize=3.5, label=_display(exp),
            ))

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Layer")
    ax.set_title(title)
    ax.set_ylim(0.45, 1.08)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    leg = ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                    borderaxespad=0, frameon=True, fontsize=8, handlelength=2)
    for text, handle in zip(leg.get_texts(), legend_handles):
        text.set_fontproperties(bold_fp if handle.get_color() == "none" else normal_fp)
        if handle.get_color() == "none":
            text.set_color("#333333")

    if _standalone:
        _save(ax.figure, output_path, pgf)
        plt.close(ax.figure)


# ── Categories line chart + early-vs-final scatter ────────────────────────────

def plot_categories(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    pgf: bool = False,
    axes=None,
) -> None:
    """Two-panel figure: (left) mean±std per distortion category vs layer;
    (right) scatter of first-conv vs embedding probe accuracy per experiment."""
    import matplotlib.pyplot as plt

    body_layers = [l for l in layers
                   if l not in ("input", "logits")
                   and any(not np.isnan(_get_acc(results, e, l)) for e in results)]
    layer_labels = [l.replace("conv_block_", "cb") for l in body_layers]
    x = list(range(len(body_layers)))

    # Identify the first body layer (earliest conv) and the embedding layer for scatter
    first_layer = body_layers[0] if body_layers else None
    embed_layer = "embedding" if "embedding" in body_layers else (body_layers[-1] if body_layers else None)

    _standalone = axes is None
    if _standalone:
        fig, (ax_lines, ax_scatter) = plt.subplots(1, 2, figsize=(13, 5),
                                                    gridspec_kw={"width_ratios": [2, 1]})
    else:
        ax_lines, ax_scatter = axes
    ax_lines.axhline(0.5, color="grey", linewidth=0.8, linestyle=":", label="chance")

    non_control_cats = {k: v for k, v in _CATEGORIES.items() if k != "Controls"}
    for cat, exps in non_control_cats.items():
        cat_exps = [e for e in exps if e in results]
        if not cat_exps:
            continue
        color = _PALETTES[cat][0]
        matrix = np.array([[_get_acc(results, e, l) for l in body_layers] for e in cat_exps])
        mean = np.nanmean(matrix, axis=0)
        std  = np.nanstd(matrix, axis=0)
        ax_lines.plot(x, mean, marker="o", markersize=4, linewidth=2, color=color,
                      label=f"{cat} (n={len(cat_exps)})")
        ax_lines.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    # Controls as dashed lines
    for i, exp in enumerate(_CATEGORIES.get("Controls", [])):
        if exp not in results:
            continue
        vals = [_get_acc(results, exp, l) for l in body_layers]
        ax_lines.plot(x, vals, linestyle="--", color="#aaa", linewidth=1,
                      label=exp.replace("_", " "), alpha=0.7)

    ax_lines.set_xticks(x)
    ax_lines.set_xticklabels(layer_labels)
    ax_lines.set_ylim(0.45, 1.05)
    ax_lines.set_ylabel("Mean probe accuracy ± 1 std")
    ax_lines.set_title("By distortion category")
    ax_lines.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_lines.legend(fontsize=8, loc="lower left")
    ax_lines.grid(axis="y", alpha=0.3)

    # Scatter: first layer vs embedding
    if first_layer and embed_layer and first_layer != embed_layer:
        ax_scatter.axline((0.5, 0.5), slope=1, color="grey", linewidth=0.8,
                          linestyle="--", label="no forgetting")
        all_exps = [e for e in results if not any(c in e for c in _CONTROL_SUFFIXES)]
        cat_lookup = {exp: cat for cat, exps in _CATEGORIES.items() for exp in exps}
        for exp in all_exps:
            x_val = _get_acc(results, exp, first_layer)
            y_val = _get_acc(results, exp, embed_layer)
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            cat = cat_lookup.get(exp, "Font style")
            color = _PALETTES.get(cat, ["#888"])[0]
            ax_scatter.scatter(x_val, y_val, color=color, s=55, zorder=3)
            ax_scatter.annotate(_display(exp), (x_val, y_val),
                                textcoords="offset points", xytext=(4, 3), fontsize=7)
        first_label = first_layer.replace("conv_block_", "cb")
        ax_scatter.set_xlabel(f"{first_label} probe accuracy")
        ax_scatter.set_ylabel(f"Embedding probe accuracy")
        ax_scatter.set_title("Early encoding vs final encoding")
        ax_scatter.legend(fontsize=8, loc="upper left")
        ax_scatter.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_scatter.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_scatter.grid(alpha=0.25)

    if _standalone:
        ax_lines.figure.tight_layout()
        _save(ax_lines.figure, output_path, pgf)
        plt.close(ax_lines.figure)


# ── Forgetting bar chart ──────────────────────────────────────────────────────

def plot_forgetting(
    results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    pgf: bool = False,
    ax=None,
) -> None:
    """Stacked bar chart: peak probe accuracy and drop-to-embedding, sorted by drop."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    body_layers = [l for l in layers if l not in ("input", "logits")
                   and any(not np.isnan(_get_acc(results, e, l)) for e in results)]
    embed_layer = "embedding" if "embedding" in body_layers else (body_layers[-1] if body_layers else None)
    if not embed_layer:
        return

    exps = [e for e in results if not any(c in e for c in _CONTROL_SUFFIXES)]
    data = []
    for exp in exps:
        peak = max((_get_acc(results, exp, l) for l in body_layers), default=float("nan"))
        embed = _get_acc(results, exp, embed_layer)
        if not np.isnan(peak) and not np.isnan(embed):
            data.append((exp, peak, embed, peak - embed))

    data.sort(key=lambda r: r[3], reverse=True)  # sort by drop descending

    names   = [_display(r[0]) for r in data]
    peaks   = [r[1] for r in data]
    embeds  = [r[2] for r in data]
    drops   = [r[3] for r in data]

    colors = cm.tab20.colors
    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(data)), 6))
    x = np.arange(len(data))

    for i, (emb, drop, color) in enumerate(zip(embeds, drops, colors)):
        ax.bar(i, emb, color=(*color[:3], 0.40))
        ax.bar(i, drop, bottom=emb, color=color)
        sign = "-" if drop >= 0 else "+"
        ax.text(i, emb + drop + 0.005, f"{sign}{abs(drop):.1%}",
                ha="center", va="bottom", fontsize=8,
                color="#cc4400" if drop > 0.05 else "#007700")

    ax.axhline(0.5, color="grey", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(0.45, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Strategic forgetting: peak accuracy vs embedding (sorted by drop)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=(0.5, 0.5, 0.5, 0.4), label="Embedding accuracy"),
                        Patch(facecolor=(0.5, 0.5, 0.5, 1.0), label="Drop to embedding")],
              fontsize=9, loc="upper right")
    if _standalone:
        ax.figure.tight_layout()
        _save(ax.figure, output_path, pgf)
        plt.close(ax.figure)


# ── Linear vs MLP comparison heatmap ─────────────────────────────────────────

def plot_linear_vs_mlp(
    linear_results: AllResults,
    mlp_results: AllResults,
    output_path: Path,
    layers: tuple[str, ...] = ALL_LAYERS,
    pgf: bool = False,
) -> None:
    """Side-by-side heatmap comparing linear (logistic regression) and MLP probes."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def sort_key(name: str) -> tuple[int, str]:
        return (1, name) if any(s in name for s in _CONTROL_SUFFIXES) else (0, name)

    # Use experiments present in both; layers present in either
    all_exps = sorted(set(linear_results) | set(mlp_results), key=sort_key)
    layer_list = [l for l in layers if l not in ("input", "logits")
                  and (any(l in linear_results.get(e, {}) for e in all_exps)
                       or any(l in mlp_results.get(e, {}) for e in all_exps))]
    layer_labels = [l.replace("conv_block_", "cb") for l in layer_list]

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    n_exp, n_layers = len(all_exps), len(layer_list)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.5 * n_layers), max(4, 0.5 * n_exp + 2)),
                              sharey=True)

    for ax, res, panel_title in zip(axes,
                                     [linear_results, mlp_results],
                                     ["Linear (logistic regression)", "MLP (64×32 hidden)"]):
        matrix = np.full((n_exp, n_layers), np.nan)
        for i, exp in enumerate(all_exps):
            for j, layer in enumerate(layer_list):
                if layer in res.get(exp, {}):
                    matrix[i, j] = res[exp][layer].test_acc
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        for i in range(n_exp):
            for j in range(n_layers):
                val = matrix[i, j]
                if not np.isnan(val):
                    # Show delta vs linear in MLP panel
                    if ax is axes[1]:
                        lin_val = linear_results.get(all_exps[i], {}).get(layer_list[j])
                        lin_val = lin_val.test_acc if lin_val and hasattr(lin_val, "test_acc") else float("nan")
                        delta = val - lin_val
                        extra = f"\n{delta:+.0%}" if not np.isnan(delta) and abs(delta) > 0.005 else ""
                        ax.text(j, i, f"{val:.0%}{extra}", ha="center", va="center",
                                fontsize=6.5, color="white" if val > 0.80 else "black")
                    else:
                        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                                fontsize=7, color="white" if val > 0.80 else "black")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layer_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(panel_title, fontsize=10)
        if ax is axes[0]:
            ax.set_yticks(range(n_exp))
            ax.set_yticklabels(all_exps, fontsize=8)
        n_non = sum(1 for e in all_exps if not any(c in e for c in _CONTROL_SUFFIXES))
        if 0 < n_non < n_exp:
            ax.axhline(n_non - 0.5, color="black", linewidth=1.5, linestyle="--")

    fig.suptitle("Linear probe vs  MLP probe  (test accuracy)", fontsize=11, y=1.01)
    axes[1].text(0.5, -0.12, "Small italic numbers on MLP panel = delta vs linear",
                 ha="center", transform=axes[1].transAxes, fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, output_path, pgf)
    plt.close(fig)


# ── PCA scatter ───────────────────────────────────────────────────────────────

def plot_pca(
    activations_dir: Path,
    results: AllResults,
    output_path: Path,
    n_samples: int = 600,
    pgf: bool = False,
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
    _save(fig, output_path, pgf)
    plt.close(fig)
