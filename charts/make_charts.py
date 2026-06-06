"""Generate per-model and cross-model charts from probe results.

Per-model charts are written to charts/<slug>/:

  heatmap.png           experiment x layer accuracy heatmap
  lines.png             one line per experiment across layers
  full_layers.png       full layer range: raw pixels → logits
  categories.png        grouped by distortion category
  forgetting.png        forgetting curve
  arch.png              architecture diagram overlay
  pca.png               PCA scatter of activations (if activations dir exists)

Collated charts (one subplot per model, plus cross-model comparisons) are written to charts/collated/:

  heatmap.png, lines.png, lines_wide.png, full_layers.png, forgetting.png, categories.png
  decodability_vs_depth.png, peak_vs_output.png

Usage:  uv run python charts/make_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from probe.config import ALL_LAYERS
from probe.plot import plot_arch, plot_categories, plot_forgetting, plot_full_layers, plot_heatmap, plot_linear_vs_mlp, plot_lines, plot_pca, plot_sparsity, plot_task_accuracy
from probe.results import AllResults, load_results

CHARTS_DIR = Path(__file__).resolve().parent

# Candidate runs, in plot order. Only those whose results.json exists are used.
# (label, results.json path, activations dir or None)
CANDIDATES: list[tuple[str, Path, Path | None]] = [
    ("CNN",                REPO_ROOT / "probe_results/full/results.json",                         None),
    ("DINOv2-S-lora",      REPO_ROOT / "dino_results/dinov2-small-lora/results.json",             REPO_ROOT / "dino_results/dinov2-small-lora/activations"),
    ("DINOv2-S-frozen",    REPO_ROOT / "dino_results/dinov2-small-frozen/results.json",           REPO_ROOT / "dino_results/dinov2-small-frozen/activations"),
    ("DINOv2-B-lora",      REPO_ROOT / "dino_results/dinov2-base-lora/results.json",              REPO_ROOT / "dino_results/dinov2-base-lora/activations"),
    ("DINOv2-B-frozen",    REPO_ROOT / "dino_results/dinov2-base-frozen/results.json",            REPO_ROOT / "dino_results/dinov2-base-frozen/activations"),
    ("DINOv2-L-lora",      REPO_ROOT / "dino_results/dinov2-large-lora/results.json",              REPO_ROOT / "dino_results/dinov2-large-lora/activations"),
    ("CLIP-B-lora",        REPO_ROOT / "dino_results/clip-vit-base-lora/results.json",            REPO_ROOT / "dino_results/clip-vit-base-lora/activations"),
    ("CLIP-B-frozen",      REPO_ROOT / "dino_results/clip-vit-base-frozen/results.json",          REPO_ROOT / "dino_results/clip-vit-base-frozen/activations"),
    ("ViT-B-lora",         REPO_ROOT / "dino_results/vit-base-supervised-lora/results.json",      REPO_ROOT / "dino_results/vit-base-supervised-lora/activations"),
    ("ViT-B-frozen",       REPO_ROOT / "dino_results/vit-base-supervised-frozen/results.json",    REPO_ROOT / "dino_results/vit-base-supervised-frozen/activations"),
    ("DINOv2-S",           REPO_ROOT / "dino_results/dinov2-small/results.json",                   REPO_ROOT / "dino_results/dinov2-small/activations"),
    ("DINOv2-B",           REPO_ROOT / "dino_results/dinov2-base/results.json",                    REPO_ROOT / "dino_results/dinov2-base/activations"),
    ("CLIP-B",             REPO_ROOT / "dino_results/clip-vit-base/results.json",                  REPO_ROOT / "dino_results/clip-vit-base/activations"),
]

def _probe_results(subdir: str) -> dict[str, Path]:
    """Build a label->Path map for a probe variant (mlp, sparse_logistic, etc.)."""
    return {
        "CNN":           REPO_ROOT / f"probe_results/full_{subdir}/results.json",
        "DINOv2-S-lora": REPO_ROOT / f"dino_results/dinov2-small-lora/{subdir}/results.json",
        "DINOv2-S-frozen":REPO_ROOT / f"dino_results/dinov2-small-frozen/{subdir}/results.json",
        "DINOv2-B-lora": REPO_ROOT / f"dino_results/dinov2-base-lora/{subdir}/results.json",
        "DINOv2-B-frozen": REPO_ROOT / f"dino_results/dinov2-base-frozen/{subdir}/results.json",
        "DINOv2-L-lora": REPO_ROOT / f"dino_results/dinov2-large-lora/{subdir}/results.json",
        "CLIP-B-lora":   REPO_ROOT / f"dino_results/clip-vit-base-lora/{subdir}/results.json",
        "CLIP-B-frozen": REPO_ROOT / f"dino_results/clip-vit-base-frozen/{subdir}/results.json",
        "ViT-B-lora":    REPO_ROOT / f"dino_results/vit-base-supervised-lora/{subdir}/results.json",
        "ViT-B-frozen":  REPO_ROOT / f"dino_results/vit-base-supervised-frozen/{subdir}/results.json",
    }

MLP_RESULTS:    dict[str, Path] = _probe_results("mlp")
SPARSE_RESULTS: dict[str, Path] = _probe_results("sparse_logistic")

TRAINING_METRICS: dict[str, Path | dict] = {
    "CNN": {
        "val_seq_acc": 0.9569, "val_char_acc": None,
        "freeze_backbone": False, "train_size": None,
    },
    "DINOv2-S-frozen":  REPO_ROOT / "dino_runs/dinov2-small-frozen/metrics.json",
    "DINOv2-S-lora":    REPO_ROOT / "dino_runs/dinov2-small-lora/metrics.json",
    "DINOv2-B-lora":    REPO_ROOT / "dino_runs/dinov2-base-lora/metrics.json",
    "DINOv2-B-frozen":  REPO_ROOT / "dino_runs/dinov2-base-frozen/metrics.json",
    "DINOv2-L-lora":    REPO_ROOT / "dino_runs/dinov2-large-lora/metrics.json",
    "CLIP-B-lora":      REPO_ROOT / "dino_runs/clip-vit-base-lora/metrics.json",
    "CLIP-B-frozen":    REPO_ROOT / "dino_runs/clip-vit-base-frozen/metrics.json",
    "ViT-B-lora":       REPO_ROOT / "dino_runs/vit-base-supervised-lora/metrics.json",
    "ViT-B-frozen":     REPO_ROOT / "dino_runs/vit-base-supervised-frozen/metrics.json",
}

TRANSCRIPTION_ACCURACY: dict[str, Path] = {
    "CNN":              REPO_ROOT / "probe_results/full/transcription_accuracy.json",
    "DINOv2-S-lora":    REPO_ROOT / "dino_results/dinov2-small-lora/transcription_accuracy.json",
    "DINOv2-S-frozen":  REPO_ROOT / "dino_results/dinov2-small-frozen/transcription_accuracy.json",
    "DINOv2-B-lora":    REPO_ROOT / "dino_results/dinov2-base-lora/transcription_accuracy.json",
    "DINOv2-B-frozen":  REPO_ROOT / "dino_results/dinov2-base-frozen/transcription_accuracy.json",
    "DINOv2-L-lora":    REPO_ROOT / "dino_results/dinov2-large-lora/transcription_accuracy.json",
    "CLIP-B-lora":      REPO_ROOT / "dino_results/clip-vit-base-lora/transcription_accuracy.json",
    "CLIP-B-frozen":    REPO_ROOT / "dino_results/clip-vit-base-frozen/transcription_accuracy.json",
    "ViT-B-lora":       REPO_ROOT / "dino_results/vit-base-supervised-lora/transcription_accuracy.json",
    "ViT-B-frozen":     REPO_ROOT / "dino_results/vit-base-supervised-frozen/transcription_accuracy.json",
    "DINOv2-S":         REPO_ROOT / "dino_results/dinov2-small/transcription_accuracy.json",
    "DINOv2-B":         REPO_ROOT / "dino_results/dinov2-base/transcription_accuracy.json",
    "CLIP-B":           REPO_ROOT / "dino_results/clip-vit-base/transcription_accuracy.json",
}

_CONTROL = ("same_data_control", "same_distribution_control")

_CATEGORIES: dict[str, list[str]] = {
    "Pixel-level noise": ["blur", "dots", "salt_pepper"],
    "Geometric":         ["rotation", "wave", "wavy_line", "easy_line", "hard_line", "two_lines"],
    "Font style":        ["bold", "italic"],
    "Controls":          ["same_data_control", "same_distribution_control"],
}
_PALETTES: dict[str, list[str]] = {
    "Pixel-level noise": ["#1f77b4", "#aec7e8", "#4a90d9"],
    "Geometric":         ["#d62728", "#ff9896", "#e07070", "#9467bd", "#c5b0d5", "#8c5294"],
    "Font style":        ["#2ca02c", "#98df8a"],
    "Controls":          ["#bdbdbd", "#969696"],
}


def _display(name: str) -> str:
    return {"wave": "letter wave", "easy_line": "horizontal line",
            "hard_line": "angled line",
            "same_data_control": "same-data control",
            "same_distribution_control": "same-distribution control",
            "dumb_control": "same-data control",
            "variation_control": "same-distribution control",
            }.get(name, name).replace("_", " ")


def _layer_label(name: str) -> str:
    if name.startswith("conv_block_"):
        return "cb " + name.split("_")[-1]
    if name.startswith("block_"):
        return "blk " + name.split("_")[-1]
    return name


def _slug(label: str) -> str:
    return label.split(" ")[0].replace("(", "").replace(")", "").lower().replace("/", "-")


def order_layers(results: AllResults) -> tuple[str, ...]:
    """Recover an input->output layer ordering for either the CNN or a ViT run."""
    layers = next(iter(results.values())).keys()
    layerset = set(layers)
    if any(l.startswith("block_") for l in layerset):  # ViT run
        blocks = sorted((l for l in layerset if l.startswith("block_")), key=lambda s: int(s.split("_")[1]))
        return tuple([l for l in ("input",) if l in layerset] + blocks
                     + [l for l in ("embedding", "logits") if l in layerset])
    return tuple(l for l in ALL_LAYERS if l in layerset)  # CNN run


def mean_curve(results: AllResults, layers: tuple[str, ...]) -> np.ndarray:
    """Mean test accuracy across non-control experiments at each layer."""
    exps = [e for e in results if not any(c in e for c in _CONTROL)]
    return np.array([
        np.mean([results[e][l].test_acc for e in exps if l in results[e]])
        for l in layers
    ])


def discover() -> list[tuple[str, AllResults, tuple[str, ...], Path | None]]:
    found = []
    for label, path, act_dir in CANDIDATES:
        if path.exists():
            res = load_results(path)
            found.append((label, res, order_layers(res), act_dir))
            print(f"  found: {label:30} ({path.relative_to(REPO_ROOT)})")
        else:
            print(f"  missing (skipped): {label:30} ({path.relative_to(REPO_ROOT)})")
    return found


def _save(fig, path: Path, pgf: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if pgf:
        fig.savefig(path.with_suffix(".pgf"), bbox_inches="tight")


def plot_per_model_accuracy(
    label: str, results: AllResults, layers: tuple[str, ...],
    out: Path, pgf: bool = False,
) -> None:
    """One line per distortion experiment, all layers including embedding and logits."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    x = list(range(len(layers)))
    xlabels = [_layer_label(l) for l in layers]
    logits_idx = next((i for i, l in enumerate(layers) if l == "logits"), None)

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(layers) + 3), 5.5))
    if logits_idx is not None:
        ax.axvspan(logits_idx - 0.45, logits_idx + 0.45, color="#d4edda", alpha=0.45, zorder=0)
        ax.text(logits_idx, 1.015, "task output\n(logits)", ha="center",
                va="bottom", fontsize=7, color="#2d6a4f")
    ax.axhline(0.5, color="#aaa", linewidth=0.9, linestyle=":", zorder=1)

    distortion_exps = [exp for cat, exps in _CATEGORIES.items()
                       if cat != "Controls" for exp in exps if exp in results]
    flat_colors: list[str] = []
    for cat, exps in _CATEGORIES.items():
        if cat == "Controls":
            continue
        palette = _PALETTES[cat]
        flat_colors.extend(palette[i % len(palette)] for i, exp in enumerate(exps) if exp in results)

    for exp, color in zip(distortion_exps, flat_colors):
        vals = [results[exp].get(l) for l in layers]
        ys = [v.test_acc if v is not None else float("nan") for v in vals]
        ax.plot(x, ys, marker="o", markersize=4, linewidth=1.6, color=color, alpha=0.85, zorder=2)

    # Controls as dashed muted baselines
    control_entries = [
        ("same_data_control",         "#999999"),
        ("same_distribution_control", "#bbbbbb"),
    ]
    for exp, color in control_entries:
        if exp not in results:
            continue
        vals = [results[exp].get(l) for l in layers]
        ys = [v.test_acc if v is not None else float("nan") for v in vals]
        ax.plot(x, ys, linestyle="--", linewidth=1.2, color=color, alpha=0.8, zorder=1)

    from matplotlib.font_manager import FontProperties
    bold_fp = FontProperties(weight="bold")
    distortion_handles = [
        mlines.Line2D([], [], color=color, linestyle="-", marker="o",
                      markersize=4, label=_display(exp))
        for exp, color in zip(distortion_exps, flat_colors)
    ]
    control_handles = [
        mlines.Line2D([], [], color="none", label="Controls"),
    ] + [
        mlines.Line2D([], [], color=color, linestyle="--", linewidth=1.2,
                      label=_display(exp))
        for exp, color in control_entries if exp in results
    ]
    legend_handles = distortion_handles + control_handles

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Linear probe test accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(f"Linear probe accuracy across layers — {label}", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    leg = ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                    borderaxespad=0, frameon=True, fontsize=8, handlelength=2.2)
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        if handle.get_color() == "none":
            text.set_fontproperties(bold_fp)
            text.set_color("#444")
    fig.tight_layout()
    _save(fig, out, pgf)
    plt.close(fig)
    print(f"  per_model_accuracy.png")


def plot_transcription_accuracy(out: Path, val_size: int = 5000, pgf: bool = False) -> None:
    """Bar chart of val sequence accuracy across all models, with binomial SE error bars.

    Groups LoRA-adapted models together; the frozen baseline is visually distinct.
    Error bars are ±1 binomial SE = sqrt(p*(1-p)/n) using val_size as n.
    """
    import json
    import matplotlib.pyplot as plt

    labels, accs, errs, colors = [], [], [], []
    for label, entry in TRAINING_METRICS.items():
        if isinstance(entry, Path):
            if not entry.exists():
                continue
            m = json.loads(entry.read_text())
        else:
            m = entry
        seq_acc = m.get("val_seq_acc") or m.get("best_val_seq_acc")
        if seq_acc is None:
            continue
        n = m.get("val_size") or val_size
        se = (seq_acc * (1 - seq_acc) / n) ** 0.5
        labels.append(label)
        accs.append(seq_acc)
        errs.append(se)
        frozen = m.get("freeze_backbone", False)
        colors.append("#b0b0b0" if frozen else "#3b6ea5")

    if not labels:
        print("  transcription_accuracy.png skipped (no metrics.json files found yet)")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(labels)), 5))
    x = range(len(labels))
    bars = ax.bar(x, accs, color=colors, width=0.6,
                  yerr=errs, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555"})

    for bar, acc, err in zip(bars, accs, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + err + 0.005,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Val sequence accuracy")
    ax.set_ylim(0, min(1.05, max(accs) + 0.12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("CAPTCHA transcription accuracy by model\n"
                 "(LoRA-adapted = blue, frozen backbone = grey)", fontsize=11)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#3b6ea5", label="LoRA-adapted"),
        Patch(facecolor="#b0b0b0", label="Frozen backbone (heads only)"),
    ], fontsize=8, loc="lower right")

    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out, pgf)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_depth_curve(models, out: Path, pgf: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(models)))
    for (label, res, layers, _act), color in zip(models, colors):
        curve = mean_curve(res, layers)
        x = np.linspace(0, 1, len(layers))
        ax.plot(x, curve, "-o", color=color, label=label, markersize=4, linewidth=2)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Relative layer depth  (0 = input  →  1 = output logits)", fontsize=11)
    ax.set_ylabel("Linear probe accuracy  (A vs. B)", fontsize=11)
    ax.set_title("Linear decodability of distortion features across network depth", fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out, pgf)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_peak_vs_output(models, out: Path, pgf: bool = False) -> None:
    import matplotlib.pyplot as plt

    labels = [m[0] for m in models]
    peaks, finals = [], []
    for _label, res, layers, _act in models:
        curve = mean_curve(res, layers)
        peaks.append(curve.max())
        finals.append(curve[-1])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 5))
    ax.bar(x - w / 2, peaks, w, label="peak across depth", color="#3b6ea5")
    ax.bar(x + w / 2, finals, w, label="at final logits", color="#a53b3b")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("mean linear-probe accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Peak vs. output decodability: distortion is attenuated toward the\n"
                 "behaviorally-invariant output, but never fully erased", fontsize=11)
    ax.legend(fontsize=9)
    for xi, (p, f) in enumerate(zip(peaks, finals)):
        ax.text(xi - w / 2, p + 0.01, f"{p:.0%}", ha="center", fontsize=8)
        ax.text(xi + w / 2, f + 0.01, f"{f:.0%}", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out, pgf)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def _short_label(label: str) -> str:
    """Compact model name for subplot titles."""
    return label.split(" ")[0]


def _compact_layer_label(label: str) -> str:
    """Short tick labels for narrow, column-oriented plots."""
    if label.startswith("block_"):
        return "b" + label.split("_", 1)[1]
    if label.startswith("conv_block_"):
        return "cb" + label.rsplit("_", 1)[1]
    if label == "embedding":
        return "emb"
    if label == "logits":
        return "log"
    return label


def collate_plots(models: list, pgf: bool = False) -> None:
    """For each chart type, produce a single figure with one subplot per model."""
    if not models:
        return

    n = len(models)
    out_dir = CHARTS_DIR / "collated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(max(6, 1.6 * 6) * n, max(4, 0.5 * 13 + 1.5)),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (label, res, layers, _act) in zip(axes, models):
        plot_heatmap(res, None, layers=layers, title=_short_label(label), ax=ax)
    fig.suptitle("Linear probe accuracy — all models", fontsize=13)
    _save(fig, out_dir / "heatmap.png", pgf)
    plt.close(fig)
    print(f"  collated/heatmap.png")

    # ── lines ─────────────────────────────────────────────────────────────────
    # Column-oriented version for LaTeX documents. One model per row keeps each
    # subplot readable after fitting the figure to a single column.
    fig, axes = plt.subplots(n, 1, figsize=(6.8, max(2.8 * n, 4.5)), constrained_layout=True)
    if n == 1:
        axes = [axes]
    legend_handles = None
    legend_labels = None
    for ax, (label, res, layers, _act) in zip(axes, models):
        plot_lines(res, None, layers=layers, title=_short_label(label), ax=ax)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()
        tick_labels = [_compact_layer_label(t.get_text()) for t in ax.get_xticklabels()]
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=7)
        ax.set_xlabel("")
    if len(axes) > 0:
        axes[-1].set_xlabel("Layer")
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            ncol=3,
            frameon=True,
            fontsize=7,
            handlelength=1.6,
            columnspacing=1.0,
        )
    fig.suptitle("Linear probe accuracy by layer — all models", fontsize=13)
    _save(fig, out_dir / "lines.png", pgf)
    plt.close(fig)
    print(f"  collated/lines.png")

    # Also preserve the old wide layout for slides or desktop inspection.
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (label, res, layers, _act) in zip(axes, models):
        plot_lines(res, None, layers=layers, title=_short_label(label), ax=ax)
    fig.suptitle("Linear probe accuracy by layer — all models", fontsize=13)
    _save(fig, out_dir / "lines_wide.png", pgf)
    plt.close(fig)
    print(f"  collated/lines_wide.png")

    # ── full_layers ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (label, res, layers, _act) in zip(axes, models):
        plot_full_layers(res, None, layers=layers, title=_short_label(label), ax=ax)
    fig.suptitle("Probe accuracy: raw pixels → logits — all models", fontsize=13)
    _save(fig, out_dir / "full_layers.png", pgf)
    plt.close(fig)
    print(f"  collated/full_layers.png")

    # ── forgetting ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(max(8, 1.3 * 11) * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (label, res, layers, _act) in zip(axes, models):
        ax.set_title(_short_label(label), fontsize=11)
        plot_forgetting(res, None, layers=layers, ax=ax)
    fig.suptitle("Strategic forgetting — all models", fontsize=13)
    _save(fig, out_dir / "forgetting.png", pgf)
    plt.close(fig)
    print(f"  collated/forgetting.png")

    # ── categories ────────────────────────────────────────────────────────────
    # Each model needs 2 sub-panels (lines + scatter); use gridspec with 2 cols per model.
    fig = plt.figure(figsize=(13 * n, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, n * 2, width_ratios=([2, 1] * n))
    for i, (label, res, layers, _act) in enumerate(models):
        ax_lines = fig.add_subplot(gs[0, i * 2])
        ax_scatter = fig.add_subplot(gs[0, i * 2 + 1])
        ax_lines.set_title(_short_label(label), fontsize=11)
        plot_categories(res, None, layers=layers, axes=(ax_lines, ax_scatter))
    fig.suptitle("By distortion category — all models", fontsize=13)
    _save(fig, out_dir / "categories.png", pgf)
    plt.close(fig)
    print(f"  collated/categories.png")


def write_transcription_accuracy_csv(out: Path) -> None:
    """Write charts/transcription_accuracy.csv — one row per model with overall accuracy.

    Accuracy is averaged across all distortion experiments (controls excluded).
    Columns: model, seq_acc, char_acc.
    """
    import csv
    import json

    _CONTROLS = {"same_data_control", "same_distribution_control", "dumb_control", "variation_control"}

    rows = []
    for label, path in TRANSCRIPTION_ACCURACY.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        seq_accs, char_accs = [], []
        for exp, splits in data.items():
            if exp in _CONTROLS:
                continue
            test = splits.get("test", {})
            for key in ("batch_a_seq_acc", "batch_b_seq_acc"):
                v = test.get(key)
                if v is not None:
                    seq_accs.append(v)
            for key in ("batch_a_char_acc", "batch_b_char_acc"):
                v = test.get(key)
                if v is not None:
                    char_accs.append(v)
        if not seq_accs:
            continue
        rows.append({
            "model": label,
            "seq_acc": round(sum(seq_accs) / len(seq_accs), 4),
            "char_acc": round(sum(char_accs) / len(char_accs), 4) if char_accs else "",
        })

    if not rows:
        print("  transcription_accuracy.csv skipped (no data)")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seq_acc", "char_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pgf", action="store_true",
                   help="Also save every chart as a same-named .pgf file for LaTeX inclusion.")
    args = p.parse_args()

    print("Discovering probe results...")
    models = discover()
    if not models:
        raise SystemExit("No results.json found — run the CNN or dino probes first.")

    for label, res, layers, act_dir in models:
        slug = _slug(label)
        out_dir = CHARTS_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{label} -> charts/{slug}/")

        plot_heatmap(res, out_dir / "heatmap.png", layers=layers, title=f"Linear probe accuracy — {label}", pgf=args.pgf)
        print(f"  heatmap.png")

        plot_lines(res, out_dir / "lines.png", layers=layers, title=f"Linear probe accuracy by layer — {label}", pgf=args.pgf)
        print(f"  lines.png")

        plot_full_layers(res, out_dir / "full_layers.png", layers=layers, title=f"Probe accuracy across full layer range — {label}", pgf=args.pgf)
        print(f"  full_layers.png")

        plot_categories(res, out_dir / "categories.png", layers=layers, pgf=args.pgf)
        print(f"  categories.png")

        plot_forgetting(res, out_dir / "forgetting.png", layers=layers, pgf=args.pgf)
        print(f"  forgetting.png")

        try:
            plot_arch(res, out_dir / "arch.png", layers=layers, pgf=args.pgf)
            print(f"  arch.png")
        except Exception as e:
            print(f"  arch.png skipped: {e}")

        if act_dir and act_dir.exists():
            try:
                plot_pca(act_dir, res, out_dir / "pca.png", pgf=args.pgf)
                print(f"  pca.png")
            except ImportError:
                print(f"  pca.png skipped (scikit-learn not available)")
            except FileNotFoundError as e:
                print(f"  pca.png skipped (missing activations: {e.filename})")
        else:
            print(f"  pca.png skipped (no activations dir)")

        plot_per_model_accuracy(label, res, layers, out_dir / "per_model_accuracy.png", pgf=args.pgf)

        mlp_path = MLP_RESULTS.get(label)
        if mlp_path and mlp_path.exists():
            mlp_res = load_results(mlp_path)
            plot_linear_vs_mlp(res, mlp_res, out_dir / "linear_vs_mlp.png", layers=layers, pgf=args.pgf)
            print(f"  linear_vs_mlp.png")

        sparse_path = SPARSE_RESULTS.get(label)
        if sparse_path and sparse_path.exists():
            sparse_res = load_results(sparse_path)
            plot_linear_vs_mlp(res, sparse_res, out_dir / "linear_vs_sparse.png", layers=layers, pgf=args.pgf)
            print(f"  linear_vs_sparse.png")
            plot_sparsity(sparse_res, layers, out_dir / "sparsity.png", title=f"Probe sparsity — {label}", pgf=args.pgf)
            print(f"  sparsity.png")

        acc_path = TRANSCRIPTION_ACCURACY.get(label)
        if acc_path and acc_path.exists():
            import json
            acc_data = json.loads(acc_path.read_text())
            plot_task_accuracy(acc_data, out_dir / "task_accuracy.png", label=label, pgf=args.pgf)
            print(f"  task_accuracy.png")
        else:
            print(f"  task_accuracy.png skipped (no transcription_accuracy.json)")

    print(f"\nCollated charts -> charts/collated/")
    collate_plots(models, pgf=args.pgf)
    plot_depth_curve(models, CHARTS_DIR / "collated" / "decodability_vs_depth.png", pgf=args.pgf)
    if len(models) > 1:
        plot_peak_vs_output(models, CHARTS_DIR / "collated" / "peak_vs_output.png", pgf=args.pgf)
    plot_transcription_accuracy(CHARTS_DIR / "transcription_accuracy.png", pgf=args.pgf)
    write_transcription_accuracy_csv(CHARTS_DIR / "transcription_accuracy.csv")
    print("Done.")


if __name__ == "__main__":
    main()
