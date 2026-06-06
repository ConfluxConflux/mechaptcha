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

  heatmap.png, lines.png, full_layers.png, forgetting.png, categories.png
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
from probe.plot import plot_arch, plot_categories, plot_forgetting, plot_full_layers, plot_heatmap, plot_linear_vs_mlp, plot_lines, plot_pca
from probe.results import AllResults, load_results

CHARTS_DIR = Path(__file__).resolve().parent

# Candidate runs, in plot order. Only those whose results.json exists are used.
# (label, results.json path, activations dir or None)
CANDIDATES: list[tuple[str, Path, Path | None]] = [
    ("CNN (trained from scratch)",       REPO_ROOT / "probe_results/full/results.json",              None),
    ("DINOv2-S (self-sup, LoRA)",        REPO_ROOT / "dino_results/dinov2-small/results.json",       REPO_ROOT / "dino_results/dinov2-small/activations"),
    ("DINOv2-B (self-sup, LoRA)",        REPO_ROOT / "dino_results/dinov2-base/results.json",        REPO_ROOT / "dino_results/dinov2-base/activations"),
    ("DINOv2-L (self-sup, LoRA)",        REPO_ROOT / "dino_results/dinov2-large/results.json",       REPO_ROOT / "dino_results/dinov2-large/activations"),
    ("CLIP-B (lang-sup, LoRA)",          REPO_ROOT / "dino_results/clip-vit-base/results.json",      REPO_ROOT / "dino_results/clip-vit-base/activations"),
    ("ViT-B/16 (supervised, LoRA)",      REPO_ROOT / "dino_results/vit-base-supervised/results.json",REPO_ROOT / "dino_results/vit-base-supervised/activations"),
]

# For each candidate, if a results.json exists at mlp_results_path, a linear-vs-mlp
# comparison chart is also generated. MLP probes reuse the same activations.
MLP_RESULTS: dict[str, Path] = {
    "DINOv2-S (self-sup, LoRA)":   REPO_ROOT / "dino_results/dinov2-small/mlp/results.json",
    "DINOv2-B (self-sup, LoRA)":   REPO_ROOT / "dino_results/dinov2-base/mlp/results.json",
    "DINOv2-L (self-sup, LoRA)":   REPO_ROOT / "dino_results/dinov2-large/mlp/results.json",
    "CLIP-B (lang-sup, LoRA)":     REPO_ROOT / "dino_results/clip-vit-base/mlp/results.json",
    "ViT-B/16 (supervised, LoRA)": REPO_ROOT / "dino_results/vit-base-supervised/mlp/results.json",
    "CNN (trained from scratch)":  REPO_ROOT / "probe_results/full_mlp/results.json",
}

_CONTROL = ("dumb_control", "variation_control")

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
    "Controls":          ["#bdbdbd", "#969696"],
}


def _display(name: str) -> str:
    return {"wave": "letter wave", "easy_line": "horizontal line",
            "hard_line": "angled line"}.get(name, name).replace("_", " ")


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

    legend_handles = [
        mlines.Line2D([], [], color=color, linestyle="-", marker="o",
                      markersize=4, label=_display(exp))
        for exp, color in zip(distortion_exps, flat_colors)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Linear probe test accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(f"Linear probe accuracy across layers — {label}", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, frameon=True, fontsize=8, handlelength=2.2)
    fig.tight_layout()
    _save(fig, out, pgf)
    plt.close(fig)
    print(f"  per_model_accuracy.png")


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
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (label, res, layers, _act) in zip(axes, models):
        plot_lines(res, None, layers=layers, title=_short_label(label), ax=ax)
    fig.suptitle("Linear probe accuracy by layer — all models", fontsize=13)
    _save(fig, out_dir / "lines.png", pgf)
    plt.close(fig)
    print(f"  collated/lines.png")

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

    print(f"\nCollated charts -> charts/collated/")
    collate_plots(models, pgf=args.pgf)
    plot_depth_curve(models, CHARTS_DIR / "collated" / "decodability_vs_depth.png", pgf=args.pgf)
    if len(models) > 1:
        plot_peak_vs_output(models, CHARTS_DIR / "collated" / "peak_vs_output.png", pgf=args.pgf)
    plot_full_layers_stacked(models, CHARTS_DIR / "collated" / "full_layers_stacked.png", pgf=args.pgf)
    print("Done.")


if __name__ == "__main__":
    main()
