"""Generate comparison charts across the CNN and pretrained-ViT probe runs.

Auto-discovers whichever `results.json` files exist (so it can be re-run as new
backbone runs finish) and writes figures into this `charts/` directory:

  heatmap_<model>.png                    per-model experiment x layer heatmap
  decodability_vs_depth.png              mean probe accuracy vs normalised depth, one line/model
  peak_vs_output.png                     peak vs final-logit decodability per model
  per-model-accuracy/<slug>.png          per-model line chart: one line per experiment,
                                         all layers including embedding and logits

Usage:  uv run python charts/make_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from probe.config import ALL_LAYERS
from probe.plot import plot_heatmap
from probe.results import AllResults, load_results

CHARTS_DIR = Path(__file__).resolve().parent

# Candidate runs, in plot order. Only those whose results.json exists are used.
CANDIDATES: list[tuple[str, Path]] = [
    ("CNN (trained from scratch)", REPO_ROOT / "probe_results/full/results.json"),
    ("DINOv2-S (self-sup, LoRA)", REPO_ROOT / "dino_results/dinov2-small/results.json"),
    ("DINOv2-B (self-sup, LoRA)", REPO_ROOT / "dino_results/dinov2-base/results.json"),
    ("CLIP-B (lang-sup, LoRA)", REPO_ROOT / "dino_results/clip-vit-base/results.json"),
]

PER_MODEL_DIR = CHARTS_DIR / "per-model-accuracy"

_CONTROL = ("dumb_control", "variation_control")

# Distortion categories and their colours — shared across all per-model plots so
# the same experiment always appears in the same colour regardless of backbone.
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
    """Short display label for a layer name, consistent across CNN and ViT."""
    if name.startswith("conv_block_"):
        return "cb " + name.split("_")[-1]
    if name.startswith("block_"):
        return "blk " + name.split("_")[-1]
    return name  # input, pool, embedding, logits — keep as-is


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


def discover() -> list[tuple[str, AllResults, tuple[str, ...]]]:
    found = []
    for label, path in CANDIDATES:
        if path.exists():
            res = load_results(path)
            found.append((label, res, order_layers(res)))
            print(f"  found: {label:30} ({path.relative_to(REPO_ROOT)})")
        else:
            print(f"  missing (skipped): {label:30} ({path.relative_to(REPO_ROOT)})")
    return found


def plot_per_model_accuracy(
    label: str,
    results: AllResults,
    layers: tuple[str, ...],
    out: Path,
    pgf: bool = False,
) -> None:
    """One line per experiment, all layers left-to-right including embedding and logits.

    Consistent style across every backbone: same colours per experiment, same y-axis
    range, same legend layout. Controls are dashed grey so distortions stay readable.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    x = list(range(len(layers)))
    xlabels = [_layer_label(l) for l in layers]

    # Mark the logits column with a subtle right-edge shade so it's clear this is
    # the task output (the behaviorally-invariant prediction).
    logits_idx = next((i for i, l in enumerate(layers) if l == "logits"), None)

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(layers) + 3), 5.5))

    if logits_idx is not None:
        ax.axvspan(logits_idx - 0.45, logits_idx + 0.45, color="#d4edda", alpha=0.45, zorder=0)

    ax.axhline(0.5, color="#aaa", linewidth=0.9, linestyle=":", zorder=1)

    # One line per distortion experiment (controls excluded).
    # Colour cycles through categories so related distortions cluster visually.
    distortion_exps = [exp for cat, exps in _CATEGORIES.items()
                       if cat != "Controls" for exp in exps if exp in results]
    # Build a flat colour list that respects category grouping.
    flat_colors: list[str] = []
    for cat, exps in _CATEGORIES.items():
        if cat == "Controls":
            continue
        palette = _PALETTES[cat]
        flat_colors.extend(palette[i % len(palette)] for i, exp in enumerate(exps) if exp in results)

    for exp, color in zip(distortion_exps, flat_colors):
        vals = [results[exp].get(l) for l in layers]
        ys = [v.test_acc if v is not None else float("nan") for v in vals]
        ax.plot(x, ys, marker="o", markersize=4, linewidth=1.6,
                linestyle="-", color=color, alpha=0.85, zorder=2)

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

    if logits_idx is not None:
        ax.text(logits_idx, 1.015, "task output\n(logits)", ha="center",
                va="bottom", fontsize=7, color="#2d6a4f")

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              frameon=True, fontsize=8, handlelength=2.2)

    fig.tight_layout()
    _save(fig, out, pgf)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_depth_curve(models, out: Path, pgf: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(models)))
    for (label, res, layers), color in zip(models, colors):
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
    for _label, res, layers in models:
        curve = mean_curve(res, layers)
        peaks.append(curve.max())
        finals.append(curve[-1])  # logits layer
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


def _save(fig, path: Path, pgf: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if pgf:
        fig.savefig(path.with_suffix(".pgf"), bbox_inches="tight")


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

    for label, res, layers in models:
        slug = label.split(" ")[0].replace("(", "").replace(")", "").lower().replace("/", "-")
        plot_heatmap(res, CHARTS_DIR / f"heatmap_{slug}.png", layers=layers,
                     title=f"Linear probe accuracy — {label}", pgf=args.pgf)
        print(f"  wrote heatmap_{slug}.png")
        plot_per_model_accuracy(label, res, layers, PER_MODEL_DIR / f"{slug}.png",
                                pgf=args.pgf)

    plot_depth_curve(models, CHARTS_DIR / "decodability_vs_depth.png", pgf=args.pgf)
    if len(models) > 1:
        plot_peak_vs_output(models, CHARTS_DIR / "peak_vs_output.png", pgf=args.pgf)
    print("Done.")


if __name__ == "__main__":
    main()
