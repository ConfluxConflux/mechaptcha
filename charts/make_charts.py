"""Generate comparison charts across the CNN and pretrained-ViT probe runs.

Auto-discovers whichever `results.json` files exist (so it can be re-run as new
backbone runs finish) and writes figures into this `charts/` directory:

  heatmap_<model>.png         per-model experiment x layer probe-accuracy heatmap
  decodability_vs_depth.png   mean probe accuracy vs normalised depth, one line/model
  peak_vs_output.png          peak vs final-logit decodability per model (retention story)

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

_CONTROL = ("dumb_control", "variation_control")


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


def plot_depth_curve(models, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(models)))
    for (label, res, layers), color in zip(models, colors):
        curve = mean_curve(res, layers)
        x = np.linspace(0, 1, len(layers))  # normalised depth: 0=input, 1=output
        ax.plot(x, curve, "-o", color=color, label=label, markersize=4, linewidth=2)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_xlabel("normalised depth  (0 = input  →  1 = logits)")
    ax.set_ylabel("mean linear-probe accuracy\n(distortion A vs B, non-control experiments)")
    ax.set_title("Distortion stays linearly decodable across depth — in every model\n"
                 "behavioral invariance ≠ representational erasure", fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_peak_vs_output(models, out: Path) -> None:
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
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    print("Discovering probe results...")
    models = discover()
    if not models:
        raise SystemExit("No results.json found — run the CNN or dino probes first.")

    for label, res, layers in models:
        slug = label.split(" ")[0].replace("(", "").replace(")", "").lower().replace("/", "-")
        plot_heatmap(res, CHARTS_DIR / f"heatmap_{slug}.png", layers=layers,
                     title=f"Linear probe accuracy — {label}")
        print(f"  wrote heatmap_{slug}.png")

    plot_depth_curve(models, CHARTS_DIR / "decodability_vs_depth.png")
    if len(models) > 1:
        plot_peak_vs_output(models, CHARTS_DIR / "peak_vs_output.png")
    print("Done.")


if __name__ == "__main__":
    main()
