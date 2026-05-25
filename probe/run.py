"""Combined CLI: extract activations (if needed) then train and report probes.

Usage:
    # Full pipeline — extract then probe all experiments
    uv run python -m probe.run --checkpoint runs/captcha-cnn/best.pt

    # Re-extract even if activations already exist
    uv run python -m probe.run --checkpoint runs/captcha-cnn/best.pt --force-extract

    # Skip extraction, probe only (requires existing activations)
    uv run python -m probe.run --probe-only

    # Single experiment
    uv run python -m probe.run --checkpoint runs/captcha-cnn/best.pt --experiment blur_vs_none

    # Different classifier or regularisation
    uv run python -m probe.run --probe-only --classifier linear_svc --C 0.1

    # MLP probe (run after linear to investigate non-linear encoding)
    uv run python -m probe.run --probe-only --classifier mlp --mlp-hidden-sizes 64 32

    # Add raw-pixel baseline and logit probe alongside the standard layers
    uv run python -m probe.run --probe-only --layers input conv_block_0 conv_block_1 conv_block_2 pool embedding logits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe.config import ALL_LAYERS, CLASSIFIERS, CONV_REDUCTIONS, HOOK_LAYERS, ProbeConfig, get_model_layers
from probe.extract import extract_experiment, load_model
from probe.fit import probe_experiment
from probe.plot import plot_heatmap, plot_lines
from probe.results import format_table, save_results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract activations and/or train linear probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--probe-only", action="store_true",
                      help="Skip extraction; use existing activations")
    mode.add_argument("--extract-only", action="store_true",
                      help="Extract activations but do not train probes")

    # Paths
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Model checkpoint (.pt). Required unless --probe-only")
    p.add_argument("--experiments", type=Path, default=Path("data/experiments"),
                   help="Experiments root directory")
    p.add_argument("--activations", type=Path, default=Path("probe_results/activations"),
                   help="Directory to read/write cached activations")
    p.add_argument("--output", type=Path, default=Path("probe_results"),
                   help="Directory for results.json and heatmap.png")
    p.add_argument("--experiment", type=str, default=None,
                   help="Run a single experiment (default: all)")

    # Extraction options
    p.add_argument("--force-extract", action="store_true",
                   help="Re-extract even if activations already exist")
    p.add_argument("--conv-reduction", choices=CONV_REDUCTIONS, default="global_avg_pool",
                   dest="conv_reduction")
    p.add_argument("--image-size", type=int, nargs=2, default=[64, 160],
                   metavar=("H", "W"), dest="image_size")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")

    # Probe options
    p.add_argument("--classifier", choices=CLASSIFIERS, default="logistic_regression")
    p.add_argument("--C", type=float, default=1.0, help="Regularisation strength")
    p.add_argument("--max-iter", type=int, default=1000, dest="max_iter")
    p.add_argument("--mlp-hidden-sizes", type=int, nargs="+", default=[64, 32],
                   metavar="N", dest="mlp_hidden_sizes",
                   help="Hidden layer sizes for --classifier mlp (default: 64 32)")
    p.add_argument("--layers", nargs="+", choices=ALL_LAYERS,
                   default=list(HOOK_LAYERS),
                   help="Which layers to probe. Include 'input' for raw-pixel baseline "
                        "or 'logits' to probe the model output.")
    p.add_argument("--no-plot", action="store_true", dest="no_plot",
                   help="Skip generating the heatmap PNG")

    return p.parse_args()


def _resolve_experiments(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.experiment:
        return [root / args.experiment]
    return sorted(p for p in root.iterdir() if p.is_dir())


def _needs_extraction(activations_dir: Path, config: ProbeConfig) -> bool:
    """True if any expected activation file is missing."""
    for split in ("train", "test"):
        for batch in ("batch_a", "batch_b"):
            for layer in config.layers:
                if not (activations_dir / f"{split}_{batch}_{layer}.npy").exists():
                    return True
    return False


def main() -> None:
    args = _parse_args()

    config = ProbeConfig(
        layers=tuple(args.layers),
        classifier=args.classifier,
        C=args.C,
        max_iter=args.max_iter,
        mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
        conv_reduction=args.conv_reduction,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
    )

    # ── Extraction ───────────────────────────────────────────────────────────
    if not args.probe_only:
        if args.checkpoint is None:
            print("Error: --checkpoint is required unless --probe-only is set.", file=sys.stderr)
            sys.exit(1)
        if not args.checkpoint.exists():
            print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)

        experiments = _resolve_experiments(args, args.experiments)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, config)
        model.to(device)

        # Narrow config to layers that actually exist in this model
        config = config.for_model(model)
        print(f"Probing layers: {config.layers}")

        for exp_dir in tqdm(experiments, desc="Extracting"):
            out = args.activations / exp_dir.name
            if not args.force_extract and not _needs_extraction(out, config):
                tqdm.write(f"  {exp_dir.name}: activations already exist, skipping")
                continue
            extract_experiment(exp_dir, model, device, out, config)
            tqdm.write(f"  {exp_dir.name} -> {out}")

    if args.extract_only:
        return

    # ── Probing ──────────────────────────────────────────────────────────────
    if args.probe_only:
        experiments = _resolve_experiments(args, args.activations)
    else:
        experiments = _resolve_experiments(args, args.experiments)
        experiments = [args.activations / e.name for e in experiments]

    all_results = {}
    for exp_dir in tqdm(experiments, desc="Probing"):
        all_results[exp_dir.name] = probe_experiment(exp_dir, config)

    print("\n" + format_table(all_results, config.layers))

    results_path = args.output / "results.json"
    save_results(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    if not args.no_plot:
        plot_path = args.output / "heatmap.png"
        plot_heatmap(all_results, plot_path, layers=config.layers)
        print(f"Heatmap saved to {plot_path}")

        lines_path = args.output / "chart_lines.png"
        plot_lines(all_results, lines_path, layers=config.layers)
        print(f"Line chart saved to {lines_path}")


if __name__ == "__main__":
    main()
