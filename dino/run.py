"""Extract DINOv2 activations then train + report linear probes across depth.

Reuses the CNN probe machinery (probe.fit / probe.results / probe.plot) verbatim —
only the activation source differs. The output is directly comparable to the CNN's:
test-accuracy of a batch_a-vs-batch_b probe at every layer, per experiment.

Usage:
    # Full pipeline: extract activations from a trained LoRA checkpoint, then probe
    uv run python -m dino.run --checkpoint dino_runs/dinov2-small-lora/best.pt \
        --experiments data/experiments/siddharthmb/2026.mechaptcha.linear-probe-experiments-giant-20260525 \
        --output dino_results/dinov2-small-lora

    # Probe only (activations already extracted)
    uv run python -m dino.run --probe-only --activations dino_results/dinov2-small-lora/activations \
        --output dino_results/dinov2-small-lora

    # Single experiment, MLP probe
    uv run python -m dino.run --probe-only --experiment hard_line --classifier mlp \
        --activations dino_results/dinov2-small-lora/activations --output dino_results/dinov2-small-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dino.config import block_layer_names
from dino.extract import extract_experiment
from dino.model import load_checkpoint, load_pretrained_only
from probe.config import CLASSIFIERS, ProbeConfig
from probe.fit import probe_experiment
from probe.plot import plot_heatmap, plot_lines, plot_task_accuracy
from probe.results import format_table, save_results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract DINOv2 activations and/or train linear probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--probe-only", action="store_true", help="Skip extraction; use existing activations")
    mode.add_argument("--extract-only", action="store_true", help="Extract activations but do not probe")

    p.add_argument("--checkpoint", type=str, default=None,
                   help="Trained LoRA checkpoint (.pt from dino.train). Required unless --probe-only or --pretrained.")
    p.add_argument("--pretrained", type=str, default=None, metavar="BACKBONE:MODEL_NAME",
                   help="Use a pretrained-only backbone with no LoRA fine-tuning. "
                        "Format: backbone:model_name, e.g. dinov2:facebook/dinov2-small or "
                        "clip:openai/clip-vit-base-patch32. Mutually exclusive with --checkpoint. "
                        "Character heads are randomly initialised so the logits layer is meaningless "
                        "but all intermediate block features are the unmodified pretrained representations.")
    p.add_argument("--experiments", type=str, default=None,
                   help="Local paired-experiment root directory (contains <exp>/labels.csv).")
    p.add_argument("--experiment", type=str, default=None, help="Run a single experiment (default: all)")
    p.add_argument("--activations", type=Path, default=None,
                   help="Dir to read/write cached activations (default: <output>/activations)")
    p.add_argument("--output", type=Path, default=Path("dino_results"),
                   help="Dir for results.json, accuracy.json, and charts")

    p.add_argument("--force-extract", action="store_true", help="Re-extract even if activations exist")
    p.add_argument("--max-train-ids", type=int, default=8000, dest="max_train_ids",
                   help="Cap on train ids per experiment during extraction (probe train set size)")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")

    p.add_argument("--layers", nargs="+", default=None,
                   help="Layers to probe (default: input, block_0..block_N, embedding, logits)")
    p.add_argument("--classifier", choices=CLASSIFIERS, default="logistic_regression")
    p.add_argument("--C", type=float, default=1.0, help="Probe regularisation strength")
    p.add_argument("--max-iter", type=int, default=1000, dest="max_iter")
    p.add_argument("--mlp-hidden-sizes", type=int, nargs="+", default=[64, 32], dest="mlp_hidden_sizes")
    p.add_argument("--no-plot", action="store_true", dest="no_plot")
    p.add_argument("--pgf", action="store_true",
                   help="Also save every chart as a same-named .pgf file for LaTeX inclusion.")
    return p.parse_args()


_EXP_NAME_ALIASES = {
    "dumb_control":      "same_data_control",
    "variation_control": "same_distribution_control",
}


def _exp_name(dir_name: str) -> str:
    """Translate legacy on-disk experiment names to canonical names."""
    return _EXP_NAME_ALIASES.get(dir_name, dir_name)


def _resolve_experiments(root: Path, single: str | None) -> list[Path]:
    if single:
        return [root / single]
    return sorted(p for p in root.iterdir() if p.is_dir())


def _infer_layers_from_activations(activations_root: Path) -> tuple[str, ...]:
    """Recover the layer ordering from cached .npy filenames (for --probe-only)."""
    exp_dirs = [d for d in activations_root.iterdir() if d.is_dir()]
    if not exp_dirs:
        raise SystemExit(f"No activation experiments found under {activations_root}")
    names = {p.name[len("train_batch_a_"):-len(".npy")]
             for p in exp_dirs[0].glob("train_batch_a_*.npy")}
    blocks = sorted((n for n in names if n.startswith("block_")), key=lambda s: int(s.split("_")[1]))
    ordered = [n for n in ("input",) if n in names] + blocks + [n for n in ("embedding", "logits") if n in names]
    return tuple(ordered)


def main() -> None:
    args = _parse_args()
    activations_root = args.activations or (args.output / "activations")

    # ── Extraction ───────────────────────────────────────────────────────────
    if not args.probe_only:
        if args.checkpoint is None and args.pretrained is None:
            raise SystemExit("Error: --checkpoint or --pretrained is required unless --probe-only is set.")
        if args.checkpoint is not None and args.pretrained is not None:
            raise SystemExit("Error: --checkpoint and --pretrained are mutually exclusive.")
        if args.experiments is None:
            raise SystemExit("Error: --experiments is required for extraction.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.pretrained:
            parts = args.pretrained.split(":", 1)
            if len(parts) != 2:
                raise SystemExit("Error: --pretrained must be in the form backbone:model_name")
            backbone, model_name = parts
            print(f"Device: {device}\nLoading pretrained backbone {backbone}:{model_name}")
            model = load_pretrained_only(backbone, model_name).to(device)
        else:
            print(f"Device: {device}\nLoading model from {args.checkpoint}")
            model = load_checkpoint(args.checkpoint, map_location=device).to(device)
        layers = tuple(args.layers) if args.layers else block_layer_names(model.num_blocks)
        print(f"Probing layers: {layers}")

        experiments = _resolve_experiments(Path(args.experiments), args.experiment)
        accuracy: dict[str, dict] = {}
        for exp_dir in tqdm(experiments, desc="Extracting"):
            name = _exp_name(exp_dir.name)
            out = activations_root / name
            expected = out / f"test_batch_b_{layers[-1]}.npy"
            if not args.force_extract and expected.exists():
                tqdm.write(f"  {name}: activations exist, skipping")
                continue
            acc = extract_experiment(exp_dir, model, device, out, layers,
                                     max_train_ids=args.max_train_ids, batch_size=args.batch_size)
            accuracy[name] = acc
            tqdm.write(f"  {name} -> {out}")

        if accuracy:
            args.output.mkdir(parents=True, exist_ok=True)
            acc_path = args.output / "transcription_accuracy.json"
            acc_path.write_text(json.dumps(accuracy, indent=2))
            print(f"\nTranscription accuracy (behavioral-invariance check) -> {acc_path}")
            _print_invariance_summary(accuracy)
            if not args.no_plot:
                plot_task_accuracy(accuracy, args.output / "task_accuracy.png")
                print(f"Task accuracy chart saved to {args.output / 'task_accuracy.png'}")
    else:
        layers = tuple(args.layers) if args.layers else _infer_layers_from_activations(activations_root)

    if args.extract_only:
        return

    # ── Probing (reuses the CNN probe pipeline unchanged) ──────────────────────
    config = ProbeConfig(
        layers=layers, classifier=args.classifier, C=args.C,
        max_iter=args.max_iter, mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
    )
    experiments = _resolve_experiments(activations_root, args.experiment)
    all_results = {e.name: probe_experiment(e, config) for e in tqdm(experiments, desc="Probing")}

    print("\n" + format_table(all_results, config.layers))
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    save_results(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    if not args.no_plot:
        plot_heatmap(all_results, args.output / "heatmap.png", layers=config.layers, pgf=args.pgf)
        plot_lines(all_results, args.output / "chart_lines.png", layers=config.layers, pgf=args.pgf)
        print(f"Charts saved to {args.output}")


def _print_invariance_summary(accuracy: dict[str, dict]) -> None:
    print(f"{'experiment':<24}{'test a seq':>12}{'test b seq':>12}{'a-b gap':>10}")
    for exp, splits in sorted(accuracy.items()):
        test = splits.get("test", {})
        a = test.get("batch_a_seq_acc")
        b = test.get("batch_b_seq_acc")
        if a is None or b is None:
            continue
        print(f"{exp:<24}{a:>12.3f}{b:>12.3f}{a - b:>+10.3f}")


if __name__ == "__main__":
    main()
