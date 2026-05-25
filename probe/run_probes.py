"""Train linear probes across all experiments and print a results table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe.probes import LAYER_NAMES, probe_experiment

_COL = 12


def _fmt(val: float) -> str:
    delta = val - 0.5
    sign = "+" if delta >= 0 else ""
    return f"{val:.1%}({sign}{delta:.1%})"


def format_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    header = f"{'experiment':<32}" + "".join(f"{l:>{_COL}}" for l in LAYER_NAMES)
    sep = "-" * len(header)
    rows = [header, sep]

    for exp_name, layer_results in sorted(results.items()):
        row = f"{exp_name:<32}"
        for layer in LAYER_NAMES:
            if layer in layer_results:
                row += f"{_fmt(layer_results[layer]['test_acc']):>{_COL}}"
            else:
                row += f"{'N/A':>{_COL}}"
        rows.append(row)

    rows.append(sep)
    rows.append("(test accuracy; +/- offset from 50% chance)")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear probes on saved activations and report accuracy per layer."
    )
    parser.add_argument("--activations", type=Path, default=Path("probe_results/activations"))
    parser.add_argument("--output", type=Path, default=Path("probe_results/results.json"))
    parser.add_argument("--experiment", type=str, default=None, help="Single experiment to probe")
    args = parser.parse_args()

    if not args.activations.exists():
        print(f"No activations found at {args.activations}. Run probe/extract.py first.")
        sys.exit(1)

    if args.experiment:
        experiment_dirs = [args.activations / args.experiment]
    else:
        experiment_dirs = sorted(p for p in args.activations.iterdir() if p.is_dir())

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for exp_dir in tqdm(experiment_dirs, desc="Probing"):
        all_results[exp_dir.name] = probe_experiment(exp_dir)

    print("\n" + format_table(all_results))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
