#!/usr/bin/env python3
"""Generate all paired-batch experiments from the fixed pool of seeds.

Usage:
    python build_experiments.py [--n 10000] [--output data/experiments]
    python build_experiments.py --distortions easy_line blur  # subset only

Output layout:
    data/experiments/
        same_data_control/          # A == B exactly; probe should be ~50%
        same_distribution_control/     # A and B from same distribution, different seeds
        easy_line/             # A has easy_line, B doesn't; everything else identical
        hard_line/
        ...one directory per distortion...
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiment_hf import (
    DEFAULT_MODEL_REPO,
    ExperimentHubConfig,
    default_repo_name,
    preflight_experiment_hf_upload,
    push_experiments_to_hub,
)
from training.distortions import DISTORTIONS
from training.experiment import (
    generate_controlled,
    generate_same_data_control,
    generate_same_distribution_control,
)
from training.fonts import download_fonts, load_fonts

FONT_DIR = Path("data/fonts")


def _generate_experiment(
    kind: str,
    key: str | None,
    n: int,
    font_dir: Path,
    output_dir: Path,
) -> str:
    fonts = load_fonts(font_dir)
    if not fonts:
        raise RuntimeError("No valid fonts found.")

    seeds = list(range(n))
    if kind == "same_data_control":
        generate_same_data_control(seeds, fonts, output_dir / "same_data_control")
        return "same_data_control"
    if kind == "same_distribution_control":
        generate_same_distribution_control(seeds, fonts, output_dir / "same_distribution_control")
        return "same_distribution_control"
    if kind == "controlled" and key is not None:
        generate_controlled(key, seeds, fonts, output_dir / key)
        return key
    raise ValueError(f"Unknown experiment job: kind={kind!r}, key={key!r}")


def _generate_experiments(
    n: int,
    fonts: dict,
    font_dir: Path,
    output_dir: Path,
    distortions_to_run: list[str],
    workers: int,
) -> None:
    seeds = list(range(n))
    jobs = (
        [("same_data_control", None), ("same_distribution_control", None)]
        + [("controlled", key) for key in distortions_to_run]
    )

    if workers <= 1:
        print("same_data_control")
        generate_same_data_control(seeds, fonts, output_dir / "same_data_control")

        print("same_distribution_control")
        generate_same_distribution_control(seeds, fonts, output_dir / "same_distribution_control")

        for key in distortions_to_run:
            print(key)
            generate_controlled(key, seeds, fonts, output_dir / key)
        return

    print(f"Running {len(jobs)} experiments with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_generate_experiment, kind, key, n, font_dir, output_dir): key or kind
            for kind, key in jobs
        }
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            print(f"  finished {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000,
                        help="Number of seed pairs in the fixed pool")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory. Defaults to data/experiments/<standard repo name>, "
                             "or data/experiments/<hf repo id> when --hf-repo-id is set.")
    parser.add_argument("--distortions", nargs="*", default=None,
                        help="Distortions to run (default: all). Controls always run.")
    parser.add_argument("--push-to-hf", action="store_true", default=False,
                        help="Push generated experiments to a Hugging Face dataset repo")
    parser.add_argument("--hf-repo-id", type=str, default=None,
                        help="HF dataset repo id, e.g. siddharthmb/2026.mechaptcha.linear-probe-experiments-20260525. "
                             "Defaults to the logged-in HF user and today's standard repo name.")
    parser.add_argument("--hf-private", action="store_true", default=False,
                        help="Create or update the HF dataset repo as private")
    parser.add_argument("--hf-model-repo", type=str, default=DEFAULT_MODEL_REPO,
                        help="Model repo to reference in the dataset card and probe command")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of experiments to generate in parallel")
    args = parser.parse_args()

    output_name = args.hf_repo_id or default_repo_name()
    output_dir = args.output or (Path("data/experiments") / output_name)
    hf_repo_id = args.hf_repo_id

    if args.push_to_hf:
        try:
            preflight = preflight_experiment_hf_upload(args.hf_repo_id, args.hf_private)
            hf_repo_id = preflight.repo_id
            print(f"Verified Hugging Face dataset repo access: {preflight.repo_url}")
        except Exception as exc:
            raise RuntimeError(f"Hugging Face upload preflight failed: {exc}") from exc

    print("Downloading/checking fonts...")
    download_fonts(FONT_DIR)
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)
    if not fonts:
        raise RuntimeError("No valid fonts found.")
    print(f"  {len(fonts)} fonts loaded")

    distortions_to_run = args.distortions if args.distortions else list(DISTORTIONS.keys())
    invalid = set(distortions_to_run) - set(DISTORTIONS.keys())
    if invalid:
        raise ValueError(f"Unknown distortion keys: {invalid}. Valid: {sorted(DISTORTIONS.keys())}")

    print(f"\nGenerating {args.n:,} pairs each for:")
    print(f"  Controls: same_data_control, same_distribution_control")
    print(f"  Distortions: {', '.join(distortions_to_run)}")
    print(f"  Output: {output_dir}\n")

    _generate_experiments(
        n=args.n,
        fonts=fonts,
        font_dir=FONT_DIR,
        output_dir=output_dir,
        distortions_to_run=distortions_to_run,
        workers=max(args.workers, 1),
    )

    print(f"\nDone. {len(distortions_to_run) + 2} experiments written to {output_dir}")

    if args.push_to_hf:
        print("\nPushing experiments to Hugging Face Hub...")
        upload = push_experiments_to_hub(
            ExperimentHubConfig(
                repo_id=hf_repo_id,
                private=args.hf_private,
                source_dir=output_dir,
                n=args.n,
                distortions=tuple(distortions_to_run),
                model_repo=args.hf_model_repo,
            )
        )
        print(f"  repo: {upload.repo_id}")
        print(f"  url:  {upload.repo_url}")
        for split, count in upload.split_counts.items():
            print(f"  {split}: {count:,} rows")


if __name__ == "__main__":
    main()
