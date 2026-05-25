#!/usr/bin/env python3
"""Generate all paired-batch experiments from the fixed pool of seeds.

Usage:
    python build_experiments.py [--n 10000] [--output data/experiments]
    python build_experiments.py --distortions easy_line blur  # subset only

Output layout:
    data/experiments/
        dumb_control/          # A == B exactly; probe should be ~50%
        variation_control/     # A and B from same distribution, different seeds
        easy_line/             # A has easy_line, B doesn't; everything else identical
        hard_line/
        ...one directory per distortion...
"""
import argparse
from pathlib import Path

from generate.distortions import ALL_DISTORTION_KEYS, DISTORTIONS
from generate.experiment import (
    generate_controlled,
    generate_dumb_control,
    generate_variation_control,
)
from generate.fonts import download_fonts, load_fonts

FONT_DIR = Path("data/fonts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000,
                        help="Number of seed pairs in the fixed pool")
    parser.add_argument("--output", type=Path, default=Path("data/experiments"))
    parser.add_argument("--distortions", nargs="*", default=None,
                        help="Distortions to run (default: all). Controls always run.")
    args = parser.parse_args()

    print("Downloading/checking fonts...")
    download_fonts(FONT_DIR)
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)
    if not fonts:
        raise RuntimeError("No valid fonts found.")
    print(f"  {len(fonts)} fonts loaded")

    seeds = list(range(args.n))

    distortions_to_run = args.distortions if args.distortions else list(DISTORTIONS.keys())
    invalid = set(distortions_to_run) - set(DISTORTIONS.keys())
    if invalid:
        raise ValueError(f"Unknown distortion keys: {invalid}. Valid: {sorted(DISTORTIONS.keys())}")

    print(f"\nGenerating {args.n:,} pairs each for:")
    print(f"  Controls: dumb_control, variation_control")
    print(f"  Distortions: {', '.join(distortions_to_run)}")
    print(f"  Output: {args.output}\n")

    print("dumb_control")
    generate_dumb_control(seeds, fonts, args.output / "dumb_control")

    print("variation_control")
    generate_variation_control(seeds, fonts, args.output / "variation_control")

    for key in distortions_to_run:
        print(key)
        generate_controlled(key, seeds, fonts, args.output / key)

    print(f"\nDone. {len(distortions_to_run) + 2} experiments written to {args.output}")


if __name__ == "__main__":
    main()
