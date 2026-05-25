#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from generate.dataset import generate_dataset

FONT_DIR = Path("data/fonts")
EXPERIMENTS_DIR = Path("experiments")
DATA_DIR = Path("data/experiments")


def make_qa_sheet(output_dir: Path, n_samples: int = 20) -> None:
    labels_path = output_dir / "labels.csv"
    with open(labels_path) as f:
        rows = list(csv.DictReader(f))

    sampled = random.sample(rows, min(n_samples, len(rows)))

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 0.6))
    fig.suptitle(f"QA Sheet — left: batch_a, right: batch_b", fontsize=10)

    for i, row in enumerate(sampled):
        sid = row["id"]
        text = row["text"]
        font = row["font_name"]

        img_a = np.array(Image.open(output_dir / "batch_a" / "images" / f"{int(sid):06d}.png"))
        img_b = np.array(Image.open(output_dir / "batch_b" / "images" / f"{int(sid):06d}.png"))

        axes[i, 0].imshow(img_a, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f'"{text}" ({font})', fontsize=6, pad=1)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_b, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = output_dir / "qa_sheet.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"QA sheet saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paired CAPTCHA datasets")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g. line_vs_none)")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to generate")
    args = parser.parse_args()

    cfg_path = EXPERIMENTS_DIR / f"{args.experiment}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = DATA_DIR / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment}")
    print(f"Generating {args.n} samples → {output_dir}")
    generate_dataset(cfg, args.n, output_dir, FONT_DIR)

    print("Building QA sheet...")
    make_qa_sheet(output_dir)


if __name__ == "__main__":
    main()
