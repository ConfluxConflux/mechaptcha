#!/usr/bin/env python3
"""Generate the labeled CAPTCHA dataset.

Usage:
    python build_dataset.py --n 150000 --output data/dataset
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from training.distortions import DISTORTIONS, ALL_DISTORTION_KEYS, sample_distortions
from training.fonts import download_fonts, load_fonts
from training.renderer import render_captcha

CHARSET = "abcdefghijklmnopqrstuvwxyz"
FONT_DIR = Path("data/fonts")


def generate(n: int, output_dir: Path) -> None:
    print("Downloading/checking fonts...")
    download_fonts(FONT_DIR)
    print("Loading fonts...")
    fonts = load_fonts(FONT_DIR)
    if not fonts:
        raise RuntimeError("No valid fonts found.")
    font_names = sorted(fonts.keys())
    print(f"  {len(font_names)} fonts loaded")

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "labels.csv"
    cols = ["id", "text", "font", "split"] + ALL_DISTORTION_KEYS

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for i in tqdm(range(n), unit="img"):
            rng = np.random.default_rng(i)

            text = "".join(rng.choice(list(CHARSET), size=5))
            font_name = font_names[int(rng.integers(0, len(font_names)))]
            font = fonts[font_name]
            distortions = sample_distortions(rng)

            split = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test")

            x_jitter = 9 if distortions["spacing_jitter"] else 0
            y_jitter = 4 if distortions["char_jitter"] else 0
            needs_jitter = x_jitter > 0 or y_jitter > 0
            img = render_captcha(
                text, font,
                rng=rng if needs_jitter else None,
                x_jitter_px=x_jitter,
                y_jitter_px=y_jitter,
            )
            arr = np.array(img)
            for key in DISTORTIONS:
                if distortions.get(key):
                    arr = DISTORTIONS[key](arr, np.random.default_rng(i))

            Image.fromarray(arr).save(img_dir / f"{i:06d}.png")

            row: dict = {"id": i, "text": text, "font": font_name, "split": split}
            row.update({k: int(v) for k, v in distortions.items()})
            writer.writerow(row)

    print(f"Done. Images: {img_dir}  Labels: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150_000)
    parser.add_argument("--output", type=Path, default=Path("data/dataset"))
    args = parser.parse_args()

    print(f"Generating {args.n:,} samples → {args.output}")
    generate(args.n, args.output)


if __name__ == "__main__":
    main()
