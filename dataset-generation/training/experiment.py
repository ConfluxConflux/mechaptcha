"""Core logic for paired-batch experiment generation."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .distortions import (
    ALL_DISTORTION_KEYS, DISTORTIONS, LINE_DISTORTIONS, sample_distortions,
)
from .fonts import load_fonts
from .renderer import render_captcha

CHARSET = "abcdefghijklmnopqrstuvwxyz"
VARIATION_SEED_OFFSET = 1_000_000

CSV_COLS = (
    ["id", "text", "font", "split"]
    + [f"a_{k}" for k in ALL_DISTORTION_KEYS]
    + [f"b_{k}" for k in ALL_DISTORTION_KEYS]
)


def _spawn_rngs(seed: int):
    """Return two independent RNGs from one seed: (sampling_rng, jitter_rng)."""
    children = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(children[0]), np.random.default_rng(children[1])


def _sample_base(seed: int, font_names: list[str]) -> tuple:
    """Deterministically sample text, font, distortions, and split from a seed."""
    sampling_rng, _ = _spawn_rngs(seed)
    text = "".join(sampling_rng.choice(list(CHARSET), size=5))
    font_name = font_names[int(sampling_rng.integers(0, len(font_names)))]
    distortions = sample_distortions(sampling_rng)
    split = "train" if seed % 10 < 8 else ("val" if seed % 10 == 8 else "test")
    return text, font_name, distortions, split


def _render(seed: int, text: str, font, distortions: dict[str, bool]) -> np.ndarray:
    """Render one CAPTCHA image given pre-determined text, font, and distortions."""
    _, jitter_rng = _spawn_rngs(seed)
    x_jitter = 9 if distortions["spacing_jitter"] else 0
    y_jitter = 4 if distortions["char_jitter"] else 0
    img = render_captcha(
        text, font,
        rng=jitter_rng if (x_jitter or y_jitter) else None,
        x_jitter_px=x_jitter,
        y_jitter_px=y_jitter,
    )
    arr = np.array(img)
    for key in DISTORTIONS:
        if distortions.get(key):
            arr = DISTORTIONS[key](arr, np.random.default_rng(seed))
    return arr


def _write_row(writer, seed, text, font_name, split, dist_a, dist_b):
    row = {"id": seed, "text": text, "font": font_name, "split": split}
    row.update({f"a_{k}": int(dist_a.get(k, False)) for k in ALL_DISTORTION_KEYS})
    row.update({f"b_{k}": int(dist_b.get(k, False)) for k in ALL_DISTORTION_KEYS})
    writer.writerow(row)


def _make_dirs(output_dir: Path):
    for batch in ("batch_a", "batch_b"):
        (output_dir / batch / "images").mkdir(parents=True, exist_ok=True)


def generate_controlled(
    target_key: str,
    seeds: list[int],
    fonts: dict,
    output_dir: Path,
) -> None:
    """Batch A has target forced ON; Batch B has target forced OFF.
    All other distortions are identical between the two batches.
    For line targets, all other line types are cleared so mutual exclusivity
    is not the source of any difference.
    """
    _make_dirs(output_dir)
    font_names = sorted(fonts.keys())

    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()

        for seed in tqdm(seeds, desc=f"  {target_key}", leave=False):
            text, font_name, dist, split = _sample_base(seed, font_names)
            font = fonts[font_name]

            # For line targets, clear all line types from the base so the only
            # difference between batches is the presence of the target line.
            if target_key in LINE_DISTORTIONS:
                for k in LINE_DISTORTIONS:
                    dist[k] = False

            dist_a = {**dist, target_key: True}
            dist_b = {**dist, target_key: False}

            Image.fromarray(_render(seed, text, font, dist_a)).save(
                output_dir / "batch_a" / "images" / f"{seed:06d}.png"
            )
            Image.fromarray(_render(seed, text, font, dist_b)).save(
                output_dir / "batch_b" / "images" / f"{seed:06d}.png"
            )
            _write_row(writer, seed, text, font_name, split, dist_a, dist_b)


def generate_same_data_control(
    seeds: list[int],
    fonts: dict,
    output_dir: Path,
) -> None:
    """Batch A = Batch B exactly (identical images, normal distortion sampling).
    A probe trained here should perform at chance — anything above 50% is a bug.
    """
    _make_dirs(output_dir)
    font_names = sorted(fonts.keys())

    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()

        for seed in tqdm(seeds, desc="  same_data_control", leave=False):
            text, font_name, dist, split = _sample_base(seed, font_names)
            font = fonts[font_name]
            arr = _render(seed, text, font, dist)
            img = Image.fromarray(arr)
            img.save(output_dir / "batch_a" / "images" / f"{seed:06d}.png")
            img.save(output_dir / "batch_b" / "images" / f"{seed:06d}.png")
            _write_row(writer, seed, text, font_name, split, dist, dist)


def generate_same_distribution_control(
    seeds: list[int],
    fonts: dict,
    output_dir: Path,
) -> None:
    """Batch A and Batch B are drawn from the same distribution but different seeds.
    Batch B uses seed + VARIATION_SEED_OFFSET. No distortion is forced on/off.
    A probe trained here should also perform at chance.
    """
    _make_dirs(output_dir)
    font_names = sorted(fonts.keys())

    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()

        for seed in tqdm(seeds, desc="  same_distribution_control", leave=False):
            b_seed = seed + VARIATION_SEED_OFFSET

            text_a, font_a, dist_a, split = _sample_base(seed, font_names)
            text_b, font_b, dist_b, _     = _sample_base(b_seed, font_names)

            Image.fromarray(_render(seed,   text_a, fonts[font_a], dist_a)).save(
                output_dir / "batch_a" / "images" / f"{seed:06d}.png"
            )
            Image.fromarray(_render(b_seed, text_b, fonts[font_b], dist_b)).save(
                output_dir / "batch_b" / "images" / f"{seed:06d}.png"
            )
            # text/font columns reflect batch_a; batch_b is derived from id + VARIATION_SEED_OFFSET
            _write_row(writer, seed, text_a, font_a, split, dist_a, dist_b)
