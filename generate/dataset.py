import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .distortions import DISTORTIONS
from .fonts import download_fonts, load_fonts
from .renderer import render_captcha


def _apply_transform(arr: np.ndarray, transform: dict, rng: np.random.Generator) -> np.ndarray:
    kind = transform.get("transform", "none")
    if kind == "none":
        return arr
    if kind == "distortion":
        name = transform["distortion"]
        return DISTORTIONS[name](arr.copy(), rng)
    raise ValueError(f"Unknown transform type: {kind!r}")


def generate_dataset(
    experiment_cfg: dict,
    n: int,
    output_dir: Path,
    font_dir: Path,
) -> None:
    charset = experiment_cfg.get("charset", "abcdefghijklmnopqrstuvwxyz")
    n_chars = experiment_cfg.get("n_chars", 5)
    batch_a_cfg = experiment_cfg["batch_a"]
    batch_b_cfg = experiment_cfg["batch_b"]

    print("Downloading/checking fonts...")
    download_fonts(font_dir)
    print("Loading validated fonts...")
    fonts = load_fonts(font_dir)
    if not fonts:
        raise RuntimeError("No valid fonts found. Check font downloads.")
    font_names = sorted(fonts.keys())
    print(f"  {len(font_names)} fonts loaded: {', '.join(font_names)}")

    a_dir = output_dir / "batch_a" / "images"
    b_dir = output_dir / "batch_b" / "images"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        Path(__file__).parent.parent / "experiments" / f"{experiment_cfg['name']}.yaml",
        output_dir / "config.yaml",
    )

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "font_name", "seed"])

        for sample_id in range(n):
            rng = np.random.default_rng(sample_id)

            text = "".join(rng.choice(list(charset), size=n_chars))
            font_name = font_names[rng.integers(0, len(font_names))]
            font = fonts[font_name]

            base_img = render_captcha(text, font)
            arr = np.array(base_img)

            arr_a = _apply_transform(arr, batch_a_cfg, np.random.default_rng(sample_id))
            arr_b = _apply_transform(arr, batch_b_cfg, np.random.default_rng(sample_id))

            Image.fromarray(arr_a).save(a_dir / f"{sample_id:06d}.png")
            Image.fromarray(arr_b).save(b_dir / f"{sample_id:06d}.png")

            writer.writerow([sample_id, text, font_name, sample_id])

            if (sample_id + 1) % max(1, n // 10) == 0:
                print(f"  {sample_id + 1}/{n} generated")

    print(f"Done. Labels: {labels_path}")
