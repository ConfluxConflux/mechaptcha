#!/usr/bin/env python3
"""Upload the generated dataset to HuggingFace Hub.

Usage:
    huggingface-cli login          # once, to authenticate
    python upload_hf.py --data data/dataset --repo jacobcohen/mechaptcha
"""
import argparse
import csv
from pathlib import Path

from datasets import Dataset, Features, Image, Value


def build_dataset(data_dir: Path, split: str) -> Dataset:
    csv_path = data_dir / "labels.csv"
    img_dir = data_dir / "images"

    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if r["split"] == split]
    if not rows:
        return None

    bool_cols = [c for c in rows[0] if c not in ("id", "text", "font", "split")]

    def generate():
        for row in rows:
            img_path = img_dir / f"{int(row['id']):06d}.png"
            yield {
                "image":  str(img_path),
                "id":     int(row["id"]),
                "text":   row["text"],
                "font":   row["font"],
                **{k: bool(int(row[k])) for k in bool_cols},
            }

    features = Features({
        "image":  Image(),
        "id":     Value("int32"),
        "text":   Value("string"),
        "font":   Value("string"),
        **{k: Value("bool") for k in bool_cols},
    })

    return Dataset.from_generator(generate, features=features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--repo", type=str, default="jacobcohen/mechaptcha")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    splits = {}
    for split in ("train", "val", "test"):
        ds = build_dataset(args.data, split)
        if ds is not None:
            splits[split] = ds
            print(f"  {split}: {len(ds):,} rows")

    print(f"Pushing to {args.repo} ...")
    for split_name, ds in splits.items():
        ds.push_to_hub(args.repo, split=split_name)
        print(f"  pushed {split_name}")

    print("Done.")


if __name__ == "__main__":
    main()
