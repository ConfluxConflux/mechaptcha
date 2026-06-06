"""Extract and cache DINOv2 activations from paired experiment images.

Writes the same ``{split}_{batch}_{layer}.npy`` layout that probe/extract.py
produces for the CNN, so probe.fit / probe.results / probe.plot work unchanged.
Also measures per-batch transcription accuracy — the behavioral-invariance check:
if the LoRA model reads batch_a (distorted) as accurately as batch_b (clean), it is
behaviorally invariant, making any residual distortion signal in the probes the
interesting result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dino.config import DinoConfig
from dino.model import DinoCaptchaModel, build_transform
from train.model.charset import decode_indices


def _raw_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Grayscale baseline transform — identical to the CNN's raw-pixel 'input' layer."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


def extract_activations_from_images(
    model: DinoCaptchaModel,
    images: list[Image.Image],
    device: torch.device,
    layers: tuple[str, ...],
    *,
    batch_size: int = 128,
    raw_image_size: tuple[int, int] = (64, 160),
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ({layer: [N, features]}, predicted_indices [N, num_chars]).

    Predicted indices are the argmax of the transcription heads, used for the
    per-batch accuracy / behavioral-invariance report.
    """
    dino_transform = build_transform(model.config)
    raw_transform = _raw_transform(raw_image_size)
    want_input = "input" in layers
    feature_layers = [l for l in layers if l != "input"]

    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in layers}
    pred_chunks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = [img.convert("L") for img in images[i:i + batch_size]]
            batch = torch.stack([dino_transform(img) for img in chunk]).to(device)

            feats = model.block_features(batch)
            for name in feature_layers:
                accumulated[name].append(feats[name].cpu().numpy())

            if want_input:
                raw = torch.stack([raw_transform(img) for img in chunk])
                accumulated["input"].append(raw.flatten(start_dim=1).numpy())

            logits = feats["logits"].reshape(len(chunk), model.config.num_chars, model.config.num_classes)
            pred_chunks.append(logits.argmax(dim=-1).cpu().numpy())

    # float16 halves storage vs float32 with negligible effect on linear probe accuracy.
    activations = {name: np.concatenate(chunks).astype(np.float16)
                   for name, chunks in accumulated.items()}
    predictions = np.concatenate(pred_chunks)
    return activations, predictions


def extract_experiment(
    experiment_dir: Path,
    model: DinoCaptchaModel,
    device: torch.device,
    output_dir: Path,
    layers: tuple[str, ...],
    *,
    splits: tuple[str, ...] = ("train", "test"),
    max_train_ids: int | None = None,
    batch_size: int = 128,
) -> dict[str, dict[str, float]]:
    """Extract one experiment (all splits, both batches); return transcription accuracy.

    Returns {split: {batch_a_seq_acc, batch_b_seq_acc, batch_a_char_acc, batch_b_char_acc}}.
    """
    labels = pd.read_csv(experiment_dir / "labels.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    alphabet = model.config.alphabet
    accuracy: dict[str, dict[str, float]] = {}

    for split in splits:
        split_rows = labels[labels["split"] == split]
        split_ids = split_rows["id"].tolist()
        if split == "train" and max_train_ids is not None:
            split_ids = split_ids[:max_train_ids]
        if not split_ids:
            continue
        texts = split_rows.set_index("id").loc[split_ids, "text"].tolist()
        np.save(output_dir / f"{split}_ids.npy", np.array(split_ids))

        accuracy[split] = {"n": len(split_ids)}
        for batch in ("batch_a", "batch_b"):
            img_dir = experiment_dir / batch / "images"
            images = [Image.open(img_dir / f"{sid:06d}.png") for sid in split_ids]
            activations, predictions = extract_activations_from_images(
                model, images, device, layers, batch_size=batch_size,
            )
            for layer, arr in activations.items():
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / f"{split}_{batch}_{layer}.npy", arr)

            seq_acc, char_acc = _transcription_accuracy(predictions, texts, alphabet)
            accuracy[split][f"{batch}_seq_acc"] = seq_acc
            accuracy[split][f"{batch}_char_acc"] = char_acc

    return accuracy


def _transcription_accuracy(
    predictions: np.ndarray, texts: list[str], alphabet: str,
) -> tuple[float, float]:
    decoded = [decode_indices(row.tolist(), alphabet) for row in predictions]
    seq_correct = sum(d == t for d, t in zip(decoded, texts))
    char_correct = sum(dc == tc for d, t in zip(decoded, texts) for dc, tc in zip(d, t))
    total_chars = sum(len(t) for t in texts)
    return seq_correct / max(1, len(texts)), char_correct / max(1, total_chars)
