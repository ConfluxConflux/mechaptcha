"""Extract and cache CNN activations from experiment images for linear probing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.model import CaptchaCNN, CaptchaModelConfig

LAYER_NAMES = ["conv_block_0", "conv_block_1", "conv_block_2", "pool", "embedding"]
IMAGE_SIZE = (64, 160)
BATCH_SIZE = 128

_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def load_model(checkpoint_path: Path) -> CaptchaCNN:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    config = CaptchaModelConfig(**model_cfg) if model_cfg else CaptchaModelConfig()
    model = CaptchaCNN(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def extract_activations(
    model: CaptchaCNN,
    image_paths: list[Path],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run images through model; return activations per layer as [N, features] arrays.

    Conv layers use global average pooling over spatial dims to keep feature count small.
    The pool layer is flattened (5120 features) to preserve its spatial structure.
    The embedding layer is returned as-is (256 features).
    """
    current: dict[str, np.ndarray] = {}
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in LAYER_NAMES}

    def make_hook(name: str):
        def hook(module, input, output):
            t = output.detach().cpu()
            if name in ("conv_block_0", "conv_block_1", "conv_block_2"):
                # [B, C, H, W] -> [B, C]
                t = t.mean(dim=(2, 3))
            else:
                # pool: [B, C, H, W] -> [B, C*H*W]; embedding: [B, D] -> [B, D]
                t = t.flatten(start_dim=1)
            current[name] = t.numpy()
        return hook

    handles = [
        model.features.conv_block_0.register_forward_hook(make_hook("conv_block_0")),
        model.features.conv_block_1.register_forward_hook(make_hook("conv_block_1")),
        model.features.conv_block_2.register_forward_hook(make_hook("conv_block_2")),
        model.pool.register_forward_hook(make_hook("pool")),
        model.embedding.register_forward_hook(make_hook("embedding")),
    ]

    with torch.no_grad():
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = torch.stack([
                _transform(Image.open(p).convert("L"))
                for p in image_paths[i:i + BATCH_SIZE]
            ]).to(device)
            model(batch)
            for name in LAYER_NAMES:
                accumulated[name].append(current[name])

    for h in handles:
        h.remove()

    return {name: np.concatenate(chunks) for name, chunks in accumulated.items()}


def extract_experiment(
    experiment_dir: Path,
    model: CaptchaCNN,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Extract and save activations for one experiment across all splits and batches."""
    labels = pd.read_csv(experiment_dir / "labels.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        split_ids = labels[labels["split"] == split]["id"].tolist()
        if not split_ids:
            continue

        np.save(output_dir / f"{split}_ids.npy", np.array(split_ids))

        for batch in ("batch_a", "batch_b"):
            img_dir = experiment_dir / batch / "images"
            paths = [img_dir / f"{sid:06d}.png" for sid in split_ids]

            activations = extract_activations(model, paths, device)
            for layer, arr in activations.items():
                np.save(output_dir / f"{split}_{batch}_{layer}.npy", arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNN activations for linear probing.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, default=Path("data/experiments"))
    parser.add_argument("--output", type=Path, default=Path("probe_results/activations"))
    parser.add_argument("--experiment", type=str, default=None, help="Single experiment (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint).to(device)

    if args.experiment:
        experiments = [args.experiments / args.experiment]
    else:
        experiments = sorted(p for p in args.experiments.iterdir() if p.is_dir())

    for exp_dir in tqdm(experiments, desc="Experiments"):
        extract_experiment(
            experiment_dir=exp_dir,
            model=model,
            device=device,
            output_dir=args.output / exp_dir.name,
        )
        tqdm.write(f"  {exp_dir.name} -> {args.output / exp_dir.name}")


if __name__ == "__main__":
    main()
