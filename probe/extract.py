"""Extract and cache CNN activations from experiment images."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.model import CaptchaCNN, CaptchaModelConfig
from train.model.charset import decode_indices
from probe.config import HOOK_LAYERS, ProbeConfig, get_model_layers


_CAPTCHA_MODEL_CONFIG_FIELDS = {
    "image_channels", "num_chars", "alphabet", "conv_channels",
    "pooled_shape", "embedding_dim", "dropout",
}


def load_model(checkpoint_path: str | Path, config: ProbeConfig | None = None) -> CaptchaCNN:
    """Load a CaptchaCNN from a training checkpoint."""
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("model_config", {})
    # Strip computed properties (num_classes, flattened_features) that aren't dataclass fields
    model_cfg = {k: v for k, v in raw_cfg.items() if k in _CAPTCHA_MODEL_CONFIG_FIELDS}
    model_config = CaptchaModelConfig(**model_cfg) if model_cfg else CaptchaModelConfig()
    model = CaptchaCNN(model_config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a local checkpoint path or download one from a HF model repo."""
    path = Path(checkpoint)
    if path.exists():
        return path

    repo_id, filename = parse_hf_checkpoint(str(checkpoint))
    if repo_id is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    from huggingface_hub import hf_hub_download

    filenames = [filename] if filename else ["best.pt", "checkpoints/best.pt", "checkpoints/last.pt"]
    errors: list[Exception] = []
    for candidate in filenames:
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="model"))
        except Exception as exc:
            errors.append(exc)
    raise FileNotFoundError(
        f"Could not find a checkpoint in HF model repo {repo_id!r}. "
        f"Tried: {', '.join(filenames)}"
    ) from errors[-1]


def parse_hf_checkpoint(value: str) -> tuple[str | None, str | None]:
    """Return (repo_id, filename) for HF repo ids or huggingface.co URLs."""
    parsed = urlparse(value)
    if parsed.netloc == "huggingface.co":
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            return None, None
        repo_id = "/".join(parts[:2])
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            return repo_id, "/".join(parts[4:])
        return repo_id, None

    if value.count("/") == 1 and not value.endswith(".pt"):
        return value, None
    return None, None


def _make_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


def _reduce_conv(t: torch.Tensor, method: str) -> np.ndarray:
    """Reduce a [B, C, H, W] conv activation to [B, features] in float16."""
    if method == "global_avg_pool":
        return t.mean(dim=(2, 3)).numpy().astype(np.float16)
    elif method == "flatten":
        return t.flatten(start_dim=1).numpy().astype(np.float16)
    raise ValueError(f"Unknown conv_reduction: {method!r}")


def extract_activations(
    model: CaptchaCNN,
    image_paths: list[Path],
    device: torch.device,
    config: ProbeConfig,
    return_logits: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], torch.Tensor]:
    """Run images through the model and return [N, features] arrays per probed layer.

    "input"  — raw pixels [B, 1, H, W] flattened; always flat regardless of conv_reduction
    conv layers — reduced according to config.conv_reduction
    "pool"   — flattened [B, 5120]; preserves spatial structure
    "embedding" — [B, 256] as-is
    "logits" — model output [B, 5, 26] flattened to [B, 130]

    When return_logits=True, returns (activations, logits_tensor) where logits_tensor
    is [N, num_chars, vocab_size] — used for transcription accuracy computation.
    """
    images = [Image.open(p).convert("L") for p in image_paths]
    return extract_activations_from_images(model, images, device, config, return_logits=return_logits)


def extract_activations_from_images(
    model: CaptchaCNN,
    images: list[Image.Image],
    device: torch.device,
    config: ProbeConfig,
    return_logits: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], torch.Tensor]:
    """Run in-memory images through the model and return activations."""
    transform = _make_transform(config.image_size)
    conv_layers = {"conv_block_0", "conv_block_1", "conv_block_2"}
    hook_layers_needed = [l for l in config.layers if l in HOOK_LAYERS]

    current: dict[str, np.ndarray] = {}
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in config.layers}
    all_logits: list[torch.Tensor] = []

    def make_hook(name: str):
        def hook(module, input, output):
            t = output.detach().cpu()
            if name in conv_layers:
                current[name] = _reduce_conv(t, config.conv_reduction)
            else:
                current[name] = t.flatten(start_dim=1).numpy()
        return hook

    handles = _register_hooks(model, hook_layers_needed, make_hook)

    with torch.no_grad():
        for i in range(0, len(images), config.batch_size):
            batch = torch.stack([
                transform(img.convert("L"))
                for img in images[i:i + config.batch_size]
            ]).to(device)

            if "input" in config.layers:
                current["input"] = batch.cpu().flatten(start_dim=1).numpy()

            logits = model(batch)
            all_logits.append(logits.detach().cpu())

            if "logits" in config.layers:
                current["logits"] = logits.detach().cpu().flatten(start_dim=1).numpy()

            for name in config.layers:
                accumulated[name].append(current[name])

    for h in handles:
        h.remove()

    activations = {name: np.concatenate(chunks) for name, chunks in accumulated.items()}
    if return_logits:
        return activations, torch.cat(all_logits, dim=0)
    return activations


def _register_hooks(model: CaptchaCNN, layers: list[str], make_hook) -> list:
    layer_map: dict[str, torch.nn.Module] = {
        name: module for name, module in model.features.named_children()
    }
    layer_map["pool"] = model.pool
    layer_map["embedding"] = model.embedding
    handles = []
    for name in layers:
        if name in layer_map:
            handles.append(layer_map[name].register_forward_hook(make_hook(name)))
    return handles


def _transcription_accuracy(
    logits: torch.Tensor, texts: list[str], alphabet: str,
) -> tuple[float, float]:
    """Seq and char accuracy from raw [B, num_chars, vocab] logits."""
    preds = logits.argmax(dim=-1).cpu().numpy()  # [B, num_chars]
    decoded = [decode_indices(row.tolist(), alphabet) for row in preds]
    seq_correct = sum(d == t for d, t in zip(decoded, texts))
    char_correct = sum(dc == tc for d, t in zip(decoded, texts) for dc, tc in zip(d, t))
    total_chars = sum(len(t) for t in texts)
    return seq_correct / max(1, len(texts)), char_correct / max(1, total_chars)


def extract_experiment(
    experiment_dir: Path,
    model: CaptchaCNN,
    device: torch.device,
    output_dir: Path,
    config: ProbeConfig,
    splits: tuple[str, ...] = ("train", "test"),
    max_train_ids: int | None = None,
) -> dict[str, dict[str, float]]:
    """Extract and save activations for one experiment; return transcription accuracy.

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
            paths = [img_dir / f"{sid:06d}.png" for sid in split_ids]
            activations, logits = extract_activations(model, paths, device, config, return_logits=True)
            for layer, arr in activations.items():
                np.save(output_dir / f"{split}_{batch}_{layer}.npy", arr)
            seq_acc, char_acc = _transcription_accuracy(logits, texts, alphabet)
            accuracy[split][f"{batch}_seq_acc"] = seq_acc
            accuracy[split][f"{batch}_char_acc"] = char_acc

    return accuracy


def extract_hf_experiments(
    dataset_id: str,
    model: CaptchaCNN,
    device: torch.device,
    output_root: Path,
    config: ProbeConfig,
    experiment: str | None = None,
    splits: tuple[str, ...] = ("train", "test"),
    force_extract: bool = False,
    max_train_ids: int | None = None,
) -> list[Path]:
    """Extract activations from a HF paired-experiment dataset repo."""
    from datasets import DatasetDict, load_dataset

    dataset = load_dataset(dataset_id)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected {dataset_id!r} to load as a DatasetDict with train/test splits.")

    available = set(dataset.keys())
    missing = set(splits) - available
    if missing:
        raise ValueError(f"HF dataset {dataset_id!r} is missing required splits: {sorted(missing)}")

    experiment_names = [experiment] if experiment else sorted(set(dataset[splits[0]]["experiment"]))
    output_dirs: list[Path] = []
    for name in experiment_names:
        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(output_dir)
        if not force_extract and not _needs_hf_extraction(output_dir, config, splits):
            continue

        for split in splits:
            split_dataset = dataset[split].filter(lambda row, exp=name: row["experiment"] == exp)
            if split == "train" and max_train_ids is not None:
                split_dataset = split_dataset.select(range(min(max_train_ids, len(split_dataset))))
            if len(split_dataset) == 0:
                continue

            ids = np.array(split_dataset["id"])
            np.save(output_dir / f"{split}_ids.npy", ids)

            for batch, image_column in (("batch_a", "image_a"), ("batch_b", "image_b")):
                images = [example[image_column] for example in split_dataset]
                activations = extract_activations_from_images(model, images, device, config)
                for layer, arr in activations.items():
                    np.save(output_dir / f"{split}_{batch}_{layer}.npy", arr)

    return output_dirs


def _needs_hf_extraction(output_dir: Path, config: ProbeConfig, splits: tuple[str, ...]) -> bool:
    for split in splits:
        for batch in ("batch_a", "batch_b"):
            for layer in config.layers:
                if not (output_dir / f"{split}_{batch}_{layer}.npy").exists():
                    return True
    return False
