"""LoRA fine-tune a DINOv2 backbone on the CAPTCHA transcription task.

The point is NOT to build the best reader — it is to obtain a model that is
*behaviorally invariant* to the distortions (transcribes batch_a and batch_b
identically) while initialised from frozen pretrained features we did not train.
LoRA lets the transcription loss reshape the backbone cheaply, so the invariance
pressure reaches the intermediate blocks we later probe.

Usage:
    # Single-GPU fine-tune on the same transcription set the CNN used
    uv run python -m dino.train --output dino_runs/dinov2-small/best.pt

    # Larger backbone, more data
    uv run python -m dino.train --model-name facebook/dinov2-base \
        --train-size 100000 --epochs 8 --output dino_runs/dinov2-base/best.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dino.config import DinoConfig
from dino.model import DinoCaptchaModel, build_transform, save_checkpoint
from train.model import CaptchaModelConfig
from train.scripts.config import TrainConfig
from train.scripts.dataset_load import load_mechaptcha_datasets
from train.scripts.evaluate import loss_fn, metric_counts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune a DINOv2 backbone to transcribe CAPTCHAs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Backbone / adapter
    p.add_argument("--backbone", choices=("dinov2", "clip", "timm"), default="dinov2",
                   help="Pretrained ViT family: dinov2 (self-supervised) or clip (language-supervised).")
    p.add_argument("--model-name", default="facebook/dinov2-small", dest="model_name",
                   help="HF backbone id. dinov2: facebook/dinov2-{small,base,large}; "
                        "clip: openai/clip-vit-{base-patch16,large-patch14}.")
    p.add_argument("--image-size", type=int, default=224, dest="image_size",
                   help="Square ViT input side; must be a multiple of the patch size 14.")
    p.add_argument("--lora-r", type=int, default=16, dest="lora_r",
                   help="LoRA rank. Higher = more adapter capacity to reshape the backbone.")
    p.add_argument("--lora-alpha", type=int, default=32, dest="lora_alpha",
                   help="LoRA scaling (effective lr multiplier alpha/r on the adapters).")
    p.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    p.add_argument("--lora-target-modules", nargs="+", default=[],
                   dest="lora_target_modules",
                   help="Attention sublayers to adapt (matched by name suffix). "
                        "Empty = backbone default (dinov2: query/value, clip: q_proj/v_proj).")
    p.add_argument("--head-pooling", choices=("cls", "mean"), default="cls", dest="head_pooling",
                   help="Which token the transcription heads read: cls summary or mean of patches.")
    p.add_argument("--freeze-backbone", action="store_true", dest="freeze_backbone",
                   help="Freeze the entire backbone (no LoRA); only the transcription heads are "
                        "trained. Useful as a baseline to measure how much LoRA adaptation matters.")

    # Data
    p.add_argument("--dataset-name", default="jacobcohen/mechaptcha", dest="dataset_name",
                   help="HF transcription dataset (same one the CNN trained on).")
    p.add_argument("--train-split", default="train", dest="train_split")
    p.add_argument("--val-split", default="val", dest="val_split")
    p.add_argument("--train-size", type=int, default=50_000, dest="train_size",
                   help="Cap on training examples (<=0 for all). LoRA needs far fewer than full FT.")
    p.add_argument("--val-size", type=int, default=5_000, dest="val_size")
    p.add_argument("--num-workers", type=int, default=8, dest="num_workers")

    # Optimisation
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=5e-4, dest="learning_rate")
    p.add_argument("--min-learning-rate", type=float, default=1e-5, dest="min_learning_rate")
    p.add_argument("--warmup-steps", type=int, default=200, dest="warmup_steps")
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--eval-every-steps", type=int, default=200, dest="eval_every_steps")
    p.add_argument("--seed", type=int, default=82)
    p.add_argument("--no-amp", action="store_false", dest="amp")

    p.add_argument("--output", type=Path, default=Path("dino_runs/dinov2-small/best.pt"),
                   help="Checkpoint path (saves LoRA adapters + heads + config).")
    return p.parse_args()


def _lr_at(step: int, total: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cos


@torch.no_grad()
def _evaluate(model: DinoCaptchaModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(4, device=device)
    for images, labels, _ids, _meta in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        totals += metric_counts(logits, labels)
    char_correct, total_chars, exact_correct, total_examples = totals.tolist()
    return {
        "char_acc": char_correct / max(1, total_chars),
        "seq_acc": exact_correct / max(1, total_examples),
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dino_config = DinoConfig(
        backbone=args.backbone,
        model_name=args.model_name,
        image_size=args.image_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        head_pooling=args.head_pooling,
    )
    if args.freeze_backbone:
        print(f"Loading backbone {dino_config.model_name} (frozen — heads only)")
    else:
        print(f"Loading backbone {dino_config.model_name} + LoRA(r={dino_config.lora_r})")
    model = DinoCaptchaModel(dino_config).to(device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train:,} ({model.num_blocks} blocks)")

    # Reuse the CNN's dataset loader, overriding only the image transform.
    train_config = TrainConfig(
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        val_split=args.val_split,
        train_size=args.train_size,
        val_size=args.val_size,
        wandb_mode="disabled",
    )
    model_config = CaptchaModelConfig(num_chars=dino_config.num_chars, alphabet=dino_config.alphabet)
    bundle = load_mechaptcha_datasets(train_config, model_config, transform=build_transform(dino_config))
    print(f"Train examples: {len(bundle.train):,}  Val examples: {len(bundle.val):,}")

    train_loader = DataLoader(
        bundle.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        bundle.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    total_steps = args.epochs * len(train_loader)
    print(f"Total optimisation steps: {total_steps}")
    step = 0
    best_seq_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for images, labels, _ids, _meta in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = _lr_at(step, total_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.1e}")

            if step % args.eval_every_steps == 0:
                metrics = _evaluate(model, val_loader, device)
                tqdm.write(f"  step {step}: val char_acc={metrics['char_acc']:.3f} "
                           f"seq_acc={metrics['seq_acc']:.3f}")
                if metrics["seq_acc"] > best_seq_acc:
                    best_seq_acc = metrics["seq_acc"]
                    save_checkpoint(model, args.output)
                    tqdm.write(f"    new best seq_acc -> saved {args.output}")
                model.train()

    # Final eval + save (covers the case where no eval step landed on the best).
    metrics = _evaluate(model, val_loader, device)
    print(f"Final: val char_acc={metrics['char_acc']:.3f} seq_acc={metrics['seq_acc']:.3f}")
    if metrics["seq_acc"] >= best_seq_acc:
        save_checkpoint(model, args.output)
        print(f"Saved final checkpoint -> {args.output}")
    print(f"Best val seq_acc: {max(best_seq_acc, metrics['seq_acc']):.3f}")


if __name__ == "__main__":
    main()
