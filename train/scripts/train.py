# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.model import CaptchaCNN, CaptchaModelConfig
from train.scripts.config import (
    TrainConfig,
    model_config_dict,
    parse_args,
    parse_conv_channels,
    train_config_from_args,
    train_config_dict,
    train_config_from_dict,
    wandb_env_config,
)
from train.scripts.dataset_load import dataset_hub_sha, load_mechaptcha_datasets
from train.scripts.evaluate import evaluate, log_wandb_summary_artifacts, loss_fn, metric_counts
from train.scripts.hf_upload import WandbRunInfo, preflight_hf_upload, upload_training_artifacts_to_hf


class WandbRun(Protocol):
    config: Any
    summary: Any
    name: str
    url: str

    def log(self, data: dict[str, object], step: int | None = None) -> None: ...

    def log_artifact(self, artifact: object) -> None: ...

    def finish(self) -> None: ...


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    distributed: bool,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader, sampler


def broadcast_config(config: TrainConfig, distributed: bool, rank: int) -> TrainConfig:
    if not distributed:
        return config

    payload = [train_config_dict(config) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return train_config_from_dict(config, cast(dict[str, object], payload[0]))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    config: TrainConfig,
    model_config: CaptchaModelConfig,
) -> None:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_config": train_config_dict(config),
            "model_config": model_config_dict(model_config),
        },
        path,
    )


def init_wandb(args: argparse.Namespace, enabled: bool) -> tuple[WandbRun | None, TrainConfig]:
    base_config = train_config_from_args(args)
    if not enabled or base_config.wandb_mode == "disabled":
        return None, base_config

    import wandb

    run = cast(
        WandbRun,
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=cast(Any, args),
        ),
    )
    resolved_config = train_config_from_dict(base_config, dict(run.config))
    resolved_config = train_config_from_dict(resolved_config, wandb_env_config())
    run.config.update(train_config_dict(resolved_config), allow_val_change=True)
    return run, resolved_config


def update_wandb_model_config(run: WandbRun | None, model_config: CaptchaModelConfig) -> None:
    if run is None:
        return

    run.config.update(
        {f"model/{key}": value for key, value in model_config_dict(model_config).items()},
        allow_val_change=True,
    )


def wandb_run_info(run: WandbRun | None) -> WandbRunInfo:
    return WandbRunInfo(
        name=getattr(run, "name", None),
        url=getattr(run, "url", None),
    )


def preflight_hf_upload_or_raise(
    config: TrainConfig,
    run: WandbRun | None,
    distributed: bool,
    is_main: bool,
) -> None:
    error_message: str | None = None
    if is_main:
        try:
            upload_info = preflight_hf_upload(config.output_dir, wandb_run_info(run))
            print(f"Verified Hugging Face model repo access: {upload_info.repo_url}", flush=True)
        except Exception as exc:
            error_message = str(exc)

    if distributed:
        payload = [error_message]
        dist.broadcast_object_list(payload, src=0)
        error_message = cast(str | None, payload[0])

    if error_message is not None:
        raise RuntimeError(f"Hugging Face upload preflight failed: {error_message}")


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if config.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    min_factor = config.min_learning_rate / config.learning_rate
    warmup_steps = max(config.warmup_steps, 0)
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, min_factor)

        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    run, config = init_wandb(args, enabled=is_main)
    config = broadcast_config(config, distributed, rank)
    if is_main:
        print(f"Output directory: {config.output_dir.resolve()}", flush=True)
    if config.upload_to_hf:
        preflight_hf_upload_or_raise(config, run, distributed, is_main)

    torch.manual_seed(config.seed + rank)
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp and device.type == "cuda"
    model_config = CaptchaModelConfig(
        image_channels=1,
        num_chars=config.num_chars,
        alphabet=config.alphabet,
        conv_channels=parse_conv_channels(config.conv_channels),
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    )
    if is_main:
        update_wandb_model_config(run, model_config)

    datasets = load_mechaptcha_datasets(config)
    dataset_sha = dataset_hub_sha(config.dataset_name) if is_main else None
    if is_main and run is not None:
        run.summary["dataset/id"] = config.dataset_name
        run.summary["dataset/sha"] = dataset_sha
    train_loader, train_sampler = make_loader(
        datasets.train,
        config.batch_size,
        config.num_workers,
        config.prefetch_factor,
        distributed,
        shuffle=True,
    )
    val_loader, _ = make_loader(
        datasets.val,
        config.batch_size,
        config.num_workers,
        config.prefetch_factor,
        distributed,
        shuffle=False,
    )
    val_output_dir = config.output_dir / "val"
    setattr(datasets.val, "failed_ids_path", val_output_dir / "val_failed_ids.json")
    setattr(datasets.val, "summary_output_dir", val_output_dir)
    checkpoints_dir = config.output_dir / "checkpoints"

    model: nn.Module = CaptchaCNN(model_config).to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if config.compile:
        model = cast(nn.Module, torch.compile(model))
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = make_lr_scheduler(optimizer, config, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    best_exact_match = -math.inf
    best_val_global_step: int | None = None
    validation_history: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        progress = tqdm(train_loader, disable=not is_main, desc=f"epoch {epoch + 1}/{config.epochs}")
        for images, labels, _example_ids, _metadata in progress:
            images = images.to(device, non_blocking=True)
            if config.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            counts = metric_counts(logits.detach(), labels)
            if distributed:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            char_accuracy = (counts[0] / counts[1]).item()
            exact_match = (counts[2] / counts[3]).item()

            global_step += 1
            if is_main and global_step % config.log_every == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/char_accuracy": char_accuracy,
                    "train/exact_match": exact_match,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "world_size": world_size,
                }
                progress.set_postfix(loss=f"{loss.item():.4f}", exact=f"{exact_match:.3f}")
                if run is not None:
                    run.log(metrics, step=global_step)

            if global_step % config.eval_every_steps == 0:
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    use_amp,
                    distributed,
                    metric_prefix="val",
                    channels_last=config.channels_last,
                    collect_distortion_metrics=True,
                )
                if is_main:
                    metrics["epoch"] = epoch + 1
                    metrics["global_step"] = global_step
                    validation_history.append(metrics.copy())
                    if run is not None:
                        run.log(metrics, step=global_step)
                    if metrics["val/exact_match"] > best_exact_match:
                        best_exact_match = metrics["val/exact_match"]
                        best_val_global_step = global_step
                        if global_step == total_steps:
                            (checkpoints_dir / "best.pt").unlink(missing_ok=True)
                        else:
                            save_checkpoint(
                                checkpoints_dir / "best.pt",
                                model,
                                optimizer,
                                scheduler,
                                epoch,
                                global_step,
                                config,
                                model_config,
                            )
                model.train()

        metrics = evaluate(
            model,
            val_loader,
            device,
            use_amp,
            distributed,
            metric_prefix="val",
            channels_last=config.channels_last,
            collect_distortion_metrics=True,
        )
        if is_main:
            metrics["epoch"] = epoch + 1
            metrics["global_step"] = global_step
            validation_history.append(metrics.copy())
            if run is not None:
                run.log(metrics, step=global_step)
            save_checkpoint(
                checkpoints_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                model_config,
            )
            if metrics["val/exact_match"] > best_exact_match:
                best_exact_match = metrics["val/exact_match"]
                best_val_global_step = global_step
                if global_step == total_steps:
                    (checkpoints_dir / "best.pt").unlink(missing_ok=True)
                else:
                    save_checkpoint(
                        checkpoints_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        config,
                        model_config,
                    )

    final_val_metrics = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        distributed,
        metric_prefix="val/final",
        channels_last=config.channels_last,
        collect_failures=True,
        collect_summary=True,
        alphabet=config.alphabet,
    )
    if is_main:
        final_val_metrics["global_step"] = global_step
        if run is not None:
            run.log(final_val_metrics, step=global_step)
            log_wandb_summary_artifacts(run, config.output_dir)
        if config.upload_to_hf:
            upload_info = upload_training_artifacts_to_hf(
                output_dir=config.output_dir,
                train_config=config,
                model_config=model_config,
                dataset_sha=dataset_sha,
                final_val_metrics=final_val_metrics,
                validation_history=validation_history,
                best_val_exact_match=best_exact_match,
                best_val_global_step=best_val_global_step,
                global_step=global_step,
                wandb_run=wandb_run_info(run),
            )
            print(f"Uploaded Hugging Face model repo: {upload_info.repo_url}", flush=True)
            if run is not None:
                run.log({"hf/repo_url": upload_info.repo_url}, step=global_step)

    if run is not None:
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
