from __future__ import annotations

import argparse
import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from train.model import CaptchaModelConfig

DEFAULT_OUTPUT_ROOT = Path("runs/captcha-cnn")


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    run_name = f"{timestamp}_slurm-{slurm_job_id}" if slurm_job_id else f"{timestamp}_local"
    return DEFAULT_OUTPUT_ROOT / run_name


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path = field(default_factory=default_output_dir)
    upload_to_hf: bool = True
    wandb_project: str = "mechaptcha"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = field(default="online", metadata={"choices": ("online", "offline", "disabled")})
    dataset_name: str = "jacobcohen/mechaptcha"
    dataset_config: str | None = None
    train_split: str = "train"
    val_split: str = "val"
    image_column: str = "image"
    text_column: str = "text"
    id_column: str = "id"
    dataset_cache_dir: str | None = None
    epochs: int = 10
    train_size: int = 100_000
    val_size: int = 10_000
    batch_size: int = 256
    num_workers: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 82
    num_chars: int = 5
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    image_height: int = 64
    image_width: int = 160
    log_every: int = 50
    eval_every_steps: int = 500
    amp: bool = True
    compile: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the simple CAPTCHA CNN.")
    add_dataclass_args(parser, TrainConfig)
    return parser.parse_args()


def add_dataclass_args(parser: argparse.ArgumentParser, config_type: type[TrainConfig]) -> None:
    type_hints = get_type_hints(config_type)
    for config_field in fields(config_type):
        name = config_field.name
        arg_name = f"--{name.replace('_', '-')}"
        field_type = type_hints[name]
        if config_field.default_factory is not MISSING:
            default = config_field.default_factory()
        else:
            default = config_field.default
        if unwrap_optional(field_type) is Path:
            default = str(default)
        kwargs: dict[str, Any] = {
            "default": default,
            "dest": name,
        }
        if "choices" in config_field.metadata:
            kwargs["choices"] = config_field.metadata["choices"]

        if field_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = cli_type(field_type)

        parser.add_argument(arg_name, **kwargs)


def cli_type(field_type: object) -> type:
    field_type = unwrap_optional(field_type)
    if field_type is Path:
        return str
    if field_type in (str, int, float):
        return field_type
    return str


def unwrap_optional(field_type: object) -> object:
    origin = get_origin(field_type)
    if origin not in (Union, UnionType):
        return field_type

    args = [arg for arg in get_args(field_type) if arg is not type(None)]
    return args[0] if len(args) == 1 else field_type


def train_config_dict(config: TrainConfig) -> dict[str, object]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    return values


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return train_config_from_dict(TrainConfig(), vars(args))


def train_config_from_dict(base: TrainConfig, values: dict[str, object]) -> TrainConfig:
    config_values = train_config_dict(base)
    train_values = values.get("train")
    if isinstance(train_values, dict):
        config_values.update(train_values)
    config_values.update({key: value for key, value in values.items() if key in config_values})
    config_values["output_dir"] = Path(str(config_values["output_dir"]))
    return TrainConfig(**cast(Any, config_values))


def wandb_env_config() -> dict[str, object]:
    config = os.environ.get("WANDB_CONFIG")
    if config is not None:
        return json.loads(config)

    chunks: list[str] = []
    chunk_idx = 0
    while True:
        chunk = os.environ.get(f"WANDB_CONFIG_{chunk_idx}")
        if chunk is None:
            break
        chunks.append(chunk)
        chunk_idx += 1
    if chunks:
        return json.loads("".join(chunks))
    return {}


def model_config_dict(config: CaptchaModelConfig) -> dict[str, object]:
    values = asdict(config)
    values["num_classes"] = config.num_classes
    values["flattened_features"] = config.flattened_features
    return values
