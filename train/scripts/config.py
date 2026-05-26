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
    batch_size: int = 1024
    num_workers: int = 16
    prefetch_factor: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    lr_schedule: str = field(default="cosine", metadata={"choices": ("constant", "cosine")})
    warmup_steps: int = 1_000
    weight_decay: float = 1e-4
    seed: int = 82
    image_height: int = 64
    image_width: int = 160
    log_every: int = 50
    eval_every_steps: int = 500
    amp: bool = True
    compile: bool = False
    channels_last: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the simple CAPTCHA CNN.")
    add_dataclass_args(parser, TrainConfig)
    add_dataclass_args(parser, CaptchaModelConfig)
    return parser.parse_args()


def add_dataclass_args(parser: argparse.ArgumentParser, config_type: type[object]) -> None:
    type_hints = get_type_hints(config_type)
    for config_field in fields(config_type):
        name = config_field.name
        arg_name = f"--{name.replace('_', '-')}"
        arg_names = [arg_name]
        underscore_arg_name = f"--{name}"
        if underscore_arg_name != arg_name:
            arg_names.append(underscore_arg_name)
        field_type = type_hints[name]
        default = cli_default(config_field.default_factory() if config_field.default_factory is not MISSING else config_field.default)
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

        parser.add_argument(*arg_names, **kwargs)


def cli_default(value: object) -> object:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return value


def cli_type(field_type: object) -> type:
    field_type = unwrap_optional(field_type)
    if field_type is Path:
        return str
    if get_origin(field_type) is tuple:
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


def model_config_from_args(args: argparse.Namespace) -> CaptchaModelConfig:
    return model_config_from_dict(CaptchaModelConfig(), vars(args))


def train_config_from_dict(base: TrainConfig, values: dict[str, object]) -> TrainConfig:
    config_values = train_config_dict(base)
    train_values = values.get("train")
    if isinstance(train_values, dict):
        config_values.update(train_values)
    config_values.update({key: value for key, value in values.items() if key in config_values})
    config_values["output_dir"] = Path(str(config_values["output_dir"]))
    return TrainConfig(**cast(Any, config_values))


def model_config_from_dict(base: CaptchaModelConfig, values: dict[str, object]) -> CaptchaModelConfig:
    config_values = asdict(base)
    model_values = values.get("model")
    if isinstance(model_values, dict):
        config_values.update(model_values)
    config_values.update({key: value for key, value in values.items() if key in config_values})

    type_hints = get_type_hints(CaptchaModelConfig)
    parsed_values = {
        key: parse_config_value(value, type_hints[key])
        for key, value in config_values.items()
    }
    return CaptchaModelConfig(**cast(Any, parsed_values))


def parse_config_value(value: object, field_type: object) -> object:
    field_type = unwrap_optional(field_type)
    if get_origin(field_type) is tuple:
        return parse_int_tuple(value)
    if field_type is int:
        return int(cast(Any, value))
    if field_type is float:
        return float(cast(Any, value))
    if field_type is str:
        return str(value)
    return value


def parse_int_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        return tuple(int(chunk) for chunk in chunks)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise TypeError(f"Expected comma-separated string or sequence of ints, got {type(value).__name__}")


def model_config_input_dict(config: CaptchaModelConfig) -> dict[str, object]:
    return {key: cli_default(value) for key, value in asdict(config).items()}


def combined_config_dict(train_config: TrainConfig, model_config: CaptchaModelConfig) -> dict[str, object]:
    return {
        **train_config_dict(train_config),
        **model_config_input_dict(model_config),
    }


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


def parse_conv_channels(value: str) -> tuple[int, ...]:
    channels = parse_int_tuple(value)
    if not channels:
        raise ValueError("conv_channels must contain at least one channel")
    return channels
