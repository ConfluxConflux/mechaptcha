from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from train.model.config import CaptchaModelConfig
from train.scripts.config import TrainConfig, model_config_dict, train_config_dict


@dataclass(frozen=True)
class WandbRunInfo:
    name: str | None
    url: str | None


@dataclass(frozen=True)
class HuggingFaceUploadInfo:
    repo_id: str
    repo_url: str


def preflight_hf_upload(output_dir: Path, wandb_run: WandbRunInfo) -> HuggingFaceUploadInfo:
    from huggingface_hub import HfApi

    repo_id = hf_repo_id(wandb_run.name or output_dir.name)
    api = HfApi()
    return create_or_get_hf_repo(api, repo_id)


def upload_training_artifacts_to_hf(
    output_dir: Path,
    train_config: TrainConfig,
    model_config: CaptchaModelConfig,
    dataset_sha: str | None,
    final_val_metrics: dict[str, float | int],
    validation_history: list[dict[str, float | int]],
    best_val_exact_match: float,
    global_step: int,
    wandb_run: WandbRunInfo,
) -> HuggingFaceUploadInfo:
    from huggingface_hub import HfApi

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_id = hf_repo_id(wandb_run.name or output_dir.name)
    metadata = build_metadata(
        repo_id=repo_id,
        output_dir=output_dir,
        train_config=train_config,
        model_config=model_config,
        dataset_sha=dataset_sha,
        final_val_metrics=final_val_metrics,
        validation_history=validation_history,
        best_val_exact_match=best_val_exact_match,
        global_step=global_step,
        wandb_run=wandb_run,
    )
    write_training_metadata(output_dir / "training_metadata.json", metadata)
    write_model_card(output_dir / "README.md", metadata)

    api = HfApi()
    upload_info = create_or_get_hf_repo(api, repo_id)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=upload_info.repo_id,
        repo_type="model",
        commit_message=f"Upload training artifacts for {repo_id}",
    )
    return upload_info


def create_or_get_hf_repo(api: object, repo_id: str) -> HuggingFaceUploadInfo:
    repo_url = api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    resolved_repo_id = getattr(repo_url, "repo_id", None) or repo_id_from_url(str(repo_url)) or repo_id
    return HuggingFaceUploadInfo(repo_id=resolved_repo_id, repo_url=str(repo_url))


def repo_id_from_url(repo_url: str) -> str | None:
    match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", repo_url)
    return match.group(1) if match else None


def hf_repo_id(run_name: str) -> str:
    return f"2026.mechaptcha.{sanitize_repo_component(run_name)}"


def sanitize_repo_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = re.sub(r"[-.]{2,}", "-", sanitized).strip(".-")
    return sanitized or "run"


def build_metadata(
    repo_id: str,
    output_dir: Path,
    train_config: TrainConfig,
    model_config: CaptchaModelConfig,
    dataset_sha: str | None,
    final_val_metrics: dict[str, float | int],
    validation_history: list[dict[str, float | int]],
    best_val_exact_match: float,
    global_step: int,
    wandb_run: WandbRunInfo,
) -> dict[str, object]:
    return {
        "hf_repo_id": repo_id,
        "dataset_id": train_config.dataset_name,
        "dataset_sha": dataset_sha,
        "dataset_config": train_config.dataset_config,
        "splits": {
            "train": train_config.train_split,
            "val": train_config.val_split,
        },
        "sample_counts": {
            "train": train_config.train_size,
            "val": train_config.val_size,
        },
        "global_step": global_step,
        "best_val_exact_match": best_val_exact_match,
        "final_validation": final_val_metrics,
        "validation_history": validation_history,
        "evaluation_summary": evaluation_summary_metadata(output_dir),
        "wandb": {
            "name": wandb_run.name,
            "url": wandb_run.url,
        },
        "train_config": train_config_dict(train_config),
        "model_config": model_config_dict(model_config),
    }


def write_training_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_model_card(path: Path, metadata: dict[str, object]) -> None:
    dataset_id = str(metadata["dataset_id"])
    final_val_metrics = metadata["final_validation"]
    if not isinstance(final_val_metrics, dict):
        raise TypeError("final_validation metadata must be a dictionary")

    path.write_text(
        "\n".join(
            [
                "---",
                "library_name: pytorch",
                "datasets:",
                f"- {dataset_id}",
                "metrics:",
                "- accuracy",
                "- loss",
                "tags:",
                "- captcha",
                "- image-classification",
                "---",
                "",
                f"# {metadata['hf_repo_id']}",
                "",
                "CAPTCHA CNN trained on the Mechaptcha dataset.",
                "",
                "## Source Links",
                "",
                f"- Dataset: `{dataset_id}`",
                f"- Weights & Biases run: {metadata['wandb']['url'] or 'not available'}",
                "",
                "## Validation Metrics",
                "",
                metrics_table(final_val_metrics),
                "",
                "## Validation History",
                "",
                validation_history_table(metadata["validation_history"]),
                "",
                "## Validation Summary Artifacts",
                "",
                "- `val_final_distortion_summary.json`: validation metrics split by distortion/style flag.",
                "- `val_final_confusion_matrix.json`: character-level confusion matrix.",
                "- `val_final_confusion_matrix.csv`: CSV version of the character-level confusion matrix.",
                "- `val_failed_ids.json`: validation example IDs that failed exact match.",
                "",
                "## Training Summary",
                "",
                f"- Global step: `{metadata['global_step']}`",
                f"- Best validation exact match: `{metadata['best_val_exact_match']}`",
                "",
                "## Configuration",
                "",
                "The full training and model configuration is included in `training_metadata.json`.",
                "",
            ]
        )
    )


def metrics_table(metrics: dict[str, object]) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key in sorted(metrics):
        lines.append(f"| `{key}` | `{metrics[key]}` |")
    return "\n".join(lines)


def validation_history_table(history: object) -> str:
    if not isinstance(history, list) or not history:
        return "No validation history was recorded."

    metric_keys = sorted({key for item in history if isinstance(item, dict) for key in item})
    lines = [
        "| " + " | ".join(f"`{key}`" for key in metric_keys) + " |",
        "| " + " | ".join("---:" for _ in metric_keys) + " |",
    ]
    for item in history:
        if not isinstance(item, dict):
            continue
        lines.append("| " + " | ".join(f"`{item.get(key, '')}`" for key in metric_keys) + " |")
    return "\n".join(lines)


def evaluation_summary_metadata(output_dir: Path) -> dict[str, object]:
    summary_files = {
        "distortion_summary": output_dir / "val_final_distortion_summary.json",
        "confusion_matrix": output_dir / "val_final_confusion_matrix.json",
        "confusion_matrix_csv": output_dir / "val_final_confusion_matrix.csv",
        "failed_ids": output_dir / "val_failed_ids.json",
    }
    metadata: dict[str, object] = {}
    for name, path in summary_files.items():
        if not path.exists():
            continue
        metadata[name] = {"path": path.name}
        if path.suffix == ".json":
            metadata[name]["content"] = json.loads(path.read_text())
    return metadata
