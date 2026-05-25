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
    best_val_global_step: int | None,
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
        best_val_global_step=best_val_global_step,
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
    best_val_global_step: int | None,
    global_step: int,
    wandb_run: WandbRunInfo,
) -> dict[str, object]:
    best_is_last = best_val_global_step == global_step
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
        "best_val_global_step": best_val_global_step,
        "checkpoints": {
            "last": {
                "path": "checkpoints/last.pt",
                "global_step": global_step,
            },
            "best": None
            if best_is_last
            else {
                "path": "checkpoints/best.pt",
                "global_step": best_val_global_step,
            },
            "best_is_last": best_is_last,
        },
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
    checkpoints = metadata["checkpoints"]
    if not isinstance(checkpoints, dict):
        raise TypeError("checkpoints metadata must be a dictionary")

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
                "## Best Validation",
                "",
                f"- Best exact match: `{metadata['best_val_exact_match']}`",
                f"- Best validation step: `{metadata['best_val_global_step']}`",
                f"- Best checkpoint: `{best_checkpoint_description(checkpoints)}`",
                f"- Final exact match: `{final_val_metrics.get('val/final/exact_match', 'not available')}`",
                f"- Final character accuracy: `{final_val_metrics.get('val/final/char_accuracy', 'not available')}`",
                f"- Final loss: `{final_val_metrics.get('val/final/loss', 'not available')}`",
                f"- Global step: `{metadata['global_step']}`",
                "",
                "## Validation Summary Artifacts",
                "",
                "- `val/val_final_distortion_summary.json`: validation metrics split by distortion/style flag.",
                "- `val/val_final_distortion_count_summary.json`: validation metrics split by number of active distortions.",
                "- `val/val_final_confusion_matrix.json`: character-level confusion matrix.",
                "- `val/val_final_confusion_matrix.csv`: CSV version of the character-level confusion matrix.",
                "- `val/val_failed_ids.json`: validation example IDs that failed exact match.",
                "- `val/figures/`: validation graphs.",
                "",
                "## Training Summary",
                "",
                f"- Global step: `{metadata['global_step']}`",
                f"- Train split: `{metadata['splits']['train']}`",
                f"- Validation split: `{metadata['splits']['val']}`",
                f"- Train sample limit: `{metadata['sample_counts']['train']}`",
                f"- Validation sample limit: `{metadata['sample_counts']['val']}`",
                "",
                "## Configuration",
                "",
                "The full training and model configuration is included in `training_metadata.json`.",
                "",
                "## Final Validation Metrics",
                "",
                metrics_table(final_val_metrics),
                "",
            ]
        )
    )


def best_checkpoint_description(checkpoints: dict[str, object]) -> str:
    if checkpoints.get("best_is_last"):
        return "same as checkpoints/last.pt; separate best checkpoint omitted"

    best = checkpoints.get("best")
    if not isinstance(best, dict):
        return "not available"

    path = best.get("path", "not available")
    step = best.get("global_step", "not available")
    return f"{path} from step {step}"


def metrics_table(metrics: dict[str, object]) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key in sorted(metrics):
        lines.append(f"| `{key}` | `{metrics[key]}` |")
    return "\n".join(lines)


def evaluation_summary_metadata(output_dir: Path) -> dict[str, object]:
    val_dir = output_dir / "val"
    summary_files = {
        "distortion_summary": val_dir / "val_final_distortion_summary.json",
        "distortion_count_summary": val_dir / "val_final_distortion_count_summary.json",
        "distortion_count_accuracy_figure": val_dir / "figures" / "val_final_distortion_count_accuracy.png",
        "distortion_count_accuracy_figure_svg": val_dir / "figures" / "val_final_distortion_count_accuracy.svg",
        "confusion_matrix": val_dir / "val_final_confusion_matrix.json",
        "confusion_matrix_csv": val_dir / "val_final_confusion_matrix.csv",
        "failed_ids": val_dir / "val_failed_ids.json",
    }
    metadata: dict[str, object] = {}
    for name, path in summary_files.items():
        if not path.exists():
            continue
        metadata[name] = {"path": str(path.relative_to(output_dir))}
        if path.suffix == ".json":
            metadata[name]["content"] = json.loads(path.read_text())
    return metadata
