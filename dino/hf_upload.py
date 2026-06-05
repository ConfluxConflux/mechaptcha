"""HuggingFace upload for LoRA-fine-tuned dino/CLIP CAPTCHA models."""
from __future__ import annotations

import json
from pathlib import Path

from train.scripts.hf_upload import HuggingFaceUploadInfo, WandbRunInfo, hf_repo_id, create_or_get_hf_repo


def preflight_hf_upload(output_dir: Path) -> HuggingFaceUploadInfo:
    from huggingface_hub import HfApi
    repo_id = _repo_id(output_dir)
    api = HfApi()
    return create_or_get_hf_repo(api, repo_id)


def upload(
    output_dir: Path,
    metrics: dict,
    wandb_run: WandbRunInfo | None = None,
) -> HuggingFaceUploadInfo:
    from huggingface_hub import HfApi
    repo_id = _repo_id(output_dir)
    transcription_accuracy = _load_transcription_accuracy(output_dir)
    _write_model_card(output_dir / "README.md", repo_id, metrics,
                      wandb_run=wandb_run, transcription_accuracy=transcription_accuracy)
    api = HfApi()
    info = create_or_get_hf_repo(api, repo_id)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=info.repo_id,
        repo_type="model",
        commit_message=f"Upload training artifacts for {repo_id}",
    )
    return info


def _repo_id(output_dir: Path) -> str:
    name = output_dir.name if output_dir.is_dir() else output_dir.parent.name
    return hf_repo_id(name)


def _load_transcription_accuracy(output_dir: Path) -> dict | None:
    path = output_dir / "transcription_accuracy.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _fmt(v, fmt=".4f") -> str:
    return f"`{v:{fmt}}`" if isinstance(v, float) else f"`{v}`"


def _write_model_card(
    path: Path,
    repo_id: str,
    metrics: dict,
    wandb_run: WandbRunInfo | None = None,
    transcription_accuracy: dict | None = None,
) -> None:
    backbone = metrics.get("backbone", "unknown")
    model_name = metrics.get("model_name", "unknown")
    frozen = metrics.get("freeze_backbone", False)
    adapter = "frozen backbone (heads only)" if frozen else f"LoRA (r={metrics.get('lora_r', '?')})"
    best_seq_acc = metrics.get("best_val_seq_acc", metrics.get("val_seq_acc", "?"))
    val_char_acc = metrics.get("val_char_acc", "?")
    train_size = metrics.get("train_size", "?")
    epochs = metrics.get("epochs", "?")
    trainable = metrics.get("trainable_params", "?")

    lines = [
        "---",
        "library_name: pytorch",
        "tags:",
        "- captcha",
        "- image-classification",
        f"- {backbone}",
        "- lora" if not frozen else "- frozen-backbone",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"CAPTCHA transcription model: **{model_name}** fine-tuned with {adapter} on the Mechaptcha dataset.",
        "",
    ]

    if wandb_run and wandb_run.url:
        lines += [f"Weights & Biases run: {wandb_run.url}", ""]

    lines += [
        "## Performance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Best val seq accuracy | {_fmt(best_seq_acc)} |",
        f"| Final val char accuracy | {_fmt(val_char_acc)} |",
        "",
    ]

    if transcription_accuracy:
        _CONTROLS = {"dumb_control", "variation_control", "same_data_control", "same_distribution_control"}
        distortions = [(exp, splits) for exp, splits in transcription_accuracy.items() if exp not in _CONTROLS]
        if distortions:
            lines += [
                "### Per-distortion accuracy (test set, seq accuracy)",
                "",
                "| Distortion | Distorted (A) | Clean (B) |",
                "| --- | ---: | ---: |",
            ]
            for exp, splits in sorted(distortions):
                test = splits.get("test") or splits.get("train") or {}
                a = test.get("batch_a_seq_acc")
                b = test.get("batch_b_seq_acc")
                a_str = f"`{a:.1%}`" if isinstance(a, float) else "—"
                b_str = f"`{b:.1%}`" if isinstance(b, float) else "—"
                lines.append(f"| {exp.replace('_', ' ')} | {a_str} | {b_str} |")
            lines.append("")

    lines += [
        "## Training",
        "",
        f"- Backbone: `{model_name}`",
        f"- Adaptation: {adapter}",
        f"- Trainable parameters: `{trainable:,}`" if isinstance(trainable, int) else f"- Trainable parameters: `{trainable}`",
        f"- Train size: `{train_size:,}`" if isinstance(train_size, int) else f"- Train size: `{train_size}`",
        f"- Epochs: `{epochs}`",
        "",
        "Full config in `metrics.json`.",
        "",
    ]

    path.write_text("\n".join(lines))
