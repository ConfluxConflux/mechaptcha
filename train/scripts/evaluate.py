# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )


def metric_counts(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    char_correct = (predictions == labels).sum()
    exact_correct = (predictions == labels).all(dim=1).sum()
    total_chars = torch.tensor(labels.numel(), device=labels.device)
    total_examples = torch.tensor(labels.shape[0], device=labels.device)
    return torch.stack((char_correct, total_chars, exact_correct, total_examples)).float()


def per_example_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    return losses.view(labels.shape).mean(dim=1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    distributed: bool,
    metric_prefix: str,
    channels_last: bool,
    collect_failures: bool = False,
    collect_summary: bool = False,
    collect_distortion_metrics: bool = False,
    alphabet: str | None = None,
) -> dict[str, float | str]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_examples = torch.tensor(0.0, device=device)
    total_counts = torch.zeros(4, device=device)
    failed_ids: list[int] = []
    metadata_columns = tuple(getattr(loader.dataset, "metadata_columns", ()))
    should_collect_distortions = collect_summary or collect_distortion_metrics
    num_classes = len(alphabet or "")
    distortion_stats = torch.zeros((len(metadata_columns), 2, 6), device=device)
    distortion_count_stats = torch.zeros((len(metadata_columns) + 1, 6), device=device)
    confusion_matrix = torch.zeros((num_classes, num_classes), device=device)

    for images, labels, example_ids, metadata in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.detach() * batch_size
        total_examples += batch_size
        total_counts += metric_counts(logits, labels)
        if collect_failures or should_collect_distortions:
            predictions = logits.argmax(dim=-1)
        if collect_failures:
            failed_mask = ~(predictions == labels).all(dim=1)
            failed_ids.extend(example_ids[failed_mask.cpu()].tolist())
        if should_collect_distortions:
            example_losses = per_example_loss(logits, labels) if collect_summary else None
            update_distortion_counts(
                distortion_stats,
                metadata,
                labels,
                predictions,
                example_losses,
            )
            update_distortion_count_counts(
                distortion_count_stats,
                metadata,
                labels,
                predictions,
                example_losses,
            )
        if collect_summary:
            update_confusion_counts(confusion_matrix, labels, predictions)

    values = torch.cat((total_loss.view(1), total_examples.view(1), total_counts))
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if should_collect_distortions:
            dist.all_reduce(distortion_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(distortion_count_stats, op=dist.ReduceOp.SUM)
        if collect_summary:
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        if collect_failures:
            gathered: list[list[int] | None] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, failed_ids)
            failed_ids = [example_id for rank_ids in gathered if rank_ids for example_id in rank_ids]

    loss_sum, example_count, char_correct, char_total, exact_correct, exact_total = values.tolist()
    metrics = {
        f"{metric_prefix}/loss": loss_sum / max(example_count, 1.0),
        f"{metric_prefix}/char_accuracy": char_correct / max(char_total, 1.0),
        f"{metric_prefix}/exact_match": exact_correct / max(exact_total, 1.0),
    }
    if collect_failures:
        metrics[f"{metric_prefix}/failed_exact_count"] = float(len(failed_ids))
        write_failed_ids(loader, failed_ids)
    if should_collect_distortions:
        metrics.update(distortion_scalar_metrics(metric_prefix, metadata_columns, distortion_stats.cpu()))
        metrics.update(distortion_count_scalar_metrics(metric_prefix, distortion_count_stats.cpu()))
    if collect_summary:
        summary_paths = write_evaluation_summary(
            loader=loader,
            metric_prefix=metric_prefix,
            distortion_stats=distortion_stats.cpu(),
            distortion_count_stats=distortion_count_stats.cpu(),
            confusion_matrix=confusion_matrix.cpu(),
            alphabet=alphabet or "",
        )
        metrics.update(summary_file_metrics(metric_prefix, summary_paths))
    return metrics


def update_distortion_counts(
    distortion_stats: torch.Tensor,
    metadata: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    example_losses: torch.Tensor | None,
) -> None:
    correct_chars = (predictions == labels).sum(dim=1).float()
    total_chars = torch.full_like(correct_chars, labels.shape[1], dtype=torch.float)
    exact_correct = (predictions == labels).all(dim=1).float()
    ones = torch.ones_like(correct_chars)

    for column_idx in range(metadata.shape[1]):
        for value_idx, mask in enumerate((~metadata[:, column_idx], metadata[:, column_idx])):
            if not mask.any():
                continue
            distortion_stats[column_idx, value_idx, 0] += mask.sum()
            if example_losses is not None:
                distortion_stats[column_idx, value_idx, 1] += example_losses[mask].sum()
            distortion_stats[column_idx, value_idx, 2] += correct_chars[mask].sum()
            distortion_stats[column_idx, value_idx, 3] += total_chars[mask].sum()
            distortion_stats[column_idx, value_idx, 4] += exact_correct[mask].sum()
            distortion_stats[column_idx, value_idx, 5] += ones[mask].sum()


def update_distortion_count_counts(
    distortion_count_stats: torch.Tensor,
    metadata: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    example_losses: torch.Tensor | None,
) -> None:
    correct_chars = (predictions == labels).sum(dim=1).float()
    total_chars = torch.full_like(correct_chars, labels.shape[1], dtype=torch.float)
    exact_correct = (predictions == labels).all(dim=1).float()
    ones = torch.ones_like(correct_chars)
    distortion_counts = metadata.sum(dim=1)

    for distortion_count in torch.unique(distortion_counts).tolist():
        count_idx = int(distortion_count)
        mask = distortion_counts == count_idx
        distortion_count_stats[count_idx, 0] += mask.sum()
        if example_losses is not None:
            distortion_count_stats[count_idx, 1] += example_losses[mask].sum()
        distortion_count_stats[count_idx, 2] += correct_chars[mask].sum()
        distortion_count_stats[count_idx, 3] += total_chars[mask].sum()
        distortion_count_stats[count_idx, 4] += exact_correct[mask].sum()
        distortion_count_stats[count_idx, 5] += ones[mask].sum()


def update_confusion_counts(confusion_matrix: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> None:
    num_classes = confusion_matrix.shape[0]
    encoded = labels.reshape(-1) * num_classes + predictions.reshape(-1)
    confusion_matrix += torch.bincount(encoded, minlength=num_classes * num_classes).view(num_classes, num_classes)


def write_evaluation_summary(
    loader: DataLoader,
    metric_prefix: str,
    distortion_stats: torch.Tensor,
    distortion_count_stats: torch.Tensor,
    confusion_matrix: torch.Tensor,
    alphabet: str,
) -> dict[str, Path]:
    dataset = loader.dataset
    output_dir = Path(getattr(dataset, "summary_output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = metric_prefix.replace("/", "_")

    distortion_path = output_dir / f"{safe_prefix}_distortion_summary.json"
    distortion_count_path = output_dir / f"{safe_prefix}_distortion_count_summary.json"
    confusion_path = output_dir / f"{safe_prefix}_confusion_matrix.json"
    confusion_csv_path = output_dir / f"{safe_prefix}_confusion_matrix.csv"

    metadata_columns = tuple(getattr(dataset, "metadata_columns", ()))
    distortion_rows = distortion_summary_rows(metadata_columns, distortion_stats)
    distortion_path.write_text(json.dumps(distortion_rows, indent=2, sort_keys=True) + "\n")
    distortion_count_rows = distortion_count_summary_rows(distortion_count_stats)
    distortion_count_path.write_text(json.dumps(distortion_count_rows, indent=2, sort_keys=True) + "\n")
    figure_paths = write_distortion_count_accuracy_figure(output_dir, safe_prefix, distortion_count_rows)

    confusion_payload = {
        "alphabet": alphabet,
        "rows_are_targets": True,
        "columns_are_predictions": True,
        "matrix": confusion_matrix.int().tolist(),
    }
    confusion_path.write_text(json.dumps(confusion_payload, indent=2, sort_keys=True) + "\n")
    write_confusion_csv(confusion_csv_path, alphabet, confusion_matrix)
    return {
        "distortion_summary": distortion_path,
        "distortion_count_summary": distortion_count_path,
        **figure_paths,
        "confusion_matrix": confusion_path,
        "confusion_matrix_csv": confusion_csv_path,
    }


def distortion_summary_rows(metadata_columns: tuple[str, ...], distortion_stats: torch.Tensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column_idx, column in enumerate(metadata_columns):
        for value_idx, present in enumerate((False, True)):
            count, loss_sum, char_correct, char_total, exact_correct, exact_total = distortion_stats[
                column_idx, value_idx
            ].tolist()
            rows.append(
                {
                    "distortion": column,
                    "present": present,
                    "examples": int(count),
                    "loss": loss_sum / max(count, 1.0),
                    "char_accuracy": char_correct / max(char_total, 1.0),
                    "exact_match": exact_correct / max(exact_total, 1.0),
                }
            )
    return rows


def distortion_count_summary_rows(distortion_count_stats: torch.Tensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for distortion_count, stats in enumerate(distortion_count_stats):
        count, loss_sum, char_correct, char_total, exact_correct, exact_total = stats.tolist()
        if count == 0:
            continue
        rows.append(
            {
                "num_distortions": distortion_count,
                "examples": int(count),
                "loss": loss_sum / max(count, 1.0),
                "char_accuracy": char_correct / max(char_total, 1.0),
                "exact_match": exact_correct / max(exact_total, 1.0),
            }
        )
    return rows


def write_distortion_count_accuracy_figure(
    output_dir: Path,
    safe_prefix: str,
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    if not rows:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{safe_prefix}_distortion_count_accuracy.png"
    svg_path = figure_dir / f"{safe_prefix}_distortion_count_accuracy.svg"

    x_values = [int(row["num_distortions"]) for row in rows]
    char_accuracy = [float(row["char_accuracy"]) for row in rows]
    exact_match = [float(row["exact_match"]) for row in rows]
    example_counts = [int(row["examples"]) for row in rows]
    char_error = [
        binomial_standard_error(accuracy, example_count * 5)
        for accuracy, example_count in zip(char_accuracy, example_counts)
    ]
    exact_error = [
        binomial_standard_error(accuracy, example_count) for accuracy, example_count in zip(exact_match, example_counts)
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(x_values, char_accuracy, yerr=char_error, marker="o", linewidth=2, capsize=4, label="Character accuracy")
    ax.errorbar(x_values, exact_match, yerr=exact_error, marker="o", linewidth=2, capsize=4, label="Exact match")
    for x_value, y_value, example_count in zip(x_values, exact_match, example_counts):
        ax.annotate(f"n={example_count}", (x_value, y_value), textcoords="offset points", xytext=(0, -16), ha="center")
    ax.set_xlabel("# active perturbations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation accuracy by number of active perturbations")
    ax.set_xticks(x_values)
    ax.set_ylim(0.0, 1.01)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)
    plt.close(fig)
    return {
        "distortion_count_accuracy_figure": png_path,
        "distortion_count_accuracy_figure_svg": svg_path,
    }


def binomial_standard_error(accuracy: float, trials: int) -> float:
    return math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / max(trials, 1))


def write_confusion_csv(path: Path, alphabet: str, confusion_matrix: torch.Tensor) -> None:
    lines = ["," + ",".join(alphabet)]
    for row_idx, char in enumerate(alphabet):
        values = [str(int(value)) for value in confusion_matrix[row_idx].tolist()]
        lines.append(",".join((char, *values)))
    path.write_text("\n".join(lines) + "\n")


def summary_file_metrics(
    metric_prefix: str,
    summary_paths: dict[str, Path],
) -> dict[str, str]:
    return {
        f"{metric_prefix}/distortion_summary_path": str(summary_paths["distortion_summary"]),
        f"{metric_prefix}/distortion_count_summary_path": str(summary_paths["distortion_count_summary"]),
        f"{metric_prefix}/confusion_matrix_path": str(summary_paths["confusion_matrix"]),
        f"{metric_prefix}/confusion_matrix_csv_path": str(summary_paths["confusion_matrix_csv"]),
        **{
            f"{metric_prefix}/{name}_path": str(path)
            for name, path in summary_paths.items()
            if name.startswith("distortion_count_accuracy_figure")
        },
    }


def distortion_scalar_metrics(
    metric_prefix: str,
    metadata_columns: tuple[str, ...],
    distortion_stats: torch.Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for column_idx, column in enumerate(metadata_columns):
        present = distortion_stats[column_idx, 1]
        absent = distortion_stats[column_idx, 0]
        present_char = present[2].item() / max(present[3].item(), 1.0)
        absent_char = absent[2].item() / max(absent[3].item(), 1.0)
        present_exact = present[4].item() / max(present[5].item(), 1.0)
        absent_exact = absent[4].item() / max(absent[5].item(), 1.0)
        metrics[f"{metric_prefix}/by_distortion/{column}/present_char_accuracy"] = present_char
        metrics[f"{metric_prefix}/by_distortion/{column}/absent_char_accuracy"] = absent_char
        metrics[f"{metric_prefix}/by_distortion/{column}/char_accuracy_delta"] = present_char - absent_char
        metrics[f"{metric_prefix}/by_distortion/{column}/present_exact_match"] = present_exact
        metrics[f"{metric_prefix}/by_distortion/{column}/absent_exact_match"] = absent_exact
        metrics[f"{metric_prefix}/by_distortion/{column}/exact_match_delta"] = present_exact - absent_exact
    return metrics


def distortion_count_scalar_metrics(
    metric_prefix: str,
    distortion_count_stats: torch.Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for distortion_count, stats in enumerate(distortion_count_stats):
        count, _loss_sum, char_correct, char_total, exact_correct, exact_total = stats.tolist()
        if count == 0:
            continue
        metrics[f"{metric_prefix}/by_distortion_count/{distortion_count}/examples"] = count
        metrics[f"{metric_prefix}/by_distortion_count/{distortion_count}/char_accuracy"] = char_correct / max(
            char_total, 1.0
        )
        metrics[f"{metric_prefix}/by_distortion_count/{distortion_count}/exact_match"] = exact_correct / max(
            exact_total, 1.0
        )
    return metrics


def write_failed_ids(loader: DataLoader, failed_ids: list[int]) -> None:
    dataset = loader.dataset
    output_path = getattr(dataset, "failed_ids_path", None)
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_ids = sorted(set(failed_ids))
    path.write_text(json.dumps(unique_ids, indent=2) + "\n")


def log_wandb_summary_artifacts(run: Any | None, output_dir: Path) -> None:
    if run is None:
        return

    val_dir = output_dir / "val"
    artifact_paths = [
        val_dir / "val_final_distortion_summary.json",
        val_dir / "val_final_distortion_count_summary.json",
        val_dir / "val_final_confusion_matrix.json",
        val_dir / "val_final_confusion_matrix.csv",
        val_dir / "val_failed_ids.json",
        val_dir / "figures" / "val_final_distortion_count_accuracy.png",
        val_dir / "figures" / "val_final_distortion_count_accuracy.svg",
    ]
    existing_paths = [path for path in artifact_paths if path.exists()]
    if not existing_paths:
        return

    import wandb

    artifact = wandb.Artifact(f"{getattr(run, 'name', None) or 'run'}-validation-summary", type="validation-summary")
    for path in existing_paths:
        artifact.add_file(str(path), name=str(path.relative_to(output_dir)))
    run.log_artifact(artifact)
