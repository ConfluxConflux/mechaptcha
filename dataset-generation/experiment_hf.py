"""Hugging Face Hub export for paired CAPTCHA probe experiments."""
from __future__ import annotations

import csv
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value

from training.distortions import ALL_DISTORTION_KEYS

CONTROL_EXPERIMENTS = {"same_data_control", "same_distribution_control"}
DEFAULT_MODEL_REPO = "siddharthmb/2026.mechaptcha.full-accuracy-target-90p-val-distortion-stats-20260525"


@dataclass(frozen=True)
class ExperimentHubConfig:
    repo_id: str | None
    private: bool
    source_dir: Path
    n: int
    distortions: tuple[str, ...]
    model_repo: str


@dataclass(frozen=True)
class ExperimentHubUpload:
    repo_id: str
    repo_url: str
    split_counts: dict[str, int]
    experiment_counts: dict[str, int]


def preflight_experiment_hf_upload(repo_id: str | None, private: bool) -> ExperimentHubUpload:
    """Verify that the target HF dataset repo can be created or written."""
    from huggingface_hub import HfApi

    api = HfApi()
    resolved_repo_id = repo_id or default_repo_id(api)
    repo_url = api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=private, exist_ok=True)
    resolved_repo_id = getattr(repo_url, "repo_id", None) or repo_id_from_url(str(repo_url)) or resolved_repo_id
    return ExperimentHubUpload(
        repo_id=resolved_repo_id,
        repo_url=str(repo_url),
        split_counts={},
        experiment_counts={},
    )


def push_experiments_to_hub(config: ExperimentHubConfig) -> ExperimentHubUpload:
    """Build a dataset repo from generated paired experiments and push it to HF."""
    from huggingface_hub import HfApi

    upload_info = preflight_experiment_hf_upload(config.repo_id, config.private)
    resolved_repo_id = upload_info.repo_id
    api = HfApi()

    dataset = build_experiment_dataset(config.source_dir)
    split_counts = {split: len(ds) for split, ds in dataset.items()}
    experiment_counts = experiment_counts_from_dataset(dataset)

    dataset.push_to_hub(resolved_repo_id, private=config.private)
    upload_card_and_metadata(
        api=api,
        repo_id=resolved_repo_id,
        config=config,
        split_counts=split_counts,
        experiment_counts=experiment_counts,
    )

    return ExperimentHubUpload(
        repo_id=resolved_repo_id,
        repo_url=upload_info.repo_url,
        split_counts=split_counts,
        experiment_counts=experiment_counts,
    )


def build_experiment_dataset(experiments_dir: Path) -> DatasetDict:
    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for experiment_dir in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        labels_path = experiment_dir / "labels.csv"
        if not labels_path.exists():
            continue
        for row in read_experiment_rows(experiment_dir, labels_path):
            split = str(row["split"])
            if split not in rows_by_split:
                rows_by_split[split] = []
            rows_by_split[split].append(row)

    features = experiment_features()
    return DatasetDict(
        {
            split: Dataset.from_list(rows, features=features)
            for split, rows in rows_by_split.items()
            if rows
        }
    )


def read_experiment_rows(experiment_dir: Path, labels_path: Path) -> list[dict[str, object]]:
    experiment = experiment_dir.name
    target_distortion = None if experiment in CONTROL_EXPERIMENTS else experiment
    control_type = experiment if experiment in CONTROL_EXPERIMENTS else "controlled_distortion"

    rows: list[dict[str, object]] = []
    with labels_path.open(newline="") as f:
        for row in csv.DictReader(f):
            seed_id = int(row["id"])
            item: dict[str, object] = {
                "experiment": experiment,
                "target_distortion": target_distortion or "",
                "control_type": control_type,
                "id": seed_id,
                "text": row["text"],
                "font": row["font"],
                "split": row["split"],
                "image_a": str(experiment_dir / "batch_a" / "images" / f"{seed_id:06d}.png"),
                "image_b": str(experiment_dir / "batch_b" / "images" / f"{seed_id:06d}.png"),
            }
            item.update({f"a_{key}": bool(int(row[f"a_{key}"])) for key in ALL_DISTORTION_KEYS})
            item.update({f"b_{key}": bool(int(row[f"b_{key}"])) for key in ALL_DISTORTION_KEYS})
            rows.append(item)
    return rows


def experiment_features() -> Features:
    return Features(
        {
            "experiment": Value("string"),
            "target_distortion": Value("string"),
            "control_type": Value("string"),
            "id": Value("int32"),
            "text": Value("string"),
            "font": Value("string"),
            "split": Value("string"),
            "image_a": Image(),
            "image_b": Image(),
            **{f"a_{key}": Value("bool") for key in ALL_DISTORTION_KEYS},
            **{f"b_{key}": Value("bool") for key in ALL_DISTORTION_KEYS},
        }
    )


def upload_card_and_metadata(
    api: object,
    repo_id: str,
    config: ExperimentHubConfig,
    split_counts: dict[str, int],
    experiment_counts: dict[str, int],
) -> None:
    metadata = {
        "repo_id": repo_id,
        "source_dir": str(config.source_dir),
        "n_seed_pairs": config.n,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_repo": config.model_repo,
        "distortions": list(config.distortions),
        "control_experiments": sorted(CONTROL_EXPERIMENTS),
        "split_counts": split_counts,
        "experiment_counts": experiment_counts,
        "schema": {
            "image_a": "Batch A image for this paired probe example.",
            "image_b": "Batch B image for this paired probe example.",
            "label_convention": "Probe label is 1 for image_a and 0 for image_b.",
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        readme = tmp_path / "README.md"
        metadata_path = tmp_path / "experiment_metadata.json"
        readme.write_text(model_card(repo_id, metadata), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="experiment_metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add experiment metadata",
        )


def model_card(repo_id: str, metadata: dict[str, object]) -> str:
    split_counts = metadata["split_counts"]
    experiment_counts = metadata["experiment_counts"]
    if not isinstance(split_counts, dict) or not isinstance(experiment_counts, dict):
        raise TypeError("split_counts and experiment_counts must be dictionaries")

    split_rows = "\n".join(f"| `{name}` | {count:,} |" for name, count in split_counts.items())
    experiment_rows = "\n".join(f"| `{name}` | {count:,} |" for name, count in sorted(experiment_counts.items()))
    distortion_list = ", ".join(f"`{name}`" for name in metadata["distortions"])
    model_repo = metadata["model_repo"]

    return "\n".join(
        [
            "---",
            "license: mit",
            "task_categories:",
            "- image-classification",
            "tags:",
            "- captcha",
            "- mechanistic-interpretability",
            "- linear-probe",
            "- paired-dataset",
            "pretty_name: Mechaptcha linear probe experiments",
            "---",
            "",
            f"# {repo_id}",
            "",
            "Paired CAPTCHA image experiments for linear probes over a trained Mechaptcha CNN.",
            "Each example contains a matched `image_a` and `image_b` pair generated from the same seed pool.",
            "",
            "## Intended Use",
            "",
            "This dataset is designed for linear probe experiments that compare activations from Batch A against Batch B.",
            "Use label `1` for `image_a` and label `0` for `image_b`.",
            "",
            "Recommended checkpoint:",
            "",
            f"- `{model_repo}`",
            "",
            "Example command:",
            "",
            "```bash",
            "uv run python -m probe.run \\",
            f"    --checkpoint {model_repo} \\",
            f"    --experiments {repo_id} \\",
            "    --activations probe_results/activations \\",
            "    --output probe",
            "```",
            "",
            "## Dataset Structure",
            "",
            "Columns:",
            "",
            "- `experiment`: experiment name, such as `easy_line`, `blur`, or `same_data_control`.",
            "- `target_distortion`: distortion being isolated for controlled experiments; empty for controls.",
            "- `control_type`: `controlled_distortion`, `same_data_control`, or `same_distribution_control`.",
            "- `id`, `text`, `font`, `split`: seed metadata shared by the pair.",
            "- `image_a`, `image_b`: paired CAPTCHA images.",
            "- `a_*`, `b_*`: boolean distortion flags used for each batch.",
            "",
            "## Splits",
            "",
            "| Split | Rows |",
            "| --- | ---: |",
            split_rows,
            "",
            "## Experiments",
            "",
            "| Experiment | Rows |",
            "| --- | ---: |",
            experiment_rows,
            "",
            "Generated controlled distortions:",
            "",
            distortion_list or "None",
            "",
            "Control experiments:",
            "",
            "- `same_data_control`: Batch A and Batch B are identical images.",
            "- `same_distribution_control`: Batch A and Batch B are independent samples from the same distribution.",
            "",
            "## Provenance",
            "",
            f"- Source directory at upload time: `{metadata['source_dir']}`",
            f"- Seed pairs per experiment requested: `{metadata['n_seed_pairs']}`",
            f"- Generated at: `{metadata['generated_at']}`",
            "",
            "Additional machine-readable metadata is in `experiment_metadata.json`.",
            "",
        ]
    )


def experiment_counts_from_dataset(dataset: DatasetDict) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for split_dataset in dataset.values():
        counts.update(split_dataset["experiment"])
    return dict(counts)


def default_repo_id(api: object) -> str:
    whoami = api.whoami()
    namespace = whoami.get("name")
    if not namespace:
        raise RuntimeError("Could not infer Hugging Face username; pass --hf-repo-id explicitly.")
    return f"{namespace}/{default_repo_name()}"


def default_repo_name() -> str:
    now = datetime.now(UTC)
    return f"{now:%Y}.mechaptcha.linear-probe-experiments-{now:%Y%m%d}"


def repo_id_from_url(repo_url: str) -> str | None:
    match = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", repo_url)
    if match:
        return match.group(1)
    match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", repo_url)
    return match.group(1) if match else None
