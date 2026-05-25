"""Train and evaluate linear probes on cached activations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LAYER_NAMES = ["conv_block_0", "conv_block_1", "conv_block_2", "pool", "embedding"]


def train_probe(
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
) -> dict[str, float]:
    """Fit logistic regression to distinguish batch_a (label=1) from batch_b (label=0)."""
    X_train = np.concatenate([train_a, train_b])
    y_train = np.array([1] * len(train_a) + [0] * len(train_b))
    X_test = np.concatenate([test_a, test_b])
    y_test = np.array([1] * len(test_a) + [0] * len(test_b))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train_s, y_train)

    return {
        "train_acc": float(clf.score(X_train_s, y_train)),
        "test_acc": float(clf.score(X_test_s, y_test)),
    }


def probe_experiment(activations_dir: Path | str) -> dict[str, dict[str, float]]:
    """Load saved activations for one experiment and train a probe at every layer."""
    activations_dir = Path(activations_dir)
    results: dict[str, dict[str, float]] = {}

    for layer in LAYER_NAMES:
        files = {
            split_batch: activations_dir / f"{split_batch}_{layer}.npy"
            for split_batch in ("train_batch_a", "train_batch_b", "test_batch_a", "test_batch_b")
        }
        if not all(p.exists() for p in files.values()):
            continue

        results[layer] = train_probe(
            train_a=np.load(files["train_batch_a"]),
            train_b=np.load(files["train_batch_b"]),
            test_a=np.load(files["test_batch_a"]),
            test_b=np.load(files["test_batch_b"]),
        )

    return results
