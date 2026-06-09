"""Train linear probes on cached activations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from probe.config import ProbeConfig
from probe.results import ProbeResult


def _make_classifier(config: ProbeConfig):
    """Return a fresh, untrained sklearn classifier based on config."""
    if config.classifier == "logistic_regression":
        return LogisticRegression(C=config.C, max_iter=config.max_iter, solver="lbfgs")
    elif config.classifier == "linear_svc":
        return LinearSVC(C=config.C, max_iter=config.max_iter)
    elif config.classifier == "mlp":
        # alpha is the L2 penalty; inverse of C to keep the interface consistent
        return MLPClassifier(
            hidden_layer_sizes=config.mlp_hidden_sizes,
            max_iter=config.max_iter,
            alpha=1.0 / config.C,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=0,
        )
    elif config.classifier == "sparse_logistic":
        # L1 penalty (l1_ratio=1) drives most weights to zero, revealing which
        # features encode the distortion. saga supports l1_ratio; liblinear does not.
        return LogisticRegression(
            C=config.C, max_iter=config.max_iter,
            l1_ratio=1.0, solver="saga",
        )
    raise ValueError(f"Unknown classifier: {config.classifier!r}")


def train_probe(
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    config: ProbeConfig,
) -> ProbeResult:
    """Fit a probe to distinguish batch_a (label=1) from batch_b (label=0)."""
    X_train = np.concatenate([train_a, train_b])
    y_train = np.array([1] * len(train_a) + [0] * len(train_b))
    X_test = np.concatenate([test_a, test_b])
    y_test = np.array([1] * len(test_a) + [0] * len(test_b))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = _make_classifier(config)
    clf.fit(X_train_s, y_train)

    sparsity = None
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        sparsity = float(np.count_nonzero(coef) / coef.size)

    return ProbeResult(
        train_acc=float(clf.score(X_train_s, y_train)),
        test_acc=float(clf.score(X_test_s, y_test)),
        sparsity=sparsity,
    )


def probe_experiment(
    activations_dir: Path,
    config: ProbeConfig,
) -> dict[str, ProbeResult]:
    """Load saved activations for one experiment and train a probe at every configured layer."""
    results: dict[str, ProbeResult] = {}

    for layer in config.layers:
        files = {
            key: activations_dir / f"{key}_{layer}.npy"
            for key in ("train_batch_a", "train_batch_b", "test_batch_a", "test_batch_b")
        }
        if not all(p.exists() for p in files.values()):
            continue

        results[layer] = train_probe(
            train_a=np.load(files["train_batch_a"]),
            train_b=np.load(files["train_batch_b"]),
            test_a=np.load(files["test_batch_a"]),
            test_b=np.load(files["test_batch_b"]),
            config=config,
        )

    return results
