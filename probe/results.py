"""ProbeResult type and utilities for saving, loading, and formatting results."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from probe.config import ALL_LAYERS

# experiment name -> layer name -> ProbeResult
AllResults = dict[str, dict[str, "ProbeResult"]]


@dataclass(frozen=True)
class ProbeResult:
    train_acc: float
    test_acc: float
    # Fraction of probe weights that are non-zero; None for dense classifiers.
    sparsity: float | None = None

    @property
    def test_delta(self) -> float:
        """How far above chance (50%) the test accuracy is."""
        return self.test_acc - 0.5

    def to_dict(self) -> dict:
        d: dict = {"train_acc": self.train_acc, "test_acc": self.test_acc}
        if self.sparsity is not None:
            d["sparsity"] = self.sparsity
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProbeResult:
        return cls(
            train_acc=d["train_acc"],
            test_acc=d["test_acc"],
            sparsity=d.get("sparsity"),
        )


def save_results(results: AllResults, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        exp: {layer: result.to_dict() for layer, result in layer_results.items()}
        for exp, layer_results in results.items()
    }
    path.write_text(json.dumps(serialisable, indent=2))


def load_results(path: Path) -> AllResults:
    raw = json.loads(path.read_text())
    return {
        exp: {layer: ProbeResult.from_dict(d) for layer, d in layer_results.items()}
        for exp, layer_results in raw.items()
    }


_COL = 16  # column width for the table


def _fmt(result: ProbeResult) -> str:
    sign = "+" if result.test_delta >= 0 else ""
    return f"{result.test_acc:.1%}({sign}{result.test_delta:.1%})"


def format_table(results: AllResults, layers: tuple[str, ...] = ALL_LAYERS) -> str:
    """Return a human-readable table of test accuracy vs chance per experiment × layer."""
    header = f"{'experiment':<32}" + "".join(f"{l:>{_COL}}" for l in layers)
    sep = "-" * len(header)
    rows = [header, sep]

    # Controls first, then experiments alphabetically
    def sort_key(name: str) -> tuple[int, str]:
        if "control" in name:
            return (0, name)
        return (1, name)

    for exp_name in sorted(results, key=sort_key):
        layer_results = results[exp_name]
        row = f"{exp_name:<32}"
        for layer in layers:
            if layer in layer_results:
                row += f"{_fmt(layer_results[layer]):>{_COL}}"
            else:
                row += f"{'—':>{_COL}}"
        rows.append(row)

    rows.append(sep)
    rows.append("(test accuracy; +/- from 50% chance)")
    return "\n".join(rows)
