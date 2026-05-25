from __future__ import annotations

from dataclasses import dataclass, field

ALL_LAYERS = ("conv_block_0", "conv_block_1", "conv_block_2", "pool", "embedding")

# How conv-layer spatial tensors [B, C, H, W] are reduced to [B, features] before probing.
# "global_avg_pool" -> [B, C]       compact; loses position info, keeps channel stats
# "flatten"         -> [B, C*H*W]   large; preserves spatial structure
CONV_REDUCTIONS = ("global_avg_pool", "flatten")

# Supported sklearn classifiers
CLASSIFIERS = ("logistic_regression", "linear_svc")


@dataclass(frozen=True)
class ProbeConfig:
    # Which layers to register hooks on and probe
    layers: tuple[str, ...] = ALL_LAYERS

    # Classifier and its regularisation strength
    classifier: str = "logistic_regression"
    C: float = 1.0
    max_iter: int = 1000

    # How conv-layer outputs are reduced to a feature vector (see CONV_REDUCTIONS)
    conv_reduction: str = "global_avg_pool"

    # Image preprocessing (must match training)
    image_size: tuple[int, int] = (64, 160)
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.classifier not in CLASSIFIERS:
            raise ValueError(f"classifier must be one of {CLASSIFIERS}, got {self.classifier!r}")
        if self.conv_reduction not in CONV_REDUCTIONS:
            raise ValueError(f"conv_reduction must be one of {CONV_REDUCTIONS}, got {self.conv_reduction!r}")
        unknown = set(self.layers) - set(ALL_LAYERS)
        if unknown:
            raise ValueError(f"Unknown layers: {unknown}. Valid: {ALL_LAYERS}")
