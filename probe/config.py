from __future__ import annotations

from dataclasses import dataclass, field

# Layers backed by forward hooks on the CNN modules.
# conv_block_N names are generated dynamically from the model; list up to _MAX_CONV_BLOCKS.
_MAX_CONV_BLOCKS = 8
HOOK_LAYERS = tuple(f"conv_block_{i}" for i in range(_MAX_CONV_BLOCKS)) + ("pool", "embedding")

# Special layers captured directly during the forward pass (no hook module)
#   "input"  — raw image pixels flattened to [B, C*H*W]; baseline before any processing
#   "logits" — model output [B, 5, 26] flattened to [B, 130]; checks distortion leakage into predictions
SPECIAL_LAYERS = ("input", "logits")

ALL_LAYERS = ("input",) + HOOK_LAYERS + ("logits",)

# How conv-layer spatial tensors [B, C, H, W] are reduced to [B, features] before probing.
# "global_avg_pool" -> [B, C]       compact; loses position info, keeps channel stats
# "flatten"         -> [B, C*H*W]   large; preserves spatial structure
CONV_REDUCTIONS = ("global_avg_pool", "flatten")

# Supported sklearn classifiers
CLASSIFIERS = ("logistic_regression", "linear_svc", "mlp")


@dataclass(frozen=True)
class ProbeConfig:
    # Which layers to probe (subset of ALL_LAYERS)
    layers: tuple[str, ...] = HOOK_LAYERS

    # Classifier and its regularisation strength
    classifier: str = "logistic_regression"
    C: float = 1.0
    max_iter: int = 1000

    # MLP hidden layer sizes (only used when classifier="mlp")
    mlp_hidden_sizes: tuple[int, ...] = (64, 32)

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
        # Layer names are not restricted to the CNN's ALL_LAYERS: other backbones
        # (e.g. the DINOv2 ViT in dino/) probe layers like 'block_0' … 'block_N'.
        # fit/results/plot key purely on layer names, so any string is valid here.

    def for_model(self, model) -> "ProbeConfig":
        """Return a copy with layers filtered to those that actually exist in model."""
        available = get_model_layers(model)
        filtered = tuple(l for l in self.layers if l in available)
        return ProbeConfig(
            layers=filtered,
            classifier=self.classifier,
            C=self.C,
            max_iter=self.max_iter,
            mlp_hidden_sizes=self.mlp_hidden_sizes,
            conv_reduction=self.conv_reduction,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )


def get_model_layers(model) -> tuple[str, ...]:
    """Return the ordered probe-able layer names for a loaded CaptchaCNN."""
    conv_names = tuple(name for name, _ in model.features.named_children())
    return ("input",) + conv_names + ("pool", "embedding", "logits")
