"""Configuration for the DINOv2-backbone CAPTCHA transfer experiment.

This mirrors train.model.config (CaptchaModelConfig) but for a frozen-pretrained
ViT backbone adapted with LoRA. The goal is to reproduce the original probe
finding on a model we did NOT train from scratch: a backbone that is *behaviorally*
invariant to the distortion (correct transcription on both batch_a and batch_b)
yet may still linearly encode the distortion in its intermediate blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from train.model.charset import DEFAULT_ALPHABET

# Token reductions for collapsing a ViT hidden state [B, 1+P, H] to [B, features]
# before probing. Analogous to probe.config.CONV_REDUCTIONS for the CNN.
#   "mean" -> mean over the P patch tokens (excludes CLS); the closest analog of
#             the CNN's global-average-pool. Keeps channel stats, drops position.
#   "cls"  -> the CLS summary token [:, 0]; what the transcription heads read from.
TOKEN_REDUCTIONS = ("mean", "cls")

# Supported pretrained ViT backbone families. Both are ViTs that expose per-block
# hidden states and a CLS token at index 0, so the probe pipeline is identical;
# they differ only in how they load, their LoRA target module names, and their
# input normalisation (handled in dino/model.py).
#   "dinov2" -> self-supervised (facebook/dinov2-*), LoRA on query/value
#   "clip"   -> language-supervised vision tower (openai/clip-vit-*), LoRA on q_proj/v_proj
BACKBONES = ("dinov2", "clip", "timm")

# Default LoRA attention targets per backbone (matched by module-name suffix).
_DEFAULT_LORA_TARGETS = {
    "dinov2": ("query", "value"),
    "clip": ("q_proj", "v_proj"),
    # timm ViTs use a single fused QKV linear; "qkv" matches blocks.N.attn.qkv.
    "timm": ("qkv",),
}


@dataclass(frozen=True)
class DinoConfig:
    """Backbone + adapter + head configuration for DinoCaptchaModel."""

    # Pretrained ViT backbone family (see BACKBONES).
    backbone: str = "dinov2"

    # HF model id for the backbone. Smaller = faster probing across depth.
    #   facebook/dinov2-small        -> ViT-S/14, hidden 384,  12 blocks
    #   facebook/dinov2-base         -> ViT-B/14, hidden 768,  12 blocks
    #   facebook/dinov2-large        -> ViT-L/14, hidden 1024, 24 blocks
    #   openai/clip-vit-base-patch16 -> CLIP ViT-B/16, hidden 768,  12 blocks
    #   openai/clip-vit-large-patch14-> CLIP ViT-L/14, hidden 1024, 24 blocks
    #   vit_base_patch16_224         -> timm supervised ViT-B/16, hidden 768, 12 blocks
    model_name: str = "facebook/dinov2-small"

    # CAPTCHA transcription target (must match the dataset labels).
    num_chars: int = 5
    alphabet: str = DEFAULT_ALPHABET

    # Square input side fed to the ViT. Must be a multiple of the patch size (14).
    image_size: int = 224

    # LoRA hyperparameters. Adapters are injected so the transcription loss can
    # reshape the backbone's behavior (and intermediate reps) WITHOUT training it
    # from scratch — this is what restores the behavioral-invariance pressure the
    # frozen backbone would otherwise never feel.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # peft matches these by module-name suffix. Empty tuple = resolve per backbone
    # (see resolved_lora_targets): query/value for DINOv2, q_proj/v_proj for CLIP.
    lora_target_modules: tuple[str, ...] = ()

    # Which token to feed the transcription heads. "cls" is the natural ViT summary.
    head_pooling: str = "cls"

    # How intermediate hidden states are reduced before probing (see TOKEN_REDUCTIONS).
    token_reduction: str = "mean"

    dropout: float = 0.0

    @property
    def num_classes(self) -> int:
        return len(self.alphabet)

    @property
    def resolved_lora_targets(self) -> tuple[str, ...]:
        """LoRA target modules, defaulting to the backbone's attention projections."""
        return self.lora_target_modules or _DEFAULT_LORA_TARGETS[self.backbone]

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if self.head_pooling not in TOKEN_REDUCTIONS:
            raise ValueError(f"head_pooling must be one of {TOKEN_REDUCTIONS}, got {self.head_pooling!r}")
        if self.token_reduction not in TOKEN_REDUCTIONS:
            raise ValueError(f"token_reduction must be one of {TOKEN_REDUCTIONS}, got {self.token_reduction!r}")
        # image_size is validated against the backbone's actual patch size in model.py.


def block_layer_names(num_blocks: int) -> tuple[str, ...]:
    """Probe-able layer names for a DINOv2 of the given depth.

    Mirrors probe.config layer naming so the existing fit/results/plot code works
    unchanged: a raw-pixel baseline, one entry per transformer block, the pooled
    embedding the heads read, and the transcription logits.
    """
    return ("input",) + tuple(f"block_{i}" for i in range(num_blocks)) + ("embedding", "logits")
