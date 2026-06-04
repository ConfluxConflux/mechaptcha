"""DINOv2 backbone + LoRA adapters + per-character transcription heads.

The backbone is loaded pretrained and wrapped with LoRA so that fine-tuning on the
CAPTCHA transcription task reshapes its behavior (and intermediate representations)
without training from scratch. Five linear heads read a pooled embedding and predict
one character each — identical output shape to the original CaptchaCNN ([B, 5, 26]).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dino.config import DinoConfig

# Per-backbone input normalisation (applied to the grayscale CAPTCHA replicated to
# 3 channels). DINOv2 uses ImageNet stats; CLIP uses its own.
_NORM_STATS = {
    "dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}


def build_transform(config: DinoConfig) -> transforms.Compose:
    """Preprocessing for feeding grayscale CAPTCHAs to the RGB ViT backbone."""
    mean, std = _NORM_STATS[config.backbone]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # replicate gray -> RGB
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_backbone(config: DinoConfig, pretrained: bool):
    """Load the pretrained ViT vision tower for the configured backbone family."""
    if config.backbone == "dinov2":
        from transformers import Dinov2Config, Dinov2Model
        if pretrained:
            return Dinov2Model.from_pretrained(config.model_name)
        return Dinov2Model(Dinov2Config.from_pretrained(config.model_name))
    if config.backbone == "clip":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        if pretrained:
            return CLIPVisionModel.from_pretrained(config.model_name)
        return CLIPVisionModel(CLIPVisionConfig.from_pretrained(config.model_name))
    raise ValueError(f"Unknown backbone: {config.backbone!r}")


class DinoCaptchaModel(nn.Module):
    """LoRA-adapted DINOv2 with 5 character-classification heads."""

    def __init__(self, config: DinoConfig | None = None, *, pretrained: bool = True) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.config = config or DinoConfig()

        backbone = _load_backbone(self.config, pretrained)

        patch = backbone.config.patch_size
        if self.config.image_size % patch != 0:
            raise ValueError(
                f"image_size {self.config.image_size} must be a multiple of the "
                f"{self.config.model_name} patch size {patch}."
            )

        lora = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.resolved_lora_targets),
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora)

        hidden = self.backbone.config.hidden_size
        self.head_dropout = nn.Dropout(self.config.dropout)
        self.character_heads = nn.ModuleList(
            nn.Linear(hidden, self.config.num_classes) for _ in range(self.config.num_chars)
        )

    @property
    def num_blocks(self) -> int:
        return self.backbone.config.num_hidden_layers

    def _pool(self, last_hidden_state: torch.Tensor, how: str) -> torch.Tensor:
        # last_hidden_state: [B, 1 + num_patches, hidden]; index 0 is the CLS token.
        if how == "cls":
            return last_hidden_state[:, 0]
        return last_hidden_state[:, 1:].mean(dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = self._pool(outputs.last_hidden_state, self.config.head_pooling)
        pooled = self.head_dropout(pooled)
        logits = [head(pooled) for head in self.character_heads]
        return torch.stack(logits, dim=1)  # [B, num_chars, num_classes]

    @torch.no_grad()
    def block_features(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return reduced features per block + pooled embedding + logits.

        Keys: ``block_0`` … ``block_{L-1}`` (token-reduced hidden states), ``embedding``
        (the pooled token the heads read), and ``logits`` ([B, num_chars*num_classes]).
        The ``input`` raw-pixel baseline is produced separately in extract.py so it can
        match the CNN's grayscale baseline rather than the 224px RGB ViT input.
        """
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        # hidden_states: tuple of length (num_blocks + 1); [0] is the embedding output,
        # [i] is the output of transformer block i-1. We label block_{i} = hidden_states[i+1].
        hidden_states = outputs.hidden_states
        reduction = self.config.token_reduction

        features: dict[str, torch.Tensor] = {}
        for i in range(self.num_blocks):
            features[f"block_{i}"] = self._pool(hidden_states[i + 1], reduction)

        features["embedding"] = self._pool(outputs.last_hidden_state, self.config.head_pooling)

        pooled = features["embedding"]
        logits = torch.stack([head(pooled) for head in self.character_heads], dim=1)
        features["logits"] = logits.flatten(start_dim=1)
        return features


def save_checkpoint(model: DinoCaptchaModel, path: str | Path) -> None:
    """Persist LoRA adapters + heads + config (not the frozen base weights)."""
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save only trainable params (LoRA deltas + heads); base weights come from HF.
    trainable = {k: v.cpu() for k, v in model.state_dict().items()
                 if "lora_" in k or k.startswith("character_heads")}
    torch.save({"dino_config": asdict(model.config), "trainable": trainable}, path)


def load_checkpoint(path: str | Path, *, map_location="cpu") -> DinoCaptchaModel:
    """Rebuild a DinoCaptchaModel from a saved checkpoint (re-downloads base weights)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    config = DinoConfig(**ckpt["dino_config"])
    model = DinoCaptchaModel(config, pretrained=True)
    missing, unexpected = model.load_state_dict(ckpt["trainable"], strict=False)
    unexpected = [k for k in unexpected]
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading DINO checkpoint: {unexpected[:5]}")
    model.eval()
    return model
