from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from train.model.config import CaptchaModelConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv(images)
        activated = self.activation(features)
        return self.pool(activated)


class CaptchaCNN(nn.Module):
    def __init__(self, config: CaptchaModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CaptchaModelConfig()

        channels = (self.config.image_channels, *self.config.conv_channels)
        self.features = nn.Sequential(
            OrderedDict(
                (f"conv_block_{idx}", ConvBlock(channels[idx], channels[idx + 1]))
                for idx in range(len(self.config.conv_channels))
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(self.config.pooled_shape)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.flattened_features, self.config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.character_heads = nn.ModuleList(
            nn.Linear(self.config.embedding_dim, self.config.num_classes)
            for _ in range(self.config.num_chars)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features)
        embedding = self.embedding(pooled)
        logits = [head(embedding) for head in self.character_heads]
        return torch.stack(logits, dim=1)
