from __future__ import annotations

from dataclasses import dataclass

from train.model.charset import DEFAULT_ALPHABET

DEFAULT_NUM_CHARS = 5


@dataclass(frozen=True)
class CaptchaModelConfig:
    image_channels: int = 1
    num_chars: int = DEFAULT_NUM_CHARS
    alphabet: str = DEFAULT_ALPHABET
    conv_channels: tuple[int, ...] = (64, 128, 256, 384)
    pooled_shape: tuple[int, int] = (4, 10)
    embedding_dim: int = 512
    dropout: float = 0.1

    @property
    def num_classes(self) -> int:
        return len(self.alphabet)

    @property
    def flattened_features(self) -> int:
        channels = self.conv_channels[-1]
        height, width = self.pooled_shape
        return channels * height * width
