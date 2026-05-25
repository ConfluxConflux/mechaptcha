# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from train.model.charset import encode_text
from train.scripts.config import TrainConfig


@dataclass(frozen=True)
class DatasetBundle:
    train: TorchDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    val: TorchDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class MechaptchaDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: Dataset,
        image_column: str,
        text_column: str,
        id_column: str,
        image_size: tuple[int, int],
        alphabet: str,
        num_chars: int,
    ) -> None:
        self.dataset = dataset
        self.image_column = image_column
        self.text_column = text_column
        self.id_column = id_column
        self.image_size = image_size
        self.alphabet = alphabet
        self.num_chars = num_chars
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.dataset[index]
        text = example[self.text_column]
        if len(text) != self.num_chars:
            raise ValueError(f"Expected {self.num_chars} characters, got {len(text)} for label {text!r}")

        image_tensor = self.transform(example[self.image_column])
        label_tensor = torch.tensor(encode_text(text, self.alphabet), dtype=torch.long)
        id_tensor = torch.tensor(int(example[self.id_column]), dtype=torch.long)
        return image_tensor, label_tensor, id_tensor


def load_mechaptcha_datasets(config: TrainConfig) -> DatasetBundle:
    train_split = load_hf_split(config, config.train_split, config.train_size)
    val_split = load_hf_split(config, config.val_split, config.val_size)
    image_size = (config.image_height, config.image_width)

    return DatasetBundle(
        train=MechaptchaDataset(
            dataset=train_split,
            image_column=config.image_column,
            text_column=config.text_column,
            id_column=config.id_column,
            image_size=image_size,
            alphabet=config.alphabet,
            num_chars=config.num_chars,
        ),
        val=MechaptchaDataset(
            dataset=val_split,
            image_column=config.image_column,
            text_column=config.text_column,
            id_column=config.id_column,
            image_size=image_size,
            alphabet=config.alphabet,
            num_chars=config.num_chars,
        ),
    )


def load_hf_split(config: TrainConfig, split: str, max_samples: int) -> Dataset:
    split_expr = f"{split}[:{max_samples}]" if max_samples > 0 else split
    dataset = load_dataset(
        path=config.dataset_name,
        name=config.dataset_config,
        split=split_expr,
        cache_dir=config.dataset_cache_dir,
    )
    return dataset
