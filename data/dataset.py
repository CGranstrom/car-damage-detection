from pathlib import Path
from random import shuffle
from typing import Any, Callable, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import DEVICE


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        validation_split: float = 0.2,
        mode: str = "train",
    ):
        """
        Initializes dataset and splits it into training and validation sets.

        Args:
            data_dir (str or Path): Directory containing images.
            transform (callable, optional): Transform to apply to images.
            validation_split (float): Fraction of data to use for validation.
            mode (str): Whether dataset if for training ("train"), validation ("val"), or testing ("test").
        """
        data_dir = Path(data_dir)
        self.filepaths = []
        for image_type in ("whole", "damaged"):
            self.filepaths.extend(
                [
                    f
                    for f in data_dir.glob(f"{image_type}/*")
                    if f.suffix.lower() in [".jpg", ".jpeg"]
                ]
            )

        shuffle(self.filepaths)
        self.targets = [fpath.parent.stem == "damaged" for fpath in self.filepaths]

        self.transform = transform
        split_idx = int(len(self.filepaths) * (1 - validation_split))

        if mode == "train":
            split_idx = int(len(self.filepaths) * (1 - validation_split))
            self.filepaths = self.filepaths[:split_idx]
            self.targets = self.targets[:split_idx]
            self.transform = transform
        elif mode == "validation":
            split_idx = int(len(self.filepaths) * (1 - validation_split))
            self.filepaths = self.filepaths[split_idx:]
            self.targets = self.targets[split_idx:]
            self.transform = transform
        elif mode == "test":
            pass
        else:
            raise ValueError("Mode should be 'train', 'validation', or 'test'.")

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, ix) -> tuple[Any, torch.Tensor]:
        filepath = self.filepaths[ix]
        target = self.targets[ix]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image.to(DEVICE), torch.tensor([target]).float().to(DEVICE)
