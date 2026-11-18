from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from reliable_derm.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int, train: bool):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor(), norm])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) normalised -> [0,1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].tolist()
        self.labels = self.df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i]), i
