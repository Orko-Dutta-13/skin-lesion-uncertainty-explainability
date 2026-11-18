from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reliable_derm.data.dataset import SkinDataset, build_transforms


def make_loader(df: pd.DataFrame, image_size: int, batch_size: int, num_workers: int,
                train: bool, device: torch.device) -> DataLoader:
    ds = SkinDataset(df, build_transforms(image_size, train))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=train and len(ds) > batch_size,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_logits(model, loader, device, amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x)
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def load_checkpoint(path: str, device: torch.device):
    from reliable_derm.models import build_model

    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = build_model(cfg["model"]["arch"], len(ck["classes"]), pretrained=False,
                        dropout=cfg["model"]["dropout"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg
