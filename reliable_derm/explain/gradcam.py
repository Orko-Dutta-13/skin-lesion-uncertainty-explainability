"""Grad-CAM (Selvaraju et al. 2017), implemented directly so every line is explainable in an interview.

    cam_c(x) = ReLU( sum_k  alpha_k^c * A^k(x) ),   alpha_k^c = mean_{i,j} d y_c / d A^k_{ij}

CLI: save a grid of "image | Grad-CAM" tiles for correct / wrong predictions, covering all classes.

    python -m reliable_derm.explain.gradcam --ckpt runs/baseline/best.pt --split test --mode wrong --per-class 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from reliable_derm.constants import CLASSES
from reliable_derm.data.dataset import SkinDataset, build_transforms, denormalize
from reliable_derm.engine import load_checkpoint
from reliable_derm.models import gradcam_target_layer
from reliable_derm.utils import get_device


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        self._h = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inp, out):
        self._acts = out
        # tensor hook instead of module backward hook: robust to in-place activations (SiLU/ReLU).
        # Skipped under torch.no_grad() (e.g. plain prediction passes while the hook is attached).
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def remove(self):
        self._h.remove()

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | int | None = None):
        """Returns (cam, logits). cam: (B,H,W) in [0,1]. class_idx=None -> predicted class."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            if class_idx is None:
                idx = logits.argmax(1)
            elif isinstance(class_idx, int):
                idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
            else:
                idx = class_idx.to(x.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx[:, None]).sum().backward()

        alpha = self._grads.mean(dim=(2, 3), keepdim=True)            # (B,K,1,1)
        cam = F.relu((alpha * self._acts).sum(1, keepdim=True))        # (B,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.detach(), logits.detach()


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """img (H,W,3) in [0,1], cam (H,W) in [0,1] -> blended heat-map overlay."""
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return (1 - alpha) * img + alpha * heat


def _select(df: pd.DataFrame, pred: np.ndarray, mode: str, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(CLASSES)):
        mask = df["label"].to_numpy() == c
        if mode == "correct":
            mask &= pred == c
        elif mode == "wrong":
            mask &= pred != c
        idx = np.flatnonzero(mask)
        if len(idx):
            keep += rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist()
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-col", default=None)
    ap.add_argument("--mode", choices=["correct", "wrong", "any"], default="any")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default=None, help="default: <ckpt dir>/gradcam_<mode>_<split>.png")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    device = get_device(a.device)
    model, cfg = load_checkpoint(a.ckpt, device)
    arch, size = cfg["model"]["arch"], cfg["data"]["image_size"]
    split_col = a.split_col or cfg["data"]["split_col"]
    df = pd.read_csv(cfg["data"]["master_csv"])
    df = df[df[split_col] == a.split].reset_index(drop=True)

    ds = SkinDataset(df, build_transforms(size, train=False))
    cam_engine = GradCAM(model, gradcam_target_layer(model, arch))

    # predictions for the whole split (no grad) to choose which samples to show
    preds = []
    with torch.no_grad():
        for s in range(0, len(ds), 64):
            xb = torch.stack([ds[i][0] for i in range(s, min(s + 64, len(ds)))]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)

    sel = _select(df, pred, a.mode, a.per_class, a.seed)
    if not sel:
        raise SystemExit("no samples matched the selection")
    x = torch.stack([ds[i][0] for i in sel]).to(device)
    cam, logits = cam_engine(x)  # explain the PREDICTED class
    prob = logits.softmax(1).cpu()
    imgs = denormalize(x).cpu().permute(0, 2, 3, 1).numpy()
    cam = cam.cpu().numpy()

    per_row = 4
    rows = int(np.ceil(len(sel) / per_row))
    fig, axes = plt.subplots(rows, per_row * 2, figsize=(per_row * 4.2, rows * 2.3))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, i in enumerate(sel):
        r, c = divmod(k, per_row)
        p = int(prob[k].argmax())
        axes[r, 2 * c].imshow(imgs[k])
        axes[r, 2 * c + 1].imshow(overlay(imgs[k], cam[k]))
        axes[r, 2 * c].set_title(f"true: {CLASSES[int(df['label'][i])]}", fontsize=8)
        ok = p == int(df["label"][i])
        axes[r, 2 * c + 1].set_title(f"pred: {CLASSES[p]} {prob[k, p]:.2f}", fontsize=8,
                                     color="green" if ok else "red")
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(a.ckpt).parent / f"gradcam_{a.mode}_{a.split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    cam_engine.remove()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
