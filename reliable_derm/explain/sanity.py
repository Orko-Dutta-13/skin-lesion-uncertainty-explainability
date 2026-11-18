"""Grad-CAM sanity check by cascading weight randomisation (Adebayo et al., 2018, "Sanity Checks for Saliency Maps").

Idea: an explanation that really depends on what the model learned must change when the learned weights are destroyed.
We re-initialise the network from the top (classifier) down to the first stage, one stage at a time, and after every
step recompute Grad-CAM for the SAME class (the originally predicted one). We then measure how similar each map is to
the original map (Spearman rank correlation over pixels).

    healthy    similarity falls towards ~0 as more layers are randomised
    suspicious similarity stays high even when everything is random -> the heat-map mostly reflects the image
               (e.g. a centred lesion / edges), not the learned model

    python -m reliable_derm.explain.sanity --ckpt runs/baseline/best.pt --split test --per-class 2

Only Conv2d/Linear weights are re-initialised (BatchNorm keeps its trained statistics so activations stay finite).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from reliable_derm.data.dataset import SkinDataset, build_transforms, denormalize
from reliable_derm.engine import load_checkpoint
from reliable_derm.explain.gradcam import GradCAM, overlay
from reliable_derm.models import gradcam_target_layer
from reliable_derm.utils import get_device


def stages_top_down(model: nn.Module, arch: str) -> list[tuple[str, nn.Module]]:
    """Parameter-carrying stages ordered from the output to the input."""
    if arch == "efficientnet_b0":
        out = [("classifier", model.classifier)]
        out += [(f"features.{i}", model.features[i]) for i in reversed(range(len(model.features)))]
        return out
    if arch == "resnet50":
        out = [("fc", model.fc)]
        out += [(f"layer{i}", getattr(model, f"layer{i}")) for i in (4, 3, 2, 1)]
        out.append(("conv1", nn.Sequential(model.conv1)))
        return out
    raise ValueError(arch)


def randomize_(module: nn.Module, gen: torch.Generator) -> int:
    """Re-initialise every Conv2d / Linear in `module` (weights ~ He-normal, biases 0). Returns #layers changed."""
    n = 0
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan_in = m.weight[0].numel()
            std = (2.0 / fan_in) ** 0.5
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * std)
                if m.bias is not None:
                    m.bias.zero_()
            n += 1
    return n


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:      # a constant map has no rank information
        return float("nan")
    return float(spearmanr(a, b)[0])


def cascade(model: nn.Module, arch: str, x: torch.Tensor, seed: int = 0):
    """Return (stage_names, sims [n_stages+1, B], cams list[n_stages+1] of (B,H,W) arrays).

    Index 0 is the original model (similarity 1 by construction). Class explained = original prediction.
    """
    device = x.device
    work = copy.deepcopy(model).eval()
    gen = torch.Generator().manual_seed(seed)

    def cam_for(m, cls):
        engine = GradCAM(m, gradcam_target_layer(m, arch))
        try:
            cam, logits = engine(x, cls)
        finally:
            engine.remove()
        return cam.cpu().numpy(), logits

    cam0, logits0 = cam_for(work, None)
    target = logits0.argmax(1)
    names, cams, sims = ["original"], [cam0], [np.ones(len(x))]
    for name, stage in stages_top_down(work, arch):
        if randomize_(stage, gen) == 0:
            continue
        cam, _ = cam_for(work, target)
        names.append(name)
        cams.append(cam)
        sims.append(np.array([_similarity(cam0[i], cam[i]) for i in range(len(x))]))
    return names, np.stack(sims), cams


def plot_cascade(names, sims, cams, imgs, out: Path, n_show: int = 4) -> None:
    mean = np.nanmean(sims, axis=1)
    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_axes([0.07, 0.62, 0.9, 0.33])
    for i in range(sims.shape[1]):
        ax.plot(range(len(names)), sims[:, i], color="0.8", lw=0.8)
    ax.plot(range(len(names)), mean, color="C3", marker="o", lw=2, label="mean over images")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman vs original Grad-CAM")
    ax.set_title("Cascading randomisation: top layers first (left) to the whole network (right)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    show = np.linspace(0, len(names) - 1, min(6, len(names))).round().astype(int)
    k_img = min(n_show, imgs.shape[0])
    for r in range(k_img):
        for c, s in enumerate(show):
            a = fig.add_axes([0.07 + c * 0.155, 0.33 - r * 0.082, 0.145, 0.078])
            a.imshow(overlay(imgs[r], cams[s][r]))
            a.axis("off")
            if r == 0:
                a.set_title(names[s], fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-col", default=None)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="default: <ckpt dir>")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = get_device(a.device)
    model, cfg = load_checkpoint(a.ckpt, device)
    arch, size = cfg["model"]["arch"], cfg["data"]["image_size"]
    split_col = a.split_col or cfg["data"]["split_col"]
    df = pd.read_csv(cfg["data"]["master_csv"])
    df = df[df[split_col] == a.split]
    df = pd.concat([g.sample(min(a.per_class, len(g)), random_state=a.seed) for _, g in df.groupby("label")])
    ds = SkinDataset(df.reset_index(drop=True), build_transforms(size, train=False))
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    imgs = denormalize(x).cpu().permute(0, 2, 3, 1).numpy()

    names, sims, cams = cascade(model, arch, x, a.seed)
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sims.T, columns=names, index=df["image"].tolist()).to_csv(out_dir / "gradcam_sanity.csv")
    plot_cascade(names, sims, cams, imgs, out_dir / "gradcam_sanity.png")

    mean, sd = np.nanmean(sims, axis=1), np.nanstd(sims, axis=1)
    print(f"{len(ds)} images; class explained = originally predicted class")
    for n, m, s, nn_ in zip(names, mean, sd, np.isnan(sims).sum(1)):
        print(f"  after randomising up to {n:<12s} Spearman {m:+.3f} ± {s:.3f}   (constant maps: {nn_})")
    print(f"saved -> {out_dir / 'gradcam_sanity.png'}  and  gradcam_sanity.csv")


if __name__ == "__main__":
    main()
