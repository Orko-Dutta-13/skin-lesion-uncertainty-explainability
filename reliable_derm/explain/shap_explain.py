"""SHAP attributions (expected gradients) for the skin-lesion classifier, compared with Grad-CAM.

    python -m reliable_derm.explain.shap_explain --ckpt runs/baseline/best.pt --split test --per-class 4

Method: shap.GradientExplainer = "expected gradients" (Erion et al. 2021), a sampling approximation of integrated
gradients averaged over a background set. Every pixel/channel gets a signed contribution to the explained logit.
    sum of attributions  ~=  logit(x)  -  mean logit(background)           ("additivity", reported as a diagnostic)
Why not KernelSHAP / PartitionExplainer? They are model-agnostic but need thousands of forward passes per image;
expected gradients needs gradients only and is the standard choice for CNNs.

What is explained: the model's PREDICTED class logit (the same class Grad-CAM explains).
Outputs in <ckpt dir>:  shap_gradcam_<split>.png,  shap_<split>.csv,  shap_<split>.npz
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
from scipy.stats import spearmanr

from reliable_derm.constants import CLASSES
from reliable_derm.data.dataset import SkinDataset, build_transforms, denormalize
from reliable_derm.engine import load_checkpoint
from reliable_derm.explain.gradcam import GradCAM, overlay
from reliable_derm.models import gradcam_target_layer
from reliable_derm.utils import get_device


def shap_attributions(model: torch.nn.Module, x: torch.Tensor, background: torch.Tensor,
                      nsamples: int = 100, seed: int = 0, batch: int = 4):
    """Expected-gradients SHAP values for the predicted class.

    Returns (attr [N,C,H,W] signed numpy, target [N] numpy, additivity_gap [N] numpy).
    additivity_gap = |sum(attr) - (logit_target(x) - mean logit_target(background))|.
    """
    import shap

    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    explainer = shap.GradientExplainer(model, background)
    attrs, targets = [], []
    for s in range(0, len(x), batch):
        sv, idx = explainer.shap_values(x[s:s + batch], nsamples=nsamples, ranked_outputs=1)
        sv = np.asarray(sv)                              # [b, C, H, W, 1]
        attrs.append(sv[..., 0])
        targets.append(idx[:, 0].cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)[:, 0])
    attr, target = np.concatenate(attrs), np.concatenate(targets)

    with torch.no_grad():
        fx = model(x).gather(1, torch.as_tensor(target, device=x.device)[:, None])[:, 0]
        fb = model(background)[:, torch.as_tensor(target, device=x.device)].mean(0)
    delta = (fx - fb).cpu().numpy()
    gap = np.abs(attr.reshape(len(x), -1).sum(1) - delta)
    return attr, target, gap


def pool_map(m: np.ndarray, size: int = 14) -> np.ndarray:
    """Average-pool (N,H,W) maps to (N,size,size). Pixel-level SHAP is noisy; Grad-CAM is only 7x7 to begin with."""
    t = torch.as_tensor(m, dtype=torch.float32)[:, None]
    return F.adaptive_avg_pool2d(t, size)[:, 0].numpy()


def agreement(cam: np.ndarray, shap_abs: np.ndarray, size: int = 14) -> np.ndarray:
    """Spearman correlation per image between pooled Grad-CAM and pooled |SHAP| maps (NaN if either is constant)."""
    a, b = pool_map(cam, size), pool_map(shap_abs, size)
    out = []
    for i in range(len(a)):
        x, y = a[i].ravel(), b[i].ravel()
        out.append(float("nan") if x.std() < 1e-8 or y.std() < 1e-8 else float(spearmanr(x, y)[0]))
    return np.array(out)


def shap_heat(attr: np.ndarray, blur: int = 9) -> np.ndarray:
    """Display map: |attribution| summed over colour channels, smoothed, scaled to [0,1]. (N,C,H,W) -> (N,H,W)."""
    m = torch.as_tensor(np.abs(attr).sum(1), dtype=torch.float32)[:, None]
    m = F.avg_pool2d(m, blur, stride=1, padding=blur // 2)[:, 0].numpy()
    lo = m.reshape(len(m), -1).min(1)[:, None, None]
    hi = m.reshape(len(m), -1).max(1)[:, None, None]
    return (m - lo) / (hi - lo + 1e-12)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-col", default=None)
    ap.add_argument("--per-class", type=int, default=4, help="images per class (8 classes -> 32 images)")
    ap.add_argument("--n-background", type=int, default=50, help="training images used as SHAP background")
    ap.add_argument("--nsamples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-plot", type=int, default=16, help="images shown in the figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = get_device(a.device)
    model, cfg = load_checkpoint(a.ckpt, device)
    arch, size = cfg["model"]["arch"], cfg["data"]["image_size"]
    split_col = a.split_col or cfg["data"]["split_col"]
    df = pd.read_csv(cfg["data"]["master_csv"])

    test = df[df[split_col] == a.split]
    test = pd.concat([g.sample(min(a.per_class, len(g)), random_state=a.seed) for _, g in test.groupby("label")])
    test = test.reset_index(drop=True)
    train = df[df[split_col] == "train"].sample(a.n_background, random_state=a.seed).reset_index(drop=True)
    tf = build_transforms(size, train=False)
    ds, ds_bg = SkinDataset(test, tf), SkinDataset(train, tf)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    bg = torch.stack([ds_bg[i][0] for i in range(len(ds_bg))]).to(device)

    attr, target, gap = shap_attributions(model, x, bg, a.nsamples, a.seed, a.batch)
    heat = shap_heat(attr)

    engine = GradCAM(model, gradcam_target_layer(model, arch))
    cam, logits = engine(x, torch.as_tensor(target, device=device))
    engine.remove()
    cam = cam.cpu().numpy()
    prob = logits.softmax(1).cpu().numpy()
    rho = agreement(cam, np.abs(attr).sum(1))

    out_dir = Path(a.ckpt).parent
    labels = test["label"].to_numpy()
    res = pd.DataFrame({"image": test["image"], "true": [CLASSES[i] for i in labels],
                        "pred": [CLASSES[i] for i in target], "correct": labels == target,
                        "p_pred": prob[np.arange(len(target)), target],
                        "spearman_gradcam_shap": rho, "additivity_gap": gap})
    res.to_csv(out_dir / f"shap_{a.split}.csv", index=False)
    np.savez_compressed(out_dir / f"shap_{a.split}.npz", shap_heat=heat.astype(np.float16),
                        gradcam=cam.astype(np.float16), images=test["image"].to_numpy(), target=target)

    imgs = denormalize(x).cpu().permute(0, 2, 3, 1).numpy()
    n = min(a.n_plot, len(x))
    keep = np.linspace(0, len(x) - 1, n).round().astype(int)
    per_row = 3
    rows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(rows, per_row * 3, figsize=(per_row * 6.0, rows * 2.2))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, i in enumerate(keep):
        r, c = divmod(k, per_row)
        ok = res["correct"][i]
        axes[r, 3 * c].imshow(imgs[i])
        axes[r, 3 * c].set_title(f"true {res['true'][i]} / pred {res['pred'][i]}", fontsize=7,
                                 color="green" if ok else "red")
        axes[r, 3 * c + 1].imshow(overlay(imgs[i], cam[i]))
        axes[r, 3 * c + 1].set_title("Grad-CAM", fontsize=7)
        axes[r, 3 * c + 2].imshow(overlay(imgs[i], heat[i]))
        axes[r, 3 * c + 2].set_title(f"SHAP |attr|  rho={rho[i]:+.2f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"shap_gradcam_{a.split}.png", dpi=150)

    rel = float(np.mean(gap) / (np.mean(np.abs(attr.reshape(len(x), -1).sum(1))) + 1e-12))
    print(f"{len(x)} test images, {len(bg)} background images, nsamples={a.nsamples}, explained class = predicted")
    print(f"Grad-CAM vs SHAP (Spearman on 14x14 maps): mean {np.nanmean(rho):+.3f}  median {np.nanmedian(rho):+.3f}"
          f"  (n={int(np.sum(~np.isnan(rho)))})")
    for ok, name in ((True, "correct"), (False, "wrong")):
        m = res["correct"].to_numpy() == ok
        if m.any():
            print(f"   {name:<7s} n={int(m.sum()):2d}  mean rho {np.nanmean(rho[m]):+.3f}")
    print(f"additivity: mean |sum(attr) - (f(x) - E f(bg))| = {np.mean(gap):.3f}  "
          f"(~{100 * rel:.0f}% of the attribution total; approximation error of the sampling)")
    print(f"saved -> {out_dir / f'shap_gradcam_{a.split}.png'}, shap_{a.split}.csv, shap_{a.split}.npz")


if __name__ == "__main__":
    main()
