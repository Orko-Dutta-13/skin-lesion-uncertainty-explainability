"""Monte-Carlo Dropout: keep the random layers switched on at test time and run S stochastic passes.

    python -m reliable_derm.mc_dropout --ckpt runs\\baseline\\best.pt --splits val test --n-passes 20

Two modes
    full  head Dropout + the StochasticDepth layers inside the EfficientNet blocks   (default)
    head  only the Dropout before the classifier (weak diversity: only the last layer is random)
BatchNorm always stays in eval mode (running statistics), otherwise predictions would depend on the batch.

Writes runs/<name>/mc_logits_<mode>_<split column>_<split>.npz  with  logits [S,N,K], labels, images.
GPU recommended: about S times the cost of one evaluation per split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from torchvision.ops import StochasticDepth

from reliable_derm.engine import load_checkpoint, make_loader
from reliable_derm.metrics import expected_calibration_error, nll
from reliable_derm.utils import get_device


def set_mc_mode(model: nn.Module, stochastic_depth: bool = True) -> tuple[int, int]:
    """eval() everywhere, then switch Dropout (and optionally StochasticDepth) layers to train().

    Returns (#dropout layers, #stochastic-depth layers) that are now random.
    """
    model.eval()
    n_drop = n_sd = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n_drop += 1
        elif stochastic_depth and isinstance(m, StochasticDepth):
            m.train()
            n_sd += 1
    return n_drop, n_sd


@torch.no_grad()
def mc_logits(model: nn.Module, loader, device: torch.device, n_passes: int = 20,
              stochastic_depth: bool = True, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits [S,N,K], labels [N]). The loader must not shuffle."""
    n_drop, n_sd = set_mc_mode(model, stochastic_depth)
    if n_drop + n_sd == 0:
        raise ValueError("model has no Dropout / StochasticDepth layers: MC Dropout would be deterministic")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batches, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        passes = torch.stack([model(x).float().cpu() for _ in range(n_passes)])  # [S, B, K]
        batches.append(passes)
        labels.append(y)
    return torch.cat(batches, dim=1).numpy(), torch.cat(labels).numpy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--split-col", default=None, help="defaults to the column used for training")
    ap.add_argument("--n-passes", type=int, default=20)
    ap.add_argument("--mode", choices=["full", "head"], default="full")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    device = get_device(a.device)
    model, cfg = load_checkpoint(a.ckpt, device)
    split_col = a.split_col or cfg["data"]["split_col"]
    df = pd.read_csv(cfg["data"]["master_csv"])
    nw = cfg["data"]["num_workers"] if a.num_workers is None else a.num_workers
    out_dir = Path(a.ckpt).parent

    for split in a.splits:
        sub = df[df[split_col] == split].reset_index(drop=True)
        loader = make_loader(sub, cfg["data"]["image_size"], a.batch_size, nw, False, device)
        z, y = mc_logits(model, loader, device, a.n_passes, a.mode == "full", a.seed)
        n_drop, n_sd = set_mc_mode(model, a.mode == "full")
        p = softmax(z, axis=-1)
        mean_p = p.mean(0)
        pass_sd = float(p.max(-1).std(0).mean())
        out = out_dir / f"mc_logits_{a.mode}_{split_col}_{split}.npz"
        np.savez_compressed(out, logits=z.astype(np.float32), labels=y, images=np.array(sub["image"].tolist()))
        print(f"{split}: {a.mode} mode, {n_drop} dropout + {n_sd} stochastic-depth layers random, "
              f"{a.n_passes} passes, N={len(y)}")
        print(f"   mean-prediction accuracy {(mean_p.argmax(1) == y).mean():.4f}  "
              f"ECE {expected_calibration_error(mean_p, y):.4f}  NLL {nll(mean_p, y):.4f}  "
              f"(avg std of max-prob across passes: {pass_sd:.4f})")
        print(f"   saved -> {out}")


if __name__ == "__main__":
    main()
