"""Evaluate a checkpoint on one split and save raw logits (needed later for calibration / UQ / drift).

    python -m reliable_derm.evaluate --ckpt runs/baseline/best.pt --split val
    python -m reliable_derm.evaluate --ckpt runs/baseline/best.pt --split test          # final numbers only
    python -m reliable_derm.evaluate --ckpt runs/baseline/best.pt --split shift_test --split-col split_shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reliable_derm.constants import CLASSES
from reliable_derm.engine import load_checkpoint, make_loader, predict_logits
from reliable_derm.metrics import compute_metrics
from reliable_derm.utils import get_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--split-col", default=None, help="defaults to the split column used for training")
    ap.add_argument("--master-csv", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = get_device(a.device)
    model, cfg = load_checkpoint(a.ckpt, device)
    split_col = a.split_col or cfg["data"]["split_col"]
    df = pd.read_csv(a.master_csv or cfg["data"]["master_csv"])
    sub = df[df[split_col] == a.split].reset_index(drop=True)
    if cfg["data"].get("limit_per_split"):
        sub = sub.head(cfg["data"]["limit_per_split"])
    if sub.empty:
        raise SystemExit(f"no rows with {split_col}=={a.split!r}")

    nw = cfg["data"]["num_workers"] if a.num_workers is None else a.num_workers
    loader = make_loader(sub, cfg["data"]["image_size"], a.batch_size, nw, False, device)
    logits, labels = predict_logits(model, loader, device)
    m = compute_metrics(logits, labels)

    out_dir = Path(a.ckpt).parent
    tag = f"{split_col}_{a.split}"
    out = sub[["image", "label", "source"]].copy()
    for i, c in enumerate(CLASSES):
        out[f"logit_{c}"] = logits[:, i]
    out.to_csv(out_dir / f"logits_{tag}.csv", index=False)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(m, indent=2))

    print(f"\n{tag}: n={m['n']}")
    for k in ("accuracy", "balanced_accuracy", "macro_auroc", "ece", "nll", "brier"):
        print(f"  {k:18s} {m[k]:.4f}")
    print("  per-class recall:", {k: round(v, 3) for k, v in m["recall_per_class"].items()})
    print(f"saved -> {out_dir / f'logits_{tag}.csv'}")


if __name__ == "__main__":
    main()
