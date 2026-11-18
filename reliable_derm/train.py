"""Fine-tune a pretrained CNN on the prepared ISIC 2019 table.

    python -m reliable_derm.train --config configs/baseline.yaml
    python -m reliable_derm.train --config configs/baseline.yaml --override train.epochs=3 data.split_col=split_naive

Only the *validation* split is looked at here (model selection). Run evaluate.py on the test
split once, at the end of an experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from reliable_derm.constants import CLASSES
from reliable_derm.engine import make_loader, predict_logits
from reliable_derm.metrics import compute_metrics
from reliable_derm.models import build_model
from reliable_derm.utils import get_device, load_config, set_seed


def _flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _maybe_mlflow(cfg):
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed -> skipping experiment tracking")
        return None
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    return mlflow


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="e.g. train.epochs=3 data.split_col=split_naive")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.override)
    set_seed(cfg["seed"])
    device = get_device(a.device)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"device={device}  out_dir={out_dir}")

    df = pd.read_csv(cfg["data"]["master_csv"])
    split_col = cfg["data"]["split_col"]
    lim = cfg["data"].get("limit_per_split")
    if lim:  # smoke-test mode
        df = pd.concat([g.sample(min(len(g), lim), random_state=cfg["seed"])
                        for _, g in df.groupby(split_col)])
    tr, va = df[df[split_col] == "train"], df[df[split_col] == "val"]
    print(f"train={len(tr)}  val={len(va)}  (split column: {split_col})")

    d, t = cfg["data"], cfg["train"]
    train_loader = make_loader(tr, d["image_size"], t["batch_size"], d["num_workers"], True, device)
    val_loader = make_loader(va, d["image_size"], t["batch_size"], d["num_workers"], False, device)

    model = build_model(cfg["model"]["arch"], len(CLASSES), cfg["model"]["pretrained"],
                        cfg["model"]["dropout"]).to(device)

    weight = None
    if t["imbalance"] == "weighted_loss":
        counts = np.bincount(tr["label"], minlength=len(CLASSES)).astype(float)
        w = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5  # sqrt-inverse frequency
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        print("class weights:", {c: round(float(v), 2) for c, v in zip(CLASSES, weight.cpu().numpy())})
    criterion = nn.CrossEntropyLoss(weight=weight)

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = max(1, len(train_loader))
    total, warm = t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mlflow = _maybe_mlflow(cfg)
    run = mlflow.start_run(run_name=Path(cfg["out_dir"]).name) if mlflow else None
    if mlflow:
        mlflow.log_params({k: v for k, v in _flatten(cfg)})

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(y)
            n += len(y)

        logits, labels = predict_logits(model, val_loader, device, amp=use_amp)
        m = compute_metrics(logits, labels)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), "val_balanced_acc": m["balanced_accuracy"],
               "val_acc": m["accuracy"], "val_auroc": m["macro_auroc"], "val_ece": m["ece"],
               "val_nll": m["nll"], "sec": round(time.time() - t0, 1)}
        history.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        if mlflow:
            mlflow.log_metrics({k: v for k, v in row.items() if k not in ("epoch", "sec") and not math.isnan(v)},
                               step=epoch)

        if m["balanced_accuracy"] > best:
            best, best_epoch = m["balanced_accuracy"], epoch
            torch.save({"model": model.state_dict(), "config": cfg, "classes": CLASSES, "epoch": epoch},
                       out_dir / "best.pt")
            (out_dir / "val_metrics.json").write_text(json.dumps(m, indent=2))
        elif epoch - best_epoch >= t["patience"]:
            print(f"early stop (no val balanced-acc improvement for {t['patience']} epochs)")
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if mlflow:
        mlflow.log_artifact(str(out_dir / "config.yaml"))
        mlflow.log_artifact(str(out_dir / "history.csv"))
        mlflow.log_metric("best_val_balanced_acc", best)
        mlflow.end_run()
    print(f"\nbest val balanced accuracy {best:.4f} at epoch {best_epoch}  ->  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
