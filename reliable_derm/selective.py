"""Selective prediction: let the model abstain on its least certain cases.

For a given uncertainty score, sort the test images from most to least certain. "Coverage" c means we
keep (answer) the c most certain images and defer the rest to a human. We report, among the kept images,
accuracy / balanced accuracy and, for melanoma, what happens to the TRUE melanomas:
    mel_detected     kept and called melanoma
    mel_deferred     handed to a human (safe)
    mel_silent_miss  kept but called something else (the dangerous case)
These three add up to 1.

AURC = area under the risk-coverage curve (risk = error rate among kept images), lower is better.
E-AURC = AURC minus the best achievable AURC for the same predictions (an oracle that defers all errors first).

Works on the saved logits (no GPU):
    python -m reliable_derm.selective
    python -m reliable_derm.selective --run-dirs runs\\baseline runs\\baseline_s1 runs\\baseline_s2 --mc-mode full

Methods compared (per run unless noted)
    single / single+TS    one deterministic network, raw softmax / temperature-scaled (T fitted on val)
    mc / mc+TS            MC Dropout mean prediction (needs reliable_derm.mc_dropout output)
    ensemble / ensemble+TS   mean over all runs (the three seeds)  -> one row, no spread
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from reliable_derm.constants import CLASSES
from reliable_derm.metrics import expected_calibration_error, multiclass_brier, nll
from reliable_derm.uncertainty import fit_temperature_members, uncertainty_scores

MEL = CLASSES.index("MEL")
COVERAGES = (0.9, 0.8, 0.7, 0.5)
GRID = np.round(np.arange(0.30, 1.0001, 0.05), 2)
METHOD_ORDER = ["single", "single+TS", "mc", "mc+TS", "ensemble", "ensemble+TS"]
PLOT = [("single+TS", "maxprob"), ("mc+TS", "entropy"), ("mc+TS", "mutual_info"),
        ("ensemble+TS", "entropy"), ("ensemble+TS", "mutual_info")]


# ----------------------------------------------------------------------------- selective-prediction maths
def rank_by_uncertainty(unc: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices from most to least certain; ties are broken randomly but reproducibly."""
    rng = np.random.default_rng(seed)
    return np.lexsort((rng.random(len(unc)), unc))


def risk_coverage(correct: np.ndarray, unc: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """coverage k/N and selective risk (error rate among the k most certain) for k = 1..N."""
    c = correct[rank_by_uncertainty(unc, seed)].astype(float)
    k = np.arange(1, len(c) + 1)
    return k / len(c), 1.0 - np.cumsum(c) / k


def aurc(correct: np.ndarray, unc: np.ndarray, seed: int = 0) -> float:
    return float(risk_coverage(correct, unc, seed)[1].mean())


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of an oracle that ranks every correct prediction before every error."""
    return aurc(correct, (~correct).astype(float))


def at_coverage(pred: np.ndarray, labels: np.ndarray, order: np.ndarray, coverage: float) -> dict:
    n = len(labels)
    k = max(1, int(np.ceil(coverage * n)))
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    mel = labels == MEL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(labels[keep], pred[keep])
    return {
        "accuracy": float((pred[keep] == labels[keep]).mean()),
        "balanced_accuracy": float(bal),
        "mel_detected": float((keep & (pred == MEL))[mel].mean()),
        "mel_deferred": float((~keep)[mel].mean()),
        "mel_silent_miss": float((keep & (pred != MEL))[mel].mean()),
    }


def evaluate_scores(probs: np.ndarray, labels: np.ndarray, scores: dict[str, np.ndarray], seed: int = 0):
    """One (row, curve) per uncertainty score. Row = scalar metrics, curve = metrics on a coverage grid."""
    pred = probs.argmax(1)
    correct = pred == labels
    calib = {"accuracy": float(correct.mean()), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
             "ece": expected_calibration_error(probs, labels), "nll": nll(probs, labels),
             "brier": multiclass_brier(probs, labels)}
    opt = optimal_aurc(correct)
    out = []
    for name, unc in scores.items():
        order = rank_by_uncertainty(unc, seed)
        a = aurc(correct, unc, seed)
        row = {"score": name, "aurc": a, "optimal_aurc": opt, "eaurc": a - opt, **calib}
        for cov in COVERAGES:
            for key, v in at_coverage(pred, labels, order, cov).items():
                row[f"{key}@{int(round(cov * 100))}"] = v
        curve = pd.DataFrame([{"coverage": g, **at_coverage(pred, labels, order, g)} for g in GRID])
        out.append((row, curve))
    return out


# ----------------------------------------------------------------------------- loading
def _read_single(run: Path, tag: str, split: str):
    df = pd.read_csv(run / f"logits_{tag}_{split}.csv")
    return (df[[f"logit_{c}" for c in CLASSES]].to_numpy(np.float64), df["label"].to_numpy(int),
            df["image"].tolist())


def _read_mc(run: Path, tag: str, mode: str, split: str):
    p = run / f"mc_logits_{mode}_{tag}_{split}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return z["logits"].astype(np.float64), z["labels"].astype(int), z["images"].tolist()


def _check_same(a: tuple, b: tuple, what: str) -> None:
    if a[2] != b[2] or not np.array_equal(a[1], b[1]):
        raise SystemExit(f"{what}: image order / labels differ between files, cannot combine them")


# ----------------------------------------------------------------------------- driver
def _fmt(mean: float, sd: float, n: int, digits: int = 3) -> str:
    return f"{mean:.{digits}f}" if n < 2 or np.isnan(sd) else f"{mean:.{digits}f}±{sd:.{digits}f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", default=["runs/baseline", "runs/baseline_s1", "runs/baseline_s2"])
    ap.add_argument("--tag", default="split")
    ap.add_argument("--mc-mode", default="full", choices=["full", "head"])
    ap.add_argument("--out", default="reports/selective")
    ap.add_argument("--seed", type=int, default=0, help="tie-breaking seed for the ranking")
    a = ap.parse_args(argv)

    runs = [Path(d) for d in a.run_dirs]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, curves = [], []
    singles_val, singles_test, ref = [], [], None

    def add(method: str, run_name: str, val_members, test_members, ts: bool, y_val, y_test) -> None:
        T = fit_temperature_members(val_members, y_val) if ts else 1.0
        probs, scores = uncertainty_scores(test_members, T)
        for row, curve in evaluate_scores(probs, y_test, scores, a.seed):
            row.update(method=method, run=run_name, T=T)
            rows.append(row)
            curves.append((method, row["score"], run_name, curve))

    for r in runs:
        for split in ("val", "test"):
            if not (r / f"logits_{a.tag}_{split}.csv").exists():
                raise SystemExit(f"missing {r / f'logits_{a.tag}_{split}.csv'}; run reliable_derm.evaluate "
                                 f"--ckpt {r}\\best.pt --split {split}")
        sv, st = _read_single(r, a.tag, "val"), _read_single(r, a.tag, "test")
        if ref is None:
            ref = (sv, st)
        _check_same(ref[0], sv, f"{r.name} val")
        _check_same(ref[1], st, f"{r.name} test")
        y_val, y_test = sv[1], st[1]
        singles_val.append(sv[0])
        singles_test.append(st[0])
        add("single", r.name, sv[0][None], st[0][None], False, y_val, y_test)
        add("single+TS", r.name, sv[0][None], st[0][None], True, y_val, y_test)

        mv, mt = _read_mc(r, a.tag, a.mc_mode, "val"), _read_mc(r, a.tag, a.mc_mode, "test")
        if mv is not None and mt is not None:
            _check_same(sv, mv, f"{r.name} MC val")
            _check_same(st, mt, f"{r.name} MC test")
            add("mc", r.name, mv[0], mt[0], False, y_val, y_test)
            add("mc+TS", r.name, mv[0], mt[0], True, y_val, y_test)
        else:
            print(f"(no MC files for {r.name}; run: python -m reliable_derm.mc_dropout --ckpt {r}\\best.pt)")

    if len(runs) > 1:
        zv, zt = np.stack(singles_val), np.stack(singles_test)
        add("ensemble", f"{len(runs)} runs", zv, zt, False, ref[0][1], ref[1][1])
        add("ensemble+TS", f"{len(runs)} runs", zv, zt, True, ref[0][1], ref[1][1])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_rows.csv", index=False)

    # ---- summary table (mean ± sample SD over runs where there is more than one run)
    show = [("ece", "ECE"), ("nll", "NLL"), ("balanced_accuracy", "BA"), ("aurc", "AURC"), ("eaurc", "E-AURC"),
            ("accuracy@80", "acc@80%"), ("balanced_accuracy@80", "BA@80%"),
            ("mel_deferred@80", "MELdefer@80"), ("mel_silent_miss@80", "MELsilent@80")]
    table, flat = [], []
    for (method, score), g in df.groupby(["method", "score"], sort=False):
        rec, frec = {"method": method, "score": score, "runs": len(g)}, {"method": method, "score": score}
        for col, label in show:
            m, s = float(g[col].mean()), float(g[col].std(ddof=1)) if len(g) > 1 else float("nan")
            rec[label], frec[label + "_mean"], frec[label + "_sd"] = _fmt(m, s, len(g)), m, s
        table.append(rec)
        flat.append(frec)
    table = pd.DataFrame(table)
    table["_o"] = table["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    table = table.sort_values(["_o", "score"]).drop(columns="_o")
    pd.DataFrame(flat).to_csv(out_dir / "summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\nTest set. Uncertainty-based abstention; mean±sd over runs ({len(runs)} runs; ensemble rows are one model).")
    print("Lower is better: ECE, NLL, AURC, E-AURC, MELsilent. MELsilent = true melanomas kept but NOT called MEL.\n")
    print(table.to_string(index=False))

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("accuracy", "risk (error rate among kept images)", True),
              ("balanced_accuracy", "balanced accuracy among kept images", False),
              ("mel_silent_miss", "true melanomas silently missed", False)]
    for method, score in PLOT:
        sel = [c for m, s, _, c in curves if m == method and s == score]
        if not sel:
            continue
        mean = pd.concat(sel).groupby("coverage").mean()
        for ax, (key, ylabel, as_risk) in zip(axes, panels):
            y = 1 - mean[key] if as_risk else mean[key]
            ax.plot(mean.index, y, marker="o", ms=3, label=f"{method} ({score})")
            ax.set_xlabel("coverage (fraction of images answered)")
            ax.set_ylabel(ylabel)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Selective prediction on the test set (curves averaged over runs where applicable)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "selective_prediction.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_dir / 'all_rows.csv'}, {out_dir / 'summary.csv'}, {out_dir / 'selective_prediction.png'}")


if __name__ == "__main__":
    main()
