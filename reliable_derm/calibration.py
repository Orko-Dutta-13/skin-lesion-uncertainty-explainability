"""Post-hoc temperature scaling (Guo et al., 2017) + reliability diagrams.

Idea: the network outputs logits z. Softmax(z / T) with a single scalar T > 0 is a recalibrated
probability. T > 1 softens (fixes over-confidence), T < 1 sharpens. T is fitted on the VALIDATION
logits by minimising NLL, then applied unchanged to the TEST logits. Dividing by a positive T never
changes the arg-max, so accuracy / balanced accuracy are identical before and after; only the
confidence (ECE, NLL, Brier) changes.

Usage (from the project root, venv active):
    python -m reliable_derm.calibration
    python -m reliable_derm.calibration --run-dirs runs\\baseline runs\\baseline_s1 runs\\baseline_s2

Needs runs/<name>/logits_split_val.csv and logits_split_test.csv (made by reliable_derm.evaluate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from reliable_derm.constants import CLASSES
from reliable_derm.metrics import compute_metrics

LOG_T_BOUNDS = (-3.0, 3.0)  # T in [0.05, 20]
SUMMARY_KEYS = ("ece", "nll", "brier", "balanced_accuracy", "macro_auroc")


# ----------------------------------------------------------------------------- core maths
def load_logits(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a logits CSV written by evaluate.py -> (logits [N, 8] in CLASSES order, labels [N])."""
    df = pd.read_csv(csv_path)
    cols = [f"logit_{c}" for c in CLASSES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks columns {missing}")
    return df[cols].to_numpy(dtype=np.float64), df["label"].to_numpy(dtype=int)


def _nll_at(log_t: float, logits: np.ndarray, labels: np.ndarray) -> float:
    logp = log_softmax(logits / np.exp(log_t), axis=1)
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """T > 0 minimising NLL on the validation set. Optimises log T so T stays positive."""
    res = minimize_scalar(_nll_at, bounds=LOG_T_BOUNDS, args=(val_logits, val_labels), method="bounded")
    if min(abs(res.x - LOG_T_BOUNDS[0]), abs(res.x - LOG_T_BOUNDS[1])) < 1e-3:
        print(f"WARNING: fitted log T = {res.x:.3f} sits on the search boundary; check your logits.")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        raise ValueError("temperature must be positive")
    return logits / T


# ----------------------------------------------------------------------------- reliability
def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Top-label reliability statistics, binned exactly like metrics.expected_calibration_error.

    Returns (edges, acc, conf, count); acc/conf are NaN for empty bins.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc, mean_conf, count = (np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int))
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi)
        count[i] = int(m.sum())
        if m.any():
            acc[i], mean_conf[i] = correct[m].mean(), conf[m].mean()
    return edges, acc, mean_conf, count


def reliability_diagram(logits_before: np.ndarray, logits_after: np.ndarray, labels: np.ndarray,
                        path: str | Path, n_bins: int = 15, title: str = "") -> None:
    """Two panels (before / after): per-bin accuracy bars vs the diagonal. Saves a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, lg, name in zip(axes, (logits_before, logits_after), ("uncalibrated", "temperature-scaled")):
        probs = softmax(lg, axis=1)
        edges, acc, _, count = reliability_bins(probs, labels, n_bins)
        centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
        ok = count > 0
        ax.bar(centers[ok], acc[ok], width=width * 0.95, color="#4C72B0", edgecolor="white", label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ece = compute_metrics(lg, labels)["ece"]
        ax.set_title(f"{name}\nECE = {ece:.3f}", fontsize=10)
        ax.set_xlabel("confidence (max probability)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy in bin")
    axes[0].legend(loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def calibrate_run(run_dir: Path, tag: str, n_bins: int, out_dir: Path) -> dict:
    val_csv, test_csv = run_dir / f"logits_{tag}_val.csv", run_dir / f"logits_{tag}_test.csv"
    for p in (val_csv, test_csv):
        if not p.exists():
            raise SystemExit(f"missing {p}. Run: python -m reliable_derm.evaluate --ckpt {run_dir}\\best.pt "
                             f"--split {p.stem.rsplit('_', 1)[-1]}")
    zv, yv = load_logits(val_csv)
    zt, yt = load_logits(test_csv)

    T = fit_temperature(zv, yv)
    before, after = compute_metrics(zt, yt), compute_metrics(apply_temperature(zt, T), yt)

    # temperature scaling must not change predictions
    assert np.array_equal(zt.argmax(1), apply_temperature(zt, T).argmax(1)), "arg-max changed: bug"
    assert abs(before["balanced_accuracy"] - after["balanced_accuracy"]) < 1e-12

    reliability_diagram(zt, apply_temperature(zt, T), yt, out_dir / f"reliability_{run_dir.name}.png",
                        n_bins=n_bins, title=f"{run_dir.name}: test set, T = {T:.2f} (fitted on val)")
    row = {"run": run_dir.name, "T": T}
    for k in SUMMARY_KEYS:
        row[f"{k}_before"], row[f"{k}_after"] = before[k], after[k]
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", default=["runs/baseline", "runs/baseline_s1", "runs/baseline_s2"])
    ap.add_argument("--tag", default="split", help="split column used for the logits files (split | split_naive)")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--out", default="reports/calibration")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [calibrate_run(Path(d), a.tag, a.n_bins, out_dir) for d in a.run_dirs]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "calibration_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nTest-set metrics, before -> after temperature scaling (T fitted on validation only)\n")
    for r in rows:
        print(f"{r['run']:14s} T={r['T']:.3f}  " + "  ".join(
            f"{k}: {r[k + '_before']:.4f}->{r[k + '_after']:.4f}" for k in ("ece", "nll", "brier")))
    if len(rows) > 1:
        print(f"\nmean ± sd over {len(rows)} runs (sample sd):")
        summary = {"T": [float(df["T"].mean()), float(df["T"].std(ddof=1))]}
        print(f"  T                  {df['T'].mean():.3f} ± {df['T'].std(ddof=1):.3f}")
        for k in SUMMARY_KEYS:
            b, af = df[f"{k}_before"], df[f"{k}_after"]
            summary[k] = {"before": [float(b.mean()), float(b.std(ddof=1))],
                          "after": [float(af.mean()), float(af.std(ddof=1))]}
            print(f"  {k:18s} {b.mean():.4f} ± {b.std(ddof=1):.4f}  ->  {af.mean():.4f} ± {af.std(ddof=1):.4f}")
        (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {out_dir / 'calibration_summary.csv'} and reliability_<run>.png in {out_dir}")


if __name__ == "__main__":
    main()
