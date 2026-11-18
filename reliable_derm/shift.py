"""Distribution-shift evaluation: what happens to accuracy, calibration and uncertainty on images from a source
(BCN20000) that the models never saw?

Protocol (column ``split_ood``, see data/splits.py: add_ood_split)
    train / val / test   HAM10000 + MSK images only, grouped by lesion      ("in distribution", ID)
    shift_test           all BCN20000 images, never seen in training         ("shifted")
Temperature scaling is fitted on the ID validation set only and then applied unchanged to both test sets.

Class-mix confound: BCN has a very different class mix from HAM + MSK (far fewer nevi, many more AK/BCC/SCC).
Balanced accuracy, per-class recall and macro AUROC do not depend on the class prior. Accuracy, ECE, NLL, Brier,
selective-prediction risk and the shift-detection AUROC do, so for the ID test set they are reported with every image
weighted by p_BCN(class) / p_ID(class) ("matched to the BCN class mix"). Unweighted ID numbers are saved too.

    python -m reliable_derm.shift
    python -m reliable_derm.shift --run-dirs runs\\shift runs\\shift_s1 runs\\shift_s2 --oracle-dirs runs\\baseline runs\\baseline_s1 runs\\baseline_s2

Inputs (made with reliable_derm.evaluate --split-col split_ood):  logits_split_ood_{val,test,shift_test}.csv
Optional "oracle" reference: the baseline models (which DID see BCN in training), evaluated on their BCN test images.
Outputs: reports/shift/{summary.csv, per_class.csv, detection.csv, bootstrap.json, shift_overview.png}
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from reliable_derm.constants import CLASSES
from reliable_derm.metrics import compute_metrics
from reliable_derm.selective import GRID, rank_by_uncertainty
from reliable_derm.uncertainty import fit_temperature_members, uncertainty_scores

K = len(CLASSES)
EPS = 1e-12


# ----------------------------------------------------------------------------- class-prior matching + weighted metrics
def class_weights(src_labels: np.ndarray, tgt_labels: np.ndarray, k: int = K) -> np.ndarray:
    """Per-image weights that make the class mix of ``src`` equal to that of ``tgt`` (mean weight over src = 1)."""
    ps = np.bincount(src_labels, minlength=k) / len(src_labels)
    pt = np.bincount(tgt_labels, minlength=k) / len(tgt_labels)
    w_c = np.where(ps > 0, pt / np.maximum(ps, EPS), 0.0)
    return w_c[src_labels]


def w_mean(v: np.ndarray, w: np.ndarray) -> float:
    return float((v * w).sum() / w.sum())


def w_ece(probs: np.ndarray, labels: np.ndarray, w: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE, equal-width bins (lo, hi], every image counted with weight w."""
    conf, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, tot = 0.0, w.sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any() and w[m].sum() > 0:
            ece += w[m].sum() / tot * abs(w_mean(correct[m], w[m]) - w_mean(conf[m], w[m]))
    return float(ece)


def w_calibration_metrics(probs: np.ndarray, labels: np.ndarray, w: np.ndarray) -> dict:
    p_true = probs[np.arange(len(labels)), labels]
    brier = ((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1)
    return {"accuracy": w_mean((probs.argmax(1) == labels).astype(float), w),
            "ece": w_ece(probs, labels, w), "nll": w_mean(-np.log(p_true + EPS), w), "brier": w_mean(brier, w)}


def w_risk_coverage(correct: np.ndarray, unc: np.ndarray, w: np.ndarray, seed: int = 0):
    """Weighted selective risk: coverage = weight share kept, risk = weighted error rate among the kept."""
    order = rank_by_uncertainty(unc, seed)
    ws, cs = w[order], correct[order].astype(float)
    cum_w = np.cumsum(ws)
    risk = 1.0 - np.cumsum(ws * cs) / cum_w
    return cum_w / cum_w[-1], risk, ws


def w_aurc(correct: np.ndarray, unc: np.ndarray, w: np.ndarray, seed: int = 0) -> float:
    _, risk, ws = w_risk_coverage(correct, unc, w, seed)
    return float((risk * ws).sum() / ws.sum())


def w_risk_at(correct: np.ndarray, unc: np.ndarray, w: np.ndarray, coverage: float, seed: int = 0) -> float:
    cov, risk, _ = w_risk_coverage(correct, unc, w, seed)
    return float(risk[min(np.searchsorted(cov, coverage - 1e-12), len(risk) - 1)])


def detection_auroc(unc_id: np.ndarray, unc_shift: np.ndarray, w_id: np.ndarray) -> float:
    """AUROC of an uncertainty score for telling shifted images (positive) from ID images (class-mix matched)."""
    y = np.r_[np.zeros(len(unc_id)), np.ones(len(unc_shift))]
    return float(roc_auc_score(y, np.r_[unc_id, unc_shift], sample_weight=np.r_[w_id, np.ones(len(unc_shift))]))


def balanced_acc_from_counts(labels: np.ndarray, pred: np.ndarray, k: int = K) -> float:
    cm = np.bincount(labels * k + pred, minlength=k * k).reshape(k, k)
    n = cm.sum(1)
    return float((np.diag(cm)[n > 0] / n[n > 0]).mean())


def bootstrap_ba(labels: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(labels)
    return np.array([balanced_acc_from_counts(labels[i], pred[i])
                     for i in (rng.integers(0, n, n) for _ in range(n_boot))])


# ----------------------------------------------------------------------------- loading
def read_logits(run: Path, tag: str, split: str):
    f = run / f"logits_{tag}_{split}.csv"
    if not f.exists():
        raise SystemExit(f"missing {f}\n  run:  python -m reliable_derm.evaluate --ckpt {run}\\best.pt "
                         f"--split {split} --split-col {tag}")
    df = pd.read_csv(f)
    return (df[[f"logit_{c}" for c in CLASSES]].to_numpy(np.float64), df["label"].to_numpy(int),
            df["image"].tolist(), df["source"].to_numpy())


def load_runs(run_dirs: list[str], tag: str, splits: tuple[str, ...]):
    """-> {split: dict(members [R,N,K], labels, images, source)} ; checks every run saw the same images."""
    out = {}
    for s in splits:
        got = [read_logits(Path(r), tag, s) for r in run_dirs]
        for r, g in zip(run_dirs[1:], got[1:]):
            if g[2] != got[0][2] or not np.array_equal(g[1], got[0][1]):
                raise SystemExit(f"{s}: image order / labels of {r} differ from {run_dirs[0]}; cannot combine runs")
        out[s] = {"members": np.stack([g[0] for g in got]), "labels": got[0][1], "images": got[0][2],
                  "source": got[0][3]}
    return out


# ----------------------------------------------------------------------------- one method on one dataset
def evaluate_members(val_members, val_labels, members, labels, w, *, temperature: bool, score: str = "maxprob",
                     T: float | None = None) -> tuple[dict, dict]:
    """Return (row, extras). Members are logits [M,N,K]; M>1 -> mean-of-softmaxes prediction."""
    if temperature and T is None:
        T = fit_temperature_members(val_members, val_labels)
    t = T if temperature else 1.0
    probs, scores = uncertainty_scores(members, t)
    pred = probs.argmax(1)
    correct = pred == labels
    unc = scores[score]
    row = {"T": t, "n": len(labels), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
           "macro_auroc": compute_metrics(np.log(probs + EPS), labels)["macro_auroc"],
           **w_calibration_metrics(probs, labels, w),
           "aurc": w_aurc(correct, unc, w), "acc@80": 1 - w_risk_at(correct, unc, w, 0.8),
           "acc@50": 1 - w_risk_at(correct, unc, w, 0.5)}
    return row, {"probs": probs, "pred": pred, "scores": scores, "T": t}


def per_class_recall(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.array([(pred[labels == c] == c).mean() if (labels == c).any() else np.nan for c in range(K)])


# ----------------------------------------------------------------------------- driver
def fmt(vals: np.ndarray, d: int = 3) -> str:
    vals = np.asarray(vals, float)
    return f"{vals.mean():.{d}f}" if len(vals) < 2 else f"{vals.mean():.{d}f}±{vals.std(ddof=1):.{d}f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", default=["runs/shift", "runs/shift_s1", "runs/shift_s2"])
    ap.add_argument("--tag", default="split_ood")
    ap.add_argument("--oracle-dirs", nargs="*", default=["runs/baseline", "runs/baseline_s1", "runs/baseline_s2"],
                    help="models trained WITH BCN (tag 'split'); their BCN test images are the no-shift reference. "
                         "Pass an empty list to skip.")
    ap.add_argument("--oracle-tag", default="split")
    ap.add_argument("--score", default="maxprob", choices=["maxprob", "entropy"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/shift")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load_runs(a.run_dirs, a.tag, ("val", "test", "shift_test"))
    val, idt, sh = D["val"], D["test"], D["shift_test"]
    if not set(np.unique(sh["source"])) <= {"BCN"}:
        raise SystemExit("shift_test must contain BCN images only")
    w_id = class_weights(idt["labels"], sh["labels"])
    ones_id, ones_sh = np.ones(len(idt["labels"])), np.ones(len(sh["labels"]))
    R = len(a.run_dirs)
    print(f"{R} shift models | val n={len(val['labels'])}  ID test n={len(idt['labels'])}  BCN shift n={len(sh['labels'])}")
    mix = pd.DataFrame({"ID test": np.bincount(idt["labels"], minlength=K) / len(idt["labels"]),
                        "BCN shift": np.bincount(sh["labels"], minlength=K) / len(sh["labels"])}, index=CLASSES)
    print("class mix:\n", mix.round(3).T.to_string())

    # ---- every method on every dataset
    rows, extras = [], {}
    methods = {"single": [[i] for i in range(R)], "ensemble": [list(range(R))]}
    for base, groups in methods.items():
        for temperature in (False, True):
            name = base + ("+TS" if temperature else "")
            for gi, idx in enumerate(groups):
                T = fit_temperature_members(val["members"][idx], val["labels"]) if temperature else None
                for ds_name, ds, w in (("ID test (raw)", idt, ones_id), ("ID test (BCN mix)", idt, w_id),
                                       ("BCN shift", sh, ones_sh)):
                    row, ex = evaluate_members(val["members"][idx], val["labels"], ds["members"][idx], ds["labels"],
                                               w, temperature=temperature, score=a.score, T=T)
                    rows.append({"method": name, "run": gi, "dataset": ds_name, **row})
                    extras[(name, gi, ds_name)] = ex
    res = pd.DataFrame(rows)

    # ---- oracle reference: models that trained on BCN, evaluated on BCN test images
    if a.oracle_dirs:
        try:
            O = load_runs(a.oracle_dirs, a.oracle_tag, ("val", "test"))
        except SystemExit as e:
            print(f"(oracle reference skipped: {e})")
            O = None
        if O:
            m = O["test"]["source"] == "BCN"
            for name, temperature, idx_groups in (("single", False, [[i] for i in range(len(a.oracle_dirs))]),
                                                  ("single+TS", True, [[i] for i in range(len(a.oracle_dirs))]),
                                                  ("ensemble", False, [list(range(len(a.oracle_dirs)))]),
                                                  ("ensemble+TS", True, [list(range(len(a.oracle_dirs)))])):
                for gi, idx in enumerate(idx_groups):
                    row, _ = evaluate_members(O["val"]["members"][idx], O["val"]["labels"],
                                              O["test"]["members"][idx][:, m], O["test"]["labels"][m],
                                              np.ones(int(m.sum())), temperature=temperature, score=a.score)
                    rows.append({"method": name, "run": gi, "dataset": "BCN (trained with BCN)", **row})
            res = pd.DataFrame(rows)
    res.to_csv(out / "all_rows.csv", index=False)

    # ---- summary table (mean ± sd over the shift models; ensemble = one model)
    cols = ["balanced_accuracy", "macro_auroc", "accuracy", "ece", "nll", "brier", "aurc", "acc@80"]
    summ = []
    for (method, ds), g in res.groupby(["method", "dataset"], sort=False):
        summ.append({"method": method, "dataset": ds, "n_models": len(g), "T": g["T"].mean(),
                     **{c: g[c].mean() for c in cols}, **{c + "_sd": g[c].std(ddof=1) if len(g) > 1 else np.nan
                                                         for c in cols}})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)

    order = ["single", "single+TS", "ensemble", "ensemble+TS"]
    ds_order = ["ID test (BCN mix)", "BCN shift"] + (["BCN (trained with BCN)"] if a.oracle_dirs and "BCN (trained with BCN)"
                                                      in set(res["dataset"]) else [])
    print(f"\nHigher is better: BA, AUROC, acc, acc@80.  Lower is better: ECE, NLL, Brier, AURC.  score = {a.score}")
    print("ID test numbers are weighted to the BCN class mix; BA/AUROC do not depend on the class mix.\n")
    table = []
    for method in order:
        for ds in ds_order:
            g = res[(res["method"] == method) & (res["dataset"] == ds)]
            if len(g):
                table.append({"method": method, "dataset": ds, **{c: fmt(g[c].to_numpy()) for c in cols}})
    print(pd.DataFrame(table).to_string(index=False))

    drop = []
    for method in order:
        a_ = res[(res["method"] == method) & (res["dataset"] == "ID test (BCN mix)")].sort_values("run")
        b_ = res[(res["method"] == method) & (res["dataset"] == "BCN shift")].sort_values("run")
        drop.append({"method": method, **{f"d_{c}": fmt(b_[c].to_numpy() - a_[c].to_numpy()) for c in
                                          ("balanced_accuracy", "macro_auroc", "ece", "nll", "acc@80")}})
    print("\nChange from ID test (BCN mix) to BCN shift:")
    print(pd.DataFrame(drop).to_string(index=False))

    # ---- shift detection from uncertainty
    det = []
    for method in ("single+TS", "ensemble+TS"):
        n_g = R if method.startswith("single") else 1
        for sc in ("maxprob", "entropy", "mutual_info"):
            vals = []
            for gi in range(n_g):
                ei, es = extras[(method, gi, "ID test (raw)")], extras[(method, gi, "BCN shift")]
                if sc in ei["scores"]:
                    vals.append(detection_auroc(ei["scores"][sc], es["scores"][sc], w_id))
            if vals:
                det.append({"method": method, "score": sc, "detection_auroc": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan})
    det = pd.DataFrame(det)
    det.to_csv(out / "detection.csv", index=False)
    print("\nCan the uncertainty score tell BCN images from ID images? (AUROC, 0.5 = no; ID weighted to BCN class mix)")
    print(det.round(3).to_string(index=False))

    # ---- per-class recall (ensemble+TS) and BA bootstrap
    ei, es = extras[("ensemble+TS", 0, "ID test (raw)")], extras[("ensemble+TS", 0, "BCN shift")]
    pc = pd.DataFrame({"class": CLASSES, "n_ID": np.bincount(idt["labels"], minlength=K),
                       "recall_ID": per_class_recall(idt["labels"], ei["pred"]),
                       "n_BCN": np.bincount(sh["labels"], minlength=K),
                       "recall_BCN": per_class_recall(sh["labels"], es["pred"])})
    pc["change"] = pc["recall_BCN"] - pc["recall_ID"]
    pc.to_csv(out / "per_class.csv", index=False)
    print("\nPer-class recall, ensemble+TS (n in the ID test set is small for DF / VASC / SCC):")
    print(pc.round(3).to_string(index=False))

    b_id, b_sh = bootstrap_ba(idt["labels"], ei["pred"], a.n_boot, 1), bootstrap_ba(sh["labels"], es["pred"], a.n_boot, 2)
    d = b_sh - b_id
    boot = {"ensemble+TS_BA_ID": [float(np.percentile(b_id, 2.5)), float(np.percentile(b_id, 97.5))],
            "ensemble+TS_BA_shift": [float(np.percentile(b_sh, 2.5)), float(np.percentile(b_sh, 97.5))],
            "ensemble+TS_BA_change": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "note": "image-level bootstrap; images of one lesion are correlated so these intervals are too narrow"}
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    print(f"\nEnsemble+TS balanced accuracy, 95% bootstrap CI: ID {boot['ensemble+TS_BA_ID'][0]:.3f}-"
          f"{boot['ensemble+TS_BA_ID'][1]:.3f}, BCN {boot['ensemble+TS_BA_shift'][0]:.3f}-"
          f"{boot['ensemble+TS_BA_shift'][1]:.3f}, change {boot['ensemble+TS_BA_change'][0]:+.3f} to "
          f"{boot['ensemble+TS_BA_change'][1]:+.3f}  (image-level, optimistic)")

    plot_overview(out / "shift_overview.png", pc, ei, es, idt, sh, w_id, a.score, det)
    print(f"\nsaved: {out / 'summary.csv'}, all_rows.csv, per_class.csv, detection.csv, bootstrap.json, shift_overview.png")


def plot_overview(path, pc, ei, es, idt, sh, w_id, score, det) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(K)
    ax[0, 0].bar(x - 0.2, pc["recall_ID"], 0.4, label="ID test (HAM+MSK)")
    ax[0, 0].bar(x + 0.2, pc["recall_BCN"], 0.4, label="BCN shift")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([f"{c}\n{ni}/{nb}" for c, ni, nb in zip(pc["class"], pc["n_ID"], pc["n_BCN"])], fontsize=8)
    ax[0, 0].set_ylabel("recall")
    ax[0, 0].set_title("Per-class recall, ensemble+TS (n ID / n BCN under each class)")
    ax[0, 0].legend()

    def reliab(probs, labels, w, n_bins=10):
        conf, corr = probs.max(1), (probs.argmax(1) == labels).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        xs, ys, ws = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 20:
                xs.append(w_mean(conf[m], w[m]))
                ys.append(w_mean(corr[m], w[m]))
                ws.append(w[m].sum())
        return np.array(xs), np.array(ys), np.array(ws)

    ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    for name, ex, ds, w in (("ID test (BCN class mix)", ei, idt, w_id), ("BCN shift", es, sh, np.ones(len(sh["labels"])))):
        xs, ys, _ = reliab(ex["probs"], ds["labels"], w)
        ax[0, 1].plot(xs, ys, "o-", label=name)
    ax[0, 1].set_xlabel("confidence")
    ax[0, 1].set_ylabel("accuracy")
    ax[0, 1].set_title("Reliability, ensemble+TS (T fitted on ID val)")
    ax[0, 1].legend()

    for name, ex, ds, w in (("ID test (BCN class mix)", ei, idt, w_id), ("BCN shift", es, sh, np.ones(len(sh["labels"])))):
        correct = ex["pred"] == ds["labels"]
        cov, risk, _ = w_risk_coverage(correct, ex["scores"][score], w)
        g = np.searchsorted(cov, GRID - 1e-12).clip(0, len(risk) - 1)
        ax[1, 0].plot(GRID, risk[g], "o-", ms=3, label=name)
    ax[1, 0].set_xlabel("coverage")
    ax[1, 0].set_ylabel("error rate among answered")
    ax[1, 0].set_title(f"Risk-coverage, ensemble+TS, score = {score}")
    ax[1, 0].legend()

    bins = np.linspace(0, max(ei["scores"]["entropy"].max(), es["scores"]["entropy"].max()), 40)
    ax[1, 1].hist(ei["scores"]["entropy"], bins, weights=w_id / w_id.sum(), alpha=0.6, label="ID test (BCN class mix)")
    ax[1, 1].hist(es["scores"]["entropy"], bins, weights=np.ones(len(sh["labels"])) / len(sh["labels"]), alpha=0.6,
                  label="BCN shift")
    row = det[(det["method"] == "ensemble+TS") & (det["score"] == "entropy")]
    t = f"  (detection AUROC {row['detection_auroc'].iloc[0]:.2f})" if len(row) else ""
    ax[1, 1].set_title("Predictive entropy, ensemble+TS" + t)
    ax[1, 1].set_xlabel("entropy (nats)")
    ax[1, 1].set_ylabel("share of images")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
