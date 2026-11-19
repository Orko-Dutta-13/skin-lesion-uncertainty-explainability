"""Per-image out-of-distribution (OOD) detection: can a detector flag BCN images as "not like my training data"?

    python -m reliable_derm.ood                      # 3 shift models, needs the features of drift.py or computes them

Question
    Step 3.2 showed that the model's own confidence is a weak shift detector (AUROC about 0.7) and Step 3.3 that
    a domain classifier trained WITH BCN images separates the sources easily (AUROC about 0.9). This module asks the
    deployment-relevant question in between: detectors that are fitted on in-distribution (ID) training data only,
    never see a BCN image, and score every image on its own. Can they beat "just use the confidence"?

Detectors (score: higher = more out-of-distribution)
    output based    msp          1 - maximum softmax probability
                    entropy      predictive entropy
                    energy       -logsumexp(logits)                                  (Liu et al. 2020)
    feature based   mahalanobis  min over classes of the squared Mahalanobis distance of the 1280-d penultimate
                                 embedding to the class mean, tied shrinkage covariance (Lee et al. 2018)
                    rel_mahal.   the same minus the distance to one class-agnostic Gaussian        (Ren et al. 2021)
                    knn          distance to the k-th nearest ID training embedding, L2-normalised (Sun et al. 2022)
Fitting uses ID TRAINING images only (HAM + MSK, split_ood = train).

Evaluation (AUROC, positive = BCN image)
    ID side = ID test images (HAM + MSK) re-weighted to the class mix of BCN, so a detector cannot win just by
    spotting that BCN has fewer melanocytic nevi. AUROC without weights and FPR@95%TPR (share of ID images flagged when
    95 % of BCN images are flagged) are also reported. Mean +- sample SD over the shift models (seeds).
Outputs in reports/ood: per_run.csv, summary.csv, ood_overview.png
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
from scipy.special import entr, logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from reliable_derm.drift import load_or_extract
from reliable_derm.engine import load_checkpoint
from reliable_derm.shift import class_weights, detection_auroc
from reliable_derm.utils import get_device

DETECTORS = ["msp", "entropy", "energy", "mahalanobis", "rel_mahalanobis", "knn"]
NAMES = {"msp": "max-softmax (msp)", "entropy": "entropy", "energy": "energy", "mahalanobis": "Mahalanobis",
         "rel_mahalanobis": "relative Mahalanobis", "knn": "kNN distance"}
FAMILY = {"msp": "output", "entropy": "output", "energy": "output",
          "mahalanobis": "feature", "rel_mahalanobis": "feature", "knn": "feature"}


# ----------------------------------------------------------------------------- output-based scores
def msp_score(logits: np.ndarray) -> np.ndarray:
    return 1.0 - softmax(logits, axis=1).max(1)


def entropy_score(logits: np.ndarray) -> np.ndarray:
    return entr(softmax(logits, axis=1)).sum(1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """-T * logsumexp(logits / T): low energy = confident = in-distribution."""
    return -temperature * logsumexp(logits / temperature, axis=1)


# ----------------------------------------------------------------------------- feature-based scores
class Mahalanobis:
    """Class-conditional Gaussians with one tied (shrunk) covariance, plus a class-agnostic background Gaussian."""

    def __init__(self, emb: np.ndarray, labels: np.ndarray):
        emb = np.asarray(emb, np.float64)
        self.classes = np.unique(labels)
        self.means = np.stack([emb[labels == c].mean(0) for c in self.classes])
        resid = emb - self.means[np.searchsorted(self.classes, labels)]
        self.prec = LedoitWolf().fit(resid).precision_
        self.bg_mean = emb.mean(0)
        self.bg_prec = LedoitWolf().fit(emb).precision_

    @staticmethod
    def _sq(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
        d = x - mu
        return np.einsum("ij,jk,ik->i", d, prec, d, optimize=True)

    def scores(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (min-over-classes squared distance, the same minus the background distance)."""
        x = np.asarray(emb, np.float64)
        per_class = np.stack([self._sq(x, mu, self.prec) for mu in self.means], 1)
        md = per_class.min(1)
        return md, md - self._sq(x, self.bg_mean, self.bg_prec)


def knn_score(train_emb: np.ndarray, emb: np.ndarray, k: int = 10, chunk: int = 2048) -> np.ndarray:
    """Euclidean distance (on L2-normalised embeddings) to the k-th nearest training embedding."""
    t = torch.as_tensor(train_emb, dtype=torch.float32)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q = torch.as_tensor(emb, dtype=torch.float32)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k = min(k, len(t))
    out = []
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk] @ t.T
        kth = sim.topk(k, dim=1).values[:, -1]
        out.append((2.0 - 2.0 * kth).clamp_min(0).sqrt())
    return torch.cat(out).numpy().astype(np.float64)


# ----------------------------------------------------------------------------- metrics
def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, w_id: np.ndarray, tpr: float = 0.95) -> float:
    """Weighted share of ID images flagged at the threshold that flags a fraction ``tpr`` of the OOD images."""
    thr = np.quantile(ood_scores, 1.0 - tpr)
    return float(w_id[id_scores >= thr].sum() / w_id.sum())


def all_scores(train: dict, query: dict, k: int, maha: Mahalanobis | None = None) -> dict[str, np.ndarray]:
    maha = maha or Mahalanobis(train["emb"], train["labels"])
    md, rmd = maha.scores(query["emb"])
    return {"msp": msp_score(query["logits"]), "entropy": entropy_score(query["logits"]),
            "energy": energy_score(query["logits"]), "mahalanobis": md, "rel_mahalanobis": rmd,
            "knn": knn_score(train["emb"], query["emb"], k)}


def evaluate_run(train: dict, id_test: dict, ood: dict, k: int = 10) -> tuple[pd.DataFrame, dict]:
    """One model -> table (detector, auroc_weighted, auroc_unweighted, fpr95_weighted) and the raw scores."""
    maha = Mahalanobis(train["emb"], train["labels"])
    s_id, s_ood = all_scores(train, id_test, k, maha), all_scores(train, ood, k, maha)
    w = class_weights(id_test["labels"], ood["labels"])
    ones = np.ones(len(id_test["labels"]))
    rows = []
    for d in DETECTORS:
        y = np.r_[np.zeros(len(s_id[d])), np.ones(len(s_ood[d]))]
        rows.append({"detector": d,
                     "auroc_weighted": detection_auroc(s_id[d], s_ood[d], w),
                     "auroc_unweighted": float(roc_auc_score(y, np.r_[s_id[d], s_ood[d]])),
                     "fpr95_weighted": fpr_at_tpr(s_id[d], s_ood[d], w)})
    return pd.DataFrame(rows), {"id": s_id, "ood": s_ood, "w": w, "ones": ones}


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    g = per_run.groupby("detector", sort=False)
    out = g.agg(auroc_mean=("auroc_weighted", "mean"), auroc_sd=("auroc_weighted", lambda v: v.std(ddof=1)),
                auroc_unweighted_mean=("auroc_unweighted", "mean"), fpr95_mean=("fpr95_weighted", "mean"),
                fpr95_sd=("fpr95_weighted", lambda v: v.std(ddof=1)), runs=("run", "nunique")).reset_index()
    out["family"] = out["detector"].map(FAMILY)
    return out


# ----------------------------------------------------------------------------- reference numbers from earlier steps
def read_reference(shift_dir: Path, drift_dir: Path) -> dict[str, float]:
    ref = {}
    try:
        det = pd.read_csv(shift_dir / "detection.csv")
        row = det[(det["method"] == "ensemble+TS") & (det["score"] == "maxprob")]
        if len(row):
            ref["ensemble max-prob, no BCN seen (Step 3.2)"] = float(row["detection_auroc"].iloc[0])
    except Exception:
        pass
    try:
        dr = pd.read_csv(drift_dir / "drift_summary.csv")
        row = dr[dr["scenario"].str.contains("class-matched")]
        if len(row):
            ref["domain classifier, trained with BCN (Step 3.3)"] = float(row["domain_auroc_image_stats"].iloc[0])
    except Exception:
        pass
    return ref


# ----------------------------------------------------------------------------- figure
def plot_overview(path: Path, summ: pd.DataFrame, example: dict, ref: dict[str, float]) -> None:
    ink, muted, bar, c_id, c_ood = "#1F2933", "#6B7785", "#3D6A99", "#3D6A99", "#D9822B"
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = [f"{NAMES[d]}  ({FAMILY[d]})" for d in summ["detector"]]
    means, sds, colors = list(summ["auroc_mean"]), list(summ["auroc_sd"].fillna(0)), [bar] * len(summ)
    for name, v in ref.items():                                   # reference numbers from earlier steps, own bars
        labels.append(name)
        means.append(v)
        sds.append(0.0)
        colors.append(c_ood)
    n = len(labels)
    y = np.arange(n)[::-1]
    a = ax[0]
    a.barh(y, means, height=0.55, color=colors, xerr=sds, ecolor=ink, error_kw={"elinewidth": 1.2, "capsize": 3})
    for yi, m, sd in zip(y, means, sds):
        a.text(m + sd + 0.01, yi, f"{m:.2f}", va="center", ha="left", fontsize=9, color=ink)
    a.axvline(0.5, color=muted, lw=1, ls=":")
    a.set_yticks(y)
    a.set_yticklabels(labels, color=ink, fontsize=9)
    a.set_xlim(0.4, 1.0)
    a.set_xlabel("AUROC, ID test (BCN class mix) vs BCN; bars: mean ± SD over models; dotted: chance", color=ink)
    a.set_title("Detectors fitted on ID training data only (blue) vs earlier steps (orange)", color=ink, loc="left",
                fontsize=10)
    best = summ.loc[summ["auroc_mean"].idxmax(), "detector"]
    b = ax[1]
    sid, sood, w = example["id"][best], example["ood"][best], example["w"]
    lo, hi = float(min(sid.min(), sood.min())), float(np.percentile(np.r_[sid, sood], 99.5))
    bins = np.linspace(lo, hi, 50)
    b.hist(np.clip(sid, lo, hi), bins=bins, weights=w / w.sum(), color=c_id, alpha=0.65, label="ID test (BCN class mix)")
    b.hist(np.clip(sood, lo, hi), bins=bins, weights=np.ones(len(sood)) / len(sood), color=c_ood, alpha=0.65,
           label="BCN (shifted)")
    b.set_xlabel(f"{NAMES[best]} score (higher = more out-of-distribution)", color=ink)
    b.set_ylabel("share of images", color=ink)
    b.set_title(f"Best detector, first model: {NAMES[best]}", color=ink, loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=9)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(colors=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", default=["runs/shift", "runs/shift_s1", "runs/shift_s2"])
    ap.add_argument("--tag", default="split_ood", help="split column; needs train / test / shift_test")
    ap.add_argument("--k", type=int, default=10, help="k of the kNN detector")
    ap.add_argument("--max-train", type=int, default=0, help="fit detectors on a random subset of the ID training images "
                    "(0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/ood")
    ap.add_argument("--shift-dir", default="reports/shift")
    ap.add_argument("--drift-dir", default="reports/drift")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = get_device(a.device)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, example = [], None
    for run in a.run_dirs:
        ckpt = Path(run) / "best.pt"
        model, cfg = load_checkpoint(str(ckpt), device)
        master = pd.read_csv(cfg["data"]["master_csv"])
        feats = {}
        for split in ("train", "test", "shift_test"):
            sub = master[master[a.tag] == split].reset_index(drop=True)
            feats[split] = load_or_extract(ckpt, model, cfg, sub, a.tag, split, device, a.recompute, a.num_workers)
        train = feats["train"]
        if a.max_train and len(train["labels"]) > a.max_train:
            idx = rng.choice(len(train["labels"]), a.max_train, replace=False)
            train = {k_: v[idx] for k_, v in train.items()}
        tab, sc = evaluate_run(train, feats["test"], feats["shift_test"], a.k)
        tab.insert(0, "run", run)
        rows.append(tab)
        example = example or sc
        print(f"{run}: fitted on {len(train['labels'])} ID train images, ID test n={len(feats['test']['labels'])}, "
              f"BCN n={len(feats['shift_test']['labels'])}")
    per_run = pd.concat(rows, ignore_index=True)
    summ = summarise(per_run)
    per_run.to_csv(out / "per_run.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    ref = read_reference(Path(a.shift_dir), Path(a.drift_dir))
    plot_overview(out / "ood_overview.png", summ, example, ref)

    n = per_run["run"].nunique()
    print(f"\nCan a detector fitted on ID training data tell BCN images from ID test images? "
          f"(AUROC, 0.5 = no; mean ± SD over {n} model(s); ID weighted to the BCN class mix)\n")
    show = summ.copy()
    show["detector"] = show["detector"].map(NAMES)
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"
    show["AUROC"] = [fmt(m, s) for m, s in zip(show["auroc_mean"], show["auroc_sd"])]
    show["AUROC (no weights)"] = show["auroc_unweighted_mean"].round(3)
    show["FPR@95%TPR"] = [fmt(m, s) for m, s in zip(show["fpr95_mean"], show["fpr95_sd"])]
    print(show[["detector", "family", "AUROC", "AUROC (no weights)", "FPR@95%TPR"]].to_string(index=False))
    if ref:
        print("\nFor comparison (earlier steps):")
        for name, v in ref.items():
            print(f"  {name}: {v:.3f}")
    print(f"\nsaved: {out / 'per_run.csv'}, summary.csv, ood_overview.png")


if __name__ == "__main__":
    main()
