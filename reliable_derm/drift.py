"""Drift monitoring with Evidently: can we SEE that the images come from a different source, without labels?

    python -m reliable_derm.drift --ckpt runs\\shift\\best.pt

What is compared (reference = in-distribution validation images, HAM + MSK)
    control            ID test images                     expected: no drift (a monitor that cries wolf is useless)
    shift              BCN images, natural class mix      drift in appearance AND class mix
    shift (matched)    BCN images sampled to the same class mix as the reference    -> appearance drift only
Monitored columns (all computable without labels, as in a real deployment)
    image statistics   brightness, contrast, saturation, sharpness, dark-border share, mean R/G/B
    embedding          8 PCA components of the 1280-d penultimate feature (PCA fitted on the reference only)
    model output       max softmax probability, predictive entropy
Two views of the same question:
    Evidently DataDriftPreset (per-column tests, share of drifted columns), pinned version in requirements.txt
    domain classifier: cross-validated AUROC of a logistic regression that tries to tell reference from current images
                       (0.5 = indistinguishable, 1.0 = perfectly separable). Needs no p-value / threshold tuning.
Outputs in reports/drift: drift_summary.csv, drift_columns.csv, drift_overview.png, evidently_<scenario>.html
(the HTML reports are several MB each: open them in a browser, do not commit them).
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
from scipy.special import entr, softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reliable_derm.data.dataset import SkinDataset, build_transforms, denormalize
from reliable_derm.engine import load_checkpoint
from reliable_derm.utils import get_device

STAT_NAMES = ["brightness", "contrast", "saturation", "sharpness", "dark_share", "mean_r", "mean_g", "mean_b"]
OUT_NAMES = ["max_prob", "entropy"]
GROUPS = ("image statistics", "embedding", "model output")


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def image_statistics(x01: torch.Tensor) -> np.ndarray:
    """Label-free appearance statistics of a batch (B,3,H,W) in [0,1] -> (B, len(STAT_NAMES))."""
    gray = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2])[:, None]
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x01.dtype, device=x01.device).view(1, 1, 3, 3)
    sharp = F.conv2d(gray, lap).flatten(1).var(1)
    sat = (x01.amax(1) - x01.amin(1)).flatten(1).mean(1)
    cols = [gray.flatten(1).mean(1), gray.flatten(1).std(1), sat, sharp, (gray < 0.1).flatten(1).float().mean(1),
            x01[:, 0].flatten(1).mean(1), x01[:, 1].flatten(1).mean(1), x01[:, 2].flatten(1).mean(1)]
    return torch.stack(cols, 1).cpu().numpy()


@torch.no_grad()
def extract_features(model: torch.nn.Module, df: pd.DataFrame, size: int, device: torch.device,
                     batch_size: int = 64, num_workers: int = 0) -> dict:
    """One pass over the images: penultimate embedding, logits and image statistics."""
    model.eval()
    captured = {}
    h = model.avgpool.register_forward_hook(lambda _m, _i, out: captured.__setitem__("f", out.flatten(1)))
    loader = torch.utils.data.DataLoader(SkinDataset(df, build_transforms(size, train=False)),
                                         batch_size=batch_size, shuffle=False, num_workers=num_workers)
    emb, logits, stats = [], [], []
    try:
        for x, _, _ in loader:
            x = x.to(device)
            logits.append(model(x).float().cpu().numpy())
            emb.append(captured["f"].float().cpu().numpy())
            stats.append(image_statistics(denormalize(x)))
    finally:
        h.remove()
    return {"emb": np.concatenate(emb), "logits": np.concatenate(logits), "stats": np.concatenate(stats),
            "labels": df["label"].to_numpy(int), "images": df["image"].to_numpy()}


def load_or_extract(ckpt: Path, model, cfg, df: pd.DataFrame, tag: str, split: str, device, recompute: bool,
                    num_workers: int) -> dict:
    f = ckpt.parent / f"features_{tag}_{split}.npz"
    if f.exists() and not recompute:
        z = np.load(f, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        if list(d["images"]) == df["image"].tolist():
            return d
    print(f"extracting features for {tag}={split} ({len(df)} images) ...")
    d = extract_features(model, df, cfg["data"]["image_size"], device, num_workers=num_workers)
    np.savez_compressed(f, **d)
    return d


# ----------------------------------------------------------------------------- sampling and tables
def class_matched_indices(labels: np.ndarray, target_labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ~n indices from ``labels`` (without replacement) with the class mix of ``target_labels``."""
    k = int(max(labels.max(), target_labels.max())) + 1
    quota = np.round(np.bincount(target_labels, minlength=k) / len(target_labels) * n).astype(int)
    idx = []
    for c in range(k):
        pool = np.flatnonzero(labels == c)
        idx.append(rng.choice(pool, size=min(quota[c], len(pool)), replace=False))
    return np.concatenate(idx)


def monitoring_table(d: dict, idx: np.ndarray, pca: PCA) -> pd.DataFrame:
    p = softmax(d["logits"][idx], axis=1)
    t = pd.DataFrame(d["stats"][idx], columns=STAT_NAMES)
    pcs = pd.DataFrame(pca.transform(d["emb"][idx]), columns=[f"emb_pc{i + 1}" for i in range(pca.n_components_)])
    out = pd.DataFrame({"max_prob": p.max(1), "entropy": entr(p).sum(1)})
    return pd.concat([t, pcs, out], axis=1).reset_index(drop=True)


def column_group(c: str) -> str:
    return "embedding" if c.startswith("emb_pc") else "model output" if c in OUT_NAMES else "image statistics"


def domain_auroc(ref: np.ndarray, cur: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """Cross-validated AUROC of a logistic regression separating reference (0) from current (1) rows."""
    X, y = np.vstack([ref, cur]), np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


# ----------------------------------------------------------------------------- Evidently
def run_evidently(ref: pd.DataFrame, cur: pd.DataFrame, html: Path | None = None) -> pd.DataFrame:
    """Evidently DataDriftPreset -> one row per column (score, test, threshold, drifted).

    Columns that are (numerically) constant in the reference or the current sample (spread below 1e-5 of their
    magnitude) make Evidently's histogram step fail, so they are not passed to it: they get a row with method "constant (not tested)" and drifted = "the constant
    value changed".
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    def is_const(x: pd.Series) -> bool:      # no usable spread relative to the magnitude (float32 resolution is ~1e-7)
        return float(np.ptp(x)) <= 1e-5 * max(float(np.abs(x).max()), 1e-12)

    const = [c for c in ref.columns if is_const(ref[c]) or is_const(cur[c])]
    use = [c for c in ref.columns if c not in const]
    rows = [{"column": c, "method": "constant (not tested)", "score": float("nan"), "threshold": float("nan"),
             "drift_strength": float("nan"), "drifted": bool(abs(cur[c].mean() - ref[c].mean()) > 1e-6)}
            for c in const]
    if use:
        dd = DataDefinition(numerical_columns=use)
        res = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur[use], data_definition=dd),
                                              Dataset.from_pandas(ref[use], data_definition=dd))
        if html is not None:
            res.save_html(str(html))
        for m in res.dict()["metrics"]:
            cfg = m.get("config", {})
            if cfg.get("column") is None or "ValueDrift" not in str(cfg.get("type", "")):
                continue
            method, thr, score = cfg.get("method", ""), cfg.get("threshold"), m["value"]
            p_value_test = ("p_value" in method.lower()) or method.lower().startswith(("k-s", "kolmogorov", "chi", "z-test"))
            drifted = score < thr if p_value_test else score > thr
            rows.append({"column": cfg["column"], "method": method, "score": float(score), "threshold": thr,
                         "drift_strength": float(1 - score) if p_value_test else float(score),   # larger = more drift
                         "drifted": bool(drifted)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", default="runs/shift/best.pt")
    ap.add_argument("--tag", default="split_ood", help="split column; needs val / test / shift_test")
    ap.add_argument("--n", type=int, default=1800, help="images per compared sample")
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/drift")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = get_device(a.device)
    ckpt = Path(a.ckpt)
    model, cfg = load_checkpoint(a.ckpt, device)
    master = pd.read_csv(cfg["data"]["master_csv"])
    feats = {}
    for split in ("val", "test", "shift_test"):
        sub = master[master[a.tag] == split].reset_index(drop=True)
        feats[split] = load_or_extract(ckpt, model, cfg, sub, a.tag, split, device, a.recompute, a.num_workers)
    rng = np.random.default_rng(a.seed)
    n = min(a.n, len(feats["val"]["labels"]), len(feats["test"]["labels"]))
    ref_idx = rng.choice(len(feats["val"]["labels"]), n, replace=False)
    scenarios = {
        "control (ID test)": ("test", rng.choice(len(feats["test"]["labels"]), n, replace=False)),
        "shift (BCN)": ("shift_test", rng.choice(len(feats["shift_test"]["labels"]), n, replace=False)),
        "shift, class-matched (BCN)": ("shift_test", class_matched_indices(
            feats["shift_test"]["labels"], feats["val"]["labels"][ref_idx], n, rng)),
    }
    pca = PCA(a.n_pca, random_state=a.seed).fit(feats["val"]["emb"][ref_idx])      # reference only
    ref_tab = monitoring_table(feats["val"], ref_idx, pca)
    pca32 = PCA(min(32, len(ref_idx) - 1), random_state=a.seed).fit(feats["val"]["emb"][ref_idx])
    print(f"reference: {len(ref_idx)} ID validation images | model {ckpt}")
    try:
        import evidently
        print(f"Evidently {evidently.__version__}")
        have_ev = True
    except ImportError:
        print("evidently not installed -> skipping the Evidently part (pip install evidently==0.7.23)")
        have_ev = False

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    col_rows, sum_rows, tabs = [], [], {}
    for name, (split, idx) in scenarios.items():
        cur_tab = monitoring_table(feats[split], idx, pca)
        tabs[name] = (cur_tab, feats[split]["labels"][idx])
        row = {"scenario": name, "n_current": len(idx)}
        if have_ev:
            ev = run_evidently(ref_tab, cur_tab, out / f"evidently_{name.split(' (')[0].replace(', ', '_').replace(' ', '_')}.html")
            ev["group"] = ev["column"].map(column_group)
            ev.insert(0, "scenario", name)
            col_rows.append(ev)
            row["share_drifted_all"] = ev["drifted"].mean()
            for g in GROUPS:
                row[f"share_drifted_{g}"] = ev.loc[ev["group"] == g, "drifted"].mean()
        emb_cols = [c for c in ref_tab.columns if c.startswith("emb_pc")]
        row["domain_auroc_embedding"] = domain_auroc(pca32.transform(feats["val"]["emb"][ref_idx]),
                                                     pca32.transform(feats[split]["emb"][idx]), a.seed)
        row["domain_auroc_image_stats"] = domain_auroc(ref_tab[STAT_NAMES].to_numpy(), cur_tab[STAT_NAMES].to_numpy(),
                                                       a.seed)
        row["domain_auroc_model_output"] = domain_auroc(ref_tab[OUT_NAMES].to_numpy(), cur_tab[OUT_NAMES].to_numpy(),
                                                        a.seed)
        sum_rows.append(row)
    summ = pd.DataFrame(sum_rows)
    summ.to_csv(out / "drift_summary.csv", index=False)
    cols = pd.concat(col_rows) if col_rows else pd.DataFrame()
    if len(cols):
        cols.to_csv(out / "drift_columns.csv", index=False)

    print("\nShare of monitored columns flagged as drifted by Evidently (reference = ID validation):")
    show = [c for c in summ.columns if c.startswith("share_") or c in ("scenario",)]
    print(summ[show].round(2).to_string(index=False) if have_ev else "(skipped)")
    print("\nDomain classifier AUROC (0.5 = reference and current indistinguishable):")
    print(summ[["scenario"] + [c for c in summ.columns if c.startswith("domain_")]].round(3).to_string(index=False))
    if len(cols):
        top = cols[cols["scenario"] == "shift, class-matched (BCN)"].sort_values("drift_strength", ascending=False, na_position="last")
        print("\nMost drifted columns, class-matched BCN vs reference (score = Evidently's statistic; for p-value tests drift means p < threshold):")
        print(top[["column", "group", "method", "score", "threshold", "drifted"]].head(8).round(3).to_string(index=False))

    plot_overview(out / "drift_overview.png", summ, ref_tab, tabs, cols, have_ev)
    print(f"\nsaved: {out}/drift_summary.csv, drift_columns.csv, drift_overview.png"
          + (", evidently_*.html" if have_ev else ""))


def plot_overview(path: Path, summ: pd.DataFrame, ref: pd.DataFrame, tabs: dict, cols: pd.DataFrame, have_ev: bool) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    names = list(summ["scenario"])
    x, wbar = np.arange(len(names)), 0.25
    if have_ev:
        for i, g in enumerate(GROUPS):
            ax[0, 0].bar(x + (i - 1) * wbar, summ[f"share_drifted_{g}"], wbar, label=g)
        ax[0, 0].set_ylabel("share of columns flagged")
        ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("Evidently: columns flagged as drifted")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    for i, (k, lab) in enumerate((("domain_auroc_image_stats", "image statistics"), ("domain_auroc_embedding", "embedding"),
                                   ("domain_auroc_model_output", "model output"))):
        ax[0, 1].bar(x + (i - 1) * wbar, summ[k], wbar, label=lab)
    ax[0, 1].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[0, 1].set_ylim(0.4, 1.02)
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax[0, 1].set_title("Domain classifier AUROC (0.5 = indistinguishable)")
    ax[0, 1].legend(fontsize=8)
    cur = tabs["shift, class-matched (BCN)"][0]
    ax[0, 2].scatter(ref["emb_pc1"], ref["emb_pc2"], s=4, alpha=0.4, label="reference (ID val)")
    ax[0, 2].scatter(cur["emb_pc1"], cur["emb_pc2"], s=4, alpha=0.4, label="BCN (class-matched)")
    ax[0, 2].set_xlabel("embedding PC1")
    ax[0, 2].set_ylabel("embedding PC2")
    ax[0, 2].set_title("Penultimate-layer embedding")
    ax[0, 2].legend(fontsize=8, markerscale=3)
    if len(cols):
        top = (cols[(cols["scenario"] == "shift, class-matched (BCN)") & (cols["group"] == "image statistics")]
               .sort_values("drift_strength", ascending=False, na_position="last")["column"].tolist()[:2])
    else:
        top = ["brightness", "sharpness"]
    for j, c in enumerate(top + ["max_prob"]):
        a = ax[1, j]
        lo, hi = np.percentile(np.r_[ref[c], cur[c]], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        a.hist(ref[c], bins, alpha=0.55, density=True, label="reference (ID val)")
        a.hist(cur[c], bins, alpha=0.55, density=True, label="BCN (class-matched)")
        a.set_title(c)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
