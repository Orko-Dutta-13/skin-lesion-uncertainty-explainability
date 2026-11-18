"""Leakage-safe splitting.

Why this exists
---------------
ISIC 2019 contains several images of the *same lesion* (HAM10000 in particular).
A random image-level split puts near-duplicate views of one lesion into both train
and test, which inflates every metric you report. We therefore split by *group*
(lesion), stratified by diagnosis.

Limitation: ISIC 2019 has no patient identifier, so the
closest available unit is the lesion. Different lesions of one patient can still land
in different splits. Images without a lesion_id are treated as singleton groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

SHIFT_HOLDOUT_SOURCE = "BCN"


def _take_one_fold(df: pd.DataFrame, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rest_idx, fold_idx) positions where fold is one StratifiedGroupKFold test fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, fold = next(sgkf.split(df, df["label"], groups=df["group"]))
    return rest, fold


def _grouped_train_val_test(df: pd.DataFrame, seed: int) -> pd.Series:
    """~71 / 14.5 / 14.5 grouped + stratified split. Returns a Series of split names."""
    out = pd.Series("train", index=df.index, dtype=object)
    rest_pos, test_pos = _take_one_fold(df, n_splits=7, seed=seed)  # ~14.3 % test
    out.iloc[test_pos] = "test"
    rest = df.iloc[rest_pos]
    rest_pos2, val_pos = _take_one_fold(rest, n_splits=6, seed=seed + 1)  # ~16.7 % of rest
    out.loc[rest.index[val_pos]] = "val"
    return out


def add_ood_split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Source-shift protocol WITH an in-distribution test set.

    BCN images -> ``shift_test`` (never seen). All other images (HAM + MSK) are split by group into
    train / val / test (~71 / 14 / 14 %). ``test`` is the in-distribution reference for ``shift_test``:
    same models, same metrics, only the image source differs. (``split_shift`` has no in-distribution test.)
    """
    out = pd.Series("shift_test", index=df.index, dtype=object)
    sub = df[df["source"] != SHIFT_HOLDOUT_SOURCE]
    if len(sub) == 0:
        return out
    rest_pos, test_pos = _take_one_fold(sub, n_splits=7, seed=seed + 3)
    out.loc[sub.index[test_pos]] = "test"
    rest = sub.iloc[rest_pos]
    rest_pos2, val_pos = _take_one_fold(rest, n_splits=6, seed=seed + 4)
    out.loc[rest.index[rest_pos2]] = "train"
    out.loc[rest.index[val_pos]] = "val"
    return out


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add four split columns to ``df`` (needs columns: label, group, source).

    split        grouped, stratified train/val/test (the main protocol)
    split_naive  random *image-level* split with the same proportions. Only used for the
                 leakage ablation ("how much do metrics inflate if you split by image?")
    split_shift  source-shift protocol: all BCN images are ``shift_test``; the rest is split
                 into grouped train/val. Used for the distribution-shift experiments.
    split_ood    like split_shift but the non-BCN part is split into train/val/TEST, so there is an
                 in-distribution test set to compare against ``shift_test`` (see add_ood_split).
    """
    df = df.copy().reset_index(drop=True)

    df["split"] = _grouped_train_val_test(df, seed)

    # naive image-level split (deliberately leaky)
    naive = pd.Series("train", index=df.index, dtype=object)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 / 7, random_state=seed)
    rest_pos, test_pos = next(sss.split(df, df["label"]))
    naive.iloc[test_pos] = "test"
    rest = df.iloc[rest_pos]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=seed + 1)
    _, val_pos = next(sss2.split(rest, rest["label"]))
    naive.loc[rest.index[val_pos]] = "val"
    df["split_naive"] = naive

    # source-shift split
    shift = pd.Series("shift_test", index=df.index, dtype=object)
    in_dist = df["source"] != SHIFT_HOLDOUT_SOURCE
    sub = df[in_dist]
    if len(sub) > 0:
        rest_pos, val_pos = _take_one_fold(sub, n_splits=7, seed=seed + 2)
        shift.loc[sub.index[rest_pos]] = "train"
        shift.loc[sub.index[val_pos]] = "val"
    df["split_shift"] = shift
    df["split_ood"] = add_ood_split(df, seed)
    return df


def leakage_report(df: pd.DataFrame, split_col: str = "split") -> dict:
    """How many val/test images share a group (lesion) with a training image?"""
    train_groups = set(df.loc[df[split_col] == "train", "group"])
    report = {}
    for s in sorted(set(df[split_col]) - {"train"}):
        sub = df[df[split_col] == s]
        leaked = sub["group"].isin(train_groups)
        report[s] = {
            "n_images": int(len(sub)),
            "n_images_sharing_group_with_train": int(leaked.sum()),
            "pct": round(100.0 * float(leaked.mean()), 2) if len(sub) else 0.0,
        }
    return report


def assert_no_group_overlap(df: pd.DataFrame, split_col: str = "split") -> None:
    groups_by_split = {s: set(g["group"]) for s, g in df.groupby(split_col)}
    names = sorted(groups_by_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = groups_by_split[a] & groups_by_split[b]
            if overlap:
                raise AssertionError(
                    f"[{split_col}] {len(overlap)} groups appear in both '{a}' and '{b}'"
                )
