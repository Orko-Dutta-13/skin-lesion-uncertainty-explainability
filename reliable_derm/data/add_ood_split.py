"""Add the ``split_ood`` column to an EXISTING master.csv (no need to re-run prepare / re-read the images).

    python -m reliable_derm.data.add_ood_split

Writes master.csv in place (a backup master.csv.bak is kept) and prints the size of every split and its
class/source composition. Existing columns, including ``split`` used by the baseline runs, are not touched.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from reliable_derm.data.splits import add_ood_split, assert_no_group_overlap, leakage_report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default="data/processed/master.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    path = Path(a.master_csv)
    df = pd.read_csv(path)
    before = {c: df[c].copy() for c in df.columns if c != "split_ood"}
    df["split_ood"] = add_ood_split(df, a.seed)
    assert_no_group_overlap(df, "split_ood")
    for c, col in before.items():
        assert df[c].equals(col), f"column {c} changed"
    bak = path.with_suffix(".csv.bak")
    if not bak.exists():
        shutil.copy(path, bak)
    df.to_csv(path, index=False)

    print("split_ood sizes:\n", df["split_ood"].value_counts().to_string())
    print("\nclass x split_ood:\n", pd.crosstab(df["label_name" if "label_name" in df else "label"], df["split_ood"]).to_string())
    print("\nsource x split_ood:\n", pd.crosstab(df["source"], df["split_ood"]).to_string())
    print("\nleakage (images sharing a lesion with train):",
          {k: v["n_images_sharing_group_with_train"] for k, v in leakage_report(df, "split_ood").items()})
    print(f"\nwritten -> {path}  (backup: {bak})")


if __name__ == "__main__":
    main()
