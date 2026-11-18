"""Build data/processed/master.csv from the raw ISIC 2019 download.

Usage
-----
python -m reliable_derm.data.prepare --raw-dir data/raw --out-dir data/processed --resize 256

What it does
------------
1. reads ground truth + metadata, merges them
2. finds every image file under --raw-dir
3. (one pass, multiprocessing) records the ORIGINAL image size and writes a resized copy
   (short side = --resize) so training is not bottlenecked by decoding 1024px+ JPEGs
4. infers the source dataset from the original size (HAM 600x450, BCN 1024x1024, else MSK/other)
5. builds the grouping key (lesion_id, or a singleton group if missing)
6. creates grouped / naive / source-shift splits, asserts no group leakage, writes reports
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

from reliable_derm.constants import CLASSES, SOURCE_BY_SIZE, SOURCE_OTHER
from reliable_derm.data.splits import assert_no_group_overlap, leakage_report, make_splits


def _find_one(raw_dir: Path, pattern: str) -> Path:
    hits = sorted(raw_dir.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' under {raw_dir}. See docs/DATA.md.")
    return hits[0]


def _index_images(raw_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in raw_dir.rglob("*.jpg")}


def _process_one(args: tuple[str, str, str | None, int]) -> tuple[str, int, int, str]:
    image, src, dst, short_side = args
    with Image.open(src) as im:
        w, h = im.size
        out_path = src
        if dst is not None:
            if not os.path.exists(dst):
                im = im.convert("RGB")
                scale = short_side / min(w, h)
                if scale < 1.0:
                    im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
                im.save(dst, quality=95)
            out_path = dst
    return image, w, h, out_path


def infer_source(w: int, h: int) -> str:
    return SOURCE_BY_SIZE.get((w, h), SOURCE_OTHER)


def build_master(raw_dir: Path, out_dir: Path, resize: int | None, workers: int, seed: int):
    gt = pd.read_csv(_find_one(raw_dir, "*Training_GroundTruth.csv"))
    meta = pd.read_csv(_find_one(raw_dir, "*Training_Metadata.csv"))
    missing = [c for c in CLASSES if c not in gt.columns]
    if missing:
        raise ValueError(f"Ground-truth CSV lacks class columns {missing}; found {list(gt.columns)}")

    df = gt.merge(meta, on="image", how="left", validate="one_to_one")
    df["label"] = df[CLASSES].to_numpy().argmax(axis=1)
    df["label_name"] = [CLASSES[i] for i in df["label"]]
    if "lesion_id" not in df.columns:
        print("WARNING: no lesion_id column in metadata -> every image becomes its own group "
              "(LEAKY). Check your metadata file.")
        df["lesion_id"] = pd.NA

    images = _index_images(raw_dir)
    absent = [i for i in df["image"] if i not in images]
    if absent:
        raise FileNotFoundError(f"{len(absent)} images from the CSV not found under {raw_dir}, "
                                f"e.g. {absent[:3]}")

    jobs = []
    if resize:
        cache = out_dir / f"images_{resize}"
        cache.mkdir(parents=True, exist_ok=True)
    for img in df["image"]:
        dst = str(cache / f"{img}.jpg") if resize else None
        jobs.append((img, str(images[img]), dst, resize or 0))

    print(f"Processing {len(jobs)} images with {workers} workers "
          f"({'resizing to short side ' + str(resize) if resize else 'no resize'}) ...")
    with Pool(workers) as pool:
        res = pool.map(_process_one, jobs, chunksize=64)
    info = pd.DataFrame(res, columns=["image", "orig_w", "orig_h", "path"])
    df = df.merge(info, on="image", how="left", validate="one_to_one")

    df["source"] = [infer_source(w, h) for w, h in zip(df["orig_w"], df["orig_h"])]
    df["group"] = df["lesion_id"].where(df["lesion_id"].notna(), "img_" + df["image"])

    df = make_splits(df, seed=seed)
    for col in ("split", "split_shift", "split_ood"):
        assert_no_group_overlap(df, col)  # hard fail if the grouped splits leak

    keep = ["image", "path", "label", "label_name", "group", "lesion_id", "source", "orig_w",
            "orig_h", "age_approx", "sex", "anatom_site_general", "split", "split_naive",
            "split_shift", "split_ood"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def write_reports(df: pd.DataFrame, out_dir: Path) -> dict:
    report = {
        "n_images": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "pct_images_without_lesion_id": round(100 * float(df["lesion_id"].isna().mean()), 2),
        "images_per_group_max": int(df.groupby("group").size().max()),
        "source_counts": df["source"].value_counts().to_dict(),
        "class_counts": df["label_name"].value_counts().to_dict(),
        "split_sizes": {c: df[c].value_counts().to_dict()
                        for c in ("split", "split_naive", "split_shift", "split_ood")},
        "leakage": {c: leakage_report(df, c) for c in ("split", "split_naive", "split_shift", "split_ood")},
        "class_by_split": pd.crosstab(df["label_name"], df["split"]).to_dict(),
        "class_by_source": pd.crosstab(df["label_name"], df["source"]).to_dict(),
    }
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--resize", type=int, default=256, help="short-side cache size; 0 = no resize")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_master(Path(a.raw_dir), out_dir, a.resize or None, a.workers, a.seed)
    df.to_csv(out_dir / "master.csv", index=False)
    rep = write_reports(df, out_dir)

    print(f"\nWrote {out_dir / 'master.csv'}  ({rep['n_images']} images, {rep['n_groups']} groups)")
    print(f"Images without lesion_id: {rep['pct_images_without_lesion_id']}%")
    print("Sources:", rep["source_counts"])
    print("Split sizes:", rep["split_sizes"]["split"])
    print("\nLeakage check (val/test images sharing a lesion with a train image):")
    for col, r in rep["leakage"].items():
        for s, v in r.items():
            print(f"  {col:12s} {s:10s} {v['n_images_sharing_group_with_train']:5d} / "
                  f"{v['n_images']:5d}  ({v['pct']}%)")
    print("\nExpected: 0% for 'split' and 'split_shift'; clearly > 0% for 'split_naive'.")


if __name__ == "__main__":
    main()
