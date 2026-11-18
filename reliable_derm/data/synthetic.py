"""Tiny fake ISIC-2019-shaped dataset for smoke tests (no download needed).

    python -m reliable_derm.data.synthetic --out data/synthetic_raw

Then run the normal pipeline against it (see Makefile `smoke`). Class = dominant colour,
lesions have 1-3 near-identical images, three "sources" with the real image sizes, and
the MSK-like images have no lesion_id -- same quirks as the real data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from reliable_derm.constants import CLASSES


def make_raw(out: Path, n_lesions: int = 160, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img_dir = out / "ISIC_2019_Training_Input"
    img_dir.mkdir(parents=True, exist_ok=True)
    palette = rng.integers(40, 215, size=(len(CLASSES), 3))
    p = np.array([0.35, 0.25, 0.12, 0.08, 0.08, 0.05, 0.04, 0.03])  # imbalanced like the real data

    rows, n_img = [], 0
    for lesion in range(n_lesions):
        c = int(rng.choice(len(CLASSES), p=p))
        source = rng.choice(["HAM", "BCN", "MSK"], p=[0.45, 0.4, 0.15])
        size = {"HAM": (600, 450), "BCN": (1024, 1024), "MSK": (640 + 16 * int(rng.integers(0, 9)), 480)}[source]
        base = rng.normal(0, 18, size=(8, 8, 3))
        for _ in range(int(rng.integers(1, 4)) if source != "MSK" else 1):
            noise = base + rng.normal(0, 4, size=(8, 8, 3))
            arr = np.clip(palette[c] + noise, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr).resize(size, Image.BILINEAR)
            name = f"ISIC_{n_img:07d}"
            im.save(img_dir / f"{name}.jpg", quality=90)
            rows.append({
                "image": name, "label": c,
                "lesion_id": (f"{'HAM' if source == 'HAM' else 'BCN'}_{lesion:07d}" if source != "MSK" else np.nan),
                "age_approx": float(rng.integers(20, 85)),
                "anatom_site_general": str(rng.choice(["torso", "head/neck", "lower extremity"])),
                "sex": str(rng.choice(["male", "female"])),
            })
            n_img += 1

    df = pd.DataFrame(rows)
    gt = pd.DataFrame({"image": df["image"]})
    for i, name in enumerate(CLASSES):
        gt[name] = (df["label"] == i).astype(float)
    gt["UNK"] = 0.0
    gt.to_csv(out / "ISIC_2019_Training_GroundTruth.csv", index=False)
    df[["image", "age_approx", "anatom_site_general", "lesion_id", "sex"]].to_csv(
        out / "ISIC_2019_Training_Metadata.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_raw")
    ap.add_argument("--n-lesions", type=int, default=160)
    a = ap.parse_args(argv)
    make_raw(Path(a.out), a.n_lesions)
    print(f"wrote synthetic raw data to {a.out}")


if __name__ == "__main__":
    main()
