import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA

from reliable_derm import drift
from reliable_derm.models import build_model


def test_image_statistics_on_known_images():
    flat = torch.full((2, 3, 32, 32), 0.5)
    s = drift.image_statistics(flat)
    names = drift.STAT_NAMES
    assert np.allclose(s[:, names.index("brightness")], 0.5, atol=1e-6)
    assert np.allclose(s[:, names.index("contrast")], 0, atol=1e-6)
    assert np.allclose(s[:, names.index("sharpness")], 0, atol=1e-6)
    assert np.allclose(s[:, names.index("saturation")], 0, atol=1e-6)
    black = drift.image_statistics(torch.zeros(1, 3, 16, 16))
    assert black[0, names.index("dark_share")] == 1.0
    noisy = torch.rand(1, 3, 32, 32)
    assert drift.image_statistics(noisy)[0, names.index("sharpness")] > 0.01


def test_class_matched_indices_follow_target_mix_without_replacement():
    rng = np.random.default_rng(0)
    pool = rng.choice(4, 5000, p=[0.1, 0.1, 0.1, 0.7])
    target = rng.choice(4, 1000, p=[0.4, 0.3, 0.2, 0.1])
    idx = drift.class_matched_indices(pool, target, 800, np.random.default_rng(1))
    assert len(set(idx)) == len(idx)
    mix = np.bincount(pool[idx], minlength=4) / len(idx)
    assert np.allclose(mix, np.bincount(target, minlength=4) / 1000, atol=0.02)


def test_domain_auroc_separates_shifted_data_and_not_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (600, 6))
    assert drift.domain_auroc(a, rng.normal(0, 1, (600, 6))) == pytest.approx(0.5, abs=0.07)
    assert drift.domain_auroc(a, rng.normal(1.5, 1, (600, 6))) > 0.9


def test_monitoring_table_columns_and_pca_uses_reference_only():
    rng = np.random.default_rng(0)
    d = {"emb": rng.normal(size=(100, 20)), "logits": rng.normal(size=(100, 8)), "stats": rng.random((100, 8))}
    pca = PCA(4).fit(d["emb"][:50])
    t = drift.monitoring_table(d, np.arange(50, 100), pca)
    assert list(t.columns) == drift.STAT_NAMES + [f"emb_pc{i}" for i in range(1, 5)] + drift.OUT_NAMES
    assert t["max_prob"].between(1 / 8, 1).all() and (t["entropy"] >= 0).all()
    assert {drift.column_group(c) for c in t.columns} == set(drift.GROUPS)


def test_run_evidently_flags_only_the_shifted_column(tmp_path):
    pytest.importorskip("evidently")
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.normal(0, 1, (1500, 3)), columns=["a", "b", "c"])
    cur = pd.DataFrame(rng.normal(0, 1, (1500, 3)), columns=["a", "b", "c"])
    cur["b"] += 1.5
    r = drift.run_evidently(ref, cur, tmp_path / "r.html").set_index("column")
    assert r.loc["b", "drifted"] and not r.loc["a", "drifted"] and not r.loc["c", "drifted"]
    assert r["drift_strength"].idxmax() == "b"
    assert (tmp_path / "r.html").stat().st_size > 1000


def test_run_evidently_survives_constant_columns(tmp_path):
    pytest.importorskip("evidently")
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 800), "dark": np.zeros(800), "same": np.full(800, 3.0)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 800), "dark": np.zeros(800), "same": np.full(800, 4.0)})
    r = drift.run_evidently(ref, cur).set_index("column")
    assert set(r.index) == {"a", "dark", "same"}
    assert not r.loc["dark", "drifted"] and r.loc["same", "drifted"] and not r.loc["a", "drifted"]
    assert r.loc["dark", "method"] == "constant (not tested)"
    # float32 noise around a constant (e.g. a saturated softmax) must not crash Evidently either
    noisy = pd.DataFrame({"a": rng.normal(0, 1, 800), "p": 0.14 + rng.normal(0, 1e-9, 800)}).astype("float32")
    assert drift.run_evidently(noisy, noisy.copy()).set_index("column").loc["p", "method"] == "constant (not tested)"


def make_images(root, n, bright, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        arr = np.clip(rng.normal(bright, 25, (40, 40, 3)), 0, 255).astype(np.uint8)
        p = root / f"{bright}_{i}.jpg"
        Image.fromarray(arr).save(p)
        rows.append({"image": p.stem, "path": str(p), "label": int(rng.integers(0, 8))})
    return pd.DataFrame(rows)


def test_extract_features_shapes_and_hook_cleanup(tmp_path):
    torch.manual_seed(0)
    m = build_model("efficientnet_b0", 8, pretrained=False).eval()
    df = make_images(tmp_path, 5, 120, 0)
    d = drift.extract_features(m, df, 64, torch.device("cpu"), batch_size=2)
    assert d["emb"].shape == (5, 1280) and d["logits"].shape == (5, 8) and d["stats"].shape == (5, 8)
    assert len(m.avgpool._forward_hooks) == 0                    # hook removed
    with torch.no_grad():
        assert np.allclose(m.classifier(torch.tensor(d["emb"], dtype=torch.float32)).numpy(), d["logits"], atol=1e-4)


def test_cli_end_to_end_with_cache(tmp_path, monkeypatch):
    torch.manual_seed(0)
    m = build_model("efficientnet_b0", 8, pretrained=False)
    cfg = {"model": {"arch": "efficientnet_b0", "dropout": 0.3, "pretrained": False},
           "data": {"master_csv": str(tmp_path / "master.csv"), "image_size": 64, "split_col": "split_ood", "num_workers": 0},
           "seed": 0}
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"model": m.state_dict(), "config": cfg, "classes": list("ABCDEFGH"), "epoch": 1}, run / "best.pt")
    parts = []
    for split, bright, n in (("val", 100, 60), ("test", 100, 60), ("shift_test", 190, 120)):
        d = make_images(tmp_path, n, bright, hash(split) % 1000)
        d["image"] = [f"{split}_{i}" for i in range(n)]
        d["label"] = np.arange(n) % 8
        d["split_ood"] = split
        parts.append(d)
    pd.concat(parts).to_csv(tmp_path / "master.csv", index=False)
    out = tmp_path / "out"
    drift.main(["--ckpt", str(run / "best.pt"), "--n", "50", "--out", str(out), "--device", "cpu"])
    s = pd.read_csv(out / "drift_summary.csv").set_index("scenario")
    assert s.loc["shift (BCN)", "domain_auroc_image_stats"] > 0.95
    assert s.loc["control (ID test)", "domain_auroc_image_stats"] < 0.75
    assert (out / "drift_overview.png").exists() and (run / "features_split_ood_val.npz").exists()
    drift.main(["--ckpt", str(run / "best.pt"), "--n", "50", "--out", str(out), "--device", "cpu"])   # uses the cache
