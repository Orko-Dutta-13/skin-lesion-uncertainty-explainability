"""Whole pipeline on fake data: prepare -> train -> evaluate -> gradcam. Takes ~1 min on CPU."""
import json

import pandas as pd

from reliable_derm.data import prepare, synthetic
from reliable_derm import evaluate, train
from reliable_derm.explain import gradcam


def test_end_to_end(tmp_path):
    raw, proc, run = tmp_path / "raw", tmp_path / "proc", tmp_path / "run"
    synthetic.make_raw(raw, n_lesions=140)
    prepare.main(["--raw-dir", str(raw), "--out-dir", str(proc), "--resize", "64", "--workers", "2"])

    master = pd.read_csv(proc / "master.csv")
    assert set(master["source"]) == {"HAM", "BCN", "MSK_other"}       # source inference from image size works
    rep = json.loads((proc / "data_report.json").read_text())
    assert rep["leakage"]["split"]["test"]["n_images_sharing_group_with_train"] == 0
    assert rep["leakage"]["split_naive"]["test"]["n_images_sharing_group_with_train"] > 0

    train.main(["--config", "configs/baseline.yaml", "--device", "cpu", "--override",
                f"data.master_csv={proc / 'master.csv'}", "data.image_size=64", "data.num_workers=0",
                "model.pretrained=false", "train.epochs=2", "train.batch_size=16", "train.amp=false",
                f"out_dir={run}", f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}"])
    assert (run / "best.pt").exists()

    evaluate.main(["--ckpt", str(run / "best.pt"), "--split", "test", "--device", "cpu"])
    assert (run / "metrics_split_test.json").exists() and (run / "logits_split_test.csv").exists()

    gradcam.main(["--ckpt", str(run / "best.pt"), "--split", "test", "--per-class", "1", "--device", "cpu"])
    assert (run / "gradcam_any_test.png").exists()
