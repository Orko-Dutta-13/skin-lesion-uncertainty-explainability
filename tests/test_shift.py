import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

from reliable_derm import shift
from reliable_derm.constants import CLASSES
from reliable_derm.metrics import expected_calibration_error, multiclass_brier, nll
from reliable_derm.selective import aurc


def synth(n, k=8, scale=2.0, noise=0.0, mix=None, seed=0):
    """Logits from a latent score: the true class gets a boost of `scale`; `noise` adds disagreement."""
    rng = np.random.default_rng(seed)
    mix = np.ones(k) / k if mix is None else np.asarray(mix) / np.sum(mix)
    y = rng.choice(k, size=n, p=mix)
    z = rng.normal(0, 1, (n, k)) + noise * rng.normal(0, 1, (n, k))
    z[np.arange(n), y] += scale
    return z, y


def test_class_weights_match_target_mix_and_have_mean_one():
    rng = np.random.default_rng(0)
    src = rng.choice(8, 5000, p=np.array([5, 1, 1, 1, 1, 1, 1, 1]) / 12)
    tgt = rng.choice(8, 3000, p=np.array([1, 1, 1, 1, 1, 1, 1, 5]) / 12)
    w = shift.class_weights(src, tgt)
    assert w.mean() == pytest.approx(1.0, abs=1e-9)
    mix = np.bincount(src, weights=w, minlength=8) / w.sum()
    assert np.allclose(mix, np.bincount(tgt, minlength=8) / len(tgt), atol=1e-9)


def test_weighted_metrics_equal_unweighted_with_unit_weights():
    z, y = synth(3000, noise=1.0)
    p = softmax(z, axis=1)
    w = np.ones(len(y))
    assert shift.w_ece(p, y, w) == pytest.approx(expected_calibration_error(p, y))
    m = shift.w_calibration_metrics(p, y, w)
    assert m["nll"] == pytest.approx(nll(p, y)) and m["brier"] == pytest.approx(multiclass_brier(p, y))
    assert m["accuracy"] == pytest.approx((p.argmax(1) == y).mean())
    unc = 1 - p.max(1)
    assert shift.w_aurc(p.argmax(1) == y, unc, w) == pytest.approx(aurc(p.argmax(1) == y, unc))


def test_weighting_changes_accuracy_towards_the_target_class_mix():
    # class 0 is always right, class 1 always wrong
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    pred = np.zeros(1000, int)
    p = np.eye(8)[pred] * 0.9 + 0.1 / 8
    w = shift.class_weights(y, np.r_[np.zeros(100, int), np.ones(900, int)])     # target is 90 % class 1
    assert shift.w_calibration_metrics(p, y, np.ones(1000))["accuracy"] == pytest.approx(0.9)
    assert shift.w_calibration_metrics(p, y, w)["accuracy"] == pytest.approx(0.1)


def test_detection_auroc_extremes():
    w = np.ones(500)
    rng = np.random.default_rng(0)
    a = rng.random(500)
    assert shift.detection_auroc(a, a + 10, w) == pytest.approx(1.0)
    assert shift.detection_auroc(a, rng.random(500), w) == pytest.approx(0.5, abs=0.06)


def test_balanced_accuracy_from_counts_and_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 8, 2000)
    pred = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 8, 2000))
    assert shift.balanced_acc_from_counts(y, pred) == pytest.approx(balanced_accuracy_score(y, pred))
    b = shift.bootstrap_ba(y, pred, n_boot=200, seed=0)
    lo, hi = np.percentile(b, [2.5, 97.5])
    assert lo < balanced_accuracy_score(y, pred) < hi and hi - lo < 0.1
    assert np.array_equal(b, shift.bootstrap_ba(y, pred, n_boot=200, seed=0))


def write_run(run, tag, split, z, y, source):
    df = pd.DataFrame(z, columns=[f"logit_{c}" for c in CLASSES])
    df.insert(0, "source", source)
    df.insert(0, "label", y)
    df.insert(0, "image", [f"{split}_{i}" for i in range(len(y))])
    df.to_csv(run / f"logits_{tag}_{split}.csv", index=False)


def test_cli_end_to_end_shows_degradation_and_writes_outputs(tmp_path):
    id_mix = [8, 1, 1, 1, 1, 1, 1, 1]
    bcn_mix = [1, 2, 2, 1, 1, 1, 1, 1]
    runs, oracle = [], []
    base = {s: synth(n, mix=m, seed=sd, scale=2.5) for s, n, m, sd in
            (("val", 1500, id_mix, 1), ("test", 2000, id_mix, 2))}
    shifted = synth(2500, mix=bcn_mix, seed=3, scale=1.0)               # weaker signal -> degradation
    rng = np.random.default_rng(0)
    for i in range(3):
        r = tmp_path / f"shift{i}"
        r.mkdir()
        runs.append(str(r))
        for s in ("val", "test"):
            z, y = base[s]
            write_run(r, "split_ood", s, z + rng.normal(0, 0.3, z.shape), y, np.where(rng.random(len(y)) < 0.8, "HAM", "MSK_other"))
        z, y = shifted
        write_run(r, "split_ood", "shift_test", z + rng.normal(0, 0.3, z.shape), y, ["BCN"] * len(y))
        o = tmp_path / f"base{i}"
        o.mkdir()
        oracle.append(str(o))
        for s, (zz, yy) in (("val", base["val"]), ("test", synth(2500, mix=bcn_mix, seed=4, scale=2.5))):
            write_run(o, "split", s, zz + rng.normal(0, 0.3, zz.shape), yy, ["BCN"] * len(yy))
    out = tmp_path / "out"
    shift.main(["--run-dirs", *runs, "--oracle-dirs", *oracle, "--n-boot", "100", "--out", str(out)])

    summ = pd.read_csv(out / "summary.csv")
    ba = summ.pivot(index="method", columns="dataset", values="balanced_accuracy")
    assert (ba["BCN shift"] < ba["ID test (BCN mix)"] - 0.05).all()           # clear degradation
    assert (ba["BCN (trained with BCN)"] > ba["BCN shift"] + 0.05).all()      # trained-with-BCN recovers it
    assert {"single", "single+TS", "ensemble", "ensemble+TS"} <= set(summ["method"])
    det = pd.read_csv(out / "detection.csv")
    assert (det["detection_auroc"] > 0.55).all()                              # weaker signal = higher entropy
    pc = pd.read_csv(out / "per_class.csv")
    assert list(pc["class"]) == CLASSES
    boot = json.loads((out / "bootstrap.json").read_text())
    assert boot["ensemble+TS_BA_change"][1] < 0                               # whole interval below zero
    assert (out / "shift_overview.png").exists()


def test_cli_rejects_non_bcn_shift_set_and_mismatched_runs(tmp_path):
    z, y = synth(200)
    for i in range(2):
        r = tmp_path / f"r{i}"
        r.mkdir()
        for s in ("val", "test", "shift_test"):
            write_run(r, "split_ood", s, z, y, ["HAM"] * len(y))
    with pytest.raises(SystemExit):
        shift.main(["--run-dirs", str(tmp_path / "r0"), str(tmp_path / "r1"), "--oracle-dirs", "--out", str(tmp_path / "o")])
