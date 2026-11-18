import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from reliable_derm import calibration
from reliable_derm.calibration import (apply_temperature, fit_temperature, load_logits,
                                       reliability_bins, reliability_diagram)
from reliable_derm.constants import CLASSES
from reliable_derm.metrics import compute_metrics


def make_logits(n=20000, k=8, factor=1.0, seed=0):
    """True logits z; labels drawn from softmax(z) (perfectly calibrated); observed logits = factor * z.

    factor > 1 -> the model is over-confident, and the best temperature is exactly `factor`.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, size=(n, k))
    p = softmax(z, axis=1)
    # vectorised categorical sampling via inverse CDF
    labels = np.minimum((p.cumsum(1) < rng.random((n, 1))).sum(1), k - 1)
    return z * factor, labels


def test_fit_recovers_known_overconfidence():
    logits, labels = make_logits(factor=2.5)
    assert fit_temperature(logits, labels) == pytest.approx(2.5, rel=0.08)


def test_fit_recovers_known_underconfidence():
    logits, labels = make_logits(factor=0.5)
    assert fit_temperature(logits, labels) == pytest.approx(0.5, rel=0.08)


def test_already_calibrated_gives_t_near_one():
    logits, labels = make_logits(factor=1.0)
    assert fit_temperature(logits, labels) == pytest.approx(1.0, abs=0.08)


def test_scaling_reduces_ece_and_nll_without_changing_predictions():
    val_logits, val_labels = make_logits(factor=2.5, seed=1)
    test_logits, test_labels = make_logits(factor=2.5, seed=2)
    T = fit_temperature(val_logits, val_labels)             # fit on val ...
    scaled = apply_temperature(test_logits, T)              # ... apply to test
    before, after = compute_metrics(test_logits, test_labels), compute_metrics(scaled, test_labels)
    assert after["ece"] < before["ece"] and after["nll"] < before["nll"]
    assert np.array_equal(test_logits.argmax(1), scaled.argmax(1))
    assert after["balanced_accuracy"] == before["balanced_accuracy"]


def test_apply_temperature_rejects_non_positive():
    with pytest.raises(ValueError):
        apply_temperature(np.zeros((2, 8)), 0.0)


def test_reliability_bins_counts_and_ranges():
    logits, labels = make_logits(n=3000, factor=2.0)
    edges, acc, conf, count = reliability_bins(softmax(logits, axis=1), labels, n_bins=10)
    assert count.sum() == 3000 and len(edges) == 11
    assert np.nanmin(acc) >= 0 and np.nanmax(acc) <= 1
    assert np.all(conf[count > 0] > edges[:-1][count > 0])  # bins are (lo, hi]


def test_reliability_bins_match_metrics_ece():
    logits, labels = make_logits(n=3000, factor=2.0)
    probs = softmax(logits, axis=1)
    _, acc, conf, count = reliability_bins(probs, labels, n_bins=15)
    ok = count > 0
    ece = float((count[ok] / count.sum() * np.abs(acc[ok] - conf[ok])).sum())
    assert ece == pytest.approx(compute_metrics(logits, labels)["ece"], abs=1e-12)


def _write_logits_csv(path, logits, labels):
    df = pd.DataFrame(logits, columns=[f"logit_{c}" for c in CLASSES])
    df.insert(0, "label", labels)
    df.insert(0, "image", [f"im{i}" for i in range(len(labels))])
    df["source"] = "HAM"
    df.to_csv(path, index=False)


def test_load_logits_roundtrip(tmp_path):
    logits, labels = make_logits(n=200)
    _write_logits_csv(tmp_path / "l.csv", logits, labels)
    z, y = load_logits(tmp_path / "l.csv")
    assert np.allclose(z, logits) and np.array_equal(y, labels)


def test_reliability_diagram_writes_png(tmp_path):
    logits, labels = make_logits(n=2000, factor=2.0)
    out = tmp_path / "sub" / "rd.png"
    reliability_diagram(logits, apply_temperature(logits, 2.0), labels, out)
    assert out.exists() and out.stat().st_size > 1000


def test_cli_end_to_end(tmp_path):
    for i, seed in enumerate((1, 2)):
        run = tmp_path / f"run{i}"
        run.mkdir()
        _write_logits_csv(run / "logits_split_val.csv", *make_logits(n=3000, factor=2.5, seed=seed))
        _write_logits_csv(run / "logits_split_test.csv", *make_logits(n=3000, factor=2.5, seed=seed + 10))
    calibration.main(["--run-dirs", str(tmp_path / "run0"), str(tmp_path / "run1"),
                      "--out", str(tmp_path / "out")])
    summary = pd.read_csv(tmp_path / "out" / "calibration_summary.csv")
    assert len(summary) == 2 and (summary["T"] > 1.5).all()
    assert (summary["ece_after"] < summary["ece_before"]).all()
    assert (tmp_path / "out" / "reliability_run0.png").exists()
    assert (tmp_path / "out" / "calibration_summary.json").exists()
