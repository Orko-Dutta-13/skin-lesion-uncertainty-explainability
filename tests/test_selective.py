import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from reliable_derm import selective
from reliable_derm.constants import CLASSES
from reliable_derm.selective import (aurc, at_coverage, evaluate_scores, optimal_aurc, rank_by_uncertainty,
                                     risk_coverage)


def make_logits(n=3000, k=8, factor=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, size=(n, k))
    labels = np.minimum((softmax(z, axis=1).cumsum(1) < rng.random((n, 1))).sum(1), k - 1)
    return z * factor, labels


def synthetic_predictions(n=5000, err=0.3, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 8, n)
    pred = labels.copy()
    wrong = rng.random(n) < err
    pred[wrong] = (labels[wrong] + 1) % 8
    return pred, labels


def test_perfect_ranking_reaches_the_optimal_aurc():
    pred, labels = synthetic_predictions()
    correct = pred == labels
    unc = (~correct).astype(float)                 # errors are exactly the uncertain ones
    assert aurc(correct, unc) == pytest.approx(optimal_aurc(correct))


def test_worst_ranking_is_worse_than_optimal_and_random_is_about_the_error_rate():
    pred, labels = synthetic_predictions(n=6000, err=0.3)
    correct = pred == labels
    assert aurc(correct, correct.astype(float)) > optimal_aurc(correct) + 0.1
    rnd = np.random.default_rng(5).random(len(labels))
    assert aurc(correct, rnd) == pytest.approx(1 - correct.mean(), abs=0.03)


def test_risk_coverage_ends_at_the_overall_error_rate():
    pred, labels = synthetic_predictions()
    correct = pred == labels
    cov, risk = risk_coverage(correct, np.random.default_rng(0).random(len(labels)))
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(1 - correct.mean())
    assert len(cov) == len(labels)


def test_at_coverage_extremes():
    pred, labels = synthetic_predictions(err=0.3)
    correct = pred == labels
    order = rank_by_uncertainty((~correct).astype(float))
    full = at_coverage(pred, labels, order, 1.0)
    assert full["accuracy"] == pytest.approx(correct.mean())
    assert full["mel_deferred"] == 0
    half = at_coverage(pred, labels, order, 0.5)        # 70 % are correct, so the best 50 % are all correct
    assert half["accuracy"] == 1.0


def test_melanoma_outcomes_sum_to_one():
    pred, labels = synthetic_predictions(err=0.4, seed=2)
    order = rank_by_uncertainty(np.random.default_rng(1).random(len(labels)))
    for cov in (1.0, 0.8, 0.5, 0.2):
        r = at_coverage(pred, labels, order, cov)
        assert r["mel_detected"] + r["mel_deferred"] + r["mel_silent_miss"] == pytest.approx(1.0)


def test_deferring_uncertain_cases_reduces_silent_melanoma_misses():
    pred, labels = synthetic_predictions(err=0.4, seed=3)
    correct = pred == labels
    order = rank_by_uncertainty((~correct).astype(float))
    assert at_coverage(pred, labels, order, 0.6)["mel_silent_miss"] < at_coverage(pred, labels, order, 1.0)["mel_silent_miss"]


def test_evaluate_scores_returns_row_and_curve_per_score():
    z, y = make_logits()
    p = softmax(z, axis=1)
    out = evaluate_scores(p, y, {"maxprob": 1 - p.max(1)})
    (row, curve), = out
    assert row["eaurc"] >= -1e-12 and row["aurc"] >= row["optimal_aurc"] - 1e-12
    assert "accuracy@80" in row and "mel_silent_miss@50" in row
    assert list(curve["coverage"]) == list(selective.GRID)


def _write_logits_csv(path, logits, labels):
    df = pd.DataFrame(logits, columns=[f"logit_{c}" for c in CLASSES])
    df.insert(0, "label", labels)
    df.insert(0, "image", [f"im{i}" for i in range(len(labels))])
    df.to_csv(path, index=False)


def test_cli_end_to_end_with_mc_and_ensemble(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(2):
        run = tmp_path / f"run{i}"
        run.mkdir()
        zv, yv = make_logits(n=2000, factor=2.0, seed=10 + i)
        zt, yt = make_logits(n=2000, factor=2.0, seed=20)         # same test images/labels for all runs
        zv, yv = make_logits(n=2000, factor=2.0, seed=10)         # same val images/labels for all runs
        zv_run, zt_run = zv + rng.normal(0, 0.5, zv.shape), zt + rng.normal(0, 0.5, zt.shape)
        _write_logits_csv(run / "logits_split_val.csv", zv_run, yv)
        _write_logits_csv(run / "logits_split_test.csv", zt_run, yt)
        np.savez_compressed(run / "mc_logits_full_split_val.npz",
                            logits=(zv_run[None] + rng.normal(0, 0.6, (6,) + zv.shape)).astype(np.float32),
                            labels=yv, images=np.array([f"im{j}" for j in range(len(yv))]))
        np.savez_compressed(run / "mc_logits_full_split_test.npz",
                            logits=(zt_run[None] + rng.normal(0, 0.6, (6,) + zt.shape)).astype(np.float32),
                            labels=yt, images=np.array([f"im{j}" for j in range(len(yt))]))
    out = tmp_path / "out"
    selective.main(["--run-dirs", str(tmp_path / "run0"), str(tmp_path / "run1"), "--out", str(out)])
    rows = pd.read_csv(out / "all_rows.csv")
    assert set(rows["method"]) == {"single", "single+TS", "mc", "mc+TS", "ensemble", "ensemble+TS"}
    assert {"maxprob", "entropy", "mutual_info"} <= set(rows["score"])
    assert (out / "summary.csv").exists() and (out / "selective_prediction.png").exists()
    ts = rows[(rows["method"] == "single+TS")]
    assert (ts["T"] > 1.3).all()                      # over-confident members -> T > 1


def test_cli_refuses_mismatched_image_order(tmp_path):
    for i in range(2):
        run = tmp_path / f"run{i}"
        run.mkdir()
        z, y = make_logits(n=300, seed=i)
        _write_logits_csv(run / "logits_split_val.csv", z, y)
        _write_logits_csv(run / "logits_split_test.csv", z, y)
    # reverse the image names of run1 so the order no longer matches
    df = pd.read_csv(tmp_path / "run1" / "logits_split_val.csv")
    df["image"] = df["image"][::-1].to_numpy()
    df.to_csv(tmp_path / "run1" / "logits_split_val.csv", index=False)
    with pytest.raises(SystemExit):
        selective.main(["--run-dirs", str(tmp_path / "run0"), str(tmp_path / "run1"), "--out", str(tmp_path / "o")])
