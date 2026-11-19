import numpy as np
import pytest
from scipy.special import logsumexp, softmax

from reliable_derm.ood import (DETECTORS, Mahalanobis, energy_score, entropy_score, evaluate_run, fpr_at_tpr, knn_score,
                               msp_score, summarise)
from reliable_derm.shift import class_weights, detection_auroc


def gaussian_world(seed=0, d=20, k=3, n=300, shift=8.0):
    """k well separated classes in d dimensions; the 'OOD' set is the same classes moved along a new direction."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(k, d))

    def draw(m, offset=0.0):
        y = rng.integers(0, k, m)
        x = centers[y] + rng.normal(0, 1, size=(m, d))
        x[:, 0] += offset
        return x, y

    xt, yt = draw(n * 3)
    xi, yi = draw(n)
    xo, yo = draw(n, offset=shift)
    return (xt, yt), (xi, yi), (xo, yo)


def feats(x, y, k=3, seed=0):
    """Pretend logits: a noisy linear read-out of the embedding (only used by the output-based scores)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(x.shape[1], k))
    return {"emb": x, "labels": y, "logits": x @ w * 0.1}


def test_energy_is_minus_logsumexp_and_temperature_scales():
    z = np.random.default_rng(0).normal(size=(5, 4))
    assert np.allclose(energy_score(z), -logsumexp(z, axis=1))
    assert np.allclose(energy_score(z, 2.0), -2.0 * logsumexp(z / 2.0, axis=1))


def test_confident_images_get_lower_ood_scores():
    sure = np.array([[10.0, 0, 0, 0]])
    unsure = np.array([[0.1, 0.0, 0.05, 0.0]])
    for f in (msp_score, entropy_score, energy_score):
        assert f(sure)[0] < f(unsure)[0]
    assert np.isclose(msp_score(sure)[0], 1 - softmax(sure, axis=1).max())


def test_mahalanobis_flags_shifted_images_and_not_clean_ones():
    from sklearn.metrics import roc_auc_score
    (xt, yt), (xi, yi), (xo, _) = gaussian_world()
    m = Mahalanobis(xt, yt)
    md_i, _ = m.scores(xi)
    md_o, _ = m.scores(xo)
    assert md_o.mean() > 2 * md_i.mean()
    y = np.r_[np.zeros(len(md_i)), np.ones(len(md_o))]
    assert roc_auc_score(y, np.r_[md_i, md_o]) > 0.95


def test_relative_mahalanobis_is_class_distance_minus_background_distance():
    (xt, yt), (xi, _), _ = gaussian_world(n=60)
    m = Mahalanobis(xt, yt)
    md, rmd = m.scores(xi)
    bg = Mahalanobis._sq(xi.astype(float), m.bg_mean, m.bg_prec)
    assert np.allclose(rmd, md - bg)
    assert (rmd <= md + 1e-9).all()                      # a squared distance is never negative


def test_mahalanobis_of_class_mean_is_small_and_handles_more_dims_than_samples():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 100))                       # d > n: needs the shrinkage covariance
    y = np.repeat([0, 1], 20)
    m = Mahalanobis(x, y)
    md, _ = m.scores(np.vstack([m.means, m.means + 50.0]))
    assert np.isfinite(md).all() and md[:2].max() < md[2:].min()


def test_knn_distance_is_zero_for_a_training_point_and_grows_with_distance():
    train = np.eye(4)                                    # four orthogonal unit vectors
    d = knn_score(train, np.array([[1.0, 0, 0, 0], [1.0, 1.0, 0, 0]]), k=1)
    assert np.isclose(d[0], 0.0, atol=1e-6) and d[1] > 0.4
    # cosine geometry: the direction counts, the length does not
    assert np.allclose(knn_score(train, np.array([[5.0, 0, 0, 0]]), k=1), 0.0, atol=1e-6)
    # k larger than the training set is clipped instead of failing
    assert np.isfinite(knn_score(train, np.array([[1.0, 1.0, 1.0, 1.0]]), k=50)).all()


def test_fpr_at_tpr_simple_case():
    ood = np.arange(100, 200, dtype=float)               # 95 % of them are >= 105
    idv = np.r_[np.arange(0, 100, dtype=float), [150.0, 160.0]]
    w = np.ones(len(idv))
    assert np.isclose(fpr_at_tpr(idv, ood, w), 2 / 102)
    w2 = w.copy()
    w2[-2:] = 0.0                                        # the two flagged ID images get zero weight
    assert fpr_at_tpr(idv, ood, w2) == 0.0


def test_evaluate_run_separates_and_uses_class_weights():
    (xt, yt), (xi, yi), (xo, yo) = gaussian_world()
    train, idt, ood = feats(xt, yt), feats(xi, yi), feats(xo, yo)
    tab, sc = evaluate_run(train, idt, ood, k=5)
    assert list(tab["detector"]) == DETECTORS
    feat = tab.set_index("detector")
    assert feat.loc["mahalanobis", "auroc_weighted"] > 0.9 and feat.loc["knn", "auroc_weighted"] > 0.9
    assert (tab["fpr95_weighted"].between(0, 1)).all()
    # the weighted AUROC is exactly the shift module's detection AUROC with class-mix weights
    w = class_weights(idt["labels"], ood["labels"])
    assert np.isclose(feat.loc["knn", "auroc_weighted"], detection_auroc(sc["id"]["knn"], sc["ood"]["knn"], w))


def test_no_shift_gives_chance_level():
    (xt, yt), (xi, yi), (xo, yo) = gaussian_world(shift=0.0, seed=3)
    tab, _ = evaluate_run(feats(xt, yt), feats(xi, yi), feats(xo, yo), k=5)
    feat = tab.set_index("detector")
    assert abs(feat.loc["mahalanobis", "auroc_weighted"] - 0.5) < 0.08
    assert abs(feat.loc["knn", "auroc_weighted"] - 0.5) < 0.08


def test_summarise_mean_and_sd_over_runs():
    (xt, yt), (xi, yi), (xo, yo) = gaussian_world()
    t1, _ = evaluate_run(feats(xt, yt), feats(xi, yi), feats(xo, yo), k=5)
    t2, _ = evaluate_run(feats(xt, yt, seed=1), feats(xi, yi, seed=1), feats(xo, yo, seed=1), k=5)
    t1.insert(0, "run", "a")
    t2.insert(0, "run", "b")
    import pandas as pd
    s = summarise(pd.concat([t1, t2], ignore_index=True)).set_index("detector")
    assert (s["runs"] == 2).all()
    both = np.array([t1.set_index("detector").loc["energy", "auroc_weighted"],
                     t2.set_index("detector").loc["energy", "auroc_weighted"]])
    assert np.isclose(s.loc["energy", "auroc_mean"], both.mean())
    assert np.isclose(s.loc["energy", "auroc_sd"], both.std(ddof=1))


def test_single_run_has_nan_sd_not_an_error():
    (xt, yt), (xi, yi), (xo, yo) = gaussian_world()
    t, _ = evaluate_run(feats(xt, yt), feats(xi, yi), feats(xo, yo), k=5)
    t.insert(0, "run", "only")
    s = summarise(t)
    assert s["auroc_sd"].isna().all() and (s["runs"] == 1).all()
