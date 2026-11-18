import numpy as np
import pytest
from scipy.special import softmax

from reliable_derm.calibration import fit_temperature
from reliable_derm.metrics import nll
from reliable_derm.uncertainty import (fit_temperature_members, mean_probs, member_probs,
                                       uncertainty_scores)


def make_logits(n=8000, k=8, factor=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, size=(n, k))
    labels = np.minimum((softmax(z, axis=1).cumsum(1) < rng.random((n, 1))).sum(1), k - 1)
    return z * factor, labels


def test_single_member_matches_ordinary_temperature_scaling():
    z, y = make_logits(factor=2.0)
    assert fit_temperature_members(z[None], y) == pytest.approx(fit_temperature(z, y), rel=1e-3)


def test_member_probs_are_distributions_and_reject_bad_temperature():
    z, _ = make_logits(n=50)
    p = member_probs(np.stack([z, z * 2]), T=1.5)
    assert p.shape == (2, 50, 8) and np.allclose(p.sum(-1), 1)
    with pytest.raises(ValueError):
        member_probs(z[None], T=0)


def test_identical_members_have_zero_mutual_information():
    z, _ = make_logits(n=200)
    _, s = uncertainty_scores(np.stack([z] * 5))
    assert np.allclose(s["mutual_info"], 0, atol=1e-9)


def test_disagreeing_members_have_mutual_information_ln2():
    n = 10
    a, b = np.full((n, 8), -20.0), np.full((n, 8), -20.0)
    a[:, 0], b[:, 1] = 20.0, 20.0          # member 1 sure of class 0, member 2 sure of class 1
    p, s = uncertainty_scores(np.stack([a, b]))
    assert np.allclose(p[:, :2], 0.5, atol=1e-6)
    assert np.allclose(s["mutual_info"], np.log(2), atol=1e-3)
    assert np.allclose(s["entropy"], np.log(2), atol=1e-3)


def test_single_member_has_no_mutual_information_score():
    z, _ = make_logits(n=20)
    _, s = uncertainty_scores(z[None])
    assert set(s) == {"maxprob", "entropy"}


def test_entropy_bounds_and_maxprob_definition():
    z, _ = make_logits(n=500, factor=0.5)
    p, s = uncertainty_scores(z[None])
    assert s["entropy"].min() >= 0 and s["entropy"].max() <= np.log(8) + 1e-9
    assert np.allclose(s["maxprob"], 1 - p.max(1))


def test_fitting_t_on_an_overconfident_ensemble_lowers_nll():
    rng = np.random.default_rng(3)
    zv, yv = make_logits(factor=1.0, seed=1)
    zt, yt = make_logits(factor=1.0, seed=2)
    # three noisy, over-confident members around the true logits
    def members(z):
        return np.stack([2.5 * z + rng.normal(0, 0.5, z.shape) for _ in range(3)])
    mv, mt = members(zv), members(zt)
    T = fit_temperature_members(mv, yv)
    assert T > 1.5
    assert nll(mean_probs(mt, T), yt) < nll(mean_probs(mt, 1.0), yt)
