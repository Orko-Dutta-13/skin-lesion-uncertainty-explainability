"""Uncertainty maths shared by MC Dropout, deep ensembles and selective prediction.

Everything works on "members": logits with shape [M, N, K]
    M = number of members:  1   a single deterministic network
                            S   S stochastic MC-Dropout passes of one network
                            3   the three seeds (a deep ensemble)
    N = images, K = 8 classes.

The predictive distribution is the MEAN of the member softmaxes (optionally with temperature T).
Uncertainty scores (higher = less certain):
    maxprob      1 - max_k p_k                     (classic confidence score)
    entropy      H(p)    total predictive entropy
    mutual_info  H(p) - mean_m H(p_m)              only if M > 1: the part caused by members DISAGREEING
                                                   (epistemic uncertainty)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import entr, softmax

LOG_T_BOUNDS = (-3.0, 3.0)  # T in [0.05, 20]


def member_probs(member_logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """[M,N,K] logits -> [M,N,K] per-member probabilities softmax(z / T)."""
    if T <= 0:
        raise ValueError("temperature must be positive")
    return softmax(member_logits / T, axis=-1)


def mean_probs(member_logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    return member_probs(member_logits, T).mean(axis=0)


def fit_temperature_members(val_members: np.ndarray, val_labels: np.ndarray) -> float:
    """T minimising the NLL of the mean-of-softmaxes on validation. For M = 1 this is ordinary
    temperature scaling; for MC Dropout / ensembles it recalibrates the averaged prediction."""
    idx = np.arange(len(val_labels))

    def nll(log_t: float) -> float:
        p = mean_probs(val_members, float(np.exp(log_t)))
        return float(-np.log(p[idx, val_labels] + 1e-12).mean())

    res = minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded")
    if min(abs(res.x - LOG_T_BOUNDS[0]), abs(res.x - LOG_T_BOUNDS[1])) < 1e-3:
        print(f"WARNING: fitted log T = {res.x:.3f} sits on the search boundary")
    return float(np.exp(res.x))


def uncertainty_scores(member_logits: np.ndarray, T: float = 1.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (mean probabilities [N,K], {score name: uncertainty [N]})."""
    pm = member_probs(member_logits, T)
    p = pm.mean(axis=0)
    total = entr(p).sum(axis=-1)
    scores = {"maxprob": 1.0 - p.max(axis=-1), "entropy": total}
    if pm.shape[0] > 1:
        expected = entr(pm).sum(axis=-1).mean(axis=0)
        scores["mutual_info"] = np.maximum(total - expected, 0.0)
    return p, scores
