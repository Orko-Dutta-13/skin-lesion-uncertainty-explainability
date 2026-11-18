from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from reliable_derm.constants import CLASSES


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())


def multiclass_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = softmax(logits, axis=1)
    pred = probs.argmax(1)
    k = probs.shape[1]
    present = [c for c in range(k) if (labels == c).any()]

    try:  # macro one-vs-rest AUROC over classes that occur in this split
        auroc = float(roc_auc_score(
            np.eye(k)[labels][:, present], probs[:, present], average="macro"))
    except ValueError:
        auroc = float("nan")

    per_class_recall = recall_score(labels, pred, labels=list(range(k)), average=None, zero_division=0)
    return {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_auroc": auroc,
        "ece": expected_calibration_error(probs, labels),
        "nll": nll(probs, labels),
        "brier": multiclass_brier(probs, labels),
        "recall_per_class": {CLASSES[i]: float(per_class_recall[i]) for i in range(k)},
    }
