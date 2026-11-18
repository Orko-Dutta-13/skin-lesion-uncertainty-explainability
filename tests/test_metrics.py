import numpy as np

from reliable_derm.metrics import compute_metrics, expected_calibration_error


def test_perfect_confident_predictions():
    labels = np.array([0, 1, 2, 0, 1, 2])
    logits = np.eye(3)[labels] * 20
    m = compute_metrics(logits, labels)
    assert m["accuracy"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert m["ece"] < 1e-6 and m["macro_auroc"] == 1.0


def test_overconfident_wrong_model_has_high_ece():
    labels = np.zeros(100, dtype=int)
    logits = np.tile([0.0, 10.0], (100, 1))      # always predicts class 1 with ~100% confidence
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert expected_calibration_error(probs, labels) > 0.95


def test_balanced_accuracy_penalises_majority_class_guessing():
    labels = np.array([0] * 90 + [1] * 10)
    logits = np.tile([5.0, 0.0], (100, 1))       # always predicts class 0
    m = compute_metrics(logits, labels)
    assert m["accuracy"] == 0.9 and abs(m["balanced_accuracy"] - 0.5) < 1e-9


def test_override_parsing_handles_scientific_notation(tmp_path):
    from reliable_derm.utils import load_config
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("train: {lr: 1.0e-3, amp: true}\n")
    cfg = load_config(cfg_file, ["train.lr=3e-4", "train.amp=false", "train.epochs=5", "out_dir=runs/x"])
    assert cfg["train"]["lr"] == 3e-4 and cfg["train"]["amp"] is False
    assert cfg["train"]["epochs"] == 5 and cfg["out_dir"] == "runs/x"
