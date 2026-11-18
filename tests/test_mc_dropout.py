import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import StochasticDepth

from reliable_derm.mc_dropout import mc_logits, set_mc_mode
from reliable_derm.models import build_model

CPU = torch.device("cpu")


def loader(n=8, size=64, batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, size, size, generator=g)
    y = torch.randint(0, 8, (n,), generator=g)
    return DataLoader(TensorDataset(x, y, torch.arange(n)), batch_size=batch, shuffle=False)


def fresh_effnet():
    """Random-init EfficientNet. In eval mode its BatchNorm running stats (0/1) make activations vanish,
    so warm them up with a few train-mode batches; otherwise every pass would trivially look identical."""
    torch.manual_seed(0)
    m = build_model("efficientnet_b0", 8, pretrained=False, dropout=0.3)
    m.train()
    with torch.no_grad():
        for _ in range(5):
            m(torch.randn(16, 3, 64, 64))
    return m.eval()


def test_full_mode_switches_dropout_and_stochastic_depth_but_not_batchnorm():
    m = build_model("efficientnet_b0", 8, pretrained=False, dropout=0.3)
    n_drop, n_sd = set_mc_mode(m, stochastic_depth=True)
    assert n_drop == 1 and n_sd > 0
    assert all(not b.training for b in m.modules() if isinstance(b, nn.BatchNorm2d))
    assert all(b.training for b in m.modules() if isinstance(b, (nn.Dropout, StochasticDepth)))


def test_head_mode_leaves_stochastic_depth_in_eval():
    m = build_model("efficientnet_b0", 8, pretrained=False, dropout=0.3)
    n_drop, n_sd = set_mc_mode(m, stochastic_depth=False)
    assert n_drop == 1 and n_sd == 0
    assert all(not b.training for b in m.modules() if isinstance(b, StochasticDepth))


def test_resnet_has_only_the_head_dropout():
    m = build_model("resnet50", 8, pretrained=False, dropout=0.3)
    assert set_mc_mode(m, stochastic_depth=True) == (1, 0)


def test_model_without_random_layers_is_rejected():
    with pytest.raises(ValueError):
        mc_logits(nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 8)), loader(), CPU, n_passes=2)


def test_full_mode_gives_varying_passes_and_correct_shapes():
    m = fresh_effnet()
    z, y = mc_logits(m, loader(), CPU, n_passes=5, stochastic_depth=True)
    assert z.shape == (5, 8, 8) and y.shape == (8,)
    assert z.std(axis=0).mean() > 1e-4                      # passes differ


def test_same_seed_reproduces_and_different_seed_differs():
    m = fresh_effnet()
    a, _ = mc_logits(m, loader(), CPU, n_passes=3, seed=1)
    b, _ = mc_logits(m, loader(), CPU, n_passes=3, seed=1)
    c, _ = mc_logits(m, loader(), CPU, n_passes=3, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_eval_mode_is_deterministic_by_comparison():
    m = fresh_effnet()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))


def test_full_mode_is_more_diverse_than_head_only():
    m = fresh_effnet()
    full, _ = mc_logits(m, loader(), CPU, n_passes=8, stochastic_depth=True)
    head, _ = mc_logits(m, loader(), CPU, n_passes=8, stochastic_depth=False)
    assert head.std(axis=0).mean() > 0                      # the head dropout alone already randomises
    assert full.std(axis=0).mean() > head.std(axis=0).mean()
