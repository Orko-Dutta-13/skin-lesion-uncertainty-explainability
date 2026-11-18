import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("shap")

from reliable_derm.explain.shap_explain import agreement, pool_map, shap_attributions, shap_heat


class Linear(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, k))

    def forward(self, x):
        return self.net(x)


def test_linear_model_is_exactly_additive_and_explains_predicted_class():
    torch.manual_seed(0)
    m = Linear().eval()
    x, bg = torch.randn(5, 3, 8, 8), torch.randn(1, 3, 8, 8)
    attr, target, gap = shap_attributions(m, x, bg, nsamples=20, seed=0, batch=2)
    assert attr.shape == (5, 3, 8, 8)
    with torch.no_grad():
        assert np.array_equal(target, m(x).argmax(1).numpy())
    assert gap.max() < 1e-3                        # linear model + single background -> exact


def test_linear_model_attribution_equals_weight_times_difference():
    torch.manual_seed(1)
    m = Linear().eval()
    x, bg = torch.randn(2, 3, 8, 8), torch.randn(1, 3, 8, 8)
    attr, target, _ = shap_attributions(m, x, bg, nsamples=10, seed=0)
    w = m.net[1].weight.detach()
    for i in range(2):
        expected = (w[target[i]] * (x[i] - bg[0]).flatten()).reshape(3, 8, 8).numpy()
        assert np.allclose(attr[i], expected, atol=1e-4)


def test_shap_attributions_are_reproducible_for_a_fixed_seed():
    torch.manual_seed(2)
    m = Linear().eval()
    x, bg = torch.randn(2, 3, 8, 8), torch.randn(3, 3, 8, 8)
    a1, _, _ = shap_attributions(m, x, bg, nsamples=10, seed=5)
    a2, _, _ = shap_attributions(m, x, bg, nsamples=10, seed=5)
    assert np.allclose(a1, a2)


def test_pooling_agreement_and_heat_are_well_behaved():
    rng = np.random.default_rng(0)
    cam = rng.random((3, 28, 28))
    assert pool_map(cam, 14).shape == (3, 14, 14)
    assert np.allclose(agreement(cam, cam), 1.0)
    assert np.isnan(agreement(cam, np.zeros_like(cam))).all()
    heat = shap_heat(rng.normal(size=(2, 3, 28, 28)), blur=5)
    assert heat.shape == (2, 28, 28) and heat.min() >= 0 and heat.max() <= 1 + 1e-6
