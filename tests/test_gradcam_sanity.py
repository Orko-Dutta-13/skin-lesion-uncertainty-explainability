import numpy as np
import pytest
import torch

from reliable_derm.explain.sanity import _similarity, cascade, randomize_, stages_top_down
from reliable_derm.models import build_model


def warm_model(arch="efficientnet_b0"):
    torch.manual_seed(0)
    m = build_model(arch, 8, pretrained=False)
    m.train()
    with torch.no_grad():
        for _ in range(5):
            m(torch.randn(16, 3, 64, 64))
    return m.eval()


def test_stage_order_is_output_to_input():
    m = build_model("efficientnet_b0", 8, pretrained=False)
    names = [n for n, _ in stages_top_down(m, "efficientnet_b0")]
    assert names[0] == "classifier" and names[-1] == "features.0"
    r = build_model("resnet50", 8, pretrained=False)
    rn = [n for n, _ in stages_top_down(r, "resnet50")]
    assert rn == ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]


def test_randomize_changes_weights_and_keeps_batchnorm():
    m = warm_model()
    before_conv = m.features[0][0].weight.clone()
    bn = m.features[0][1]
    mean_before = bn.running_mean.clone()
    gen = torch.Generator().manual_seed(1)
    assert randomize_(m.features[0], gen) == 1
    assert not torch.equal(before_conv, m.features[0][0].weight)
    assert torch.equal(mean_before, bn.running_mean)


def test_similarity_handles_constant_maps_and_identity():
    a = np.random.default_rng(0).random((16, 16))
    assert _similarity(a, a) == pytest.approx(1.0)
    assert np.isnan(_similarity(a, np.zeros_like(a)))
    assert _similarity(a, -a) == pytest.approx(-1.0)


def test_cascade_shapes_original_untouched_and_deterministic():
    m = warm_model()
    w = m.features[3][0].block[0][0].weight.clone()
    x = torch.randn(3, 3, 64, 64)
    names, sims, cams = cascade(m, "efficientnet_b0", x, seed=0)
    assert names[0] == "original" and names[1] == "classifier" and names[-1] == "features.0"
    assert sims.shape == (len(names), 3) and len(cams) == len(names)
    assert np.allclose(sims[0], 1.0)
    assert torch.equal(w, m.features[3][0].block[0][0].weight)          # the caller's model is not modified
    names2, sims2, _ = cascade(m, "efficientnet_b0", x, seed=0)
    assert np.allclose(sims, sims2, equal_nan=True)


def test_randomising_the_whole_network_changes_the_map():
    m = warm_model()
    x = torch.randn(4, 3, 64, 64)
    _, sims, _ = cascade(m, "efficientnet_b0", x, seed=0)
    last = sims[-1]
    assert np.nanmean(last) < 0.9
