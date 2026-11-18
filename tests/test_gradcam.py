import pytest
import torch

from reliable_derm.explain.gradcam import GradCAM
from reliable_derm.models import build_model, gradcam_target_layer


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
def test_gradcam_shape_range_and_target_class(arch):
    torch.manual_seed(0)
    model = build_model(arch, 8, pretrained=False, dropout=0.3).eval()
    cam_engine = GradCAM(model, gradcam_target_layer(model, arch))
    x = torch.randn(2, 3, 96, 96)
    cam, logits = cam_engine(x)
    assert cam.shape == (2, 96, 96) and logits.shape == (2, 8)
    assert torch.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    # explaining a different class must (generically) give a different map
    cam_other, _ = cam_engine(x, class_idx=int((logits.argmax(1)[0] + 1) % 8))
    assert not torch.allclose(cam[0], cam_other[0])
    cam_engine.remove()
