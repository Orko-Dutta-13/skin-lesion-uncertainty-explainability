from __future__ import annotations

import torch.nn as nn
from torchvision import models as tv


def build_model(arch: str = "efficientnet_b0", num_classes: int = 8, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    """ImageNet-pretrained backbone + dropout head.

    The dropout before the classifier is what MC Dropout will re-enable at test time in Week 2.
    """
    if arch == "efficientnet_b0":
        w = tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
    elif arch == "resnet50":
        w = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tv.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    else:
        raise ValueError(f"unknown arch {arch!r} (use efficientnet_b0 or resnet50)")
    return m


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional stage = last feature map before global pooling."""
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    raise ValueError(arch)
