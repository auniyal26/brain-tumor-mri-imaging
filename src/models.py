from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 backbone with a new classification head.
    Matches notebook behavior: ImageNet weights + replaced FC.
    """
    if pretrained:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        m = models.resnet18(weights=None)

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_stageA(pretrained: bool = True) -> nn.Module:
    """Stage A: binary gate (0=no_tumor, 1=tumor)."""
    return build_resnet18(num_classes=2, pretrained=pretrained)


def build_stageB(pretrained: bool = True) -> nn.Module:
    """Stage B: tumor subtype (0=glioma, 1=meningioma, 2=pituitary)."""
    return build_resnet18(num_classes=3, pretrained=pretrained)


# --------- self-check ----------
if __name__ == "__main__":
    a = build_stageA(pretrained=False)
    b = build_stageB(pretrained=False)
    print("StageA fc out:", a.fc.out_features)  # 2
    print("StageB fc out:", b.fc.out_features)  # 3
    print("OK: models.py sanity check passed")
