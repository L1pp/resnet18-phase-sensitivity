"""Torchvision backbones with a 2-d coordinate head. No ImageNet normalize."""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from phase1_gap_rep.common import build_model
from phase2_overnight.backbones import build_backbone

from .trainer import replace_xy_head


def build_xy_backbone(arch: str, mlp_hidden: int | None = None) -> Tuple[nn.Module, str]:
    if arch == "resnet18":
        model = build_model()
        replace_xy_head(model, hidden=mlp_hidden)
        return model, "resnet18"
    model, tag = build_backbone(arch, "standard")
    replace_xy_head(model, hidden=mlp_hidden)
    return model, tag
