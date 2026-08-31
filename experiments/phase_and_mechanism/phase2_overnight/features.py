"""Stage-wise global pooled features. ResNet-style by default; ConvNeXt stages when present."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from phase1_gap_rep.common import BATCH_SIZE, images_to_tensor


def _pool(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)


@torch.no_grad()
def extract_stage_gaps(
    model: nn.Module,
    images: np.ndarray,
    batch_size: int = BATCH_SIZE,
    stages: List[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Return dict of layer name -> (N, C) float32 on CPU."""
    model.eval()
    dev = next(model.parameters()).device
    want = stages or ["layer1", "layer2", "layer3", "layer4", "gap"]
    buckets: Dict[str, List[np.ndarray]] = {name: [] for name in want}
    n = len(images)
    for start in range(0, n, batch_size):
        batch = images_to_tensor(images[start : start + batch_size]).to(dev)
        feats = _forward_stages(model, batch)
        for name in want:
            if name not in feats:
                continue
            buckets[name].append(feats[name].float().cpu().numpy())
    return {name: np.concatenate(items, axis=0) if items else np.zeros((n, 0), np.float32) for name, items in buckets.items()}


def _forward_stages(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "layer1") and hasattr(backbone, "conv1"):
        conv1 = backbone.conv1
        in_ch = int(getattr(conv1, "in_channels", 3))
        if in_ch == 5 and x.shape[1] == 3:
            b, _, h, w = x.shape
            yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            x = torch.cat([x, xx, yy], dim=1)
        return _resnet_stages(backbone, x)
    if hasattr(model, "layer1") and hasattr(model, "layer4") and hasattr(model, "conv1"):
        return _resnet_stages(model, x)
    if hasattr(model, "features"):
        return _features_backbone_stages(model, x)
    raise RuntimeError(f"unsupported backbone type {type(model)}")


def _resnet_stages(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    l1 = model.layer1(z)
    l2 = model.layer2(l1)
    l3 = model.layer3(l2)
    l4 = model.layer4(l3)
    gap = torch.flatten(model.avgpool(l4), 1)
    return {
        "layer1": _pool(l1),
        "layer2": _pool(l2),
        "layer3": _pool(l3),
        "layer4": _pool(l4),
        "gap": gap,
    }


def _features_backbone_stages(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    feats = model.features
    stages = {}
    z = x
    children = list(feats.children())
    # ConvNeXt: 4 stages; EfficientNet/MobileNet: sequential blocks.
    if len(children) >= 4 and all(isinstance(ch, nn.Sequential) or hasattr(ch, "forward") for ch in children[:4]):
        # try grouping into ~4 chunks
        n = len(children)
        cuts = [max(1, n * k // 4) for k in range(1, 5)]
        prev = 0
        names = ["layer1", "layer2", "layer3", "layer4"]
        for name, cut in zip(names, cuts):
            for child in children[prev:cut]:
                z = child(z)
            stages[name] = _pool(z) if z.ndim == 4 else z
            prev = cut
    else:
        z = feats(x)
        stages["layer4"] = _pool(z) if z.ndim == 4 else z
    if z.ndim == 4:
        if hasattr(model, "avgpool"):
            gap = torch.flatten(model.avgpool(z), 1)
        else:
            gap = _pool(z)
    else:
        gap = z
    stages["gap"] = gap
    return stages
