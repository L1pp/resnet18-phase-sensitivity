"""TorchVision CNN factories with GAP + Linear(D, 6). No ImageNet normalize. weights=None."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_gap_rep.common import COORDINATE_DIM, build_model

BLUR_KERNEL = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))


class BlurPool(nn.Module):
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        kernel = torch.tensor(BLUR_KERNEL, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        self.channels = int(channels)
        self.stride = int(stride)
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel.to(dtype=x.dtype), stride=self.stride, padding=1, groups=self.channels)


class ConvWithOptionalBlur(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        stride_value = conv.stride[0] if isinstance(conv.stride, tuple) else int(conv.stride)
        self.conv = conv
        if stride_value == 2:
            self.conv.stride = (1, 1)
            self.blur = BlurPool(conv.out_channels, stride=2)
        elif stride_value == 1:
            self.blur = nn.Identity()
        else:
            raise ValueError(f"anti-alias conversion only supports stride 1/2, got {stride_value}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.conv(x))


class AntiAliasBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, old_block: nn.Module):
        super().__init__()
        self.conv1 = ConvWithOptionalBlur(old_block.conv1)
        self.bn1 = old_block.bn1
        self.relu = old_block.relu
        self.conv2 = old_block.conv2
        self.bn2 = old_block.bn2
        self.downsample = None
        if old_block.downsample is not None:
            old_conv = old_block.downsample[0]
            old_bn = old_block.downsample[1]
            self.downsample = nn.Sequential(ConvWithOptionalBlur(old_conv), old_bn)
        self.stride = old_block.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _replace_linear(module: nn.Module, dim: int = COORDINATE_DIM) -> None:
    if isinstance(module, nn.Linear):
        raise RuntimeError("pass the parent")
    if hasattr(module, "fc") and isinstance(module.fc, nn.Linear):
        module.fc = nn.Linear(module.fc.in_features, dim)
        return
    clf = getattr(module, "classifier", None)
    if isinstance(clf, nn.Linear):
        module.classifier = nn.Linear(clf.in_features, dim)
        return
    if isinstance(clf, nn.Sequential):
        for i in range(len(clf) - 1, -1, -1):
            if isinstance(clf[i], nn.Linear):
                in_f = clf[i].in_features
                clf[i] = nn.Linear(in_f, dim)
                return
    raise RuntimeError(f"cannot replace classifier on {type(module)}")


def _tv(name: str):
    import torchvision.models as models

    builders = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "densenet121": models.densenet121,
        "convnext_tiny": models.convnext_tiny,
        "efficientnet_b0": models.efficientnet_b0,
        "mobilenet_v3_large": models.mobilenet_v3_large,
    }
    if name not in builders:
        raise ValueError(name)
    try:
        return builders[name](weights=None)
    except TypeError:
        return builders[name](pretrained=False)


def apply_circular(model: nn.Module) -> nn.Module:
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and (mod.padding != 0 and mod.padding != (0, 0)):
            mod.padding_mode = "circular"
    return model


def apply_antialias_resnet18(model: nn.Module) -> nn.Module:
    model.conv1 = ConvWithOptionalBlur(model.conv1)
    old_pool = model.maxpool
    old_pool.stride = 1
    model.maxpool = nn.Sequential(old_pool, BlurPool(64, stride=2))
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        old_layer = getattr(model, layer_name)
        new_layer = nn.Sequential(*(AntiAliasBasicBlock(block) for block in old_layer))
        setattr(model, layer_name, new_layer)
    return model


class CoordConvResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_model()
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(5, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=old.bias is not None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        x5 = torch.cat([x, xx, yy], dim=1)
        return self.backbone(x5)


def build_backbone(name: str, variant: str = "standard") -> Tuple[nn.Module, str]:
    """variant: standard | circular | antialiased | circular_aa | coordconv."""
    if variant == "coordconv":
        if name != "resnet18":
            raise ValueError("CoordConv only implemented for resnet18")
        return CoordConvResNet18(), "resnet18_coordconv"
    if name == "resnet18" and variant == "standard":
        return build_model(), "resnet18"
    model = _tv(name)
    _replace_linear(model)
    tag = name
    if variant == "circular":
        apply_circular(model)
        tag = f"{name}_circular"
    elif variant == "antialiased":
        if name != "resnet18":
            raise ValueError("antialias only implemented for resnet18")
        apply_antialias_resnet18(model)
        tag = "resnet18_antialiased"
    elif variant == "circular_aa":
        if name != "resnet18":
            raise ValueError("circular+AA only implemented for resnet18")
        apply_antialias_resnet18(model)
        apply_circular(model)
        tag = "resnet18_circular_aa"
    elif variant != "standard":
        raise ValueError(variant)
    return model, tag


def try_build(name: str) -> Tuple[bool, str]:
    try:
        m, tag = build_backbone(name, "standard")
        x = torch.zeros(1, 3, 224, 224)
        y = m(x)
        if tuple(y.shape) != (1, 6):
            return False, f"bad out {tuple(y.shape)}"
        return True, tag
    except Exception as exc:
        return False, str(exc)
