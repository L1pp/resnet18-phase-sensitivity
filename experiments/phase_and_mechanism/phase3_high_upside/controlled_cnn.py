"""Controlled 1.1M CNN. S0–S6 change only padding / stride / AA / coord channels."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BLUR_KERNEL = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
CHANNELS: Tuple[int, ...] = (64, 128, 128, 256, 256)
STRIDE_STAGES = (1, 3)  # 0-based conv index after stem: block1 and block3


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


def _padding_mode(circular: bool) -> str:
    return "circular" if circular else "zeros"


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, circular: bool, antialias: bool):
        super().__init__()
        self.stride = int(stride)
        self.antialias = bool(antialias) and int(stride) == 2
        conv_stride = 1 if self.antialias else int(stride)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=conv_stride,
            padding=1,
            padding_mode=_padding_mode(circular),
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.blur = BlurPool(out_ch, stride=2) if self.antialias else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.act(self.bn(self.conv(x))))


class ControlledCNN(nn.Module):
    """Five 3x3 stages 64-128-128-256-256, GAP, Linear(2)."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in {"S0", "S1", "S2", "S3", "S4", "S5", "S6"}:
            raise ValueError(variant)
        self.variant = variant
        self.circular = variant in {"S0", "S1", "S4", "S6"}
        self.use_stride = variant in {"S1", "S3", "S4", "S5"}
        self.antialias = variant in {"S4", "S5"}
        self.coordconv = variant == "S6"
        in_ch = 5 if self.coordconv else 3
        blocks: List[nn.Module] = []
        prev = in_ch
        for i, ch in enumerate(CHANNELS):
            stride = 2 if (self.use_stride and i in STRIDE_STAGES) else 1
            blocks.append(ConvBlock(prev, ch, stride, self.circular, self.antialias))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(CHANNELS[-1], 2)

    def _maybe_coords(self, x: torch.Tensor) -> torch.Tensor:
        if not self.coordconv:
            return x
        if x.shape[1] == 5:
            return x
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._maybe_coords(x)
        for block in self.blocks:
            z = block(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        gap = torch.flatten(self.gap(feat), 1)
        return self.head(gap)

    def count_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def build_controlled(variant: str) -> ControlledCNN:
    return ControlledCNN(variant)


def variant_spec(variant: str) -> Dict[str, object]:
    model = ControlledCNN(variant)
    return {
        "variant": variant,
        "circular": model.circular,
        "use_stride": model.use_stride,
        "antialias": model.antialias,
        "coordconv": model.coordconv,
        "channels": list(CHANNELS),
        "n_params": model.count_params(),
        "note": "S0 is circular+stride1+GAP. Phase2 circular-ResNet is NOT S0.",
    }


def all_variant_specs() -> Dict[str, Dict[str, object]]:
    return {name: variant_spec(name) for name in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")}
