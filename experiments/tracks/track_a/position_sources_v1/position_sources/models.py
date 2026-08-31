"""ResNet18 variants for padding, stride and strict-torus controls.

The implementation is intentionally self-contained rather than importing a
historical torchvision model.  This makes the padding and valid-core rules
visible in the new package and avoids accidental checkpoint compatibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = (
    "zero_s32",
    "reflection_s32",
    "circular_s32",
    "valid_core_s32",
    "true_valid_s32",
    "valid_core_aa32",
    "valid_core_s1",
    "torus_s32",
    "torus_s1",
)
REFERENCE_VARIANTS = ("zero_s32_bn_reference",)


def seed_all(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility
        torch.use_deterministic_algorithms(True)


def _groups(channels: int, requested: int) -> int:
    value = min(int(channels), int(requested))
    while value > 1 and channels % value != 0:
        value -= 1
    return max(1, value)


def norm2d(channels: int, groups: int = 32, *, normalization: str = "groupnorm") -> nn.Module:
    if str(normalization) == "batchnorm2d":
        return nn.BatchNorm2d(int(channels))
    if str(normalization) != "groupnorm":
        raise ValueError(f"unsupported normalization: {normalization}")
    return nn.GroupNorm(_groups(channels, groups), channels)


def _conv3(
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
    padding: int,
    padding_mode: str,
) -> nn.Conv2d:
    return nn.Conv2d(
        int(in_channels),
        int(out_channels),
        kernel_size=3,
        stride=int(stride),
        padding=int(padding),
        padding_mode=str(padding_mode),
        bias=False,
    )


class BlurPool(nn.Module):
    """Fixed depthwise binomial low-pass followed by decimation."""

    def __init__(self, channels: int, *, padding_mode: str = "valid") -> None:
        super().__init__()
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 16.0
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3).repeat(int(channels), 1, 1, 1))
        self.channels = int(channels)
        self.padding_mode = str(padding_mode)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "valid":
            return F.conv2d(value, self.kernel, stride=2, padding=0, groups=self.channels)
        if self.padding_mode not in {"reflect", "circular", "replicate"}:
            raise ValueError(f"unsupported BlurPool padding mode: {self.padding_mode}")
        value = F.pad(value, (1, 1, 1, 1), mode=self.padding_mode)
        return F.conv2d(value, self.kernel, stride=2, padding=0, groups=self.channels)


class SpatialMaxPool(nn.Module):
    """MaxPool with explicit non-zero boundary semantics."""

    def __init__(self, *, stride: int, padding: int, padding_mode: str) -> None:
        super().__init__()
        self.stride = int(stride)
        self.padding = int(padding)
        self.padding_mode = str(padding_mode)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            return F.max_pool2d(value, kernel_size=3, stride=self.stride, padding=0)
        if self.padding_mode == "zeros":
            return F.max_pool2d(value, kernel_size=3, stride=self.stride, padding=self.padding)
        if self.padding_mode not in {"reflect", "circular", "replicate"}:
            raise ValueError(f"unsupported maxpool padding mode: {self.padding_mode}")
        value = F.pad(value, (self.padding,) * 4, mode=self.padding_mode)
        return F.max_pool2d(value, kernel_size=3, stride=self.stride, padding=0)


def _center_crop(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    height, width = value.shape[-2:]
    target_height, target_width = target.shape[-2:]
    if height < target_height or width < target_width:
        raise RuntimeError(f"shortcut is smaller than main branch: {value.shape} vs {target.shape}")
    delta_h = height - target_height
    delta_w = width - target_width
    if delta_h % 2 or delta_w % 2:
        raise RuntimeError(f"center crop is not symmetric: {value.shape} -> {target.shape}")
    top = delta_h // 2
    left = delta_w // 2
    return value[..., top : top + target_height, left : left + target_width]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        padding_mode: str,
        true_valid: bool,
        groups: int,
        normalization: str,
    ) -> None:
        super().__init__()
        padding = 0 if true_valid else 1
        self.conv1 = _conv3(in_channels, out_channels, stride=stride, padding=padding, padding_mode=padding_mode)
        self.norm1 = norm2d(out_channels, groups, normalization=normalization)
        self.conv2 = _conv3(out_channels, out_channels, stride=1, padding=padding, padding_mode=padding_mode)
        self.norm2 = norm2d(out_channels, groups, normalization=normalization)
        self.relu = nn.ReLU(inplace=True)
        self.true_valid = bool(true_valid)
        self.downsample: nn.Module | None
        if stride != 1 or in_channels != out_channels:
            if self.true_valid:
                # A 3x3 projection makes the even-size spatial difference
                # exactly two, so the shortcut crop is genuinely symmetric.
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=0,
                        padding_mode=padding_mode,
                        bias=False,
                    ),
                    norm2d(out_channels, groups, normalization=normalization),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        padding_mode=padding_mode,
                        bias=False,
                    ),
                    norm2d(out_channels, groups, normalization=normalization),
                )
        else:
            self.downsample = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = self.relu(self.norm1(self.conv1(value)))
        output = self.norm2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(value)
        if self.true_valid and identity.shape[-2:] != output.shape[-2:]:
            identity = _center_crop(identity, output)
        if identity.shape[-2:] != output.shape[-2:]:
            raise RuntimeError(f"residual spatial mismatch: {identity.shape} vs {output.shape}")
        return self.relu(output + identity)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    padding_mode: str
    true_valid: bool
    valid_core: bool
    anti_alias: bool
    total_stride: int
    torus: bool


def parse_variant(name: str) -> VariantSpec:
    value = str(name)
    if value not in VARIANTS:
        raise ValueError(f"unknown variant {value!r}; expected one of {VARIANTS}")
    torus = value.startswith("torus_")
    true_valid = value.startswith("true_valid_")
    valid_core = value.startswith("valid_core_")
    anti_alias = value == "valid_core_aa32"
    if torus:
        padding_mode = "circular"
    elif value.startswith("reflection"):
        padding_mode = "reflect"
    elif value.startswith("circular"):
        padding_mode = "circular"
    else:
        padding_mode = "zeros"
    total_stride = 1 if value.endswith("_s1") else 32
    return VariantSpec(value, padding_mode, true_valid, valid_core, anti_alias, total_stride, torus)


class SourceResNet18(nn.Module):
    """ResNet18 + GAP + linear coordinate head for Track A."""

    variant_family = "position_sources_v1"

    def __init__(
        self,
        variant: str,
        *,
        groups: int = 32,
        input_size: int | None = None,
        normalization: str = "groupnorm",
    ) -> None:
        super().__init__()
        self.spec = parse_variant(variant)
        self.variant = self.spec.name
        self.groups = int(groups)
        self.input_size = None if input_size is None else int(input_size)
        self.normalization = str(normalization)
        self.reference_only = self.normalization == "batchnorm2d"
        true_valid = self.spec.true_valid
        padding_mode = self.spec.padding_mode
        if self.spec.anti_alias:
            stem_stride = 1
        else:
            stem_stride = 2 if self.spec.total_stride == 32 else 1
        stem_padding = 0 if true_valid else 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stem_stride, padding=stem_padding, padding_mode=padding_mode, bias=False)
        self.norm1 = norm2d(64, self.groups, normalization=self.normalization)
        self.relu = nn.ReLU(inplace=True)
        pool_stride = 2 if self.spec.total_stride == 32 else 1
        pool_padding = 0 if true_valid else 1
        self.maxpool = SpatialMaxPool(stride=pool_stride, padding=pool_padding, padding_mode=padding_mode)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=1 if self.spec.anti_alias or self.spec.total_stride == 1 else 2)
        self.layer3 = self._make_layer(128, 256, 2, stride=1 if self.spec.anti_alias or self.spec.total_stride == 1 else 2)
        self.layer4 = self._make_layer(256, 512, 2, stride=1 if self.spec.anti_alias or self.spec.total_stride == 1 else 2)
        blur_padding = "circular" if self.spec.torus else "valid"
        self.blur_stem = BlurPool(64, padding_mode=blur_padding) if self.spec.anti_alias else None
        self.blur2 = BlurPool(128, padding_mode=blur_padding) if self.spec.anti_alias else None
        self.blur3 = BlurPool(256, padding_mode=blur_padding) if self.spec.anti_alias else None
        self.blur4 = BlurPool(512, padding_mode=blur_padding) if self.spec.anti_alias else None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 4 if self.spec.torus else 2
        self.fc = nn.Linear(512, self.output_dim)
        self._rf_ops = self._build_rf_ops()

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, *, stride: int) -> nn.Sequential:
        layers = [
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                padding_mode=self.spec.padding_mode,
                true_valid=self.spec.true_valid,
                groups=self.groups,
                normalization=self.normalization,
            )
        ]
        for _ in range(1, int(blocks)):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    padding_mode=self.spec.padding_mode,
                    true_valid=self.spec.true_valid,
                    groups=self.groups,
                    normalization=self.normalization,
                )
            )
        return nn.Sequential(*layers)

    def _build_rf_ops(self) -> list[tuple[int, int, int]]:
        """Return (kernel, stride, padding) for the main receptive-field path."""

        true_valid = self.spec.true_valid
        pad7 = 0 if true_valid else 3
        pad3 = 0 if true_valid else 1
        ops: list[tuple[int, int, int]] = [(7, 1 if self.spec.anti_alias else (2 if self.spec.total_stride == 32 else 1), pad7)]
        ops.append((3, 2 if self.spec.total_stride == 32 else 1, 0 if true_valid else 1))
        for stage, blocks in enumerate((2, 2, 2, 2)):
            stride = 1 if self.spec.anti_alias or self.spec.total_stride == 1 or stage == 0 else 2
            for block in range(blocks):
                conv_stride = stride if block == 0 else 1
                ops.append((3, conv_stride, pad3))
                ops.append((3, 1, pad3))
            if self.spec.anti_alias and stage in (1, 2, 3):
                ops.append((3, 2, 0 if not self.spec.torus else 1))
        if self.spec.anti_alias:
            # The stem blur occurs before maxpool; insert it immediately after
            # the stem operation in the conservative path calculation.
            stem = ops.pop(0)
            ops = [stem, (3, 2, 0 if not self.spec.torus else 1)] + ops
        return ops

    def receptive_field(self) -> dict[str, float]:
        receptive = 1.0
        jump = 1.0
        start = 0.5
        for kernel, stride, padding in self._rf_ops:
            start += ((float(kernel) - 1.0) / 2.0 - float(padding)) * jump
            receptive += (float(kernel) - 1.0) * jump
            jump *= float(stride)
        return {"size": receptive, "jump": jump, "start": start}

    def valid_core_info(self, *, input_size: int, feature_shape: tuple[int, int] | None = None) -> dict[str, int | float | list[int]]:
        geometry = self.receptive_field()
        if feature_shape is None:
            with torch.no_grad():
                probe = torch.zeros(1, 3, int(input_size), int(input_size))
                feature_shape = tuple(int(item) for item in self._forward_spatial(probe).shape[-2:])
        height, width = map(int, feature_shape)
        indices_h: list[int] = []
        indices_w: list[int] = []
        radius = (float(geometry["size"]) - 1.0) / 2.0
        for index in range(height):
            center = float(geometry["start"]) + float(index) * float(geometry["jump"])
            if center - radius >= 0.0 and center + radius <= float(input_size):
                indices_h.append(index)
        for index in range(width):
            center = float(geometry["start"]) + float(index) * float(geometry["jump"])
            if center - radius >= 0.0 and center + radius <= float(input_size):
                indices_w.append(index)
        if self.spec.valid_core and (not indices_h or not indices_w):
            raise RuntimeError(
                f"no valid-core cells for input={input_size}, feature={feature_shape}, geometry={geometry}"
            )
        return {
            "receptive_field": float(geometry["size"]),
            "jump": float(geometry["jump"]),
            "start": float(geometry["start"]),
            "height_indices": indices_h,
            "width_indices": indices_w,
        }

    def _forward_spatial(self, value: torch.Tensor) -> torch.Tensor:
        value = self.relu(self.norm1(self.conv1(value)))
        if self.blur_stem is not None:
            value = self.blur_stem(value)
        value = self.maxpool(value)
        value = self.layer1(value)
        value = self.layer2(value)
        if self.blur2 is not None:
            value = self.blur2(value)
        value = self.layer3(value)
        if self.blur3 is not None:
            value = self.blur3(value)
        value = self.layer4(value)
        if self.blur4 is not None:
            value = self.blur4(value)
        return value

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        spatial = self._forward_spatial(value)
        if self.spec.valid_core:
            info = self.valid_core_info(input_size=int(value.shape[-1]), feature_shape=tuple(spatial.shape[-2:]))
            h = info["height_indices"]
            w = info["width_indices"]
            assert isinstance(h, list) and isinstance(w, list)
            spatial = spatial[..., h[0] : h[-1] + 1, w[0] : w[-1] + 1]
        pooled = self.avgpool(spatial)
        return torch.flatten(pooled, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(f"expected NCHW RGB input, got {tuple(value.shape)}")
        return self.fc(self.forward_features(value))


def build_reference_model(variant: str = "zero_s32_bn_reference", *, input_size: int | None = None, groups: int = 32) -> SourceResNet18:
    if str(variant) not in REFERENCE_VARIANTS:
        raise ValueError(f"unknown reference-only variant {variant!r}; expected one of {REFERENCE_VARIANTS}")
    model = SourceResNet18("zero_s32", input_size=input_size, groups=groups, normalization="batchnorm2d")
    model.variant = str(variant)
    model.reference_only = True
    return model


def build_model(variant: str, *, input_size: int | None = None, groups: int = 32) -> SourceResNet18:
    if str(variant) in REFERENCE_VARIANTS:
        return build_reference_model(str(variant), input_size=input_size, groups=groups)
    return SourceResNet18(variant, input_size=input_size, groups=groups)


__all__ = [
    "VARIANTS",
    "BlurPool",
    "REFERENCE_VARIANTS",
    "SourceResNet18",
    "VariantSpec",
    "build_model",
    "build_reference_model",
    "parse_variant",
    "seed_all",
]
