from __future__ import annotations

import copy
import hashlib
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import CONFIG, MODEL_SPECS


STAGE_NAMES = ("stem", "layer1", "layer2", "layer3", "layer4", "gap")


def seed_everything(seed: int | None = None) -> None:
    value = int(CONFIG["seed"] if seed is None else seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes and bytes in deterministic key order."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _conv3(in_channels: int, out_channels: int, *, stride: int, padding_mode: str) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        padding_mode=padding_mode,
        bias=False,
    )


class SpatialMaxPool(nn.Module):
    def __init__(self, *, stride: int, padding_mode: str) -> None:
        super().__init__()
        self.stride = int(stride)
        self.padding_mode = str(padding_mode)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "zeros":
            return F.max_pool2d(value, kernel_size=3, stride=self.stride, padding=1)
        value = F.pad(value, (1, 1, 1, 1), mode=self.padding_mode)
        return F.max_pool2d(value, kernel_size=3, stride=self.stride, padding=0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int, padding_mode: str) -> None:
        super().__init__()
        self.conv1 = _conv3(in_channels, out_channels, stride=stride, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(out_channels, out_channels, stride=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        out = self.relu(self.bn1(self.conv1(value)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(value)
        return self.relu(out + identity)


class ToyResNet18(nn.Module):
    def __init__(self, *, padding_mode: str, total_stride: int) -> None:
        super().__init__()
        if padding_mode not in {"zeros", "reflect", "circular"}:
            raise ValueError(f"unsupported padding mode: {padding_mode}")
        if total_stride not in {1, 32}:
            raise ValueError(f"unsupported total stride: {total_stride}")
        self.padding_mode = padding_mode
        self.total_stride = int(total_stride)
        stride_stem = 2 if total_stride == 32 else 1
        stride_pool = 2 if total_stride == 32 else 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stride_stem, padding=3, padding_mode=padding_mode, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = SpatialMaxPool(stride=stride_pool, padding_mode=padding_mode)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2 if total_stride == 32 else 1)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2 if total_stride == 32 else 1)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2 if total_stride == 32 else 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self._initialize()

    def _make_layer(self, in_channels: int, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(in_channels, out_channels, stride=stride, padding_mode=self.padding_mode)]
        layers.extend(BasicBlock(out_channels, out_channels, stride=1, padding_mode=self.padding_mode) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward_stages(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.maxpool(value)
        stages: dict[str, torch.Tensor] = {"stem": value}
        value = self.layer1(value)
        stages["layer1"] = value
        value = self.layer2(value)
        stages["layer2"] = value
        value = self.layer3(value)
        stages["layer3"] = value
        value = self.layer4(value)
        stages["layer4"] = value
        return stages

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        spatial = self.forward_stages(value)["layer4"]
        return torch.flatten(self.avgpool(spatial), 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(value))


def build_models(*, device: torch.device | str = "cpu") -> dict[str, ToyResNet18]:
    seed_everything()
    base = ToyResNet18(padding_mode="zeros", total_stride=32)
    shared_state = copy.deepcopy(base.state_dict())
    models: dict[str, ToyResNet18] = {}
    for name, spec in MODEL_SPECS.items():
        model = ToyResNet18(padding_mode=str(spec["padding_mode"]), total_stride=int(spec["total_stride"]))
        model.load_state_dict(shared_state, strict=True)
        models[name] = model.to(device)
    return models


def build_models_with_shared_state(
    *, device: torch.device | str = "cpu"
) -> tuple[dict[str, ToyResNet18], dict[str, torch.Tensor], str]:
    """Build all conditions from one exact base state dict and expose its hash."""

    seed_everything()
    base = ToyResNet18(padding_mode="zeros", total_stride=32)
    shared_state = copy.deepcopy(base.state_dict())
    digest = state_dict_hash(shared_state)
    models: dict[str, ToyResNet18] = {}
    for name, spec in MODEL_SPECS.items():
        model = ToyResNet18(padding_mode=str(spec["padding_mode"]), total_stride=int(spec["total_stride"]))
        model.load_state_dict(shared_state, strict=True)
        model.to(device)
        model.eval()
        models[name] = model
    return models, shared_state, digest


def extract_stage_features(model: ToyResNet18, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract spatial stages and fc-excluded GAP features in eval/no-grad mode."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        stages = model.forward_stages(value)
        gap = torch.flatten(model.avgpool(stages["layer4"]), 1)
    if was_training:
        model.train()
    return {**stages, "gap": gap}


def extract_pooled_features(model: ToyResNet18, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return one compact vector per spatial stage plus the final GAP vector."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        stages = model.forward_stages(value)
        pooled = {
            f"{name}_gap": torch.flatten(model.avgpool(value_), 1)
            for name, value_ in stages.items()
        }
        pooled["gap"] = torch.flatten(model.avgpool(stages["layer4"]), 1)
    if was_training:
        model.train()
    return pooled


def model_shape_report(model: ToyResNet18, *, image_size: int | None = None) -> dict[str, object]:
    size = int(CONFIG["image_size"] if image_size is None else image_size)
    with torch.no_grad():
        sample = torch.zeros(1, 3, size, size, device=next(model.parameters()).device)
        stages = model.forward_stages(sample)
        gap = model.forward_features(sample)
    return {
        "padding_mode": model.padding_mode,
        "total_stride": model.total_stride,
        "stage_shapes": {name: list(value.shape) for name, value in stages.items()},
        "gap_shape": list(gap.shape),
        "fc_shape": list(model.fc.weight.shape),
    }
