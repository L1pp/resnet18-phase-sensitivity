from __future__ import annotations

import copy
import random
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEED = 20260825
IMAGE_SIZE = 64
POOL_DENOMINATOR = 4096.0
CONDITION_MODES: Mapping[str, str] = {
    "Z": "zero",
    "V": "valid",
    "C": "circular",
}


def seed_everything(seed: int = SEED) -> None:
    """Set the one experiment seed; no additional seed is introduced."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class MiniCNN(nn.Module):
    """Four S1 3x3 convolutions with explicit condition-specific boundaries."""

    def __init__(self, *, mode: str) -> None:
        super().__init__()
        if mode not in {"zero", "valid", "circular"}:
            raise ValueError(f"unsupported MiniCNN mode: {mode}")
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(32, 4)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -1.0 / 32**0.5, 1.0 / 32**0.5)

    def _pad(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode == "valid":
            return value
        if self.mode == "zero":
            return F.pad(value, (1, 1, 1, 1), mode="constant", value=0.0)
        return F.pad(value, (1, 1, 1, 1), mode="circular")

    def forward_stages(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.relu(self.conv1(self._pad(value)))
        stages = {"conv1": value}
        value = self.relu(self.conv2(self._pad(value)))
        stages["conv2"] = value
        value = self.relu(self.conv3(self._pad(value)))
        stages["conv3"] = value
        value = self.relu(self.conv4(self._pad(value)))
        stages["conv4"] = value
        return stages

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        spatial = self.forward_stages(value)["conv4"]
        # The approved protocol fixes the denominator at 64*64, including V.
        return spatial.sum(dim=(2, 3)) / POOL_DENOMINATOR

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(value))


def build_shared_models(*, device: torch.device | str = "cpu") -> dict[str, MiniCNN]:
    """Copy one base state_dict directly into the three condition models."""

    seed_everything()
    base = MiniCNN(mode="zero")
    shared_state = copy.deepcopy(base.state_dict())
    models: dict[str, MiniCNN] = {}
    for condition, mode in CONDITION_MODES.items():
        model = MiniCNN(mode=mode)
        model.load_state_dict(shared_state, strict=True)
        model.to(device)
        model.eval()
        models[condition] = model
    return models


def model_shape_report(model: MiniCNN, *, image_size: int = IMAGE_SIZE) -> dict[str, object]:
    with torch.no_grad():
        sample = torch.zeros(1, 3, int(image_size), int(image_size), device=next(model.parameters()).device)
        stages = model.forward_stages(sample)
        features = model.forward_features(sample)
    return {
        "mode": model.mode,
        "stage_shapes": {name: list(value.shape) for name, value in stages.items()},
        "pool_shape": list(features.shape),
        "head_shape": list(model.head.weight.shape),
    }


__all__ = [
    "CONDITION_MODES",
    "IMAGE_SIZE",
    "MiniCNN",
    "POOL_DENOMINATOR",
    "SEED",
    "build_shared_models",
    "model_shape_report",
    "seed_everything",
]
