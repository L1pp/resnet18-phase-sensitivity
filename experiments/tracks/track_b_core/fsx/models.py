from __future__ import annotations

import random
from typing import Any

import numpy as np

TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models
except Exception as exc:  # local input preparation must remain usable without Torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tv_models = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc


def require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None or torch is None or nn is None or tv_models is None:
        raise RuntimeError(f"Torch/Torchvision runtime unavailable: {TORCH_IMPORT_ERROR}")


def seed_all(seed: int, deterministic_algorithms: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


if nn is not None:
    class FrozenBatchNorm2d(nn.BatchNorm2d):
        """Ordinary BN calibrated once, then fixed-statistics with trainable affine."""

        def __init__(self, channels: int) -> None:
            super().__init__(channels)
            self._stats_frozen = False

        def freeze_running_stats(self) -> None:
            self._stats_frozen = True
            super().train(False)

        def train(self, mode: bool = True) -> "FrozenBatchNorm2d":
            return super().train(False if self._stats_frozen else mode)


    class TinySmokeCNN(nn.Module):
        """Cheap structural substitute used only by non-formal local smoke runs.

        It deliberately exposes ``layer4``, ``fc`` and GAP features like the
        production ResNet.  A smoke config records this substitution and can
        never be formal-eligible.
        """

        def __init__(self, normalization: str) -> None:
            super().__init__()
            norm = _norm_layer(normalization)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                norm(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
                norm(32),
                nn.ReLU(inplace=False),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                norm(32),
                nn.ReLU(inplace=False),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 2)
            self.normalization = normalization

        def forward_gap_features(self, value):
            value = self.layer4(self.stem(value))
            return self.avgpool(value).flatten(1)

        def forward(self, value):
            return self.fc(self.forward_gap_features(value))
else:
    class FrozenBatchNorm2d:  # type: ignore[no-redef]
        def __init__(self, channels: int) -> None:
            require_torch()

    class TinySmokeCNN:  # type: ignore[no-redef]
        def __init__(self, normalization: str) -> None:
            require_torch()


def _norm_layer(kind: str):
    require_torch()
    if kind == "bn":
        return nn.BatchNorm2d
    if kind == "frozen_bn":
        return FrozenBatchNorm2d
    if kind == "gn":
        def factory(channels: int):
            if channels % 32:
                raise ValueError(f"GN32 incompatible with {channels} channels")
            return nn.GroupNorm(32, channels)
        return factory
    raise ValueError(f"unsupported normalization {kind}")


def build_resnet18(normalization: str, run_seed: int):
    """Seed before construction; caller may clone one state for a matched pair."""
    require_torch()
    seed_all(run_seed)
    model = tv_models.resnet18(weights=None, norm_layer=_norm_layer(normalization))
    model.fc = nn.Linear(int(model.fc.in_features), 2)
    model.normalization = normalization
    model.run_seed = int(run_seed)
    return model


def build_smoke_cnn(normalization: str, run_seed: int):
    require_torch()
    seed_all(run_seed)
    model = TinySmokeCNN(normalization)
    model.run_seed = int(run_seed)
    return model


def forward_gap_features(model: Any, value: Any):
    """Return the exact tensor feeding ``fc`` for ResNet or smoke CNN."""
    require_torch()
    if hasattr(model, "forward_gap_features"):
        return model.forward_gap_features(value)
    x = model.conv1(value)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return model.avgpool(x).flatten(1)


def calibrate_and_freeze_bn(model: Any, calibration_images: Any) -> dict[str, int]:
    require_torch()
    if not isinstance(calibration_images, torch.Tensor) or calibration_images.ndim != 4 or calibration_images.shape[0] != 256:
        raise ValueError("FrozenBN requires exactly one [256,C,H,W] calibration batch")
    layers = [module for module in model.modules() if isinstance(module, FrozenBatchNorm2d)]
    if not layers:
        raise ValueError("model contains no FrozenBatchNorm2d layers")
    if any(layer._stats_frozen for layer in layers):
        raise ValueError("FrozenBN calibration cannot be repeated")
    model.train(True)
    with torch.no_grad():
        model(calibration_images)
    for layer in layers:
        layer.freeze_running_stats()
        if layer.affine:
            layer.weight.requires_grad_(True)
            layer.bias.requires_grad_(True)
    model.train(True)
    return {"calibration_rows": 256, "forward_calls": 1, "frozen_bn_layers": len(layers)}


def set_trainable_regime(model: Any, regime: str) -> list[str]:
    require_torch()
    if regime not in {"head_only", "layer4", "full", "lp_ft"}:
        raise ValueError(f"unknown B6 regime {regime}")
    for parameter in model.parameters():
        parameter.requires_grad_(regime == "full")
    if regime in {"head_only", "lp_ft"}:
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
    elif regime == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def unfreeze_lp_ft(model: Any) -> list[str]:
    require_torch()
    newly_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            parameter.requires_grad_(True)
            newly_trainable.append(name)
    return newly_trainable


def clone_state(model: Any) -> dict[str, Any]:
    require_torch()
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def build_explicit_xy_mlp(run_seed: int):
    require_torch()
    torch.set_num_threads(1)
    seed_all(run_seed, deterministic_algorithms=True)
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ).to(dtype=torch.float32, device="cpu")
