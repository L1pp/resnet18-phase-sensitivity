"""Vanilla CNN model and exact-initialization helpers."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models.resnet import BasicBlock


def seed_all(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass


class VanillaResNet18(tv_models.ResNet):
    """Torchvision ResNet18 with the legacy vanilla checkpoint key layout.

    Subclassing torchvision's ResNet is intentional: old exact-init files use
    keys such as ``conv1.weight`` and ``layer4.1.conv2.weight``.  A wrapper
    with a ``backbone.`` prefix would silently create a different protocol.
    """

    variant = "vanilla"

    def __init__(self) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


class TinyRegressor(nn.Module):
    """Small injectable model used only by local synthetic tests/smoke."""

    variant = "tiny"

    def __init__(self, image_size: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(8, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


def build_model(*, tiny: bool = False) -> nn.Module:
    return TinyRegressor() if bool(tiny) else VanillaResNet18()


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def backbone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        return {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
    # Legacy-compatible ResNet keeps all backbone modules at the root.  Only
    # the regression fc is plastic in head_only.
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("fc.")
    }


def backbone_hash(model: nn.Module) -> str:
    return state_dict_hash(backbone_state_dict(model))


def backbone_parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable backbone parameters, excluding the regression fc.

    This parameter-only identity is the causal full-plasticity gate.  The
    historical ``backbone_hash`` intentionally includes buffers for a useful
    diagnostic, but BatchNorm running-stat changes must not count as backbone
    parameter learning.
    """

    return {
        key: value.detach().cpu().clone()
        for key, value in model.named_parameters()
        if not key.startswith("fc.") and not key.startswith("backbone.fc.")
    }


def backbone_parameter_hash(model: nn.Module) -> str:
    return state_dict_hash(backbone_parameter_state(model))


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_exact_init(model: nn.Module, path: str | Path, *, seed: int, protocol_hash: str) -> dict[str, Any]:
    state = clone_state_dict(model)
    payload = {
        "schema_version": 1,
        "kind": "exact_init",
        "variant": str(getattr(model, "variant", "vanilla")),
        "seed": int(seed),
        "protocol_hash": str(protocol_hash),
        "model_state": state,
        "init_hash": state_dict_hash(state),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return {key: value for key, value in payload.items() if key != "model_state"}


def load_exact_init(path: str | Path, model: nn.Module, *, expected_seed: int | None = None, expected_protocol_hash: str | None = None, allow_source_protocol_mismatch: bool = False) -> dict[str, Any]:
    payload = _torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise ValueError(f"invalid exact init checkpoint: {path}")
    if expected_seed is not None and int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("exact-init seed mismatch")
    if expected_protocol_hash is not None and not allow_source_protocol_mismatch and str(payload.get("protocol_hash")) != str(expected_protocol_hash):
        raise ValueError("exact-init protocol hash mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    expected_hash = state_dict_hash(payload["model_state"])
    if str(payload.get("init_hash")) != expected_hash:
        raise ValueError("exact-init hash is invalid")
    return dict(payload)


__all__ = [
    "TinyRegressor",
    "VanillaResNet18",
    "backbone_hash",
    "backbone_parameter_hash",
    "backbone_parameter_state",
    "backbone_state_dict",
    "build_model",
    "clone_state_dict",
    "load_exact_init",
    "save_exact_init",
    "seed_all",
    "state_dict_hash",
]
