"""Small, self-contained model and reproducibility helpers for S1.

This module deliberately has no dependency on the historical ``today_shortcycle``
package.  The caller is responsible for calling :func:`seed_all` *before*
calling :func:`build_model`; ``build_model`` never changes the process RNG state.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models


VANILLA_VARIANTS = {"vanilla", "standard", "resnet18", "cnn"}
COORDCONV_VARIANTS = {"coordconv", "coord_conv", "resnet18_coordconv"}


def canonical_variant(variant: str | None) -> str:
    """Return the two protocol variant names used in checkpoint metadata."""

    value = "vanilla" if variant is None else str(variant).strip().lower()
    if value in VANILLA_VARIANTS:
        return "vanilla"
    if value in COORDCONV_VARIANTS:
        return "coordconv"
    raise ValueError(f"unknown S1 model variant: {variant!r}")


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, Torch and CUDA without constructing a model.

    The S1 launcher calls this before ``build_model`` in each independent
    process.  Deterministic algorithm settings are part of the run protocol;
    ``warn_only`` keeps unsupported operators visible in logs without changing
    a batch size or silently substituting a different training procedure.
    """

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
    except (AttributeError, TypeError):
        # Older Torch releases may not expose the warn_only keyword.  The
        # cuDNN settings above still provide the historical fallback.
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass


class CoordConvResNet18(nn.Module):
    """ResNet18 with two explicit image-coordinate channels.

    The coordinate channels are generated at forward time as x/y linspaces in
    ``[-1, 1]``.  The RGB branch is otherwise the ordinary torchvision
    ``resnet18(weights=None)``.  Keeping the backbone as an attribute also
    makes the five-channel stem and checkpoint structure explicit.
    """

    variant = "coordconv"

    def __init__(self) -> None:
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            5,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        )
        backbone.fc = nn.Linear(backbone.fc.in_features, 2)
        self.backbone = backbone

    @property
    def conv1(self) -> nn.Conv2d:
        """Expose the stem for lightweight structural audits."""

        return self.backbone.conv1

    @property
    def fc(self) -> nn.Linear:
        return self.backbone.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
        channels = int(x.shape[1])
        if channels == 3:
            b, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
            yy = yy.view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
            xx = xx.view(1, 1, 1, w).expand(b, 1, h, w)
            x = torch.cat((x, xx, yy), dim=1)
        elif channels != 5:
            raise ValueError(f"CoordConv expects 3 RGB or 5-channel input, got {channels}")
        return self.backbone(x)


def build_model(variant: str = "vanilla") -> nn.Module:
    """Build a fresh S1 model.

    ``seed_all`` is intentionally not called here.  This ordering is required
    for corrected seed semantics and is tested by the S1 smoke suite.
    """

    name = canonical_variant(variant)
    if name == "coordconv":
        return CoordConvResNet18()
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # A plain attribute is sufficient for the runner and does not enter the
    # state_dict, so vanilla checkpoint keys remain torchvision-compatible.
    model.variant = "vanilla"  # type: ignore[attr-defined]
    return model


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a state dict using sorted keys and contiguous tensor bytes."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load checkpoints across Torch versions with an explicit CPU default."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_state(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    variant: str | None = None,
    device: str | torch.device = "cpu",
) -> Any:
    """Load an S1 model checkpoint.

    With no ``model`` argument this returns the checkpoint payload, allowing a
    caller to inspect its frozen ``variant`` and hashes.  When ``model`` is
    supplied, its state is populated and the same payload is returned.  A bare
    state-dict file is accepted for compatibility, but formal S1 checkpoints
    always carry the metadata fields written by ``train_run``.
    """

    payload = _torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path!s} is not a mapping")
    state = payload.get("model_state")
    if state is None:
        # A raw state_dict has tensor values and no metadata.  Keep the return
        # shape stable by wrapping it in the normal payload contract.
        if not payload or not all(torch.is_tensor(v) for v in payload.values()):
            raise KeyError(f"checkpoint {path!s} lacks model_state")
        state = dict(payload)
        payload = {"model_state": state, "variant": canonical_variant(variant)}
    if model is not None:
        model.load_state_dict(state, strict=True)
        model.to(device)
    return payload


__all__ = [
    "CoordConvResNet18",
    "build_model",
    "canonical_variant",
    "load_model_state",
    "seed_all",
    "state_dict_hash",
]
