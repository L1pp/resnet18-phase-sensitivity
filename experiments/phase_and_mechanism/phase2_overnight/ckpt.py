"""Load Phase 1.8 slim checkpoints and generic Phase 2 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from phase1_gap_rep.common import RESULTS_ROOT, build_model, device

from .backbones import build_backbone


def phase18_slim_path(seed: int) -> Path:
    return RESULTS_ROOT / "phase1_8_unseen_t" / f"seed_{seed}" / "checkpoints" / "resnet18_best_slim.pt"


def load_phase18_slim(seed: int) -> nn.Module:
    path = phase18_slim_path(seed)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device(), weights_only=False)
    model = build_model().to(device())
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def save_slim(model: nn.Module, path: Path, extra: Dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_arch_checkpoint(path: Path, arch: str, variant: str = "standard") -> nn.Module:
    payload = torch.load(path, map_location=device(), weights_only=False)
    model, _ = build_backbone(arch, variant)
    model.to(device())
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
