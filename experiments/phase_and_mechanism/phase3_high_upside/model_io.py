"""Load Phase 1.8 / Phase 2 checkpoints and build random-init twins."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from phase1_gap_rep.common import build_model, device, seed_everything
from phase2_overnight.backbones import build_backbone
from phase2_overnight.ckpt import load_arch_checkpoint, load_phase18_slim

from .ckpt_locator import ModelRef
from .protocol import SEED_RANDOM_INIT


def load_ref(ref: ModelRef) -> nn.Module:
    path = Path(ref.path)
    if ref.kind == "phase18":
        if path.name.startswith("resnet18") or "phase1_8" in str(path):
            try:
                seed = int(ref.seed)
                return load_phase18_slim(seed)
            except FileNotFoundError:
                pass
        payload = torch.load(path, map_location=device(), weights_only=False)
        model = build_model().to(device())
        model.load_state_dict(payload["model_state"])
        model.eval()
        return model
    if ref.kind in {"phase2", "init"}:
        return load_arch_checkpoint(path, ref.arch, ref.variant)
    raise ValueError(ref.kind)


def build_random_twin(arch: str, variant: str = "standard", seed: int = SEED_RANDOM_INIT) -> Tuple[nn.Module, str]:
    seed_everything(seed)
    if arch == "resnet18" and variant == "standard":
        model = build_model().to(device())
        model.eval()
        return model, "resnet18"
    model, tag = build_backbone(arch, variant)
    model.to(device())
    model.eval()
    return model, tag
