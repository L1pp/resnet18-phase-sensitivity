"""Find Phase 1.8 / Phase 2 slim checkpoints on the cloud VM. Never retrain Phase 1.8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from phase1_gap_rep.common import REPO_ROOT

from .io_util import dump_json
from .protocol import PHASE18_SEEDS, PHASE3_ROOT


@dataclass
class ModelRef:
    name: str
    path: str
    arch: str
    variant: str
    seed: int
    kind: str
    role: str
    found: bool
    notes: str = ""


def _exists(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    return None


def _phase18_candidates(seed: int) -> List[Path]:
    root = REPO_ROOT / "results" / "phase1_gap_rep" / "phase1_8_unseen_t" / f"seed_{seed}" / "checkpoints"
    return [
        root / "resnet18_best_slim.pt",
        root / "best_slim.pt",
        root / "resnet18_best.pt",
    ]


def _phase2_candidates(run_dir: str) -> List[Path]:
    root = REPO_ROOT / "results" / "phase2_overnight_discovery" / "models" / run_dir / "checkpoints"
    return [root / "best_slim.pt", root / "last_slim.pt"]


def _first(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        hit = _exists(path)
        if hit is not None:
            return hit
    return None


def locate_models() -> List[ModelRef]:
    refs: List[ModelRef] = []
    for seed in PHASE18_SEEDS:
        hit = _first(_phase18_candidates(seed))
        refs.append(
            ModelRef(
                name=f"phase18_r18_{seed}",
                path=str(hit) if hit else "",
                arch="resnet18",
                variant="standard",
                seed=int(seed),
                kind="phase18",
                role="primary",
                found=hit is not None,
                notes="Phase 1.8 may have resume=true in old config; do not retrain.",
            )
        )
    dn = _first(_phase2_candidates("densenet121__G64__20260820"))
    refs.append(
        ModelRef(
            name="phase2_densenet121_G64",
            path=str(dn) if dn else "",
            arch="densenet121",
            variant="standard",
            seed=20260820,
            kind="phase2",
            role="primary",
            found=dn is not None,
        )
    )
    r18 = _first(_phase2_candidates("resnet18__G64__20260820"))
    refs.append(
        ModelRef(
            name="phase2_resnet18_G64",
            path=str(r18) if r18 else "",
            arch="resnet18",
            variant="standard",
            seed=20260820,
            kind="phase2",
            role="secondary",
            found=r18 is not None,
        )
    )
    eff = _first(_phase2_candidates("efficientnet_b0__G64__20260820_fp32"))
    refs.append(
        ModelRef(
            name="phase2_efficientnet_b0_G64_fp32",
            path=str(eff) if eff else "",
            arch="efficientnet_b0",
            variant="standard",
            seed=20260820,
            kind="phase2",
            role="secondary",
            found=eff is not None,
            notes="fp32+clip recovery only. Do not mix with AMP failure.",
        )
    )
    for seed in PHASE18_SEEDS:
        init = _first(
            [
                REPO_ROOT
                / "results"
                / "phase1_gap_rep"
                / "phase1_8_unseen_t"
                / f"seed_{seed}"
                / "checkpoints"
                / "init.pt",
                REPO_ROOT
                / "results"
                / "phase2_overnight_discovery"
                / "models"
                / "resnet18__G64__20260820"
                / "checkpoints"
                / "init.pt",
            ]
        )
        refs.append(
            ModelRef(
                name=f"init_r18_{seed}",
                path=str(init) if init else "",
                arch="resnet18",
                variant="standard",
                seed=int(seed),
                kind="init",
                role="control",
                found=init is not None,
                notes="If missing, Track A builds a fresh random-init control.",
            )
        )
        break
    return refs


def primary_trained(refs: Optional[List[ModelRef]] = None) -> List[ModelRef]:
    refs = refs if refs is not None else locate_models()
    return [r for r in refs if r.role == "primary" and r.found]


def dump_locator() -> Dict[str, Any]:
    refs = locate_models()
    payload = {
        "models": [asdict(r) for r in refs],
        "n_found": int(sum(r.found for r in refs)),
        "n_primary_found": int(sum(r.found and r.role == "primary" for r in refs)),
    }
    dump_json(PHASE3_ROOT / "config" / "ckpt_locator.json", payload)
    return payload
