"""Original conditional branches I-A / I-B / I-C plus atlas Sparse4 extras."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from phase1_gap_rep.common import dump_json

from .protocol import PHASE2_ROOT, SEED_ATLAS, record_failure
from .train_regime import dense_eval_model, model_dir, n_steps_ref, train_regime

OUT = PHASE2_ROOT / "branches"


def _mae(blob: Dict[str, Any], key: str) -> float:
    try:
        return float(blob[key]["t_mae_px"])
    except Exception:
        return float("nan")


def run_branches(deh: Dict[str, Any]) -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    steps = n_steps_ref()
    by = {row["regime"]: row for row in deh.get("regimes", []) if row.get("ok")}
    log: Dict[str, Any] = {"ran": []}

    def dense(reg):
        return by.get(reg, {}).get("dense", {})

    c16_out = _mae(dense("C16"), "outside_hull")
    c4_out = _mae(dense("C4"), "outside_hull")
    g16_in = _mae(dense("G16"), "interpolation")
    c16_in = _mae(dense("C16"), "in_hull")
    g9_in = _mae(dense("G9"), "interpolation")
    g4_in = _mae(dense("G4"), "interpolation")

    # I-A: C9/C4 already trained. If C4 still extrapolates, LeftHalf.
    if c16_out <= 1.0 and c4_out <= 2.0:
        try:
            tr = train_regime("resnet18", "LeftHalf", SEED_ATLAS, steps=steps)
            ev = dense_eval_model("resnet18", "LeftHalf", SEED_ATLAS)
            log["ran"].append({"name": "I-A LeftHalf", "train": tr, "dense": ev})
        except Exception as exc:
            log["ran"].append({"name": "I-A LeftHalf", "error": str(exc)})
            record_failure("I-A_LeftHalf", str(exc))

    # I-B CoordConv + circular on C16
    if g16_in <= 1.0 and c16_out >= 3.0 and c16_out >= 3.0 * (c16_in if c16_in == c16_in else 1):
        for variant in ("coordconv", "circular"):
            try:
                tr = train_regime("resnet18", "C16", SEED_ATLAS, variant=variant, steps=steps)
                ev = dense_eval_model("resnet18", "C16", SEED_ATLAS, variant=variant)
                log["ran"].append({"name": f"I-B {variant}", "train": tr, "dense": ev})
            except Exception as exc:
                log["ran"].append({"name": f"I-B {variant}", "error": str(exc), "skipped_ok": True})
                record_failure(f"I-B_{variant}", str(exc), {"skipped_ok": True})

    # I-C G2x G2y
    if g9_in <= 1.0 and g4_in <= 1.5:
        for reg in ("G2x", "G2y"):
            try:
                tr = train_regime("resnet18", reg, SEED_ATLAS, steps=steps)
                ev = dense_eval_model("resnet18", reg, SEED_ATLAS)
                log["ran"].append({"name": f"I-C {reg}", "train": tr, "dense": ev})
            except Exception as exc:
                log["ran"].append({"name": f"I-C {reg}", "error": str(exc)})
                record_failure(f"I-C_{reg}", str(exc))

    dump_json(OUT / "original_branches.json", log)
    return log
