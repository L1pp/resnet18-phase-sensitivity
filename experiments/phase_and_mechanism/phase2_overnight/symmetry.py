"""R18 symmetry ladder and extra Sparse4 for other architectures."""

from __future__ import annotations

from typing import Any, Dict, List

from phase1_gap_rep.common import dump_json

from .protocol import ARCH_ATLAS, PHASE2_ROOT, SEED_ATLAS, record_failure
from .train_regime import dense_eval_model, model_dir, n_steps_ref, train_regime

OUT = PHASE2_ROOT / "symmetry"


def run_symmetry_and_extras(atlas: Dict[str, Any], deh: Dict[str, Any]) -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    steps = n_steps_ref()
    log: List[Dict[str, Any]] = []

    def interp_mae(arch: str, regime: str) -> float:
        path = model_dir(arch, regime, SEED_ATLAS) / "tables" / "dense_eval.json"
        if not path.exists():
            return float("nan")
        blob = __import__("json").loads(path.read_text(encoding="utf-8"))
        return float(blob.get("interpolation", {}).get("t_mae_px", float("nan")))

    def out_mae(regime: str) -> float:
        path = model_dir("resnet18", regime, SEED_ATLAS) / "tables" / "dense_eval.json"
        if not path.exists():
            return float("nan")
        blob = __import__("json").loads(path.read_text(encoding="utf-8"))
        return float(blob.get("outside_hull", {}).get("t_mae_px", float("nan")))

    for arch in ARCH_ATLAS:
        if arch == "resnet18":
            continue
        s9 = interp_mae(arch, "G9")
        if s9 <= 1.0:
            try:
                tr = train_regime(arch, "G4", SEED_ATLAS, steps=steps)
                ev = dense_eval_model(arch, "G4", SEED_ATLAS)
                log.append({"name": f"{arch} Sparse4", "ok": True, "train": tr, "dense": ev})
                g4 = float(ev.get("interpolation", {}).get("t_mae_px", 99))
                if g4 <= 1.5:
                    for reg in ("G2x", "G2y"):
                        tr2 = train_regime(arch, reg, SEED_ATLAS, steps=steps)
                        ev2 = dense_eval_model(arch, reg, SEED_ATLAS)
                        log.append({"name": f"{arch} {reg}", "ok": True, "dense": ev2})
            except Exception as exc:
                log.append({"name": f"{arch} Sparse4", "ok": False, "error": str(exc)})
                record_failure(f"{arch}_Sparse4", str(exc))

    for variant in ("circular", "antialiased", "circular_aa"):
        try:
            tr = train_regime("resnet18", "G9", SEED_ATLAS, variant=variant, steps=steps)
            ev = dense_eval_model("resnet18", "G9", SEED_ATLAS, variant=variant)
            log.append({"name": f"R18 {variant} Sparse9", "ok": True, "train": tr, "dense": ev})
        except Exception as exc:
            log.append({"name": f"R18 {variant} Sparse9", "ok": False, "error": str(exc), "skip_ok": True})
            record_failure(f"R18_{variant}_Sparse9", str(exc), {"skip_ok": True})

    c9_out = out_mae("C9")
    if c9_out == c9_out and c9_out >= 5.0:
        for variant in ("circular", "antialiased", "circular_aa"):
            try:
                tr = train_regime("resnet18", "C9", SEED_ATLAS, variant=variant, steps=steps)
                ev = dense_eval_model("resnet18", "C9", SEED_ATLAS, variant=variant)
                log.append({"name": f"R18 {variant} Central9", "ok": True, "train": tr, "dense": ev})
            except Exception as exc:
                log.append({"name": f"R18 {variant} Central9", "ok": False, "error": str(exc), "skip_ok": True})
                record_failure(f"R18_{variant}_Central9", str(exc), {"skip_ok": True})

    try:
        tr = train_regime("resnet18", "G9", SEED_ATLAS, variant="coordconv", steps=steps)
        ev = dense_eval_model("resnet18", "G9", SEED_ATLAS, variant="coordconv")
        log.append({"name": "R18 CoordConv Sparse9", "ok": True, "train": tr, "dense": ev})
    except Exception as exc:
        log.append({"name": "R18 CoordConv Sparse9", "ok": False, "error": str(exc), "skip_ok": True})
        record_failure("R18_CoordConv_Sparse9", str(exc), {"skip_ok": True})

    dump_json(OUT / "panel.json", log)
    return {"ran": log}
