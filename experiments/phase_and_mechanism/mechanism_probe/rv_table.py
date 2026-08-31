"""S4a: support-vs-dense amplification from already computed F1 tangents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .const import DRIFT_ROOT, RESULTS_ROOT, STAGES
from .blob_io import ensure_results, write_json


def _pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _from_local(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    rand = blob.get("random_direction_amplification") or {}
    return {
        "source": str(path),
        "amp_unit": _pick(blob, "unit_fd", "amplification"),
        "amp_gd": _pick(blob, "gd", "amplification"),
        "amp_adam": _pick(blob, "adam", "amplification"),
        "d_box_gd": _pick(blob, "gd", "d_mean_box"),
        "d_box_adam": _pick(blob, "adam", "d_mean_box"),
        "rand_median": rand.get("median"),
        "rand_max": rand.get("max"),
        "rand_all": rand.get("all"),
    }


def _from_amd(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    rand = blob.get("random_directions") or {}
    gd = blob.get("gd") or {}
    adam = blob.get("adam") or {}
    unit = blob.get("unit") or {}
    amp_gd = None
    if gd.get("rms_delta_support"):
        amp_gd = float(gd["rms_delta_dense"]) / max(float(gd["rms_delta_support"]), 1e-12)
    amp_adam = None
    if adam.get("rms_delta_support"):
        amp_adam = float(adam["rms_delta_dense"]) / max(float(adam["rms_delta_support"]), 1e-12)
    return {
        "source": str(path),
        "amp_unit": unit.get("amplification"),
        "amp_gd": amp_gd,
        "amp_adam": amp_adam,
        "d_box_gd": gd.get("d_mean_box_mae_px"),
        "d_box_adam": adam.get("d_mean_box_mae_px"),
        "rand_median": rand.get("amplification_median"),
        "rand_max": rand.get("amplification_max"),
        "rand_all": rand.get("amplification"),
        "rms_delta_dense_adam": adam.get("rms_delta_dense"),
        "rms_delta_support_adam": adam.get("rms_delta_support"),
    }


def _from_a10(path: Path) -> Optional[Dict[str, Any]]:
    return _from_amd(path)


def run() -> Dict[str, Any]:
    ensure_results()
    table: Dict[str, Any] = {"note": "R(v)=rms(delta_dense)/rms(delta_support); existing F1 only"}
    for machine, loader, root in (
        ("local_2d", _from_local, DRIFT_ROOT / "local" / "t1_2d"),
        ("amd_2d", _from_amd, DRIFT_ROOT / "amd"),
        ("a10_6d", _from_a10, DRIFT_ROOT / "a10" / "unfreeze_g64_corners4"),
        ("local_6d_probe", _from_local, DRIFT_ROOT / "local" / "t1_6d_probe"),
    ):
        table[machine] = {}
        for stage in STAGES:
            path = root / stage / "t1.json"
            if machine.startswith("amd") or machine.startswith("a10"):
                alt = root / stage / "step0_t1" / "t1.json"
                path = alt if alt.exists() else path
            rec = loader(path)
            table[machine][stage] = rec if rec is not None else {"missing": str(path)}

    ranking = []
    for machine in ("local_2d", "amd_2d"):
        amps = []
        for stage in STAGES:
            rec = table[machine].get(stage) or {}
            amps.append(
                {
                    "stage": stage,
                    "amp_unit": rec.get("amp_unit"),
                    "amp_adam": rec.get("amp_adam"),
                    "amp_gd": rec.get("amp_gd"),
                    "rand_median": rec.get("rand_median"),
                    "d_box_adam": rec.get("d_box_adam"),
                }
            )
        ranking.append({"machine": machine, "rows": amps})
    table["ranking_note"] = (
        "unit/random R(v) already ~0.7-1.1 across head/l4/full; "
        "Adam d_box is the quantity that separates shock, not R(v) on random directions."
    )
    table["ranking"] = ranking
    write_json(RESULTS_ROOT / "s4_rv_existing.json", table)
    from .plots import plot_rv

    plot_rv(table)
    print("[S4a] wrote", RESULTS_ROOT / "s4_rv_existing.json", flush=True)
    return table
