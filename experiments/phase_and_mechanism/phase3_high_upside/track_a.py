"""Track A: emergent translation-group operators on frozen trained GAP features."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from phase1_gap_rep.common import IMAGE_SIZE, setup_matplotlib_chinese
from phase2_overnight.features import extract_stage_gaps

from .ckpt_locator import ModelRef, dump_locator, locate_models, primary_trained
from .group_law import run_group_law_battery
from .head_semantics import evaluate_head_battery, extract_head
from .io_util import dump_json
from .model_io import build_random_twin, load_ref
from .operators import (
    apply_affine,
    fit_operator,
    pack_operator,
    random_same_rank_operator,
    shuffle_pairs,
)
from .protocol import (
    ALL_DELTAS,
    EVAL_SHAPE_IDS,
    PHASE3_ROOT,
    SEED_RANDOM_INIT,
    delta_key,
    in_safe_box,
    load_protocol,
    operator_origins_px,
    origin_split,
    record_failure,
    shape_split,
)
from .render_bezier import render_at

OUT = PHASE3_ROOT / "track_a"
STAGES = ["layer3", "layer4", "gap"]


def _split_pairs(
    z: np.ndarray,
    zp: np.ndarray,
    shape_ids: Sequence[int],
    origin_ids: Sequence[int],
    valid: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    proto = load_protocol()
    s_split = proto["shape_split"]
    o_split = proto["origin_split"]
    sid = np.asarray(shape_ids, dtype=int)
    oid = np.asarray(origin_ids, dtype=int)
    valid = np.asarray(valid, dtype=bool)

    def _mask(shapes: Sequence[int], origins: Sequence[int]) -> np.ndarray:
        return valid & np.isin(sid, np.asarray(shapes, dtype=int)) & np.isin(oid, np.asarray(origins, dtype=int))

    out = {}
    for name, shapes, origins in (
        ("fit", s_split["fit"], o_split["fit"]),
        ("val", s_split["val"], o_split["val"]),
        ("test", s_split["test"], o_split["test"]),
    ):
        mask = _mask(shapes, origins)
        out[name] = (z[mask], zp[mask])
    return out


def extract_operator_features(model, origins: np.ndarray, batch_size: int = 64) -> Dict[str, Any]:
    n_s = len(EVAL_SHAPE_IDS)
    n_o = len(origins)
    feats: Dict[str, Dict[str, np.ndarray]] = {}
    valid = {}
    for dkey, delta in [("src", (0.0, 0.0))] + [(delta_key(d), d) for d in ALL_DELTAS]:
        dest = origins + np.asarray(delta, dtype=np.float64).reshape(1, 2)
        ok = in_safe_box(dest)
        valid[dkey] = ok
        jobs = [(si, oi, int(EVAL_SHAPE_IDS[si]), dest[oi]) for si in range(n_s) for oi in range(n_o) if ok[oi]]
        bucket: Dict[str, np.ndarray] = {}
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]
            images = np.stack([render_at(sid, t) for _si, _oi, sid, t in chunk], axis=0)
            extracted = extract_stage_gaps(model, images, stages=STAGES)
            if not bucket:
                for name, arr in extracted.items():
                    bucket[name] = np.full((n_s, n_o, arr.shape[-1]), np.nan, dtype=np.float32)
            for local, (si, oi, _sid, _t) in enumerate(chunk):
                for name, arr in extracted.items():
                    bucket[name][si, oi] = arr[local]
        feats[dkey] = bucket
    return {"features": feats, "valid": {k: v.astype(bool) for k, v in valid.items()}, "origins": origins}


def _pairs_from_store(store: Mapping[str, Any], stage: str, dkey: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = store["features"]["src"][stage]
    dst = store["features"][dkey][stage]
    ok = np.asarray(store["valid"][dkey], dtype=bool)
    n_s, n_o, dim = src.shape
    z, zp, sids, oids, valid = [], [], [], [], []
    for si, sid in enumerate(EVAL_SHAPE_IDS):
        for oi in range(n_o):
            good = bool(ok[oi]) and np.isfinite(src[si, oi]).all() and np.isfinite(dst[si, oi]).all()
            z.append(src[si, oi])
            zp.append(dst[si, oi])
            sids.append(sid)
            oids.append(oi)
            valid.append(good)
    return (
        np.asarray(z, dtype=np.float64),
        np.asarray(zp, dtype=np.float64),
        np.asarray(sids, dtype=int),
        np.asarray(oids, dtype=int),
        np.asarray(valid, dtype=bool),
    )


def fit_all_deltas(store: Mapping[str, Any], stage: str = "gap") -> Dict[str, Any]:
    ops = {}
    packed = {}
    for delta in ALL_DELTAS:
        dkey = delta_key(delta)
        z, zp, sids, oids, valid = _pairs_from_store(store, stage, dkey)
        splits = _split_pairs(z, zp, sids, oids, valid)
        if min(len(splits["fit"][0]), len(splits["val"][0]), len(splits["test"][0])) < 8:
            packed[dkey] = {"skipped": True, "reason": "too_few_pairs"}
            continue
        result = fit_operator(*splits["fit"], *splits["val"], *splits["test"])
        ops[dkey] = {"matrix": result["matrix"], "bias": result["bias"], **pack_operator(result)}
        packed[dkey] = pack_operator(result)
        packed[dkey]["delta"] = [float(delta[0]), float(delta[1])]
    return {"ops": ops, "packed": packed}


def _test_features(store: Mapping[str, Any], stage: str = "gap") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    proto = load_protocol()
    src = store["features"]["src"][stage]
    s_test = set(int(x) for x in proto["shape_split"]["test"])
    o_test = set(int(x) for x in proto["origin_split"]["test"])
    zs = []
    for si, sid in enumerate(EVAL_SHAPE_IDS):
        if sid not in s_test:
            continue
        for oi in o_test:
            row = src[si, oi]
            if np.isfinite(row).all():
                zs.append(row)
    z = np.asarray(zs, dtype=np.float64)
    targets = {}
    for delta in ALL_DELTAS:
        dkey = delta_key(delta)
        dst = store["features"][dkey][stage]
        ok = np.asarray(store["valid"][dkey], dtype=bool)
        rows = []
        src_rows = []
        for si, sid in enumerate(EVAL_SHAPE_IDS):
            if sid not in s_test:
                continue
            for oi in o_test:
                if not ok[oi]:
                    continue
                if np.isfinite(src[si, oi]).all() and np.isfinite(dst[si, oi]).all():
                    src_rows.append(src[si, oi])
                    rows.append(dst[si, oi])
        if rows:
            targets[dkey] = np.asarray(rows, dtype=np.float64)
            targets[dkey + "__src"] = np.asarray(src_rows, dtype=np.float64)
    return z, targets


def _controls(store: Mapping[str, Any], ops: Mapping[str, Any], stage: str = "gap") -> Dict[str, Any]:
    out = {}
    for delta in ALL_DELTAS[:6]:
        dkey = delta_key(delta)
        if dkey not in ops:
            continue
        z, zp, sids, oids, valid = _pairs_from_store(store, stage, dkey)
        splits = _split_pairs(z, zp, sids, oids, valid)
        z_f, zp_f = shuffle_pairs(*splits["fit"], seed=123)
        z_v, zp_v = shuffle_pairs(*splits["val"], seed=124)
        z_t, zp_t = shuffle_pairs(*splits["test"], seed=125)
        shuf = fit_operator(z_f, zp_f, z_v, zp_v, z_t, zp_t)
        rm, rb = random_same_rank_operator(ops[dkey]["matrix"], ops[dkey]["bias"], seed=126)
        pred_r = apply_affine(splits["test"][0], rm, rb)
        out[dkey] = {
            "shuffled_e_test": shuf["e_test"],
            "true_e_test": ops[dkey]["e_test"],
            "random_rank_e_test": float(
                __import__("phase3_high_upside.operators", fromlist=["normalized_error"]).normalized_error(pred_r, splits["test"][1])
            ),
        }
    return out


def _plot_e_delta(packed: Mapping[str, Any], path: Path) -> None:
    plt = setup_matplotlib_chinese()
    keys = [k for k, v in packed.items() if not v.get("skipped")]
    xs = [packed[k]["e_test"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(range(len(keys)), xs)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("held-out E_Δ")
    ax.set_title("Track A 各 Δ 的 held-out 归一化重建误差")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_one_model(ref: ModelRef, origins: np.ndarray) -> Dict[str, Any]:
    out = OUT / ref.name
    out.mkdir(parents=True, exist_ok=True)
    model = load_ref(ref)
    store = extract_operator_features(model, origins)
    np.savez_compressed(out / "features_meta.npz", origins=origins)
    fitted = fit_all_deltas(store, "gap")
    z_test, targets = _test_features(store, "gap")
    law = run_group_law_battery(fitted["ops"], z_test, {k: v for k, v in targets.items() if not k.endswith("__src")})
    try:
        head = extract_head(model)
        head_rep = evaluate_head_battery(head, z_test, fitted["ops"], ALL_DELTAS, already_px=False)
    except Exception as exc:
        head_rep = {"error": str(exc)}
    controls = _controls(store, fitted["ops"], "gap")
    dump_json(out / "operators.json", fitted["packed"])
    dump_json(out / "group_law.json", law)
    dump_json(out / "head_semantics.json", head_rep)
    dump_json(out / "controls.json", controls)
    _plot_e_delta(fitted["packed"], out / "figures" / "e_delta.png")
    if fitted["ops"]:
        np.savez_compressed(
            out / "operators_matrices.npz",
            **{f"{k}_M": v["matrix"] for k, v in fitted["ops"].items()},
            **{f"{k}_b": v["bias"] for k, v in fitted["ops"].items()},
        )
    flag = _high_upside_flag(fitted["packed"], law, head_rep, controls)
    summary = {
        "model": ref.name,
        "arch": ref.arch,
        "seed": ref.seed,
        "n_ops": len(fitted["ops"]),
        "mean_e_test": float(np.mean([v["e_test"] for v in fitted["packed"].values() if "e_test" in v])) if fitted["ops"] else None,
        "group_law_summary": law.get("summary"),
        "head_mean_shift_mae_px": head_rep.get("mean_shift_mae_px"),
        "high_upside_flag": flag,
    }
    dump_json(out / "summary.json", summary)
    return summary


def _high_upside_flag(packed, law, head_rep, controls) -> Optional[str]:
    e_vals = [v["e_test"] for v in packed.values() if "e_test" in v]
    if not e_vals:
        return None
    mean_e = float(np.mean(e_vals))
    shuf = [c["shuffled_e_test"] for c in controls.values()]
    law_s = law.get("summary") or {}
    head_mae = head_rep.get("mean_shift_mae_px")
    if (
        mean_e < 0.35
        and shuf
        and mean_e < 0.6 * float(np.mean(shuf))
        and (law_s.get("mean_e_composed") is not None and law_s["mean_e_composed"] < 0.45)
        and (law_s.get("mean_e_roundtrip") is not None and law_s["mean_e_roundtrip"] < 0.45)
        and (head_mae is not None and head_mae < 8.0)
    ):
        return "HIGH_UPSIDE: emergent approximate translation-group action"
    return None


def run_track_a() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    load_protocol()
    loc = dump_locator()
    origins = operator_origins_px()
    o_split = origin_split(len(origins))
    dump_json(OUT / "splits.json", {"shape_split": shape_split(), "origin_split": o_split, "n_origins": int(len(origins))})
    refs = primary_trained()
    secondaries = [r for r in locate_models() if r.role == "secondary" and r.found]
    rows = []
    for ref in refs + secondaries:
        try:
            rows.append(run_one_model(ref, origins))
        except Exception as exc:
            tb = traceback.format_exc()
            record_failure(f"track_a_{ref.name}", str(exc), {"tb": tb[-3000:]})
            rows.append({"model": ref.name, "ok": False, "error": str(exc)})
    try:
        model, _tag = build_random_twin("resnet18", seed=SEED_RANDOM_INIT)
        fake = ModelRef(
            name="random_init_resnet18",
            path="",
            arch="resnet18",
            variant="standard",
            seed=SEED_RANDOM_INIT,
            kind="phase18",
            role="control",
            found=True,
            notes="fresh random init",
        )
        # bypass load_ref
        out = OUT / fake.name
        out.mkdir(parents=True, exist_ok=True)
        store = extract_operator_features(model, origins)
        fitted = fit_all_deltas(store, "gap")
        z_test, targets = _test_features(store, "gap")
        law = run_group_law_battery(fitted["ops"], z_test, {k: v for k, v in targets.items() if not k.endswith("__src")})
        dump_json(out / "operators.json", fitted["packed"])
        dump_json(out / "group_law.json", law)
        rows.append({"model": fake.name, "mean_e_test": float(np.mean([v["e_test"] for v in fitted["packed"].values() if "e_test" in v])) if fitted["ops"] else None, "control": True})
    except Exception as exc:
        record_failure("track_a_random_init", str(exc), {"tb": traceback.format_exc()[-2000:]})
        rows.append({"model": "random_init_resnet18", "ok": False, "error": str(exc)})
    payload = {"locator": loc, "rows": rows}
    dump_json(OUT / "panel.json", payload)
    return payload
