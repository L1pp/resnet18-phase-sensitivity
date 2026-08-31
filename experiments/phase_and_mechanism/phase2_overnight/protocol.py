"""Frozen Phase 2 protocol: grids, regimes, shape splits, hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase1_gap_rep.common import (
    COORD_SCALE,
    REPO_ROOT,
    dump_json,
    fingerprint,
    profile_spec,
)
from phase1_gap_rep.generate_data import build_shapes, translation_grid

GEOMETRY_FP = "195f37870e2cdb8685a7"
PHASE2_ROOT = REPO_ROOT / "results" / "phase2_overnight_discovery"
CONFIG_DIR = PHASE2_ROOT / "config"
PROTOCOL_PATH = CONFIG_DIR / "frozen_protocol.json"

SEED_EVAL_SHAPES = 20260810
SEED_ATLAS = 20260820
SEED_TRAJECTORY = 20260813
SEED_H_SPLIT = 20260820
PHASE18_SEEDS = (20260810, 20260811, 20260812)

N_TX = 8
N_TY = 8
N_DENSE = 41
N_EVAL_SHAPES = 32
N_TRAJ_SHAPES = 16
VAL_FRAC = 0.125

T_LOW_PX = 59.0
T_HIGH_PX = 164.0

ARCH_ATLAS = (
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "efficientnet_b0",
    "mobilenet_v3_large",
)

TRAJECTORY_EPOCHS = (1, 2, 5, 10, 20, 40, 60, 80, 100)


def tid_from_xy(tx: int, ty: int) -> int:
    return int(tx) * N_TY + int(ty)


def xy_from_tid(tid: int) -> Tuple[int, int]:
    return int(tid) // N_TY, int(tid) % N_TY


def _tids_from_levels(levels: Sequence[int]) -> List[int]:
    return [tid_from_xy(tx, ty) for tx in levels for ty in levels]


def regime_tids() -> Dict[str, List[int]]:
    all_tids = list(range(N_TX * N_TY))
    checker = [tid_from_xy(tx, ty) for tx in range(N_TX) for ty in range(N_TY) if (tx + ty) % 2 == 0]
    corners = [tid_from_xy(0, 0), tid_from_xy(0, 7), tid_from_xy(7, 0), tid_from_xy(7, 7)]
    g2x = [tid_from_xy(0, 3), tid_from_xy(7, 3)]
    g2y = [tid_from_xy(3, 0), tid_from_xy(3, 7)]
    left = [tid_from_xy(tx, ty) for tx in range(4) for ty in range(N_TY)]
    return {
        "G64": all_tids,
        "Full64": all_tids,
        "G32": sorted(checker),
        "G16": _tids_from_levels([0, 2, 5, 7]),
        "G9": _tids_from_levels([0, 3, 7]),
        "Sparse9": _tids_from_levels([0, 3, 7]),
        "G4": corners,
        "Sparse4": corners,
        "C16": _tids_from_levels([2, 3, 4, 5]),
        "C9": _tids_from_levels([2, 3, 4]),
        "Central9": _tids_from_levels([2, 3, 4]),
        "C4": _tids_from_levels([3, 4]),
        "G2x": g2x,
        "G2y": g2y,
        "LeftHalf": left,
    }


def integer_grid_px() -> np.ndarray:
    xs = np.linspace(T_LOW_PX, T_HIGH_PX, N_TX)
    ys = np.linspace(T_LOW_PX, T_HIGH_PX, N_TY)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


def dense_grid_px() -> np.ndarray:
    xs = np.linspace(T_LOW_PX, T_HIGH_PX, N_DENSE)
    ys = np.linspace(T_LOW_PX, T_HIGH_PX, N_DENSE)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


def dense_xy_index(flat: int) -> Tuple[int, int]:
    return int(flat) // N_DENSE, int(flat) % N_DENSE


def farthest_point_shape_ids(n_keep: int, seed: int = SEED_EVAL_SHAPES) -> List[int]:
    spec = profile_spec("factorial")
    _shapes, records = build_shapes(spec)
    feats = np.array(
        [
            [
                rec["theta_deg"] / 135.0,
                (rec["chord_px"] - 48.0) / (90.0 - 48.0),
                (rec["curve_sign"] + 1.0) / 2.0,
                (rec["alpha"] + 0.40) / 0.80,
            ]
            for rec in records
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, feats.shape[0]))
    chosen = [start]
    dist = np.linalg.norm(feats - feats[start], axis=1)
    dist[start] = -1.0
    while len(chosen) < n_keep:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
        for idx in chosen:
            dist[idx] = -1.0
    return sorted(chosen)


def shape_split(seed: int = SEED_H_SPLIT) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(64).astype(int).tolist()
    return {"train": sorted(perm[:48]), "val": sorted(perm[48:56]), "test": sorted(perm[56:64])}


def pair_split(pairs: Sequence[Tuple[int, int]], seed: int, val_frac: float = VAL_FRAC) -> Dict[str, List[List[int]]]:
    grouped: Dict[int, List[Tuple[int, int]]] = {}
    for sid, tid in pairs:
        grouped.setdefault(int(tid), []).append((int(sid), int(tid)))
    rng = np.random.default_rng(seed)
    train: List[List[int]] = []
    val: List[List[int]] = []
    for tid in sorted(grouped):
        items = list(grouped[tid])
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) >= 8 else max(1, len(items) // 8 or (1 if len(items) > 1 else 0))
        if n_val >= len(items):
            n_val = max(0, len(items) - 1)
        val.extend([list(p) for p in items[:n_val]])
        train.extend([list(p) for p in items[n_val:]])
    return {"train": train, "val": val}


def sha_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def build_protocol() -> Dict[str, Any]:
    spec = profile_spec("factorial")
    geo = fingerprint(spec)
    if geo != GEOMETRY_FP:
        raise RuntimeError(f"geometry fingerprint {geo} != {GEOMETRY_FP}")
    shapes_px, records = build_shapes(spec)
    grid, low, high, q_min, q_max = translation_grid(shapes_px, spec)
    if abs(float(low[0]) - T_LOW_PX) > 1e-6 or abs(float(high[0]) - T_HIGH_PX) > 1e-6:
        raise RuntimeError(f"safe box drifted: low={low} high={high}")
    eval_shapes = farthest_point_shape_ids(N_EVAL_SHAPES, SEED_EVAL_SHAPES)
    traj_shapes = eval_shapes[:N_TRAJ_SHAPES]
    dense = dense_grid_px()
    integer = integer_grid_px()
    if np.max(np.abs(integer - grid)) > 1e-8:
        raise RuntimeError("integer grid does not match generate_data.translation_grid")
    regimes = {name: [int(x) for x in tids] for name, tids in regime_tids().items()}
    hsplit = shape_split(SEED_H_SPLIT)
    core = {
        "schema": 1,
        "kind": "phase2_overnight_discovery",
        "geometry_fingerprint": geo,
        "t_low_px": [T_LOW_PX, T_LOW_PX],
        "t_high_px": [T_HIGH_PX, T_HIGH_PX],
        "n_dense": N_DENSE,
        "eval_shape_ids": eval_shapes,
        "traj_shape_ids": traj_shapes,
        "regimes": regimes,
        "shape_split_h": hsplit,
        "seeds": {
            "eval_shapes": SEED_EVAL_SHAPES,
            "atlas": SEED_ATLAS,
            "trajectory": SEED_TRAJECTORY,
            "h_split": SEED_H_SPLIT,
            "phase18": list(PHASE18_SEEDS),
        },
        "architectures": list(ARCH_ATLAS),
        "atlas_regimes": ["G64", "G9", "C9"],
        "atlas_grid": [
            {"arch": arch, "regime": regime}
            for arch in ARCH_ATLAS
            for regime in ("G64", "G9", "C9")
        ],
        "atlas_reuse": {
            "resnet18/G64": "Full64",
            "resnet18/G9": "Sparse9",
            "resnet18/C9": "Central9",
            "resnet18/G4": "Sparse4_insanity_required_by_D",
        },
        "content_primitives": ["quadratic", "line", "arc"],
        "val_frac": VAL_FRAC,
        "note": "A/B local on phase1.8 slim; cloud trains C–H then atlas. No ImageNet normalize.",
    }
    digest = sha_payload(core)
    payload = {
        **core,
        "protocol_sha256": digest,
        "protocol_hash": digest[:20],
        "dense_t_px": dense.tolist(),
        "integer_t_px": integer.tolist(),
        "q_min_px": q_min.astype(float).tolist(),
        "q_max_px": q_max.astype(float).tolist(),
        "shape_records": records,
        "aliases": {"Full64": "G64", "Sparse9": "G9", "Central9": "C9", "Sparse4": "G4"},
    }
    return payload


def freeze_protocol(force: bool = False) -> Dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists() and not force:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        print(f"[phase2] protocol already frozen hash={payload['protocol_hash']}")
        return payload
    payload = build_protocol()
    dump_json(PROTOCOL_PATH, payload)
    print(f"[phase2] froze protocol hash={payload['protocol_hash']} path={PROTOCOL_PATH}")
    return payload


def load_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        return freeze_protocol()
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def canonical_regime(name: str) -> str:
    aliases = {"Full64": "G64", "Sparse9": "G9", "Central9": "C9", "Sparse4": "G4"}
    return aliases.get(name, name)


def record_failure(name: str, error: str, extra: Mapping[str, Any] | None = None) -> None:
    fail_dir = PHASE2_ROOT / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:120]
    payload: Dict[str, Any] = {"name": name, "error": error}
    if extra:
        payload.update(dict(extra))
    dump_json(fail_dir / f"{safe}.json", payload)


def summarize_failures() -> List[Dict[str, Any]]:
    fail_dir = PHASE2_ROOT / "failures"
    if not fail_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(fail_dir.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row: Dict[str, Any] = {"name": blob.get("name", path.stem), "error": blob.get("error", "")}
        if blob.get("tb"):
            row["tb"] = blob["tb"][-2000:]
        rows.append(row)
    return rows


if __name__ == "__main__":
    freeze_protocol(force=True)
