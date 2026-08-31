"""Frozen Phase 3 protocol. Do not change splits, deltas, or hashes on the cloud."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase1_gap_rep.common import REPO_ROOT, environment_versions, fingerprint, profile_spec

from . import RENDERER_VERSION
from .io_util import dump_json, file_sha256, load_json, sha_payload

GEOMETRY_FP = "195f37870e2cdb8685a7"
PHASE2_PROTOCOL_HASH = "1b0f4c56d5c234a4afe2"

PHASE3_ROOT = REPO_ROOT / "results" / "phase3_high_upside_discovery"
CONFIG_DIR = PHASE3_ROOT / "config"
PROTOCOL_PATH = CONFIG_DIR / "frozen_protocol.json"
FAILURES_DIR = PHASE3_ROOT / "failures"
PROGRESS_PATH = PHASE3_ROOT / "progress.json"

T_LOW_PX = 59.0
T_HIGH_PX = 164.0
N_DENSE_A = 33
IMAGE_SIZE_A = 224
IMAGE_SIZE_B = 96
IMAGE_SIZE_C = 224

# Phase 2 farthest-point 32-shape eval set. Frozen.
EVAL_SHAPE_IDS: Tuple[int, ...] = (
    1, 2, 4, 7, 9, 10, 12, 13, 14, 15, 21, 22, 32, 33, 34, 35,
    40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 52, 55, 57, 60, 62, 63,
)

SEED_SHAPE_SPLIT = 20260831
SEED_ORIGIN_SPLIT = 20260832
SEED_TRACK_B = 20260833
SEED_TRACK_C = 20260830
SEED_TRACK_E = 20260820
SEED_TRACK_F = 20260834
SEED_RANDOM_INIT = 20260835

PHASE18_SEEDS: Tuple[int, ...] = (20260810, 20260811, 20260812)

# Affine operator shifts in pixels.
DELTA_X: Tuple[Tuple[float, float], ...] = (
    (-0.5, 0.0), (0.5, 0.0), (-1.0, 0.0), (1.0, 0.0),
    (-2.0, 0.0), (2.0, 0.0), (-4.0, 0.0), (4.0, 0.0),
    (-8.0, 0.0), (8.0, 0.0),
)
DELTA_Y: Tuple[Tuple[float, float], ...] = (
    (0.0, -0.5), (0.0, 0.5), (0.0, -1.0), (0.0, 1.0),
    (0.0, -2.0), (0.0, 2.0), (0.0, -4.0), (0.0, 4.0),
    (0.0, -8.0), (0.0, 8.0),
)
DELTA_DIAG: Tuple[Tuple[float, float], ...] = (
    (1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (4.0, -4.0),
)
ALL_DELTAS: Tuple[Tuple[float, float], ...] = DELTA_X + DELTA_Y + DELTA_DIAG

RIDGE_ALPHAS: Tuple[float, ...] = tuple(float(x) for x in np.logspace(-4, 4, 9))
REDUCED_RANKS: Tuple[Any, ...] = (8, 16, 32, 64, 128, 256, "full")

OPERATOR_ORIGIN_STRIDE = 3
ORIGIN_FIT_FRAC = 0.70
ORIGIN_VAL_FRAC = 0.15

PRIMITIVE_FAMILIES: Tuple[str, ...] = (
    "line", "circle", "arc", "quadratic", "polygon", "blob",
)
TRACK_D_SIZES: Tuple[int, ...] = (160, 192, 224, 256, 320)
TRACK_E_ARCH: Tuple[str, ...] = ("resnet18", "densenet121", "efficientnet_b0")
TRACK_E_MODES: Tuple[str, ...] = ("R0", "R1", "R2", "R3")
TRACK_E_REGIMES: Tuple[str, ...] = ("G64", "G9")

AMP_ALLOWED_ARCH: Tuple[str, ...] = ("resnet18", "densenet121")
N_STEPS_REF = 5600
BATCH_SIZE = 64
L1_WEIGHT = 0.25
LR = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4

TRACK_B_VARIANTS: Tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
TRACK_B_REGIMES: Tuple[str, ...] = ("dense", "sparse")

TIME_BUDGET_SEC = 20 * 3600
MIN_DISK_GB = 30.0


def dense_grid_a_px() -> np.ndarray:
    xs = np.linspace(T_LOW_PX, T_HIGH_PX, N_DENSE_A)
    ys = np.linspace(T_LOW_PX, T_HIGH_PX, N_DENSE_A)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


def operator_origin_indices() -> np.ndarray:
    idx = np.arange(0, N_DENSE_A, OPERATOR_ORIGIN_STRIDE, dtype=int)
    return idx


def operator_origins_px() -> np.ndarray:
    grid = dense_grid_a_px().reshape(N_DENSE_A, N_DENSE_A, 2)
    idx = operator_origin_indices()
    return grid[np.ix_(idx, idx)].reshape(-1, 2)


def shape_split(seed: int = SEED_SHAPE_SPLIT) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(EVAL_SHAPE_IDS)).astype(int)
    ids = np.asarray(EVAL_SHAPE_IDS, dtype=int)
    ordered = ids[perm]
    return {
        "fit": [int(x) for x in ordered[:16]],
        "val": [int(x) for x in ordered[16:24]],
        "test": [int(x) for x in ordered[24:32]],
    }


def origin_split(n_origin: int, seed: int = SEED_ORIGIN_SPLIT) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_origin).astype(int).tolist()
    n_fit = int(round(n_origin * ORIGIN_FIT_FRAC))
    n_val = int(round(n_origin * ORIGIN_VAL_FRAC))
    n_fit = min(max(n_fit, 1), n_origin - 2) if n_origin >= 3 else n_origin
    n_val = min(max(n_val, 1), n_origin - n_fit - 1) if n_origin - n_fit >= 2 else max(0, n_origin - n_fit)
    return {
        "fit": [int(x) for x in perm[:n_fit]],
        "val": [int(x) for x in perm[n_fit : n_fit + n_val]],
        "test": [int(x) for x in perm[n_fit + n_val :]],
    }


def _canon_delta_comp(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-12 else value


def delta_key(delta: Sequence[float]) -> str:
    dx, dy = _canon_delta_comp(delta[0]), _canon_delta_comp(delta[1])
    return f"{dx:g}_{dy:g}"


def parse_delta_key(key: str) -> Tuple[float, float]:
    parts = key.split("_")
    return float(parts[0]), float(parts[1])


def in_safe_box(t_px: np.ndarray, margin: float = 0.0) -> np.ndarray:
    t = np.asarray(t_px, dtype=np.float64)
    if t.ndim == 1:
        t = t[None, :]
    ok = (
        (t[:, 0] >= T_LOW_PX + margin)
        & (t[:, 0] <= T_HIGH_PX - margin)
        & (t[:, 1] >= T_LOW_PX + margin)
        & (t[:, 1] <= T_HIGH_PX - margin)
    )
    return ok


def build_protocol() -> Dict[str, Any]:
    geo = fingerprint(profile_spec("factorial"))
    if geo != GEOMETRY_FP:
        raise RuntimeError(f"geometry fingerprint {geo} != {GEOMETRY_FP}")
    split = shape_split()
    origins = operator_origins_px()
    o_split = origin_split(len(origins))
    core = {
        "schema": 1,
        "kind": "phase3_high_upside_discovery",
        "geometry_fingerprint": geo,
        "phase2_protocol_hash": PHASE2_PROTOCOL_HASH,
        "renderer_version": RENDERER_VERSION,
        "t_low_px": T_LOW_PX,
        "t_high_px": T_HIGH_PX,
        "n_dense_a": N_DENSE_A,
        "eval_shape_ids": list(EVAL_SHAPE_IDS),
        "shape_split": split,
        "origin_split": o_split,
        "n_operator_origins": int(len(origins)),
        "operator_origin_stride": OPERATOR_ORIGIN_STRIDE,
        "deltas": [list(d) for d in ALL_DELTAS],
        "ridge_alphas": list(RIDGE_ALPHAS),
        "reduced_ranks": list(REDUCED_RANKS),
        "seeds": {
            "shape_split": SEED_SHAPE_SPLIT,
            "origin_split": SEED_ORIGIN_SPLIT,
            "track_b": SEED_TRACK_B,
            "track_c": SEED_TRACK_C,
            "track_e": SEED_TRACK_E,
            "track_f": SEED_TRACK_F,
            "random_init": SEED_RANDOM_INIT,
            "phase18": list(PHASE18_SEEDS),
        },
        "primitive_families": list(PRIMITIVE_FAMILIES),
        "track_d_sizes": list(TRACK_D_SIZES),
        "track_e_arch": list(TRACK_E_ARCH),
        "track_e_modes": list(TRACK_E_MODES),
        "track_e_regimes": list(TRACK_E_REGIMES),
        "amp_allowed_arch": list(AMP_ALLOWED_ARCH),
        "n_steps_ref": N_STEPS_REF,
        "resume_default": False,
        "note": "Do not change splits/deltas. Do not treat Phase2 circular-ResNet as Track B S0.",
    }
    digest = sha_payload(core)
    return {**core, "protocol_sha256": digest, "protocol_hash": digest[:20]}


def freeze_protocol(force: bool = False) -> Dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists() and not force:
        return load_json(PROTOCOL_PATH)
    payload = build_protocol()
    dump_json(PROTOCOL_PATH, payload)
    return payload


def load_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        return freeze_protocol()
    return load_json(PROTOCOL_PATH)


def package_code_files() -> List[Path]:
    root = Path(__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if p.is_file())
    files += sorted(p for p in root.rglob("*.md") if p.is_file())
    return files


def code_hashes() -> Dict[str, Dict[str, Any]]:
    root = Path(__file__).resolve().parent
    out: Dict[str, Dict[str, Any]] = {}
    for path in package_code_files():
        rel = path.relative_to(root).as_posix()
        out[rel] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return out


def git_snapshot() -> Dict[str, Any]:
    info: Dict[str, Any] = {"head": None, "dirty": None, "error": None}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True)
        info["head"] = head
        info["dirty"] = bool(dirty.strip())
    except Exception as exc:
        info["error"] = str(exc)
    return info


def gpu_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        snap["cuda_available"] = bool(torch.cuda.is_available())
        snap["torch"] = torch.__version__
        if torch.cuda.is_available():
            snap["device_name"] = torch.cuda.get_device_name(0)
            snap["device_count"] = int(torch.cuda.device_count())
            snap["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        snap["error"] = str(exc)
    return snap


def record_preflight() -> Dict[str, Any]:
    proto = freeze_protocol(force=False)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "environment": environment_versions(),
        "gpu": gpu_snapshot(),
        "git": git_snapshot(),
        "protocol_hash": proto["protocol_hash"],
        "geometry_fingerprint": proto["geometry_fingerprint"],
        "renderer_version": RENDERER_VERSION,
        "code_hashes": code_hashes(),
        "resume_default": False,
    }
    dump_json(CONFIG_DIR / "preflight.json", payload)
    return payload


def record_failure(name: str, error: str, extra: Mapping[str, Any] | None = None) -> None:
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:120]
    payload: Dict[str, Any] = {
        "name": name,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(dict(extra))
    dump_json(FAILURES_DIR / f"{safe}.json", payload)


def summarize_failures() -> List[Dict[str, Any]]:
    if not FAILURES_DIR.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(FAILURES_DIR.glob("*.json")):
        try:
            blob = load_json(path)
        except Exception:
            continue
        rows.append({"name": blob.get("name", path.stem), "error": blob.get("error", "")})
    return rows
