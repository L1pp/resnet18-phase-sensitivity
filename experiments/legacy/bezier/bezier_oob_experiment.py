"""Bezier OOB / in-frame data prep, preview, and training.

Profiles under ``results/bezier_oob/{quadratic,cubic}``:

- Quadratic: forced OOB:inframe = 2:1. Endpoints >=5 px inside; OOB controls
  leave [0,1] but stay within 100 px. Labels: left-to-right endpoints only.
- Cubic: all controls in-frame. Endpoints left-to-right; interior P1/P2 ordered
  by (x, y) lexicographic (P1 strictly before P2; coincident controls rejected).

Training: AdamW + MSE+0.25*L1. Fresh runs use fixed or scheduled LR as configured.
Resume (epoch 40 -> 100): CosineAnnealingLR restart 0.01 -> 0.001 over 60 epochs.
Sample counts are not part of the geometry fingerprint (ckpt reusable after expand).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 20260810
IMAGE_SIZE = 224
SUPER_SAMPLE = 4
CURVE_SAMPLES = 256
STROKE_WIDTH = 3.0
COORD_SCALE = float(IMAGE_SIZE - 1)
ENDPOINT_MARGIN_PX = 5.0
MAX_OUT_PX = 100.0  # quadratic forced-OOB protrusion cap
ENDPOINT_MARGIN = ENDPOINT_MARGIN_PX / COORD_SCALE
MAX_OUT = MAX_OUT_PX / COORD_SCALE
MIN_X_GAP = 0.20
MIN_CHORD = 0.08
MIN_FOREGROUND = 12
OOB_RATIO = 2  # quadratic only: OOB:inframe = 2:1
INFRAME_RATIO = 1
MAX_EPOCHS = 100
RESUME_FROM_EPOCH = 40
RESUME_EPOCHS = 60  # cosine window for resume: 0.01 -> 0.001
BATCH_SIZE = 64
LR = 0.01
LR_MIN = 0.001
L1_WEIGHT = 0.25
AUG_SHIFT_PX = 4
EARLY_STOP_MAE_PX = 1.0
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class Profile:
    name: str
    kind: str  # "quadratic" | "cubic"
    train_count: int
    val_count: int
    test_count: int


PROFILES: Dict[str, Profile] = {
    "quadratic": Profile("quadratic", "quadratic", 10000, 1000, 1000),
    "cubic": Profile("cubic", "cubic", 10000, 1000, 1000),
}

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "bezier_oob"
SPLITS = ("train", "val", "test")


def profile_spec(profile: str | Profile) -> Profile:
    if isinstance(profile, Profile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile!r}; choose {sorted(PROFILES)}") from exc


def dirs(profile: str | Profile) -> Dict[str, Path]:
    root = RESULTS_ROOT / profile_spec(profile).name
    return {name: root / name for name in ("config", "manifest", "data", "figures", "tables", "checkpoints")} | {"root": root}


def coordinate_dim(profile: str | Profile) -> int:
    return 6 if profile_spec(profile).kind == "quadratic" else 8


def target_names(profile: str | Profile) -> Tuple[str, ...]:
    if profile_spec(profile).kind == "quadratic":
        return ("p0x", "p0y", "p1x", "p1y", "p2x", "p2y")
    return ("p0x", "p0y", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y")


def config(profile: str | Profile) -> Dict[str, Any]:
    spec = profile_spec(profile)
    if spec.kind == "quadratic":
        mix = {"oob": OOB_RATIO, "inframe": INFRAME_RATIO, "mode": "forced_2to1"}
        max_out_px = MAX_OUT_PX
        control_support = [-MAX_OUT, 1.0 + MAX_OUT]
    else:
        mix = {
            "mode": "all_controls_inframe",
            "note": "endpoints and P1/P2 all inside image; no oob controls",
            "controls_ordered_by_xy": True,
        }
        max_out_px = 0.0
        control_support = [0.0, 1.0]
    return {
        "schema": 1,
        "profile": asdict(spec),
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "supersample": SUPER_SAMPLE,
        "resampling": "LANCZOS",
        "curve_samples": CURVE_SAMPLES,
        "stroke_width": STROKE_WIDTH,
        "targets": list(target_names(spec)),
        "mix": mix,
        "geometry": {
            "endpoint_margin_px": ENDPOINT_MARGIN_PX,
            "endpoint_range": [ENDPOINT_MARGIN, 1.0 - ENDPOINT_MARGIN],
            "max_control_out_px": max_out_px,
            "control_support": control_support,
            "min_x_gap": MIN_X_GAP,
            "min_chord": MIN_CHORD,
            "min_foreground_pixels": MIN_FOREGROUND,
            "cubic_controls_ordered_by_xy": spec.kind == "cubic",
        },
        "label_policy": {
            "storage": "raw_normalized_canvas_coords_may_leave_01",
            "train_normalization": "zero_mean_unit_variance_from_train_split",
            "no_sigmoid_01_clamp": True,
            "cubic_p1_p2_lex_xy": spec.kind == "cubic",
        },
        "augmentation_planned": {
            "translate_px": 4,
            "keep_endpoints_visible": True,
            "controls_may_enter_frame_after_shift": True,
        },
        "training_planned": {
            "loss": f"MSE + {L1_WEIGHT} * L1",
            "optimizer": "AdamW",
            "lr": LR,
            "lr_min": LR_MIN,
            "lr_schedule": f"resume: CosineAnnealingLR(T_max={RESUME_EPOCHS}, eta_min={LR_MIN}) restart from {LR}",
            "resume_from_epoch": RESUME_FROM_EPOCH,
            "early_stop_val_mae_px": EARLY_STOP_MAE_PX,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "aug_translate_px": AUG_SHIFT_PX,
            "norm_stats": "frozen_on_expand",
        },
    }


def geometry_fingerprint(profile: str | Profile) -> str:
    """Hash geometry/task identity only; sample counts do not affect the hash."""
    cfg = config(profile)
    payload = {k: cfg[k] for k in cfg if k not in {"training_planned", "augmentation_planned", "label_policy"}}
    prof = dict(payload.get("profile") or {})
    for key in ("train_count", "val_count", "test_count"):
        prof.pop(key, None)
    payload["profile"] = prof
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def fingerprint(profile: str | Profile) -> str:
    """Alias for geometry fingerprint (counts are intentionally excluded)."""
    return geometry_fingerprint(profile)


def data_fingerprint(profile: str | Profile) -> str:
    """Fingerprint frozen in prepare artifacts; fall back to geometry hash."""
    path = dirs(profile)["manifest"] / "manifest.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("geometry_fingerprint") or payload.get("fingerprint") or fingerprint(profile))
    return fingerprint(profile)


def _ckpt_compatible(ckpt: Mapping[str, Any], profile: str | Profile) -> bool:
    """True if checkpoint can be used with current geometry (counts may differ)."""
    spec = profile_spec(profile)
    if ckpt.get("profile") not in (None, spec.name):
        return False
    geo = geometry_fingerprint(spec)
    if ckpt.get("geometry_fingerprint") == geo:
        return True
    if ckpt.get("fingerprint") == geo:
        return True
    # Legacy count-coupled fingerprints: allow when profile matches and weights load.
    return "model_state" in ckpt


def _split_seed(profile: str | Profile, split: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{profile_spec(profile).name}:{split}".encode()).digest()[:8], "little") % (2**32 - 1)


def curve_points(kind: str, parameters: Sequence[float], count: int = CURVE_SAMPLES) -> np.ndarray:
    if kind == "quadratic":
        p = np.asarray(parameters, dtype=np.float64).reshape(3, 2)
        t = np.linspace(0.0, 1.0, int(count), dtype=np.float64)[:, None]
        return (1.0 - t) ** 2 * p[0] + 2.0 * (1.0 - t) * t * p[1] + t**2 * p[2]
    p = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    t = np.linspace(0.0, 1.0, int(count), dtype=np.float64)[:, None]
    omt = 1.0 - t
    return omt**3 * p[0] + 3.0 * omt**2 * t * p[1] + 3.0 * omt * t**2 * p[2] + t**3 * p[3]


def chord_distance(kind: str, parameters: Sequence[float]) -> float:
    pts = np.asarray(parameters, dtype=np.float64).reshape((-1, 2))
    if kind == "quadratic":
        a, b, c = pts
        chord = c - a
        length = float(np.linalg.norm(chord))
        if length <= 1e-12:
            return 0.0
        offset = b - a
        return float(abs(chord[0] * offset[1] - chord[1] * offset[0]) / length)
    # cubic: mean distance of interior controls to endpoint chord
    a, b, c, d = pts
    chord = d - a
    length = float(np.linalg.norm(chord))
    if length <= 1e-12:
        return 0.0
    def dist(p: np.ndarray) -> float:
        offset = p - a
        return abs(chord[0] * offset[1] - chord[1] * offset[0]) / length
    return 0.5 * (dist(b) + dist(c))


def _control_xy_key(pt: np.ndarray) -> Tuple[float, float]:
    return float(pt[0]), float(pt[1])


def _cubic_controls_ordered(pts: np.ndarray) -> bool:
    """True iff interior P1 is strictly before P2 in (x, y) lexicographic order."""
    p1, p2 = pts[1], pts[2]
    return _control_xy_key(p1) < _control_xy_key(p2)


def canonicalize(kind: str, parameters: Sequence[float]) -> np.ndarray:
    pts = np.asarray(parameters, dtype=np.float32).reshape((-1, 2)).copy()
    if pts[0, 0] > pts[-1, 0]:
        pts = pts[::-1].copy()
    if kind == "cubic":
        # Fix P1/P2 label order by spatial (x, y); render must use this order.
        if _control_xy_key(pts[1]) > _control_xy_key(pts[2]):
            pts[1], pts[2] = pts[2].copy(), pts[1].copy()
    return pts.reshape(-1)


def render_curve(kind: str, parameters: Sequence[float]) -> np.ndarray:
    from PIL import Image, ImageDraw

    points = curve_points(kind, parameters, CURVE_SAMPLES)
    high_size = IMAGE_SIZE * SUPER_SAMPLE
    image = Image.new("L", (high_size, high_size), color=0)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * COORD_SCALE * SUPER_SAMPLE, float(y) * COORD_SCALE * SUPER_SAMPLE) for x, y in points]
    width = int(round(STROKE_WIDTH * SUPER_SAMPLE))
    try:
        draw.line(xy, fill=255, width=width, joint="curve")
    except TypeError:
        draw.line(xy, fill=255, width=width)
    resampling = getattr(Image, "Resampling", Image)
    out = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling.LANCZOS), dtype=np.uint8)
    if out.shape != (IMAGE_SIZE, IMAGE_SIZE) or out.dtype != np.uint8:
        raise RuntimeError(f"bad render: {out.shape} {out.dtype}")
    return out


def _endpoints_ok(pts: np.ndarray) -> bool:
    ends = pts[[0, -1]]
    lo, hi = ENDPOINT_MARGIN, 1.0 - ENDPOINT_MARGIN
    return bool(np.all((ends > lo) & (ends < hi)) and pts[0, 0] < pts[-1, 0] and (pts[-1, 0] - pts[0, 0]) >= MIN_X_GAP)


def _controls_in_support(controls: np.ndarray, kind: str) -> bool:
    if kind == "quadratic":
        lo, hi = -MAX_OUT, 1.0 + MAX_OUT
    else:
        lo, hi = 0.0, 1.0
    return bool(np.all((controls >= lo) & (controls <= hi)))


def _is_oob(controls: np.ndarray) -> bool:
    return bool(np.any(controls < 0.0) or np.any(controls > 1.0))


def _is_inframe(controls: np.ndarray) -> bool:
    return bool(np.all((controls >= 0.0) & (controls <= 1.0)))


def _sample_endpoints(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    lo, hi = ENDPOINT_MARGIN + 1e-3, 1.0 - ENDPOINT_MARGIN - 1e-3
    p0x, pNx = rng.uniform(lo, hi, size=2)
    p0y, pNy = rng.uniform(lo, hi, size=2)
    return float(p0x), float(p0y), float(pNx), float(pNy)


def _sample_control_inframe(rng: np.random.Generator) -> Tuple[float, float]:
    # keep a tiny inset so stroke antialias stays away from hard clip when possible
    return float(rng.uniform(0.02, 0.98)), float(rng.uniform(0.02, 0.98))


def _sample_control_oob(rng: np.random.Generator) -> Tuple[float, float]:
    """Sample one control with at least one coordinate outside [0,1], within MAX_OUT (quadratic)."""
    while True:
        out_x = bool(rng.integers(0, 2))
        out_y = bool(rng.integers(0, 2))
        if not out_x and not out_y:
            out_x = True
        if out_x:
            side = -1 if rng.random() < 0.5 else 1
            x = float(rng.uniform(-MAX_OUT, 0.0 - 1e-3)) if side < 0 else float(rng.uniform(1.0 + 1e-3, 1.0 + MAX_OUT))
        else:
            x = float(rng.uniform(0.0, 1.0))
        if out_y:
            side = -1 if rng.random() < 0.5 else 1
            y = float(rng.uniform(-MAX_OUT, 0.0 - 1e-3)) if side < 0 else float(rng.uniform(1.0 + 1e-3, 1.0 + MAX_OUT))
        else:
            y = float(rng.uniform(0.0, 1.0))
        if _is_oob(np.asarray([x, y])) and _controls_in_support(np.asarray([x, y]), "quadratic"):
            return x, y


def _accept_geometry(kind: str, params: np.ndarray, want_oob: bool | None) -> bool:
    pts = params.reshape((-1, 2))
    if not _endpoints_ok(pts):
        return False
    controls = pts[1:-1].reshape(-1)
    if not _controls_in_support(controls, kind):
        return False
    if kind == "cubic":
        if not _is_inframe(controls):
            return False
        # Require strict (x,y) order; coincident P1==P2 rejected.
        if not _cubic_controls_ordered(pts):
            return False
    if want_oob is True and not _is_oob(controls):
        return False
    if want_oob is False and not _is_inframe(controls):
        return False
    if chord_distance(kind, params) < MIN_CHORD:
        return False
    image = render_curve(kind, params)
    if int(np.count_nonzero(image >= 8)) < MIN_FOREGROUND:
        return False
    for end in (pts[0], pts[-1]):
        ex = int(np.clip(round(float(end[0]) * COORD_SCALE), 0, IMAGE_SIZE - 1))
        ey = int(np.clip(round(float(end[1]) * COORD_SCALE), 0, IMAGE_SIZE - 1))
        patch = image[max(0, ey - 2) : min(IMAGE_SIZE, ey + 3), max(0, ex - 2) : min(IMAGE_SIZE, ex + 3)]
        if int(np.count_nonzero(patch >= 8)) == 0:
            return False
    return True


def sample_parameters(kind: str, want_oob: bool | None, rng: np.random.Generator, max_tries: int = 20000) -> np.ndarray:
    for _ in range(max_tries):
        p0x, p0y, pNx, pNy = _sample_endpoints(rng)
        if kind == "quadratic":
            if want_oob:
                c1x, c1y = _sample_control_oob(rng)
            else:
                c1x, c1y = _sample_control_inframe(rng)
            cand = canonicalize(kind, (p0x, p0y, c1x, c1y, pNx, pNy))
        else:
            # cubic: all controls strictly in-frame
            c1x, c1y = _sample_control_inframe(rng)
            c2x, c2y = _sample_control_inframe(rng)
            cand = canonicalize(kind, (p0x, p0y, c1x, c1y, c2x, c2y, pNx, pNy))
        if _accept_geometry(kind, cand, want_oob if kind == "quadratic" else False):
            return cand.astype(np.float32)
    raise RuntimeError(f"failed to sample {kind} want_oob={want_oob}")


def _type_for_index(kind: str, index: int, controls: np.ndarray | None = None) -> str:
    if kind == "cubic":
        return "inframe"
    return "inframe" if (index % (OOB_RATIO + INFRAME_RATIO)) >= OOB_RATIO else "oob"


def _generate_split(profile: str | Profile, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], np.ndarray]:
    spec = profile_spec(profile)
    n = {"train": spec.train_count, "val": spec.val_count, "test": spec.test_count}[split]
    kind = spec.kind
    dim = coordinate_dim(spec)
    rng = np.random.default_rng(_split_seed(spec, split))
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    targets = np.empty((n, dim), dtype=np.float32)
    types = np.empty((n,), dtype="U16")
    ids = np.asarray([f"{spec.name}-{split}-{i:06d}" for i in range(n)], dtype="U64")
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        if kind == "quadratic":
            sample_type = _type_for_index(kind, i)
            want_oob: bool | None = sample_type == "oob"
        else:
            want_oob = False
            sample_type = "inframe"
        params = sample_parameters(kind, want_oob, rng)
        image = render_curve(kind, params)
        images[i] = image
        targets[i] = params
        pts = params.reshape((-1, 2))
        controls = pts[1:-1].reshape(-1)
        types[i] = sample_type
        rows.append(
            {
                "id": str(ids[i]),
                "split": split,
                "type": sample_type,
                "chord_distance": chord_distance(kind, params),
                "foreground_pixels": int(np.count_nonzero(image >= 8)),
                "control_out_max_px": float(np.max(np.maximum(0.0 - controls, controls - 1.0)) * COORD_SCALE) if controls.size else 0.0,
                "render_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "parameter_sha256": hashlib.sha256(params.tobytes()).hexdigest(),
            }
        )
        if (i + 1) % 250 == 0 or i + 1 == n:
            print(f"[prepare:{spec.name}:{split}] {i + 1}/{n}")
    return images, targets, ids, rows, types


def prepare(profile: str, force: bool = False, expand: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    d = dirs(spec)
    manifest_path = d["manifest"] / "manifest.json"
    geo_fp = geometry_fingerprint(spec)
    if manifest_path.exists() and not force and not expand:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts_ok = existing.get("counts") == {
            "train": spec.train_count,
            "val": spec.val_count,
            "test": spec.test_count,
        }
        fp_ok = existing.get("geometry_fingerprint", existing.get("fingerprint")) == geo_fp
        if fp_ok and counts_ok and all((d["data"] / f"{s}.npz").exists() for s in SPLITS):
            print(f"[prepare] cache ready: {d['root']}")
            return existing
        raise RuntimeError("existing manifest differs or cache incomplete; use --force or --expand")
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    (d["config"] / "config.json").write_text(
        json.dumps({**config(spec), "fingerprint": geo_fp, "geometry_fingerprint": geo_fp}, indent=2),
        encoding="utf-8",
    )
    entries: Dict[str, Any] = {}
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    type_counts: Dict[str, Dict[str, int]] = {}
    train_targets_for_norm: np.ndarray | None = None
    for split in SPLITS:
        images, targets, ids, rows, types = _generate_split(spec, split)
        npz = d["data"] / f"{split}.npz"
        np.savez_compressed(
            npz,
            images=images,
            targets=targets,
            ids=ids,
            types=types,
            fingerprint=np.asarray(geo_fp),
        )
        (d["manifest"] / f"{split}.json").write_text(json.dumps({"split": split, "records": rows}, indent=2), encoding="utf-8")
        hashes = [r["render_sha256"] for r in rows]
        if len(set(ids.tolist())) != len(ids) or len(set(hashes)) != len(hashes):
            raise RuntimeError(f"duplicate ID/raster in {split}")
        if all_ids.intersection(ids.tolist()) or all_hashes.intersection(hashes):
            raise RuntimeError(f"cross-split collision at {split}")
        all_ids.update(ids.tolist())
        all_hashes.update(hashes)
        tc = {"oob": int(np.sum(types == "oob")), "inframe": int(np.sum(types == "inframe"))}
        type_counts[split] = tc
        entries[split] = {
            "count": len(ids),
            "npz": str(npz.relative_to(d["root"])),
            "type_counts": tc,
            "image_shape": list(images.shape),
            "target_shape": list(targets.shape),
        }
        if split == "train":
            train_targets_for_norm = targets
        print(f"[prepare:{spec.name}:{split}] types={tc}")
    assert train_targets_for_norm is not None
    norm_path = d["config"] / "norm_stats.json"
    keep_norm = expand and norm_path.exists()
    if keep_norm:
        print(f"[prepare:{spec.name}] keeping existing norm_stats (expand/resume)")
        norm = json.loads(norm_path.read_text(encoding="utf-8"))
        if list(norm.get("target_names") or []) != list(target_names(spec)):
            raise RuntimeError("existing norm_stats target_names mismatch; refuse to expand")
    else:
        mean = train_targets_for_norm.mean(axis=0).astype(np.float64)
        std = train_targets_for_norm.std(axis=0).astype(np.float64)
        std = np.maximum(std, 1e-6)
        norm = {
            "target_names": list(target_names(spec)),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "note": "Apply (x-mean)/std for training targets and invert for metrics in px via *COORD_SCALE on raw coords.",
        }
        norm_path.write_text(json.dumps(norm, indent=2), encoding="utf-8")
    manifest = {
        "schema": 1,
        "profile": spec.name,
        "kind": spec.kind,
        "fingerprint": geo_fp,
        "geometry_fingerprint": geo_fp,
        "counts": {k: v["count"] for k, v in entries.items()},
        "type_counts": type_counts,
        "splits": entries,
        "norm_stats": "config/norm_stats.json",
        "norm_stats_frozen_on_expand": keep_norm,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    meta = {
        "stage": "prepare",
        "mode": "expand" if expand else "full",
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "fingerprint": geo_fp,
        "geometry_fingerprint": geo_fp,
        "config": config(spec),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "type_counts": type_counts,
        "norm_stats_frozen_on_expand": keep_norm,
    }
    (d["config"] / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[prepare] complete: {d['root']}")
    return manifest


def _load_norm_stats(profile: str | Profile) -> Tuple[np.ndarray, np.ndarray]:
    path = dirs(profile)["config"] / "norm_stats.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    return mean, std


def _load_split_arrays(profile: str | Profile, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = dirs(profile)["data"] / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare")
    expected = data_fingerprint(profile)
    with np.load(path, allow_pickle=False) as payload:
        cache_fp = str(np.asarray(payload["fingerprint"]).reshape(-1)[0])
        if cache_fp != expected:
            raise RuntimeError(f"fingerprint mismatch in {path}: {cache_fp} != {expected}")
        images = np.asarray(payload["images"])
        targets = np.asarray(payload["targets"], dtype=np.float32)
        ids = np.asarray(payload["ids"])
    return images, targets, ids


def _translate_uint8(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return image
    h, w = image.shape
    out = np.zeros_like(image)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(w, w - dx)
    src_y1 = min(h, h - dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


class OobDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Images + raw coords; optional small translate that keeps endpoints on-canvas."""

    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
        max_shift_px: int = AUG_SHIFT_PX,
    ) -> None:
        self.images = images
        self.targets = targets.astype(np.float32, copy=False)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)
        self.augment = augment
        self.max_shift_px = int(max_shift_px)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        target = self.targets[index].astype(np.float32, copy=True)
        if self.augment and self.max_shift_px > 0:
            pts = target.reshape((-1, 2))
            ends = pts[[0, -1]]
            # keep endpoints at least ~1 px inside after shift
            max_left = int(np.floor(float(ends[:, 0].min()) * COORD_SCALE) - 1)
            max_right = int(np.floor(float((1.0 - ends[:, 0].max()) * COORD_SCALE) - 1))
            max_up = int(np.floor(float(ends[:, 1].min()) * COORD_SCALE) - 1)
            max_down = int(np.floor(float((1.0 - ends[:, 1].max()) * COORD_SCALE) - 1))
            dx_lo, dx_hi = -min(self.max_shift_px, max(0, max_left)), min(self.max_shift_px, max(0, max_right))
            dy_lo, dy_hi = -min(self.max_shift_px, max(0, max_up)), min(self.max_shift_px, max(0, max_down))
            dx = random.randint(dx_lo, dx_hi) if dx_hi >= dx_lo else 0
            dy = random.randint(dy_lo, dy_hi) if dy_hi >= dy_lo else 0
            if dx or dy:
                image = _translate_uint8(image, dx, dy)
                target[0::2] += dx / COORD_SCALE
                target[1::2] += dy / COORD_SCALE
        normed = (target - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        return x, torch.from_numpy(normed).float()


def build_model(dim: int) -> nn.Module:
    from torchvision import models

    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, dim)
    return model


def _combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target) + L1_WEIGHT * nn.functional.l1_loss(pred, target)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_context(dev: torch.device, enabled: bool):
    from contextlib import nullcontext

    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16)


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_history_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def train(profile: str, force: bool = False, resume: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    d = dirs(spec)
    best_path = d["checkpoints"] / "resnet18_best.pt"
    last_path = d["checkpoints"] / "last.pt"
    history_path = d["tables"] / "history.csv"
    geo_fp = geometry_fingerprint(spec)
    if resume and force:
        raise RuntimeError("use either --resume or --force, not both")
    if best_path.exists() and not force and not resume:
        raise RuntimeError(f"checkpoint exists: {best_path}; use --force or --resume")
    mean, std = _load_norm_stats(spec)
    train_images, train_targets, _ = _load_split_arrays(spec, "train")
    val_images, val_targets, _ = _load_split_arrays(spec, "val")
    dim = coordinate_dim(spec)
    dev = _device()
    amp = dev.type == "cuda"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = build_model(dim).to(dev)
    start_epoch = 0
    history: List[Dict[str, Any]] = []
    best_val = math.inf
    best_mae = math.inf
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"missing {last_path} for --resume")
        ckpt = torch.load(last_path, map_location=dev, weights_only=False)
        if not _ckpt_compatible(ckpt, spec):
            raise RuntimeError("resume checkpoint incompatible with current geometry/profile")
        start_epoch = int(ckpt.get("epoch", -1))
        if start_epoch != RESUME_FROM_EPOCH:
            raise RuntimeError(f"expected last.pt epoch={RESUME_FROM_EPOCH}, got {start_epoch}")
        model.load_state_dict(ckpt["model_state"])
        history = _read_history_rows(history_path)
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            if "val_mae_px" in best_ckpt:
                best_mae = float(best_ckpt["val_mae_px"])
            # approximate best_val from history if possible
            for row in history:
                try:
                    vl = float(row["val_loss"])
                    if vl < best_val:
                        best_val = vl
                        best_mae = float(row["val_coordinate_mae_px"])
                except (KeyError, TypeError, ValueError):
                    pass
        print(
            f"[train:{spec.name}] resume from epoch {start_epoch}; "
            f"cosine restart lr={LR} -> {LR_MIN} over {RESUME_EPOCHS} epochs; "
            f"legacy_fp={ckpt.get('fingerprint')} geo_fp={geo_fp}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if resume:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RESUME_EPOCHS, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = DataLoader(
        OobDataset(train_images, train_targets, mean, std, augment=True, max_shift_px=AUG_SHIFT_PX),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED + start_epoch),
    )
    val_loader = DataLoader(
        OobDataset(val_images, val_targets, mean, std, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    d["checkpoints"].mkdir(parents=True, exist_ok=True)
    d["tables"].mkdir(parents=True, exist_ok=True)
    stopped_reason = "max_epochs"
    mean_t = torch.from_numpy(mean).to(dev)
    std_t = torch.from_numpy(std).to(dev)
    epoch = start_epoch
    for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(dev, amp):
                pred = model(x)
                loss = _combined_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_val = float(loss.detach())
            if not math.isfinite(loss_val):
                raise RuntimeError(f"non-finite train loss at epoch {epoch}")
            total += loss_val * len(x)
            count += len(x)
        model.eval()
        val_loss = 0.0
        preds_raw: List[np.ndarray] = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                with _amp_context(dev, False):
                    pred = model(x)
                    loss = _combined_loss(pred, y)
                val_loss += float(loss) * len(x)
                raw = pred * std_t + mean_t
                preds_raw.append(raw.cpu().numpy())
        val_loss /= max(1, len(val_targets))
        train_loss = total / max(1, count)
        pred_raw = np.concatenate(preds_raw, axis=0)
        val_mae_px = float(np.mean(np.abs(pred_raw - val_targets)) * COORD_SCALE)
        lr_used = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_coordinate_mae_px": val_mae_px,
            "lr_used": lr_used,
            "lr_next": lr_next,
        }
        history.append(row)
        print(
            f"[train:{spec.name}] epoch {epoch}/{MAX_EPOCHS} "
            f"train={train_loss:.6g} val={val_loss:.6g} mae={val_mae_px:.3f}px lr={lr_used:.3g}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_mae = val_mae_px
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "fingerprint": geo_fp,
                    "geometry_fingerprint": geo_fp,
                    "profile": spec.name,
                    "val_mae_px": val_mae_px,
                    "mean": mean,
                    "std": std,
                },
                best_path,
            )
        if val_mae_px <= EARLY_STOP_MAE_PX:
            stopped_reason = f"val_mae_px<={EARLY_STOP_MAE_PX}"
            print(f"[train:{spec.name}] early stop: reached {val_mae_px:.3f}px <= {EARLY_STOP_MAE_PX}px at epoch {epoch}")
            break
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "fingerprint": geo_fp,
            "geometry_fingerprint": geo_fp,
            "profile": spec.name,
            "mean": mean,
            "std": std,
        },
        last_path,
    )
    # de-duplicate history by epoch (keep last row per epoch) then write
    by_epoch: Dict[int, Dict[str, Any]] = {}
    for row in history:
        by_epoch[int(float(row["epoch"]))] = {
            "epoch": int(float(row["epoch"])),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "val_coordinate_mae_px": float(row["val_coordinate_mae_px"]),
            "lr_used": float(row["lr_used"]),
            "lr_next": float(row["lr_next"]),
        }
    merged = [by_epoch[k] for k in sorted(by_epoch)]
    _write_history(history_path, merged)
    summary = {
        "profile": spec.name,
        "best_val_loss": best_val if best_val < math.inf else None,
        "best_val_mae_px": best_mae if best_mae < math.inf else None,
        "final_epoch": epoch,
        "stopped_reason": stopped_reason,
        "history_epochs": len(merged),
        "resumed_from_epoch": start_epoch if resume else None,
        "geometry_fingerprint": geo_fp,
    }
    (d["tables"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    meta = {
        "stage": "train",
        "resume": resume,
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "fingerprint": geo_fp,
        "geometry_fingerprint": geo_fp,
        "summary": summary,
        "training": config(spec)["training_planned"],
    }
    (d["config"] / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def analyze(profile: str, force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    d = dirs(spec)
    out = d["tables"] / "metrics.json"
    if out.exists() and not force:
        raise RuntimeError(f"metrics exists: {out}; use --force")
    mean, std = _load_norm_stats(spec)
    images, targets, ids = _load_split_arrays(spec, "test")
    types_path = d["data"] / "test.npz"
    with np.load(types_path, allow_pickle=False) as payload:
        types = np.asarray(payload["types"]) if "types" in payload.files else np.array(["na"] * len(targets))
    ckpt = torch.load(d["checkpoints"] / "resnet18_best.pt", map_location=_device(), weights_only=False)
    if not _ckpt_compatible(ckpt, spec):
        raise RuntimeError("checkpoint incompatible with current geometry/profile")
    if ckpt.get("geometry_fingerprint") != geometry_fingerprint(spec):
        print(
            f"[analyze:{spec.name}] note: using legacy ckpt fingerprint={ckpt.get('fingerprint')} "
            f"with geometry_fingerprint={geometry_fingerprint(spec)}"
        )
    model = build_model(coordinate_dim(spec)).to(_device())
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    loader = DataLoader(OobDataset(images, targets, mean, std, augment=False), batch_size=BATCH_SIZE, num_workers=0)
    preds: List[np.ndarray] = []
    mean_t = torch.from_numpy(mean).to(_device())
    std_t = torch.from_numpy(std).to(_device())
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(_device())
            pred = model(x) * std_t + mean_t
            preds.append(pred.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    abs_err = np.abs(pred - targets)
    mae_px = float(np.mean(abs_err) * COORD_SCALE)
    endpoint_idx = (0, 1, coordinate_dim(spec) - 2, coordinate_dim(spec) - 1)
    interior_idx = tuple(i for i in range(coordinate_dim(spec)) if i not in endpoint_idx)
    metrics = {
        "profile": spec.name,
        "split": "test",
        "count": int(len(targets)),
        "mae_px": mae_px,
        "rmse_px": float(np.sqrt(np.mean(abs_err**2)) * COORD_SCALE),
        "endpoint_mae_px": float(np.mean(abs_err[:, list(endpoint_idx)]) * COORD_SCALE),
        "interior_mae_px": float(np.mean(abs_err[:, list(interior_idx)]) * COORD_SCALE),
        "best_checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "best_checkpoint_val_mae_px": float(ckpt.get("val_mae_px", float("nan"))),
    }
    # per-type breakdown when available
    for label in sorted(set(types.tolist())):
        mask = types == label
        if not np.any(mask):
            continue
        metrics[f"mae_px_{label}"] = float(np.mean(abs_err[mask]) * COORD_SCALE)
        metrics[f"count_{label}"] = int(np.sum(mask))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_history(d["tables"] / "metrics.csv", [metrics])
    print(f"[analyze:{spec.name}] test MAE={mae_px:.3f}px endpoint={metrics['endpoint_mae_px']:.3f}px interior={metrics['interior_mae_px']:.3f}px")
    return metrics


def _setup_matplotlib_chinese():
    """Configure matplotlib for Chinese labels on Windows."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_mae() -> Path:
    """Plot validation MAE curves for quadratic and cubic side by side."""
    plt = _setup_matplotlib_chinese()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, name in zip(axes, ("quadratic", "cubic")):
        path = dirs(name)["tables"] / "history.csv"
        rows = _read_history_rows(path)
        if not rows:
            raise FileNotFoundError(f"missing or empty history: {path}")
        epochs = [int(float(r["epoch"])) for r in rows]
        mae = [float(r["val_coordinate_mae_px"]) for r in rows]
        ax.plot(epochs, mae, "-", color="#1976D2", lw=1.6, label="验证 MAE")
        best_i = int(np.argmin(mae))
        ax.scatter([epochs[best_i]], [mae[best_i]], c="#D32F2F", s=36, zorder=3, label=f"最佳 ep{epochs[best_i]}")
        for mark in (20, 40):
            if epochs[0] <= mark <= epochs[-1]:
                ax.axvline(mark, color="#9E9E9E", ls="--", lw=0.9, alpha=0.8)
        # Clip extreme early spikes so late-stage trend remains readable.
        finite = [v for v in mae if math.isfinite(v)]
        y_cap = float(min(max(finite), max(25.0, float(np.percentile(finite, 85)) * 1.4)))
        ax.set_ylim(0.0, y_cap)
        y_top = y_cap * 0.95
        for mark, label in ((20, "续训@20"), (40, "续训@40")):
            if epochs[0] <= mark <= epochs[-1]:
                ax.text(mark + 0.8, y_top, label, fontsize=8, color="#616161", rotation=90, va="top")
        ax.set_title(f"{'二次' if name == 'quadratic' else '三次'} Bézier")
        ax.set_xlabel("epoch")
        ax.set_ylabel("验证坐标 MAE (px)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Bezier OOB：验证集坐标 MAE 曲线", fontsize=12)
    fig.tight_layout()
    out_dir = RESULTS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "val_mae_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[plot-mae] wrote {out}")
    return out


def plot_recon(profile: str = "cubic", count: int = 10) -> Path:
    """Overlay GT vs prediction curves on test images."""
    plt = _setup_matplotlib_chinese()
    spec = profile_spec(profile)
    d = dirs(spec)
    mean, std = _load_norm_stats(spec)
    images, targets, ids = _load_split_arrays(spec, "test")
    ckpt_path = d["checkpoints"] / "resnet18_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=_device(), weights_only=False)
    if not _ckpt_compatible(ckpt, spec):
        raise RuntimeError("checkpoint incompatible with current geometry/profile")
    model = build_model(coordinate_dim(spec)).to(_device())
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rng = np.random.default_rng(SEED)
    n = min(count, len(targets))
    chosen = np.sort(rng.choice(len(targets), size=n, replace=False))
    # predict only chosen indices
    preds = np.zeros((n, coordinate_dim(spec)), dtype=np.float32)
    mean_t = torch.from_numpy(mean).to(_device())
    std_t = torch.from_numpy(std).to(_device())
    with torch.no_grad():
        for slot, index in enumerate(chosen):
            x = torch.from_numpy(np.ascontiguousarray(images[index])).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
            x = x.unsqueeze(0).to(_device())
            pred = model(x) * std_t + mean_t
            preds[slot] = pred.squeeze(0).cpu().numpy()
    pad = MAX_OUT if spec.kind == "quadratic" else 0.05
    cols = 5
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.0))
    axes_list = np.atleast_1d(axes).ravel()
    for ax in axes_list:
        ax.set_axis_off()
    for slot, index in enumerate(chosen):
        ax = axes_list[slot]
        ax.set_axis_on()
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255, extent=(0, 1, 1, 0), origin="upper")
        gt = targets[index]
        pr = preds[slot]
        gt_curve = curve_points(spec.kind, gt, 128)
        pr_curve = curve_points(spec.kind, pr, 128)
        gt_ctrl = gt.reshape((-1, 2))
        pr_ctrl = pr.reshape((-1, 2))
        ax.plot(gt_curve[:, 0], gt_curve[:, 1], color="#4FC3F7", lw=1.8, label="真值曲线")
        ax.plot(pr_curve[:, 0], pr_curve[:, 1], color="#FF7043", lw=1.6, ls="--", label="预测曲线")
        ax.scatter(gt_ctrl[:, 0], gt_ctrl[:, 1], c="#81C784", s=22, zorder=3, label="真值控制点")
        ax.scatter(pr_ctrl[:, 0], pr_ctrl[:, 1], c="#E57373", s=22, marker="x", zorder=4, label="预测控制点")
        ax.plot(gt_ctrl[:, 0], gt_ctrl[:, 1], color="#81C784", ls=":", lw=0.7, alpha=0.7)
        ax.plot(pr_ctrl[:, 0], pr_ctrl[:, 1], color="#E57373", ls=":", lw=0.7, alpha=0.7)
        mae_px = float(np.mean(np.abs(pr - gt)) * COORD_SCALE)
        ax.set_xlim(-pad - 0.05, 1.0 + pad + 0.05)
        ax.set_ylim(1.0 + pad + 0.05, -pad - 0.05)
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="white", lw=0.8, alpha=0.7))
        ax.set_title(f"#{index}  MAE={mae_px:.2f}px", fontsize=8)
        if slot == 0:
            ax.legend(fontsize=6, loc="lower left")
    kind_cn = "二次" if spec.kind == "quadratic" else "三次"
    fig.suptitle(f"{kind_cn}测试集还原（蓝实线=真值，橙虚线=预测）", fontsize=11)
    fig.tight_layout()
    out = d["figures"] / f"test_recon_{n}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[plot-recon:{spec.name}] wrote {out}")
    return out


def preview(profile: str, count: int = 10) -> Path:
    plt = _setup_matplotlib_chinese()

    spec = profile_spec(profile)
    d = dirs(spec)
    path = d["data"] / "train.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare first")
    with np.load(path, allow_pickle=False) as payload:
        images = np.asarray(payload["images"])
        targets = np.asarray(payload["targets"])
        types = np.asarray(payload["types"])
        ids = np.asarray(payload["ids"])
    # Quadratic preview keeps 2:1 oob/inframe; cubic just shows the first N samples.
    if spec.kind == "quadratic":
        n_oob = int(round(count * OOB_RATIO / (OOB_RATIO + INFRAME_RATIO)))
        n_in = count - n_oob
        oob_idx = np.flatnonzero(types == "oob")
        in_idx = np.flatnonzero(types == "inframe")
        if len(oob_idx) < n_oob or len(in_idx) < n_in:
            chosen = np.arange(count)
        else:
            chosen = np.concatenate([oob_idx[:n_oob], in_idx[:n_in]])
    else:
        chosen = np.arange(min(count, len(images)))
    pad = MAX_OUT if spec.kind == "quadratic" else 0.05
    cols = 5
    rows = int(math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    axes_list = np.atleast_1d(axes).ravel()
    names = target_names(spec)
    for ax in axes_list:
        ax.set_axis_off()
    for slot, index in enumerate(chosen):
        ax = axes_list[slot]
        ax.set_axis_on()
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255, extent=(0, 1, 1, 0), origin="upper")
        params = targets[index]
        pts = curve_points(spec.kind, params, 128)
        ctrl = params.reshape((-1, 2))
        ax.plot(pts[:, 0], pts[:, 1], color="#4FC3F7", lw=1.5, label="curve")
        ax.scatter(ctrl[0, 0], ctrl[0, 1], c="#81C784", s=28, zorder=3, label="P0")
        ax.scatter(ctrl[-1, 0], ctrl[-1, 1], c="#FFD54F", s=28, zorder=3, label="Pend")
        if spec.kind == "quadratic":
            ax.scatter(ctrl[1, 0], ctrl[1, 1], c="#E57373", s=28, zorder=3, label="P1")
            ax.plot([ctrl[0, 0], ctrl[1, 0], ctrl[2, 0]], [ctrl[0, 1], ctrl[1, 1], ctrl[2, 1]], color="#E57373", ls="--", lw=0.8, alpha=0.8)
        else:
            ax.scatter(ctrl[1, 0], ctrl[1, 1], c="#E57373", s=28, zorder=3, label="P1")
            ax.scatter(ctrl[2, 0], ctrl[2, 1], c="#BA68C8", s=28, zorder=3, label="P2")
            ax.plot(ctrl[:, 0], ctrl[:, 1], color="#E57373", ls="--", lw=0.8, alpha=0.8)
        ax.set_xlim(-pad - 0.05, 1.0 + pad + 0.05)
        ax.set_ylim(1.0 + pad + 0.05, -pad - 0.05)
        ax.set_title(f"{types[index]} | {ids[index].split('-')[-1]}", fontsize=8)
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="white", lw=0.8, alpha=0.7))
    title_note = (
        "2:1 oob/inframe"
        if spec.kind == "quadratic"
        else "all controls in-frame; P1<=P2 by (x,y)"
    )
    fig.suptitle(f"{spec.name} train preview ({title_note}; box = image frame)", fontsize=11)
    fig.tight_layout()
    out = d["figures"] / "train_preview_10.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    # also write a small CSV describing the shown samples
    rows = []
    for index in chosen:
        row = {"id": str(ids[index]), "type": str(types[index])}
        for k, name in enumerate(names):
            row[name] = float(targets[index, k])
            row[f"{name}_px"] = float(targets[index, k] * COORD_SCALE)
        rows.append(row)
    csv_path = d["tables"] / "train_preview_10.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[preview:{spec.name}] wrote {out}")
    return out


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / den) if den > 1e-15 else float("nan")


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def diagnose_curve(profile: str, force: bool = False) -> Dict[str, Any]:
    """No training: correlate control-point MAE with curve RMSE, re-render MAE, and stroke IoU."""
    plt = _setup_matplotlib_chinese()
    spec = profile_spec(profile)
    d = dirs(spec)
    sample_path = d["tables"] / "curve_diag_samples.csv"
    summary_path = d["tables"] / "curve_diag_summary.json"
    if sample_path.exists() and summary_path.exists() and not force:
        raise RuntimeError(f"diagnose outputs exist under {d['tables']}; use --force")
    mean, std = _load_norm_stats(spec)
    images, targets, ids = _load_split_arrays(spec, "test")
    ckpt = torch.load(d["checkpoints"] / "resnet18_best.pt", map_location=_device(), weights_only=False)
    if not _ckpt_compatible(ckpt, spec):
        raise RuntimeError("checkpoint incompatible with current geometry/profile")
    model = build_model(coordinate_dim(spec)).to(_device())
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    loader = DataLoader(OobDataset(images, targets, mean, std, augment=False), batch_size=BATCH_SIZE, num_workers=0)
    preds: List[np.ndarray] = []
    mean_t = torch.from_numpy(mean).to(_device())
    std_t = torch.from_numpy(std).to(_device())
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(_device())
            pred = model(x) * std_t + mean_t
            preds.append(pred.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    dim = coordinate_dim(spec)
    endpoint_idx = (0, 1, dim - 2, dim - 1)
    interior_idx = tuple(i for i in range(dim) if i not in endpoint_idx)
    rows: List[Dict[str, Any]] = []
    control_maes: List[float] = []
    curve_rmses: List[float] = []
    render_maes: List[float] = []
    endpoint_maes: List[float] = []
    interior_maes: List[float] = []
    stroke_ious: List[float] = []
    stroke_recalls: List[float] = []
    stroke_precs: List[float] = []
    # Keep a few overlays for the visual panel (high/mid/low IoU later).
    overlay_candidates: List[Tuple[int, float, np.ndarray, np.ndarray]] = []
    curve_n = 64
    stroke_thr = 0.2  # intensity threshold on [0,1] render to define stroke pixels
    for i in range(len(targets)):
        gt = targets[i]
        pr = pred[i]
        abs_err = np.abs(pr - gt)
        control_mae_px = float(np.mean(abs_err) * COORD_SCALE)
        endpoint_mae_px = float(np.mean(abs_err[list(endpoint_idx)]) * COORD_SCALE)
        interior_mae_px = float(np.mean(abs_err[list(interior_idx)]) * COORD_SCALE)
        gt_curve = curve_points(spec.kind, gt, curve_n)
        pr_curve = curve_points(spec.kind, pr, curve_n)
        curve_rmse_px = float(np.sqrt(np.mean((pr_curve - gt_curve) ** 2)) * COORD_SCALE)
        gt_render = render_curve(spec.kind, gt).astype(np.float64) / 255.0
        pr_render = render_curve(spec.kind, pr).astype(np.float64) / 255.0
        render_mae_01 = float(np.mean(np.abs(pr_render - gt_render)))
        gt_mask = gt_render >= stroke_thr
        pr_mask = pr_render >= stroke_thr
        inter = float(np.logical_and(gt_mask, pr_mask).sum())
        union = float(np.logical_or(gt_mask, pr_mask).sum())
        gt_n = float(gt_mask.sum())
        pr_n = float(pr_mask.sum())
        stroke_iou = inter / union if union > 0 else float("nan")
        stroke_recall = inter / gt_n if gt_n > 0 else float("nan")  # 真值笔画被预测盖住的比例
        stroke_prec = inter / pr_n if pr_n > 0 else float("nan")
        rows.append(
            {
                "id": str(ids[i]),
                "control_mae_px": control_mae_px,
                "endpoint_mae_px": endpoint_mae_px,
                "interior_mae_px": interior_mae_px,
                "curve_rmse_px": curve_rmse_px,
                "render_mae_01": render_mae_01,
                "stroke_iou": stroke_iou,
                "stroke_recall": stroke_recall,
                "stroke_precision": stroke_prec,
            }
        )
        control_maes.append(control_mae_px)
        curve_rmses.append(curve_rmse_px)
        render_maes.append(render_mae_01)
        endpoint_maes.append(endpoint_mae_px)
        interior_maes.append(interior_mae_px)
        stroke_ious.append(stroke_iou)
        stroke_recalls.append(stroke_recall)
        stroke_precs.append(stroke_prec)
        overlay_candidates.append((i, stroke_iou, gt_mask, pr_mask))
        if (i + 1) % 200 == 0 or i + 1 == len(targets):
            print(f"[diagnose-curve:{spec.name}] {i + 1}/{len(targets)}")

    def _stats(vals: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "p90": float(np.nanquantile(arr, 0.9)),
            "p10": float(np.nanquantile(arr, 0.1)),
        }

    control_arr = np.asarray(control_maes, dtype=np.float64)
    curve_arr = np.asarray(curve_rmses, dtype=np.float64)
    iou_arr = np.asarray(stroke_ious, dtype=np.float64)
    ambiguous_rate = float(np.mean((control_arr > 3.0) & (curve_arr < 1.5)))
    high_control_high_iou = float(np.mean((control_arr > 3.0) & (iou_arr >= 0.5)))
    summary = {
        "profile": spec.name,
        "split": "test",
        "count": len(rows),
        "best_checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "stroke_threshold": stroke_thr,
        "control_mae_px": _stats(control_maes),
        "endpoint_mae_px": _stats(endpoint_maes),
        "interior_mae_px": _stats(interior_maes),
        "curve_rmse_px": _stats(curve_rmses),
        "render_mae_01": _stats(render_maes),
        "stroke_iou": _stats(stroke_ious),
        "stroke_recall": _stats(stroke_recalls),
        "stroke_precision": _stats(stroke_precs),
        "corr_control_vs_curve_pearson": _pearson(control_maes, curve_rmses),
        "corr_control_vs_curve_spearman": _spearman(control_maes, curve_rmses),
        "corr_control_vs_render_pearson": _pearson(control_maes, render_maes),
        "corr_control_vs_render_spearman": _spearman(control_maes, render_maes),
        "corr_control_vs_stroke_iou_pearson": _pearson(control_maes, stroke_ious),
        "corr_control_vs_stroke_iou_spearman": _spearman(control_maes, stroke_ious),
        "rate_control_mae_gt_3_and_curve_rmse_lt_1_5": ambiguous_rate,
        "rate_control_mae_gt_3_and_stroke_iou_ge_0_5": high_control_high_iou,
        "note": (
            "curve_rmse=same-t samples; render_mae=GT vs pred full-image MAE; "
            "stroke_iou=overlap of thresholded stroke pixels (visual curve coincidence)."
        ),
    }
    d["tables"].mkdir(parents=True, exist_ok=True)
    d["figures"].mkdir(parents=True, exist_ok=True)
    _write_history(sample_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(control_maes, curve_rmses, s=10, alpha=0.35, c="#1976D2")
    ax.set_xlabel("控制点 MAE (px)")
    ax.set_ylabel("同 t 曲线 RMSE (px)")
    ax.set_title(f"{spec.name}：控制点误差 vs 曲线误差")
    ax.grid(True, alpha=0.25)
    ax.axvline(3.0, color="#9E9E9E", ls="--", lw=0.8)
    ax.axhline(1.5, color="#9E9E9E", ls="--", lw=0.8)
    fig.tight_layout()
    curve_fig = d["figures"] / "control_vs_curve_scatter.png"
    fig.savefig(curve_fig, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(control_maes, render_maes, s=10, alpha=0.35, c="#E64A19")
    ax.set_xlabel("控制点 MAE (px)")
    ax.set_ylabel("重渲染 MAE [0,1]")
    ax.set_title(f"{spec.name}：控制点误差 vs 重渲染误差")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    render_fig = d["figures"] / "control_vs_render_scatter.png"
    fig.savefig(render_fig, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(control_maes, stroke_ious, s=10, alpha=0.35, c="#2E7D32")
    ax.set_xlabel("控制点 MAE (px)")
    ax.set_ylabel("笔画重合率 IoU")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{spec.name}：控制点误差 vs 图像笔画重合率")
    ax.grid(True, alpha=0.25)
    ax.axvline(3.0, color="#9E9E9E", ls="--", lw=0.8)
    ax.axhline(0.5, color="#9E9E9E", ls="--", lw=0.8)
    fig.tight_layout()
    iou_scatter = d["figures"] / "control_vs_stroke_iou_scatter.png"
    fig.savefig(iou_scatter, dpi=160)
    plt.close(fig)

    # Visual panel: 10 samples spanning IoU ranks (直观看重合)
    overlay_candidates.sort(key=lambda item: item[1])
    n_show = min(10, len(overlay_candidates))
    if n_show >= 10:
        pick_pos = [0, 1, 2, len(overlay_candidates) // 4, len(overlay_candidates) // 3, len(overlay_candidates) // 2, (2 * len(overlay_candidates)) // 3, (3 * len(overlay_candidates)) // 4, len(overlay_candidates) - 2, len(overlay_candidates) - 1]
        chosen = [overlay_candidates[p] for p in pick_pos]
    else:
        chosen = overlay_candidates[:n_show]
    cols = 5
    rows_n = int(math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.8, rows_n * 3.0))
    axes_list = np.atleast_1d(axes).ravel()
    for ax in axes_list:
        ax.set_axis_off()
    for slot, (index, iou, gt_mask, pr_mask) in enumerate(chosen):
        ax = axes_list[slot]
        ax.set_axis_on()
        # RGB overlay: GT only cyan, pred only orange, overlap white
        rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        only_gt = gt_mask & ~pr_mask
        only_pr = pr_mask & ~gt_mask
        both = gt_mask & pr_mask
        rgb[only_gt] = (0.25, 0.75, 1.0)
        rgb[only_pr] = (1.0, 0.45, 0.2)
        rgb[both] = (1.0, 1.0, 1.0)
        ax.imshow(rgb, origin="upper")
        ax.set_title(
            f"#{index} IoU={iou:.2f}\nctrl={control_maes[index]:.1f}px",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{'二次' if spec.kind == 'quadratic' else '三次'}笔画重合：青=仅真值，橙=仅预测，白=重合",
        fontsize=11,
    )
    fig.tight_layout()
    overlap_fig = d["figures"] / "stroke_overlap_panel.png"
    fig.savefig(overlap_fig, dpi=160)
    plt.close(fig)

    print(
        f"[diagnose-curve:{spec.name}] control={summary['control_mae_px']['mean']:.3f}px "
        f"curve={summary['curve_rmse_px']['mean']:.3f}px "
        f"stroke_iou={summary['stroke_iou']['mean']:.3f} "
        f"corr(ctrl,iou)={summary['corr_control_vs_stroke_iou_pearson']:.3f}"
    )
    print(f"[diagnose-curve:{spec.name}] wrote {sample_path}")
    print(f"[diagnose-curve:{spec.name}] wrote {summary_path}")
    print(f"[diagnose-curve:{spec.name}] wrote {curve_fig}")
    print(f"[diagnose-curve:{spec.name}] wrote {render_fig}")
    print(f"[diagnose-curve:{spec.name}] wrote {iou_scatter}")
    print(f"[diagnose-curve:{spec.name}] wrote {overlap_fig}")
    return summary


def check(profile: str = "quadratic") -> int:
    spec = profile_spec(profile)
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"profile={spec.name} kind={spec.kind} train={spec.train_count} val={spec.val_count} test={spec.test_count}")
    if spec.kind == "quadratic":
        print(f"endpoint_margin_px={ENDPOINT_MARGIN_PX} max_out_px={MAX_OUT_PX} mix={OOB_RATIO}:{INFRAME_RATIO}")
    else:
        print(f"endpoint_margin_px={ENDPOINT_MARGIN_PX} cubic_controls=inframe_only; P1/P2 ordered by (x,y)")
    print(f"geometry_fingerprint={geometry_fingerprint(spec)}")
    print(f"max_epochs={MAX_EPOCHS} resume_from={RESUME_FROM_EPOCH} lr={LR}->{LR_MIN} cosine_T={RESUME_EPOCHS}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("check", "prepare", "preview", "train", "analyze", "plot-mae", "plot-recon", "diagnose-curve"),
        nargs="?",
        default="check",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quadratic")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--expand", action="store_true", help="regenerate splits at new counts; keep existing norm_stats")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt at RESUME_FROM_EPOCH to max_epochs with cosine restart")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--all-profiles", action="store_true", help="run stage for quadratic and cubic")
    args = parser.parse_args(argv)
    if args.stage == "plot-mae":
        plot_mae()
        return 0
    profiles = list(PROFILES) if args.all_profiles else [args.profile]
    for name in profiles:
        if args.stage == "check":
            check(name)
        elif args.stage == "prepare":
            prepare(name, force=args.force, expand=args.expand)
        elif args.stage == "preview":
            preview(name, count=args.count)
        elif args.stage == "train":
            train(name, force=args.force, resume=args.resume)
        elif args.stage == "analyze":
            analyze(name, force=args.force)
        elif args.stage == "plot-recon":
            plot_recon(name, count=args.count)
        elif args.stage == "diagnose-curve":
            diagnose_curve(name, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
