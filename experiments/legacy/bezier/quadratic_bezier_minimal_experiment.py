"""Minimal quadratic-Bezier inverse experiment.

Only this file owns ``results/quadratic_bezier_minimal/{profile}``; the older
phase and cubic experiments are never imported or modified.  Data are rendered
once with a 4x Pillow canvas and LANCZOS reduction, then frozen in uint8 NPZ
files.  The ``check`` stage is read-only; generation, training and analysis are
explicit pipeline stages.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 20260810
IMAGE_SIZE = 224
SUPER_SAMPLE = 4
SUPERSAMPLE = SUPER_SAMPLE
CURVE_SAMPLES = 256
STROKE_WIDTH = 3.0
COORDINATE_DIM = 6
MIN_X_GAP = 0.35
MIN_DISTANCE = 0.08
ENDPOINT_MIN = 0.15
ENDPOINT_MAX = 0.85
INTERIOR_MIN = 0.10
INTERIOR_MAX = 0.90
TARGET_NAMES = ("p0x", "p0y", "p1x", "p1y", "p2x", "p2y")
ENDPOINT_INDICES = (0, 1, 4, 5)
INTERIOR_INDICES = (2, 3)
AUGMENT_MAX_SHIFT_PX = 12
COORD_SCALE = float(IMAGE_SIZE - 1)


@dataclass(frozen=True)
class Profile:
    name: str
    train_count: int
    val_count: int
    test_count: int
    epochs: int
    batch_size: int
    identifiability_count: int
    inverse_count: int
    inverse_restarts: int


PROFILES: Dict[str, Profile] = {
    "smoke": Profile("smoke", 256, 64, 64, 1, 64, 8, 2, 2),
    "minimal": Profile("minimal", 10000, 256, 512, 40, 64, 64, 16, 4),
}
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "quadratic_bezier_minimal"
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
    return {name: root / name for name in ("config", "manifest", "data", "checkpoints", "tables", "figures")} | {"root": root}


def config(profile: str | Profile) -> Dict[str, Any]:
    spec = profile_spec(profile)
    return {
        "schema": 1,
        "profile": asdict(spec),
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "supersample": SUPER_SAMPLE,
        "resampling": "LANCZOS",
        "curve_samples": CURVE_SAMPLES,
        "stroke_width": STROKE_WIDTH,
        "targets": list(TARGET_NAMES),
        "canonical": {"p0x_less_than_p2x": True, "min_x_gap": MIN_X_GAP, "endpoints_open_range": [ENDPOINT_MIN, ENDPOINT_MAX]},
        "p1_support": [INTERIOR_MIN, INTERIOR_MAX],
        "p1_chord_distance_min": MIN_DISTANCE,
        "renderer": "Pillow high-resolution polyline then LANCZOS",
        "model": {"name": "torchvision.resnet18", "weights": None, "avgpool": "AdaptiveAvgPool2d((1,1))", "head": "Linear(512,6)", "sigmoid": False},
        "loss": {"name": "MSELoss"},
        "augmentation": {"train_only": True, "translate_px": AUGMENT_MAX_SHIFT_PX, "keep_controls_in_frame": True},
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "amp": True,
            "scheduler": "CosineAnnealingLR(T_max=epochs,eta_min=1e-5)",
            "early_stop_patience": 10,
        },
        "identifiability": {"delta": 0.75 / 223.0, "delta_sensitivity_px": [0.5, 0.75, 1.0], "sensitivity_samples": 16, "normalization": IMAGE_SIZE * IMAGE_SIZE},
        "inverse": {
            "method": "SciPy L-BFGS-B",
            "samples": spec.inverse_count,
            "restarts": spec.inverse_restarts,
            "finite_difference_delta": 0.75 / 223.0,
            "initial_scale_px": [2.0, 20.0],
            "joint_render_mse_threshold": 1e-3,
            "local_only": True,
        },
    }


def fingerprint(profile: str | Profile) -> str:
    """Hash only frozen-data fields so optimizer/epoch sweeps reuse the same NPZ."""
    spec = profile_spec(profile)
    payload = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "supersample": SUPER_SAMPLE,
        "resampling": "LANCZOS",
        "curve_samples": CURVE_SAMPLES,
        "stroke_width": STROKE_WIDTH,
        "targets": list(TARGET_NAMES),
        "canonical": {"p0x_less_than_p2x": True, "min_x_gap": MIN_X_GAP, "endpoints_open_range": [ENDPOINT_MIN, ENDPOINT_MAX]},
        "p1_support": [INTERIOR_MIN, INTERIOR_MAX],
        "p1_chord_distance_min": MIN_DISTANCE,
        "renderer": "Pillow high-resolution polyline then LANCZOS",
        "profile": spec.name,
        "counts": {"train": spec.train_count, "val": spec.val_count, "test": spec.test_count},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_metadata(profile: str | Profile, stage: str, extra: Mapping[str, Any] | None = None) -> None:
    path = dirs(profile)["config"] / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(), "fingerprint": fingerprint(profile), "config": config(profile), "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__}}
    if extra:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def quadratic_points(parameters: Sequence[float], count: int = CURVE_SAMPLES) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.float64).reshape(3, 2)
    t = np.linspace(0.0, 1.0, int(count), dtype=np.float64)[:, None]
    return (1.0 - t) ** 2 * p[0] + 2.0 * (1.0 - t) * t * p[1] + t**2 * p[2]


def chord_distance(parameters: Sequence[float]) -> float:
    p = np.asarray(parameters, dtype=np.float64).reshape(3, 2)
    chord = p[2] - p[0]
    length = float(np.linalg.norm(chord))
    if length <= 1e-12:
        return 0.0
    offset = p[1] - p[0]
    return float(abs(chord[0] * offset[1] - chord[1] * offset[0]) / length)


def canonicalize(parameters: Sequence[float]) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.float32).reshape(3, 2).copy()
    if p[0, 0] > p[2, 0]:
        p = p[::-1].copy()
    return p.reshape(-1)


def valid_parameters(parameters: Sequence[float]) -> bool:
    p = np.asarray(parameters, dtype=np.float64).reshape(3, 2)
    return bool(
        np.all(np.isfinite(p))
        and ENDPOINT_MIN < p[0, 0] < ENDPOINT_MAX
        and ENDPOINT_MIN < p[0, 1] < ENDPOINT_MAX
        and ENDPOINT_MIN < p[2, 0] < ENDPOINT_MAX
        and ENDPOINT_MIN < p[2, 1] < ENDPOINT_MAX
        and p[0, 0] < p[2, 0]
        and p[2, 0] - p[0, 0] >= MIN_X_GAP
        and np.all((p[1] >= INTERIOR_MIN) & (p[1] <= INTERIOR_MAX))
        and chord_distance(p) >= MIN_DISTANCE
    )


def render_quadratic(parameters: Sequence[float]) -> np.ndarray:
    from PIL import Image, ImageDraw

    points = quadratic_points(parameters, CURVE_SAMPLES)
    high_size = IMAGE_SIZE * SUPER_SAMPLE
    image = Image.new("L", (high_size, high_size), color=0)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * (IMAGE_SIZE - 1) * SUPER_SAMPLE, float(y) * (IMAGE_SIZE - 1) * SUPER_SAMPLE) for x, y in points]
    try:
        draw.line(xy, fill=255, width=int(round(STROKE_WIDTH * SUPER_SAMPLE)), joint="curve")
    except TypeError:
        draw.line(xy, fill=255, width=int(round(STROKE_WIDTH * SUPER_SAMPLE)))
    resampling = getattr(Image, "Resampling", Image)
    out = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling.LANCZOS), dtype=np.uint8)
    if out.shape != (IMAGE_SIZE, IMAGE_SIZE) or out.dtype != np.uint8 or out.ndim != 2:
        raise RuntimeError(f"renderer invariant: {out.shape} {out.dtype}")
    return out


def _split_seed(profile: str | Profile, split: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{profile_spec(profile).name}:{split}".encode()).digest()[:8], "little") % (2**32 - 1)


def _sample_parameters(split: str, rng: np.random.Generator) -> np.ndarray:
    while True:
        p0x, p2x = rng.uniform(ENDPOINT_MIN + 1e-3, ENDPOINT_MAX - 1e-3, size=2)
        p0y, p2y = rng.uniform(ENDPOINT_MIN + 1e-3, ENDPOINT_MAX - 1e-3, size=2)
        p1 = rng.uniform(INTERIOR_MIN, INTERIOR_MAX, size=2)
        candidate = canonicalize((p0x, p0y, p1[0], p1[1], p2x, p2y))
        if valid_parameters(candidate):
            return candidate


def _generate_split(profile: str | Profile, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    n = profile_spec(profile).train_count if split == "train" else profile_spec(profile).val_count if split == "val" else profile_spec(profile).test_count
    rng = np.random.default_rng(_split_seed(profile, split))
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    targets = np.empty((n, COORDINATE_DIM), dtype=np.float32)
    ids = np.asarray([f"{profile_spec(profile).name}-{split}-{i:06d}" for i in range(n)], dtype="U64")
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        target = _sample_parameters(split, rng)
        image = render_quadratic(target)
        images[i], targets[i] = image, target
        rows.append({"id": str(ids[i]), "split": split, "chord_distance": chord_distance(target), "render_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "parameter_sha256": hashlib.sha256(target.tobytes()).hexdigest()})
    return images, targets, ids, rows


def prepare(profile: str = "minimal", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    d = dirs(spec)
    manifest_path = d["manifest"] / "manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint(spec):
            cache_paths = [d["data"] / f"{split}.npz" for split in SPLITS]
            if all(path.exists() for path in cache_paths):
                print(f"[prepare] cache ready: {d['root']}")
                return existing
            print(f"[prepare] manifest exists but local NPZ cache is incomplete; rebuilding {d['data']}")
        else:
            raise RuntimeError("existing manifest fingerprint differs; use --force")
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    (d["config"] / "config.json").write_text(json.dumps({**config(spec), "fingerprint": fingerprint(spec)}, indent=2), encoding="utf-8")
    entries: Dict[str, Any] = {}
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    for split in SPLITS:
        images, targets, ids, rows = _generate_split(spec, split)
        npz = d["data"] / f"{split}.npz"
        np.savez_compressed(npz, images=images, targets=targets, ids=ids, fingerprint=np.asarray(fingerprint(spec)))
        (d["manifest"] / f"{split}.json").write_text(json.dumps({"split": split, "records": rows}), encoding="utf-8")
        hashes = [r["render_sha256"] for r in rows]
        if len(set(ids.tolist())) != len(ids) or len(set(hashes)) != len(hashes):
            raise RuntimeError(f"duplicate ID/raster hash in {split}")
        if all_ids.intersection(ids.tolist()) or all_hashes.intersection(hashes):
            raise RuntimeError(f"cross-split ID/raster collision at {split}")
        all_ids.update(ids.tolist()); all_hashes.update(hashes)
        entries[split] = {"count": len(ids), "npz": str(npz.relative_to(d["root"])), "ids": ids.tolist(), "image_shape": list(images.shape), "image_dtype": str(images.dtype), "target_shape": list(targets.shape), "target_dtype": str(targets.dtype), "parameter_sha256": [r["parameter_sha256"] for r in rows], "render_sha256": hashes, "chord_distance_min": float(min(r["chord_distance"] for r in rows)), "chord_distance_max": float(max(r["chord_distance"] for r in rows))}
    manifest = {"schema": 1, "profile": spec.name, "fingerprint": fingerprint(spec), "counts": {k: v["count"] for k, v in entries.items()}, "splits": entries, "id_unique": len(all_ids) == sum(v["count"] for v in entries.values()), "render_hash_unique": len(all_hashes) == sum(v["count"] for v in entries.values()), "invariants": {"canonical": True, "endpoint_open_range": [ENDPOINT_MIN, ENDPOINT_MAX], "min_x_gap": MIN_X_GAP, "p1_support": [INTERIOR_MIN, INTERIOR_MAX], "p1_chord_distance_min": MIN_DISTANCE, "npz_fingerprint": fingerprint(spec)}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_metadata(spec, "prepare", {"counts": manifest["counts"]})
    print(f"[prepare] complete: {d['root']}")
    return manifest


def build_model() -> nn.Module:
    from torchvision import models

    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, COORDINATE_DIM)
    return model


def _translate_uint8(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift image content by (dx, dy) pixels with black fill; no wrap-around."""
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


class OfflineDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, targets: np.ndarray, augment: bool = False, max_shift_px: int = AUGMENT_MAX_SHIFT_PX) -> None:
        self.images, self.targets = images, targets
        self.augment = augment
        self.max_shift_px = int(max_shift_px)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        target = self.targets[index].astype(np.float32, copy=True)
        if self.augment and self.max_shift_px > 0:
            xs = target[0::2]
            ys = target[1::2]
            max_left = int(np.floor(float(xs.min()) * COORD_SCALE))
            max_right = int(np.floor(float(1.0 - xs.max()) * COORD_SCALE))
            max_up = int(np.floor(float(ys.min()) * COORD_SCALE))
            max_down = int(np.floor(float(1.0 - ys.max()) * COORD_SCALE))
            dx_lo, dx_hi = -min(self.max_shift_px, max_left), min(self.max_shift_px, max_right)
            dy_lo, dy_hi = -min(self.max_shift_px, max_up), min(self.max_shift_px, max_down)
            dx = random.randint(dx_lo, dx_hi) if dx_hi >= dx_lo else 0
            dy = random.randint(dy_lo, dy_hi) if dy_hi >= dy_lo else 0
            if dx or dy:
                image = _translate_uint8(image, dx, dy)
                target[0::2] += dx / COORD_SCALE
                target[1::2] += dy / COORD_SCALE
        x = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        return x, torch.from_numpy(target).float()


def _load_npz(profile: str | Profile, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = dirs(profile)["data"] / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare")
    with np.load(path, allow_pickle=False) as payload:
        cache_fp = str(np.asarray(payload["fingerprint"]).reshape(-1)[0])
        images, targets, ids = np.asarray(payload["images"]), np.asarray(payload["targets"]), np.asarray(payload["ids"])
    if cache_fp != fingerprint(profile) or images.dtype != np.uint8 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE) or targets.shape[1] != COORDINATE_DIM:
        raise RuntimeError(f"invalid/fingerprint-mismatched cache {path}")
    return images, targets.astype(np.float32), ids


def _amp_context(dev: torch.device, enabled: bool):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16)


def train(profile: str = "minimal", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile); d = dirs(spec)
    _load_npz(spec, "train"); _load_npz(spec, "val")
    best = d["checkpoints"] / "resnet18_best.pt"
    if best.exists() and not force:
        raise RuntimeError(f"checkpoint exists: {best}; use --force")
    train_images, train_targets, _ = _load_npz(spec, "train"); val_images, val_targets, _ = _load_npz(spec, "val")
    dev = device(); amp = dev.type == "cuda" and torch.cuda.is_available(); seed_everything()
    model = build_model().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(
        OfflineDataset(train_images, train_targets, augment=True, max_shift_px=AUGMENT_MAX_SHIFT_PX),
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    history: List[Dict[str, Any]] = []; best_val = math.inf; epochs_no_improve = 0; early_stop_patience = 10
    (d["config"] / "config.json").write_text(json.dumps({**config(spec), "fingerprint": fingerprint(spec)}, indent=2), encoding="utf-8")
    for epoch in range(1, spec.epochs + 1):
        model.train(); total = 0.0; count = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev); optimizer.zero_grad(set_to_none=True)
            with _amp_context(dev, amp): out = model(x); loss = loss_fn(out, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += float(loss.detach()) * len(x); count += len(x)
        model.eval(); val_loss = 0.0; val_out: List[np.ndarray] = []
        with torch.no_grad():
            for x, y in DataLoader(OfflineDataset(val_images, val_targets, augment=False), batch_size=spec.batch_size, num_workers=0):
                with _amp_context(dev, False): pred = model(x.to(dev)); val_loss += float(loss_fn(pred, y.to(dev))) * len(x); val_out.append(pred.cpu().numpy())
        val_loss /= max(1, len(val_targets)); train_loss = total / max(1, count); val_pred = np.concatenate(val_out); val_mae_px = float(np.mean(np.abs(val_pred - val_targets)) * COORD_SCALE)
        lr_used = float(optimizer.param_groups[0]["lr"]); scheduler.step(); lr_next = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_coordinate_mae_px": val_mae_px, "lr_used": lr_used, "lr_next": lr_next}; history.append(row); print(f"[train:{spec.name}] epoch {epoch}/{spec.epochs} train={train_loss:.6g} val={val_loss:.6g} mae={val_mae_px:.3f}px lr={lr_used:.2g}")
        if val_loss < best_val:
            best_val = val_loss; epochs_no_improve = 0; payload = {"model_state": model.state_dict(), "epoch": epoch, "fingerprint": fingerprint(spec), "profile": spec.name}; d["checkpoints"].mkdir(parents=True, exist_ok=True); torch.save(payload, best)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"[train:{spec.name}] early stopping at epoch {epoch} ({early_stop_patience} epochs without improvement)")
            break
    torch.save({"model_state": model.state_dict(), "epoch": epoch, "fingerprint": fingerprint(spec), "profile": spec.name}, d["checkpoints"] / "last.pt")
    write_rows(d["tables"] / "history.csv", history); _write_metadata(spec, "train", {"best_val_loss": best_val, "early_stopped": epochs_no_improve >= early_stop_patience, "final_epoch": epoch})
    return {"best_val_loss": best_val, "epochs": len(history)}


def _load_model(profile: str | Profile) -> nn.Module:
    path = dirs(profile)["checkpoints"] / "resnet18_best.pt"
    if not path.exists(): raise FileNotFoundError(f"missing checkpoint {path}")
    payload = torch.load(path, map_location=device(), weights_only=False)
    if payload.get("fingerprint") != fingerprint(profile): raise RuntimeError("checkpoint fingerprint mismatch")
    model = build_model().to(device()); model.load_state_dict(payload["model_state"]); model.eval(); return model


def _predict(model: nn.Module, images: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out: List[np.ndarray] = []; dev = next(model.parameters()).device
    with torch.no_grad():
        for x, _ in DataLoader(OfflineDataset(images, targets), batch_size=64, num_workers=0): out.append(model(x.to(dev)).cpu().numpy())
    return np.concatenate(out)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float); a -= a.mean(); b -= b.mean(); den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-15 else float("nan")


def _rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float); order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), float); ranks[order] = np.arange(1, len(values) + 1); return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return _pearson(_rank(a), _rank(b))


def metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    e = predictions - targets; ae = np.abs(e); result = {"count": float(len(targets)), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(e * e))), "mae_px": float(ae.mean() * 223), "rmse_px": float(np.sqrt(np.mean(e * e)) * 223), "maxabs_px": float(ae.max() * 223), "endpoint_mae_px": float(ae[:, ENDPOINT_INDICES].mean() * 223), "interior_mae_px": float(ae[:, INTERIOR_INDICES].mean() * 223), "endpoint_rmse_px": float(np.sqrt(np.mean(e[:, ENDPOINT_INDICES] ** 2)) * 223), "interior_rmse_px": float(np.sqrt(np.mean(e[:, INTERIOR_INDICES] ** 2)) * 223)}
    for i, name in enumerate(TARGET_NAMES): result[f"{name}_mae"] = float(ae[:, i].mean()); result[f"{name}_mae_px"] = float(ae[:, i].mean() * 223); result[f"{name}_rmse_px"] = float(np.sqrt(np.mean(e[:, i] ** 2)) * 223)
    curve_e = np.stack([quadratic_points(p, 64) for p in predictions]) - np.stack([quadratic_points(t, 64) for t in targets]); result["curve_rmse_px"] = float(np.sqrt(np.mean(curve_e ** 2)) * 223); result["curve_mae_px"] = float(np.mean(np.abs(curve_e)) * 223)
    pred_render = np.stack([render_quadratic(p) for p in predictions]); true_render = np.stack([render_quadratic(t) for t in targets]); result["pred_render_mae_0_1"] = float(np.mean(np.abs(pred_render.astype(float) - true_render.astype(float))) / 255.0)
    distances = np.asarray([chord_distance(t) for t in targets]);
    for name, mask in (("near", distances < 0.20), ("mid", (distances >= 0.20) & (distances < 0.35)), ("far", distances >= 0.35)):
        result[f"{name}_count"] = float(mask.sum()); result[f"{name}_mae_px"] = float(ae[mask].mean() * 223) if mask.any() else float("nan")
    return result


def _plot_analysis(profile: str | Profile, images: np.ndarray, ids: np.ndarray, targets: np.ndarray, predictions: np.ndarray, row: Mapping[str, Any]) -> None:
    import matplotlib; matplotlib.use("Agg", force=True); import matplotlib.pyplot as plt
    f = dirs(profile)["figures"]; f.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(); ax.bar(["overall", "endpoint", "interior", "curve"], [row["mae_px"], row["endpoint_mae_px"], row["interior_mae_px"], row["curve_rmse_px"]]); ax.set_ylabel("error (px)"); fig.tight_layout(); fig.savefig(f / "metric_bars.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(TARGET_NAMES, [row[f"{name}_mae_px"] for name in TARGET_NAMES]); ax.set_ylabel("coordinate MAE (px)"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(f / "per_coordinate_mae.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    history_path = dirs(profile)["tables"] / "history.csv"
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        if history:
            epochs = [int(r["epoch"]) for r in history]
            ax.plot(epochs, [float(r["train_loss"]) for r in history], "o-", label="train loss")
            ax.plot(epochs, [float(r["val_loss"]) for r in history], "o-", label="val loss")
            ax2 = ax.twinx(); ax2.plot(epochs, [float(r["val_coordinate_mae_px"]) for r in history], "s--", color="tab:red", label="val MAE px"); ax2.set_ylabel("coordinate MAE (px)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("loss and validation MAE"); ax.grid(alpha=.2); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f / "loss_mae.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4)); true = targets.reshape(-1, 3, 2); pred = predictions.reshape(-1, 3, 2)
    for i, ax in enumerate(axes):
        ax.scatter(true[:, i, 0], pred[:, i, 0], s=4, alpha=.3, label="x"); ax.scatter(true[:, i, 1], pred[:, i, 1], s=4, alpha=.3, label="y"); ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("true"); ax.set_ylabel("predicted"); ax.set_title(f"P{i}"); ax.legend(); ax.grid(alpha=.2); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f / "p0_p1_p2_scatter.png", dpi=160); plt.close(fig)
    rng = np.random.default_rng(_split_seed(profile, "overlay")); idx = rng.choice(len(targets), size=min(6, len(targets)), replace=False); fig, axes = plt.subplots(2, 3, figsize=(12, 8)); axes = axes.flat
    for panel, i in enumerate(idx):
        gt_control = targets[i].reshape(3, 2)
        pred_control = predictions[i].reshape(3, 2)
        axes[panel].imshow(images[i], cmap="gray", extent=(0, 1, 1, 0))
        axes[panel].plot(*quadratic_points(targets[i], 128).T, "c-", label="GT curve")
        axes[panel].plot(*quadratic_points(predictions[i], 128).T, "r--", label="pred curve")
        axes[panel].plot(gt_control[:, 0], gt_control[:, 1], "co:", ms=4, label="GT controls")
        axes[panel].plot(pred_control[:, 0], pred_control[:, 1], "rx:", ms=5, label="pred controls")
        axes[panel].set_title(str(ids[i])); axes[panel].set_xlim(.05, .95); axes[panel].set_ylim(.95, .05)
        if panel == 0:
            axes[panel].legend(fontsize=6)
    for ax in axes[len(idx):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(f / "overlay.png", dpi=160); plt.close(fig)


def analyze(profile: str = "minimal", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile); d = dirs(spec); metric_path = d["tables"] / "metrics.csv"
    if metric_path.exists() and not force: raise RuntimeError(f"metrics exists: {metric_path}; use --force")
    model = _load_model(spec); images, targets, ids = _load_npz(spec, "test"); predictions = _predict(model, images, targets); row = {"split": "test", **metrics(targets, predictions)}; write_rows(metric_path, [row]); rows = []
    for i, sample_id in enumerate(ids):
        rows.append({
            "id": str(sample_id),
            "sample_mae_px": float(np.mean(np.abs(predictions[i] - targets[i])) * 223),
            "chord_distance": chord_distance(targets[i]),
            **{f"true_{n}": float(targets[i, j]) for j, n in enumerate(TARGET_NAMES)},
            **{f"pred_{n}": float(predictions[i, j]) for j, n in enumerate(TARGET_NAMES)},
            **{f"error_{n}_px": float((predictions[i, j] - targets[i, j]) * 223) for j, n in enumerate(TARGET_NAMES)},
        })
    write_rows(d["tables"] / "predictions.csv", rows); _plot_analysis(spec, images, ids, targets, predictions, row); _write_metadata(spec, "analyze", row); print(f"[analyze:{spec.name}] {row}"); return row


def _jacobian_spectrum(theta: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray, int, float]:
    jacobian = np.empty((IMAGE_SIZE * IMAGE_SIZE, COORDINATE_DIM), dtype=np.float64)
    for k in range(COORDINATE_DIM):
        plus, minus = theta.copy(), theta.copy(); plus[k] += delta; minus[k] -= delta
        jacobian[:, k] = (render_quadratic(plus).astype(float).ravel() - render_quadratic(minus).astype(float).ravel()) / (2 * delta * 255)
    jtj = jacobian.T @ jacobian / (IMAGE_SIZE * IMAGE_SIZE)
    eigenvalues = np.maximum(np.linalg.eigvalsh(jtj), 0)
    sigma = np.sqrt(eigenvalues)[::-1]
    tolerance = max(sigma[0] * 1e-8, 1e-12)
    rank = int((sigma > tolerance).sum())
    condition = float(sigma[0] / sigma[-1]) if sigma[-1] > tolerance else float("inf")
    return jtj, sigma, rank, condition


def identifiability(profile: str = "minimal", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile); d = dirs(spec); out = d["tables"] / "identifiability_samples.csv"
    if out.exists() and not force: raise RuntimeError(f"identifiability exists: {out}; use --force")
    model = _load_model(spec); images, targets, ids = _load_npz(spec, "test"); predictions = _predict(model, images, targets); rng = np.random.default_rng(_split_seed(spec, "identifiability")); idx = rng.choice(len(targets), size=min(spec.identifiability_count, len(targets)), replace=False); delta = .75 / 223.0; rows: List[Dict[str, Any]] = []
    for i in idx:
        theta = targets[i].astype(float); jtj, sigma, rank, cond = _jacobian_spectrum(theta, delta); rows.append({"id": str(ids[i]), "sigma_min": float(sigma[-1]), "sigma_max": float(sigma[0]), "rank": rank, "condition": cond, "prediction_error_px": float(np.mean(np.abs(predictions[i] - targets[i])) * 223), **{f"sigma_{k+1}": float(v) for k, v in enumerate(sigma)}, **{f"jtj_{r}_{c}": float(jtj[r, c]) for r in range(COORDINATE_DIM) for c in range(COORDINATE_DIM)}})
    write_rows(out, rows)
    valid = [r for r in rows if np.isfinite(r["condition"]) and r["sigma_min"] > 0]
    sigma_log = [math.log10(r["sigma_min"]) for r in valid]
    errors = [r["prediction_error_px"] for r in valid]
    condition_log = [math.log10(max(r["condition"], 1.0)) for r in valid]
    summary = {
        "samples": len(rows),
        "finite_condition_samples": len(valid),
        "delta_normalized": delta,
        "delta_px": delta * 223,
        "rank_deficient": sum(r["rank"] < 6 for r in rows),
        "condition_infinite": sum(not np.isfinite(r["condition"]) for r in rows),
        "sigma_min_median": float(np.median([r["sigma_min"] for r in rows])),
        "condition_median": float(np.median([r["condition"] for r in valid])) if valid else float("nan"),
        "condition_p90": float(np.quantile([r["condition"] for r in valid], 0.9)) if valid else float("nan"),
        "log_sigma_min_error_pearson": _pearson(sigma_log, errors) if len(valid) > 1 else float("nan"),
        "log_sigma_min_error_spearman": _spearman(sigma_log, errors) if len(valid) > 1 else float("nan"),
        "log_condition_error_pearson": _pearson(condition_log, errors) if len(valid) > 1 else float("nan"),
        "log_condition_error_spearman": _spearman(condition_log, errors) if len(valid) > 1 else float("nan"),
    }
    write_rows(d["tables"] / "identifiability_summary.csv", [summary])
    import matplotlib; matplotlib.use("Agg", force=True); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(sigma_log, errors, s=14, alpha=.65); axes[0].set_xlabel("log10(sigma_min)"); axes[0].set_ylabel("ResNet sample MAE (px)"); axes[0].grid(alpha=.2)
    axes[1].scatter(condition_log, errors, s=14, alpha=.65); axes[1].set_xlabel("log10(condition)"); axes[1].set_ylabel("ResNet sample MAE (px)"); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(d["figures"] / "identifiability_vs_error.png", dpi=160); plt.close(fig)
    singular = np.asarray([[r[f"sigma_{k+1}"] for k in range(6)] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.boxplot(singular, tick_labels=[f"s{k+1}" for k in range(6)]); ax.set_yscale("log"); ax.set_ylabel("normalized raster-Jacobian singular value"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(d["figures"] / "singular_values.png", dpi=160); plt.close(fig)
    sensitivity_rows: List[Dict[str, Any]] = []
    sensitivity_indices = idx[: min(16, len(idx))]
    for i in sensitivity_indices:
        for delta_px in (.5, .75, 1.0):
            _, sigma_step, rank_step, condition_step = _jacobian_spectrum(targets[i].astype(float), delta_px / 223.0)
            sensitivity_rows.append({"id": str(ids[i]), "delta_px": delta_px, "sigma_min": float(sigma_step[-1]), "condition": condition_step, "rank": rank_step})
    write_rows(d["tables"] / "identifiability_delta_sensitivity.csv", sensitivity_rows)
    sensitivity_summary: List[Dict[str, Any]] = []
    by_delta = {delta_px: [r for r in sensitivity_rows if r["delta_px"] == delta_px] for delta_px in (.5, .75, 1.0)}
    for delta_px in (.5, 1.0):
        reference = by_delta[.75]; comparison = by_delta[delta_px]
        sensitivity_summary.append({"reference_delta_px": .75, "comparison_delta_px": delta_px, "log_sigma_min_spearman": _spearman([math.log10(max(r["sigma_min"], 1e-30)) for r in reference], [math.log10(max(r["sigma_min"], 1e-30)) for r in comparison]), "log_condition_spearman": _spearman([math.log10(max(r["condition"], 1.0)) for r in reference], [math.log10(max(r["condition"], 1.0)) for r in comparison]), "rank_changes": sum(a["rank"] != b["rank"] for a, b in zip(reference, comparison))})
    write_rows(d["tables"] / "identifiability_delta_sensitivity_summary.csv", sensitivity_summary)
    _write_metadata(spec, "identifiability", summary); print(f"[identifiability:{spec.name}] {summary}"); return summary


def inverse_render(profile: str | Profile, force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile); d = dirs(spec); out = d["tables"] / "inverse_render.csv"
    if out.exists() and not force: raise RuntimeError(f"inverse result exists: {out}; use --force")
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise RuntimeError("inverse stage requires scipy") from exc
    _, targets, ids = _load_npz(spec, "test"); rng = np.random.default_rng(_split_seed(spec, "inverse")); choose = rng.choice(len(targets), size=min(spec.inverse_count, len(targets)), replace=False); rows: List[Dict[str, Any]] = []; restart_rows: List[Dict[str, Any]] = []
    bounds = [(ENDPOINT_MIN, ENDPOINT_MAX), (ENDPOINT_MIN, ENDPOINT_MAX), (INTERIOR_MIN, INTERIOR_MAX), (INTERIOR_MIN, INTERIOR_MAX), (ENDPOINT_MIN, ENDPOINT_MAX), (ENDPOINT_MIN, ENDPOINT_MAX)]
    lower = np.asarray([b[0] for b in bounds]); upper = np.asarray([b[1] for b in bounds]); finite_delta = .75 / 223.0
    restart_scales_px = np.geomspace(2.0, 20.0, num=spec.inverse_restarts)
    for i in choose:
        truth = targets[i]; target_image = render_quadratic(truth).astype(float) / 255; attempts = []
        for restart, scale_px in enumerate(restart_scales_px):
            x0 = truth.astype(float).copy()
            for _ in range(100):
                candidate = np.clip(truth + rng.normal(0.0, scale_px / 223.0, size=COORDINATE_DIM), lower + 1e-5, upper - 1e-5)
                candidate = canonicalize(candidate).astype(float)
                if valid_parameters(candidate):
                    x0 = candidate
                    break
            def objective(x: np.ndarray) -> float:
                p = canonicalize(x)
                if not valid_parameters(p):
                    return 1.0 + float((max(0.0, MIN_X_GAP - (p[4] - p[0]))) ** 2)
                return float(np.mean((render_quadratic(p).astype(float) / 255 - target_image) ** 2))
            initial_loss = objective(x0); initial_mae = float(np.mean(np.abs(x0 - truth)) * 223)
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80, "ftol": 1e-12, "gtol": 1e-8, "eps": finite_delta, "maxls": 30})
            solution = canonicalize(result.x); loss, mae = objective(solution), float(np.mean(np.abs(solution - truth)) * 223); curve_rmse = float(np.sqrt(np.mean((quadratic_points(solution, 64) - quadratic_points(truth, 64)) ** 2)) * 223); attempts.append((loss, mae, solution, curve_rmse)); restart_rows.append({"id": str(ids[i]), "restart": restart, "initial_scale_px": float(scale_px), "initial_control_mae_px": initial_mae, "initial_render_loss": initial_loss, "initial_p0x": x0[0], "initial_p0y": x0[1], "initial_p1x": x0[2], "initial_p1y": x0[3], "initial_p2x": x0[4], "initial_p2y": x0[5], "render_loss": loss, "control_mae_px": mae, "curve_rmse_px": curve_rmse, "optimizer_success": bool(result.success), "iterations": int(result.nit), "function_evaluations": int(result.nfev), "message": str(result.message), **{f"solution_{name}": float(solution[k]) for k, name in enumerate(TARGET_NAMES)}})
        best = min(attempts, key=lambda item: item[0]); best_error = np.abs(best[2] - truth) * 223; rows.append({"id": str(ids[i]), "best_render_loss": best[0], "best_control_mae_px": best[1], "best_endpoint_mae_px": float(best_error[list(ENDPOINT_INDICES)].mean()), "best_interior_mae_px": float(best_error[list(INTERIOR_INDICES)].mean()), "best_curve_rmse_px": best[3], "success_le_1px": best[1] <= 1.0, "joint_success": best[1] <= 1.0 and best[0] <= 1e-3, "restart_dispersion_px": float(np.mean([np.linalg.norm(item[2] - best[2]) * 223 for item in attempts])), **{f"true_{name}": float(truth[k]) for k, name in enumerate(TARGET_NAMES)}, **{f"solution_{name}": float(best[2][k]) for k, name in enumerate(TARGET_NAMES)}})
    write_rows(out, rows); write_rows(d["tables"] / "inverse_render_restarts.csv", restart_rows); summary = {"samples": len(rows), "control_mae_le_1px_rate": float(np.mean([r["success_le_1px"] for r in rows])) if rows else float("nan"), "joint_control_le_1px_and_render_mse_le_1e_3_rate": float(np.mean([r["joint_success"] for r in rows])) if rows else float("nan"), "best_control_mae_px_mean": float(np.mean([r["best_control_mae_px"] for r in rows])) if rows else float("nan"), "best_control_mae_px_median": float(np.median([r["best_control_mae_px"] for r in rows])) if rows else float("nan"), "best_control_mae_px_p90": float(np.quantile([r["best_control_mae_px"] for r in rows], .9)) if rows else float("nan"), "best_endpoint_mae_px_mean": float(np.mean([r["best_endpoint_mae_px"] for r in rows])) if rows else float("nan"), "best_interior_mae_px_mean": float(np.mean([r["best_interior_mae_px"] for r in rows])) if rows else float("nan"), "best_curve_rmse_px_mean": float(np.mean([r["best_curve_rmse_px"] for r in rows])) if rows else float("nan"), "best_render_loss_mean": float(np.mean([r["best_render_loss"] for r in rows])) if rows else float("nan"), "best_render_loss_median": float(np.median([r["best_render_loss"] for r in rows])) if rows else float("nan"), "restart_dispersion_px_mean": float(np.mean([r["restart_dispersion_px"] for r in rows])) if rows else float("nan"), "local_only": True, "finite_difference_delta_px": finite_delta * 223, "initializations": spec.inverse_restarts, "initialization_records": len(restart_rows)}; write_rows(d["tables"] / "inverse_summary.csv", [summary]); _write_metadata(spec, "inverse_render", summary); print(f"[inverse:{spec.name}] {summary}"); return summary


def _read_single_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    converted: Dict[str, Any] = {}
    for key, value in rows[0].items():
        if value in ("True", "False"):
            converted[key] = value == "True"
        else:
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
    return converted


def summarize(profile: str | Profile = "minimal", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile); d = dirs(spec); output = d["tables"] / "summary.json"
    if output.exists() and not force:
        raise RuntimeError(f"summary exists: {output}; use --force")
    network = _read_single_csv(d["tables"] / "metrics.csv")
    identifiability_result = _read_single_csv(d["tables"] / "identifiability_summary.csv")
    inverse_result = _read_single_csv(d["tables"] / "inverse_summary.csv")
    with (d["tables"] / "history.csv").open("r", newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    best_history = min(history, key=lambda row: float(row["val_loss"]))
    result = {
        "profile": spec.name,
        "fingerprint": fingerprint(spec),
        "best_validation_epoch": int(best_history["epoch"]),
        "last_epoch": len(history),
        "best_validation_at_last_epoch": int(best_history["epoch"]) == len(history),
        "network": network,
        "raster_jacobian": identifiability_result,
        "local_inverse_rendering": inverse_result,
        "limits": ["single seed", "restricted fully visible non-degenerate quadratic curves", "inverse rendering is local and initialization-dependent", "best validation occurs at the final epoch if best_validation_at_last_epoch is true"],
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    flat = {"profile": spec.name, "fingerprint": fingerprint(spec), "best_validation_epoch": result["best_validation_epoch"], "last_epoch": len(history), "best_validation_at_last_epoch": result["best_validation_at_last_epoch"], **{f"network_{k}": v for k, v in network.items()}, **{f"jacobian_{k}": v for k, v in identifiability_result.items()}, **{f"inverse_{k}": v for k, v in inverse_result.items()}}
    write_rows(d["tables"] / "summary.csv", [flat]); _write_metadata(spec, "summarize", {"best_validation_epoch": result["best_validation_epoch"], "summary": str(output.relative_to(d["root"]))}); print(f"[summary:{spec.name}] network MAE={network['mae_px']:.3f}px; P1={network['interior_mae_px']:.3f}px; local inverse median={inverse_result['best_control_mae_px_median']:.3f}px; full-rank={int(identifiability_result['samples'] - identifiability_result['rank_deficient'])}/{int(identifiability_result['samples'])}")
    return result


def check(profile: str = "smoke") -> int:
    spec = profile_spec(profile); print(f"Python: {sys.version.split()[0]}"); print(f"PyTorch: {torch.__version__}"); print(f"CUDA: {torch.cuda.is_available()}")
    try:
        import torchvision, PIL
        print(f"torchvision: {torchvision.__version__}; Pillow: {PIL.__version__}")
        sample = canonicalize((.18, .25, .42, .72, .78, .60)); image = render_quadratic(sample); assert image.shape == (224, 224) and image.dtype == np.uint8 and valid_parameters(sample)
        model = build_model(); assert isinstance(model.avgpool, nn.AdaptiveAvgPool2d) and tuple(model.avgpool.output_size) == (1, 1) and isinstance(model.fc, nn.Linear) and model.fc.out_features == 6
        with torch.no_grad(): out = model(torch.from_numpy(image.copy()).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float().div_(255))
        assert tuple(out.shape) == (1, 6); print(f"renderer=224x224 uint8 grayscale supersample={SUPER_SAMPLE} LANCZOS; model output={tuple(out.shape)}")
    except Exception as exc:
        print(f"check failed: {exc}"); return 1
    print(f"profile={spec.name} train={spec.train_count} val={spec.val_count} test={spec.test_count} epochs={spec.epochs} fingerprint={fingerprint(spec)}"); print("check completed (no data generation/training)"); return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage", nargs="?", choices=("check", "prepare", "train", "analyze", "identifiability", "inverse", "summarize", "all"), default="check"); parser.add_argument("--profile", choices=tuple(PROFILES), default="minimal"); parser.add_argument("--force", action="store_true"); args = parser.parse_args(argv)
    if args.stage == "check": return check(args.profile)
    if args.stage == "prepare": prepare(args.profile, args.force); return 0
    if args.stage == "train": train(args.profile, args.force); return 0
    if args.stage == "analyze": analyze(args.profile, args.force); return 0
    if args.stage == "identifiability": identifiability(args.profile, args.force); return 0
    if args.stage == "inverse": inverse_render(args.profile, args.force); return 0
    if args.stage == "summarize": summarize(args.profile, args.force); return 0
    if args.stage == "all": prepare(args.profile, args.force); train(args.profile, args.force); analyze(args.profile, args.force); identifiability(args.profile, args.force); inverse_render(args.profile, args.force); summarize(args.profile, args.force); return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
