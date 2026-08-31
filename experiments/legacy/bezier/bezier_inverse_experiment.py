"""Cubic-Bezier inverse rendering experiment for a randomly initialized ResNet18.

The experiment is intentionally self contained.  A split is rendered once with
Pillow (4x supersampling followed by LANCZOS downsampling), frozen as a uint8
single-channel NPZ file, and then consumed by the training and analysis stages.
The default ``full`` profile is deliberately large, while ``smoke`` is useful
for checking the complete pipeline without committing to a long run::

    python bezier_inverse_experiment.py check
    python bezier_inverse_experiment.py prepare --profile smoke
    python bezier_inverse_experiment.py train --profile smoke
    python bezier_inverse_experiment.py analyze --profile smoke
    python bezier_inverse_experiment.py identifiability --profile smoke

No stage implicitly touches the older ``results/main`` or ``results/baseline``
artifacts.  All files written by this script live below
``results/bezier_inverse/{profile}``.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED = 20260810
IMAGE_SIZE = 224
SUPER_SAMPLE = 4
# A spelling alias is kept for small external scripts that use the term from
# the design document.
SUPERSAMPLE = SUPER_SAMPLE
STROKE_WIDTH = 3.0
CURVE_SAMPLES = 256
VISIBLE_THRESHOLD = 0.65
MIN_X_GAP = 0.20
MIN_FOREGROUND_PIXELS = 12
ENDPOINT_MIN = 0.15
ENDPOINT_MAX = 0.85
CONTROL_SUPPORT_MIN = -0.25
CONTROL_SUPPORT_MAX = 1.25
EXTRA_LOW_MIN = -0.40
EXTRA_LOW_MAX = -0.25
EXTRA_HIGH_MIN = 1.25
EXTRA_HIGH_MAX = 1.40
COORDINATE_DIM = 8
TARGET_NAMES = ("p0x", "p0y", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y")
ENDPOINT_INDICES = (0, 1, 6, 7)
CONTROL_INDICES = (2, 3, 4, 5)
TEST_SPLITS = ("test_id", "test_hole", "test_extra")


@dataclass(frozen=True)
class Profile:
    """Dataset/training sizes for one reproducible profile."""

    name: str
    train_count: int
    val_count: int
    test_count: int
    epochs: int
    batch_size: int
    patience: int
    identifiability_count: int = 200


PROFILE_CONFIGS: Dict[str, Profile] = {
    # The three test sets each contain 1,000 samples in the formal run.
    "full": Profile("full", 8000, 1000, 1000, 30, 64, 5, 200),
    "reduced": Profile("reduced", 4000, 500, 500, 20, 64, 6, 200),
    # Smoke has 500 train and 100 in validation and in each test split.
    "smoke": Profile("smoke", 500, 100, 100, 2, 32, 3, 100),
}

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "bezier_inverse"


def profile_spec(profile: str | Profile) -> Profile:
    if isinstance(profile, Profile):
        return profile
    try:
        return PROFILE_CONFIGS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILE_CONFIGS)}") from exc


def output_root(profile: str | Profile) -> Path:
    return RESULTS_ROOT / profile_spec(profile).name


def output_dirs(profile: str | Profile) -> Dict[str, Path]:
    root = output_root(profile)
    return {
        "root": root,
        "config": root / "config",
        "manifest": root / "manifest",
        "data": root / "data",
        "checkpoints": root / "checkpoints",
        "tables": root / "tables",
        "figures": root / "figures",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def experiment_config(profile: str | Profile) -> Dict[str, Any]:
    spec = profile_spec(profile)
    return {
        "schema_version": 1,
        "seed": SEED,
        "profile": asdict(spec),
        "image_size": IMAGE_SIZE,
        "supersample": SUPER_SAMPLE,
        "resampling": "LANCZOS",
        "stroke_width": STROKE_WIDTH,
        "curve_samples": CURVE_SAMPLES,
        "visible_threshold": VISIBLE_THRESHOLD,
        "canonical": {
            "p0x_less_than_p3x": True,
            "minimum_x_gap": MIN_X_GAP,
            "endpoint_coordinate_range_strict": [0.15, 0.85],
            "canonicalize_reverse_control_order": True,
        },
        "target_names": list(TARGET_NAMES),
        "test_splits": list(TEST_SPLITS),
        "model": {
            "name": "torchvision.resnet18",
            "weights": None,
            "keep_avgpool": True,
            "head": "Linear(512, 8)",
            "sigmoid": False,
        },
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "scheduler": "ReduceLROnPlateau", "scheduler_patience": 2},
        "loss": {"name": "SmoothL1Loss", "beta": 0.02},
        "amp": True,
        "early_stopping": {"patience": spec.patience, "min_delta": 1e-5, "min_epochs": 10},
        "generation": {
            "train_val_exclude_hole": True,
            "id_independent": True,
            "strict_hole_and_extra": True,
            "visible_fraction_is_arc_length_weighted": True,
            "p0_p3_support": [0.15, 0.85],
            "p1_p2_train_val_id_support": [-0.25, 1.25],
            "hole_definition": "p1x in [0.35,0.55] AND p2y in [0.45,0.65]",
            "extra_definition": "any P1/P2 coordinate in [-0.40,-0.25) or (1.25,1.40]",
        },
    }


def config_fingerprint(profile: str | Profile) -> str:
    payload = json.dumps(_jsonable(experiment_config(profile)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Reproducibility and small output helpers
# ---------------------------------------------------------------------------


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


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - environment-specific
        return f"unavailable: {exc}"


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionaries with a deterministic union of field names."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


# Existing project scripts call this helper by its private spelling; retaining
# the alias makes the style familiar without importing the older experiment.
_write_rows = write_rows


def write_run_metadata(profile: str | Profile, stage: str, extra: Mapping[str, Any] | None = None) -> None:
    """Record stage/environment metadata inside the profile-specific config dir."""

    dirs = output_dirs(profile)
    dirs["config"].mkdir(parents=True, exist_ok=True)
    path = dirs["config"] / "run_metadata.json"
    previous: Dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    now = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
    entry: Dict[str, Any] = {
        "stage": stage,
        "timestamp_utc": now,
        "profile": profile_spec(profile).name,
        "device": str(choose_device()),
        "fingerprint": config_fingerprint(profile),
    }
    if extra:
        entry.update(dict(extra))
    history = list(previous.get("stage_history", []))
    history.append(entry)
    payload = {
        "last_stage": stage,
        "last_updated_utc": now,
        "stage_history": history,
        "fingerprint": config_fingerprint(profile),
        "config": experiment_config(profile),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": _safe_package_version("torch"),
            "torchvision": _safe_package_version("torchvision"),
            "numpy": _safe_package_version("numpy"),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_config(profile: str | Profile) -> None:
    dirs = output_dirs(profile)
    dirs["config"].mkdir(parents=True, exist_ok=True)
    config = experiment_config(profile)
    config["fingerprint"] = config_fingerprint(profile)
    (dirs["config"] / "config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _split_seed(profile: str | Profile, split: str) -> int:
    material = f"{SEED}:{profile_spec(profile).name}:{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**32 - 1)


# ---------------------------------------------------------------------------
# Cubic Bezier geometry and Pillow renderer
# ---------------------------------------------------------------------------


def bezier_points(parameters: Sequence[float], count: int = CURVE_SAMPLES) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    t = np.linspace(0.0, 1.0, int(count), dtype=np.float64)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * values[0]
        + 3.0 * omt**2 * t * values[1]
        + 3.0 * omt * t**2 * values[2]
        + t**3 * values[3]
    )


def _visibility_stats(parameters: Sequence[float], count: int = CURVE_SAMPLES) -> Dict[str, Any]:
    """Return arc-length weighted visibility and simple geometric diagnostics.

    A point-count ratio can over-weight a slowly moving part of a curve.  We
    therefore classify segment midpoints and weight each segment by its length;
    this remains deterministic and is a close numerical approximation to the
    visible arc-length fraction.
    """

    points = bezier_points(parameters, count)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    midpoints = (points[:-1] + points[1:]) * 0.5
    inside_midpoints = np.logical_and.reduce(
        (midpoints[:, 0] >= 0.0, midpoints[:, 0] <= 1.0, midpoints[:, 1] >= 0.0, midpoints[:, 1] <= 1.0)
    )
    visible_length = float(np.sum(segment_lengths[inside_midpoints]))
    fraction = visible_length / total_length if total_length > 1e-15 else 0.0
    return {"visible_fraction": float(fraction), "curve_length": total_length}


def visible_fraction(parameters: Sequence[float], count: int = CURVE_SAMPLES) -> float:
    return float(_visibility_stats(parameters, count)["visible_fraction"])


def is_canonical(parameters: Sequence[float]) -> bool:
    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    endpoint_range = np.all((values[[0, 3]] > ENDPOINT_MIN) & (values[[0, 3]] < ENDPOINT_MAX))
    return bool(
        endpoint_range
        and values[0, 0] < values[3, 0]
        and values[3, 0] - values[0, 0] >= MIN_X_GAP - 1e-12
    )


def canonicalize_parameters(parameters: Sequence[float]) -> np.ndarray:
    """Canonicalize the whole control-point order, including P1/P2.

    Reversing a cubic is ``P0,P1,P2,P3 -> P3,P2,P1,P0``.  Canonicalization is
    performed before split predicates, so a reversal cannot silently move a
    hole/extra sample into a different class.
    """

    values = np.asarray(parameters, dtype=np.float32).reshape(4, 2).copy()
    if values[0, 0] > values[3, 0]:
        values = values[::-1].copy()
    return values.reshape(-1)


def is_non_degenerate(parameters: Sequence[float]) -> bool:
    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(values)):
        return False
    points = bezier_points(values, 64)
    chord = float(np.linalg.norm(values[3] - values[0]))
    arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    span = np.ptp(points, axis=0)
    # Canonical x-gap already supplies the useful lower bound.  The extra
    # checks reject collapsed control polygons and completely flat curves.
    return bool(chord >= MIN_X_GAP and arc >= chord * (1.0 + 1e-4) and np.max(span) > 1e-3)


def _control_fractions(values: np.ndarray) -> Tuple[float, float]:
    gap = max(float(values[3, 0] - values[0, 0]), 1e-8)
    return (
        float((values[1, 0] - values[0, 0]) / gap),
        float((values[2, 0] - values[0, 0]) / gap),
    )


def is_hole(parameters: Sequence[float]) -> bool:
    """User-defined hole: ``P1.x`` band AND ``P2.y`` band."""
    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    tol = 1e-6
    return bool(0.35 - tol <= values[1, 0] <= 0.55 + tol and 0.45 - tol <= values[2, 1] <= 0.65 + tol)


def is_extra(parameters: Sequence[float]) -> bool:
    """Extra split flag based on support, not on canvas visibility."""
    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    controls = values[1:3]
    return bool(
        np.any((controls >= EXTRA_LOW_MIN) & (controls < EXTRA_LOW_MAX))
        or np.any((controls > EXTRA_HIGH_MIN) & (controls <= EXTRA_HIGH_MAX))
    )


def has_canvas_outside_coordinate(parameters: Sequence[float]) -> bool:
    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    return bool(np.any((values < 0.0) | (values > 1.0)))


def render_bezier(parameters: Sequence[float]) -> np.ndarray:
    """Render a cubic Bezier as a uint8 ``(224, 224)`` grayscale image.

    Rendering happens on a 4x canvas with a high-quality Pillow polyline and is
    reduced using LANCZOS.  The use of a single ``L`` image guarantees one
    channel and avoids an accidental RGB/RGBA NPZ cache.
    """

    from PIL import Image, ImageDraw

    points = bezier_points(parameters, CURVE_SAMPLES)
    scale = float(SUPER_SAMPLE)
    high_size = IMAGE_SIZE * SUPER_SAMPLE
    image = Image.new("L", (high_size, high_size), color=0)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * (IMAGE_SIZE - 1) * scale, float(y) * (IMAGE_SIZE - 1) * scale) for x, y in points]
    width = max(1, int(round(STROKE_WIDTH * scale)))
    try:
        draw.line(xy, fill=255, width=width, joint="curve")
    except TypeError:  # older Pillow without ``joint``
        draw.line(xy, fill=255, width=width)
    resampling = getattr(Image, "Resampling", Image)
    result = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling.LANCZOS)
    array = np.asarray(result, dtype=np.uint8)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE) or array.dtype != np.uint8:
        raise RuntimeError(f"renderer invariant failed: shape={array.shape}, dtype={array.dtype}")
    return array


# A backwards-friendly descriptive alias used by a few notebooks.
render_curve = render_bezier


def _candidate_reasons(parameters: Sequence[float], split: str) -> List[str]:
    """Return rejection reasons; an empty list means the candidate is valid."""

    values = np.asarray(parameters, dtype=np.float64).reshape(4, 2)
    reasons: List[str] = []
    if not is_canonical(values):
        reasons.append("noncanonical")
    if not is_non_degenerate(values):
        reasons.append("degenerate")
    if visible_fraction(values) < VISIBLE_THRESHOLD:
        reasons.append("not_visible")
    hole = is_hole(values)
    extra = is_extra(values)
    endpoint_values = values[[0, 3]]
    controls = values[1:3]
    endpoint_valid = bool(np.all((endpoint_values > ENDPOINT_MIN) & (endpoint_values < ENDPOINT_MAX)) and values[0, 0] < values[3, 0])
    control_in_training_support = bool(np.all((controls >= CONTROL_SUPPORT_MIN) & (controls <= CONTROL_SUPPORT_MAX)))
    if not endpoint_valid and "noncanonical" not in reasons:
        reasons.append("endpoint_support")
    if split in ("train", "val", "test_id", "test_hole") and not control_in_training_support:
        reasons.append("control_support")
    if split in ("train", "val", "test_id"):
        if hole:
            reasons.append("hole_excluded")
        if extra:
            reasons.append("extra_excluded")
    elif split == "test_hole":
        if not hole:
            reasons.append("not_hole")
        if extra:
            reasons.append("extra_excluded")
    elif split == "test_extra":
        if not extra:
            reasons.append("not_extra")
        if hole:
            reasons.append("hole_excluded")
    else:
        raise ValueError(f"unknown split {split!r}")
    return reasons


def _candidate(parameters: Sequence[float], split: str) -> bool:
    return not _candidate_reasons(parameters, split)


def _sample_parameters(split: str, rng: np.random.Generator) -> np.ndarray:
    """Draw one split-specific candidate; acceptance is checked separately."""

    # Endpoints always live strictly in the compact support.  The random
    # orientation is canonicalized below, rather than assuming the sampler's
    # initial ordering is already left-to-right.
    p0x = float(rng.uniform(ENDPOINT_MIN + 1e-3, ENDPOINT_MAX - 1e-3))
    p3x = float(rng.uniform(ENDPOINT_MIN + 1e-3, ENDPOINT_MAX - 1e-3))
    p0y = float(rng.uniform(ENDPOINT_MIN + 1e-3, ENDPOINT_MAX - 1e-3))
    p3y = float(rng.uniform(ENDPOINT_MIN + 1e-3, ENDPOINT_MAX - 1e-3))
    # All in-domain controls share the same support.  Hole and extra are then
    # strict constructions over this common base distribution.
    f1, f2 = rng.uniform(0.15, 0.85, size=2)
    c1x = float(rng.uniform(CONTROL_SUPPORT_MIN, CONTROL_SUPPORT_MAX))
    c2x = float(rng.uniform(CONTROL_SUPPORT_MIN, CONTROL_SUPPORT_MAX))
    c1y = float(rng.uniform(CONTROL_SUPPORT_MIN, CONTROL_SUPPORT_MAX))
    c2y = float(rng.uniform(CONTROL_SUPPORT_MIN, CONTROL_SUPPORT_MAX))
    if split == "test_hole":
        c1x = float(rng.uniform(0.35, 0.55))
        c2y = float(rng.uniform(0.45, 0.65))
    elif split == "test_extra":
        extra_value = float(rng.choice([rng.uniform(EXTRA_LOW_MIN, EXTRA_LOW_MAX), rng.uniform(EXTRA_HIGH_MIN, EXTRA_HIGH_MAX)]))
        # Select one of the four P1/P2 coordinates for the support extension;
        # every other control coordinate remains in training support.
        slot = int(rng.integers(0, 4))
        controls = np.asarray((c1x, c1y, c2x, c2y), dtype=np.float32)
        controls[slot] = extra_value
        c1x, c1y, c2x, c2y = [float(v) for v in controls]

    parameters = np.asarray((p0x, p0y, c1x, c1y, c2x, c2y, p3x, p3y), dtype=np.float32)
    return canonicalize_parameters(parameters)


def generate_sample(split: str, rng: np.random.Generator, max_attempts: int = 20000) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate and render one accepted sample with auditable filter flags."""

    rejected: Dict[str, int] = {}
    for attempts in range(1, max_attempts + 1):
        parameters = _sample_parameters(split, rng)
        reasons = _candidate_reasons(parameters, split)
        if reasons:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
            continue
        image = render_bezier(parameters)
        visibility = _visibility_stats(parameters)
        foreground = int(np.count_nonzero(image >= 8))
        if foreground < MIN_FOREGROUND_PIXELS:
            rejected["foreground_too_small"] = rejected.get("foreground_too_small", 0) + 1
            continue
        ys, xs = np.nonzero(image >= 8)
        record = {
            "visible_fraction": visibility["visible_fraction"],
            "curve_length": visibility["curve_length"],
            "foreground_pixels": foreground,
            "foreground_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "non_degenerate": is_non_degenerate(parameters),
            "canonical": is_canonical(parameters),
            "hole": is_hole(parameters),
            "extra": is_extra(parameters),
            "canvas_outside": has_canvas_outside_coordinate(parameters),
            "attempts": attempts,
            "reject_reasons": rejected,
        }
        return image, {"parameters": parameters, **record}
    raise RuntimeError(f"unable to generate accepted {split} sample after {max_attempts} attempts")


def _split_count(spec: Profile, split: str) -> int:
    if split == "train":
        return spec.train_count
    if split == "val":
        return spec.val_count
    return spec.test_count


def _generate_split(profile: str | Profile, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    count = _split_count(profile_spec(profile), split)
    rng = np.random.default_rng(_split_seed(profile, split))
    images = np.empty((count, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    targets = np.empty((count, COORDINATE_DIM), dtype=np.float32)
    ids = np.asarray([f"{profile_spec(profile).name}-{split}-{i:06d}" for i in range(count)], dtype="U64")
    records: List[Dict[str, Any]] = []
    for index in range(count):
        image, record = generate_sample(split, rng)
        parameters = np.asarray(record.pop("parameters"), dtype=np.float32)
        images[index] = image
        targets[index] = parameters
        records.append({"id": str(ids[index]), "split": split, **record})
    if images.dtype != np.uint8 or images.ndim != 3:
        raise RuntimeError("offline image invariant failed")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"duplicate IDs inside split {split}")
    return images, targets, ids, records


def _load_manifest(profile: str | Profile) -> Dict[str, Any]:
    path = output_dirs(profile)["manifest"] / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest {path}; run prepare first")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid manifest {path}: {exc}") from exc
    expected = config_fingerprint(profile)
    if manifest.get("fingerprint") != expected:
        raise RuntimeError(
            f"manifest fingerprint mismatch ({manifest.get('fingerprint')} != {expected}); use prepare --force"
        )
    return manifest


def _load_npz(profile: str | Profile, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = output_dirs(profile)["data"] / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen split {path}; run prepare first")
    with np.load(path, allow_pickle=False) as payload:
        images = np.asarray(payload["images"])
        targets = np.asarray(payload["targets"])
        ids = np.asarray(payload["ids"])
        cache_fingerprint = str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) if "fingerprint" in payload else ""
    expected_fingerprint = config_fingerprint(profile)
    if cache_fingerprint != expected_fingerprint:
        raise RuntimeError(f"NPZ fingerprint mismatch for {path}: {cache_fingerprint!r} != {expected_fingerprint!r}")
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"invalid image cache {path}: shape={images.shape}, dtype={images.dtype}")
    if targets.dtype not in (np.float32, np.float64) or targets.shape != (len(images), COORDINATE_DIM):
        raise RuntimeError(f"invalid target cache {path}: shape={targets.shape}, dtype={targets.dtype}")
    if ids.shape[0] != len(images):
        raise RuntimeError(f"invalid ID cache {path}")
    return images, targets.astype(np.float32, copy=False), ids


def prepare_data(profile: str = "full", force: bool = False) -> Dict[str, Any]:
    """Generate all frozen NPZ splits and the auditable manifest."""

    spec = profile_spec(profile)
    dirs = output_dirs(spec)
    manifest_path = dirs["manifest"] / "manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == config_fingerprint(spec):
            print(f"[prepare] cache already ready: {dirs['root']} (use --force to regenerate)")
            return existing
        raise RuntimeError(f"existing cache fingerprint mismatch at {manifest_path}; use --force")

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    _write_config(spec)
    split_entries: Dict[str, Any] = {}
    all_ids: List[str] = []
    raster_hash_owner: Dict[str, str] = {}
    parameter_hash_owner: Dict[str, str] = {}
    cross_split_raster_collisions: List[Dict[str, str]] = []
    cross_split_parameter_collisions: List[Dict[str, str]] = []
    for split in ("train", "val", *TEST_SPLITS):
        images, targets, ids, records = _generate_split(spec, split)
        npz_path = dirs["data"] / f"{split}.npz"
        np.savez_compressed(
            npz_path,
            images=images,
            targets=targets,
            ids=ids,
            fingerprint=np.asarray(config_fingerprint(spec), dtype="U32"),
        )
        # Keep one compact per-split manifest alongside the aggregate manifest.
        (dirs["manifest"] / f"{split}.json").write_text(
            json.dumps(_jsonable({"split": split, "records": records}), ensure_ascii=False), encoding="utf-8"
        )
        values = targets.reshape(-1, 4, 2)
        raster_hashes = [hashlib.sha256(images[i].tobytes()).hexdigest() for i in range(len(images))]
        parameter_hashes = [hashlib.sha256(np.asarray(targets[i], dtype=np.float32).tobytes()).hexdigest() for i in range(len(targets))]
        if len(set(raster_hashes)) != len(raster_hashes) or len(set(parameter_hashes)) != len(parameter_hashes):
            raise RuntimeError(f"within-split exact collision in {split}")
        for digest in raster_hashes:
            owner = raster_hash_owner.get(digest)
            if owner is not None and owner != split:
                cross_split_raster_collisions.append({"hash": digest, "first_split": owner, "second_split": split})
            else:
                raster_hash_owner[digest] = split
        for digest in parameter_hashes:
            owner = parameter_hash_owner.get(digest)
            if owner is not None and owner != split:
                cross_split_parameter_collisions.append({"hash": digest, "first_split": owner, "second_split": split})
            else:
                parameter_hash_owner[digest] = split
        reject_counts: Dict[str, int] = {}
        for record in records:
            for reason, value in record.get("reject_reasons", {}).items():
                reject_counts[reason] = reject_counts.get(reason, 0) + int(value)
        foreground_values = [int(record["foreground_pixels"]) for record in records]
        bboxes = [record["foreground_bbox"] for record in records if record.get("foreground_bbox") is not None]
        entry = {
            "count": int(len(ids)),
            "npz": str(npz_path.relative_to(dirs["root"])),
            "ids": [str(value) for value in ids.tolist()],
            "image_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "npz_fingerprint": config_fingerprint(spec),
            "target_shape": list(targets.shape),
            "target_dtype": str(targets.dtype),
            "visible_fraction_min": float(min(r["visible_fraction"] for r in records)),
            "visible_fraction_mean": float(np.mean([r["visible_fraction"] for r in records])),
            "curve_length_min": float(min(r["curve_length"] for r in records)),
            "curve_length_mean": float(np.mean([r["curve_length"] for r in records])),
            "foreground_pixels_min": int(min(foreground_values)),
            "foreground_pixels_mean": float(np.mean(foreground_values)),
            "foreground_bbox_union": [
                int(min(box[0] for box in bboxes)),
                int(min(box[1] for box in bboxes)),
                int(max(box[2] for box in bboxes)),
                int(max(box[3] for box in bboxes)),
            ] if bboxes else None,
            "reject_counts": reject_counts,
            "raster_hash_unique": len(set(raster_hashes)) == len(raster_hashes),
            "parameter_hash_unique": len(set(parameter_hashes)) == len(parameter_hashes),
            "raster_hashes": raster_hashes,
            "parameter_hashes": parameter_hashes,
            "canonical_all": bool(all(r["canonical"] for r in records)),
            "non_degenerate_all": bool(all(r["non_degenerate"] for r in records)),
            "hole_count": int(sum(bool(r["hole"]) for r in records)),
            "extra_count": int(sum(bool(r["extra"]) for r in records)),
            "canvas_outside_count": int(sum(bool(r["canvas_outside"]) for r in records)),
            "target_min": values.min(axis=(0, 1)).tolist(),
            "target_max": values.max(axis=(0, 1)).tolist(),
            "endpoint_support_observed_min": values[:, [0, 3]].min(axis=(0, 1)).tolist(),
            "endpoint_support_observed_max": values[:, [0, 3]].max(axis=(0, 1)).tolist(),
            "control_support_observed_min": values[:, [1, 2]].min(axis=(0, 1)).tolist(),
            "control_support_observed_max": values[:, [1, 2]].max(axis=(0, 1)).tolist(),
        }
        split_entries[split] = entry
        all_ids.extend(str(value) for value in ids.tolist())
        print(f"[prepare] {split}: {len(ids)} frozen samples -> {npz_path}")

    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("ID independence invariant failed across splits")
    for split in ("train", "val"):
        if split_entries[split]["hole_count"] != 0:
            raise RuntimeError(f"{split} unexpectedly contains held-out hole samples")
    if split_entries["test_hole"]["hole_count"] != split_entries["test_hole"]["count"]:
        raise RuntimeError("test_hole is not strict")
    if split_entries["test_extra"]["extra_count"] != split_entries["test_extra"]["count"]:
        raise RuntimeError("test_extra is not strict")
    if split_entries["test_id"]["hole_count"] or split_entries["test_id"]["extra_count"]:
        raise RuntimeError("test_id unexpectedly contains hole/extra samples")
    if split_entries["test_extra"]["hole_count"]:
        raise RuntimeError("test_extra unexpectedly overlaps hole samples")
    for split in ("test_id", "test_extra"):
        if split_entries[split]["hole_count"] != 0:
            raise RuntimeError(f"{split} unexpectedly contains held-out hole samples")
    if split_entries["test_id"]["extra_count"] != 0:
        raise RuntimeError("test_id unexpectedly contains extrapolation-support samples")
    if cross_split_raster_collisions or cross_split_parameter_collisions:
        raise RuntimeError(
            f"cross-split exact collision: raster={len(cross_split_raster_collisions)}, "
            f"parameters={len(cross_split_parameter_collisions)}"
        )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "profile": spec.name,
        "fingerprint": config_fingerprint(spec),
        "seed": SEED,
        "counts": {split: entry["count"] for split, entry in split_entries.items()},
        "splits": split_entries,
        "id_count": len(all_ids),
        "id_unique": len(set(all_ids)) == len(all_ids),
        "generation_invariants": {
            "canonical": True,
            "minimum_x_gap": MIN_X_GAP,
            "visible_fraction_minimum": VISIBLE_THRESHOLD,
            "visible_fraction_definition": "arc-length weighted segment-midpoint centerline fraction",
            "non_degenerate": True,
            "train_val_exclude_hole": True,
            "strict_test_hole": True,
            "strict_test_extra": True,
            "arc_length_weighted_visible_fraction": True,
            "foreground_pixels_minimum": MIN_FOREGROUND_PIXELS,
        },
        "cross_split_collisions": {
            "raster": cross_split_raster_collisions,
            "parameters": cross_split_parameter_collisions,
        },
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_metadata(spec, "prepare", {"counts": manifest["counts"], "id_unique": manifest["id_unique"]})
    print(f"[prepare] complete: {dirs['root']} fingerprint={manifest['fingerprint']}")
    return manifest


# ---------------------------------------------------------------------------
# Model and offline training
# ---------------------------------------------------------------------------


def build_model() -> nn.Module:
    """Return the standard torchvision ResNet18 with only an 8-D linear head."""

    from torchvision import models

    try:
        model = models.resnet18(weights=None)
    except TypeError:  # torchvision < 0.13 compatibility
        model = models.resnet18(pretrained=False)
    # Keep the canonical AdaptiveAvgPool2d((1, 1)); replacing only ``fc`` is
    # important because the experiment is about the inverse geometry, not a
    # larger spatial head.
    model.fc = nn.Linear(512, COORDINATE_DIM)
    return model


class OfflineBezierDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, targets: np.ndarray) -> None:
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep the frozen cache single-channel, then replicate at the model
        # boundary because an unmodified torchvision ResNet18 stem expects
        # three channels.  No learnable/architectural stem change is made.
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        target = torch.from_numpy(self.targets[index]).float()
        return image, target


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        from contextlib import nullcontext

        return nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - old torch fallback
        from contextlib import nullcontext

        return nullcontext()


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old torch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _predict_model(
    model: nn.Module,
    images: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> Tuple[float, np.ndarray]:
    loader = DataLoader(OfflineBezierDataset(images, targets), batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.SmoothL1Loss(beta=0.02, reduction="sum")
    model.eval()
    total = 0.0
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for batch_images, batch_targets in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                prediction = model(batch_images)
                loss = criterion(prediction, batch_targets)
            total += float(loss.detach().item())
            outputs.append(prediction.detach().float().cpu().numpy())
    return total / max(1, len(targets) * COORDINATE_DIM), np.concatenate(outputs, axis=0)


def _plot_training_history(profile: str | Profile, history: Sequence[Mapping[str, Any]]) -> None:
    """Save the required loss and coordinate-MAE learning curves."""

    if not history:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epochs = np.asarray([int(row["epoch"]) for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_smooth_l1"]) for row in history], "o-", label="train (online)")
    axes[0].plot(epochs, [float(row["val_smooth_l1"]) for row in history], "o-", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("SmoothL1 loss per coordinate")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [float(row["train_coordinate_mae_px"]) for row in history], "o-", label="train (online)")
    axes[1].plot(epochs, [float(row["val_coordinate_mae_px"]) for row in history], "o-", label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("coordinate MAE (pixels)")
    axes[1].set_title("Training and validation coordinate MAE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    figure_dir = output_dirs(profile)["figures"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "training_history.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def train(profile: str = "full", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    manifest = _load_manifest(spec)
    dirs = output_dirs(spec)
    best_path = dirs["checkpoints"] / "resnet18_best.pt"
    if best_path.exists() and not force:
        raise RuntimeError(f"checkpoint exists at {best_path}; use --force to retrain")
    dirs["checkpoints"].mkdir(parents=True, exist_ok=True)
    train_images, train_targets, _ = _load_npz(spec, "train")
    val_images, val_targets, _ = _load_npz(spec, "val")
    device = choose_device()
    amp_enabled = bool(torch.cuda.is_available() and device.type == "cuda")
    seed_everything(SEED)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = _make_scaler(amp_enabled)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        OfflineBezierDataset(train_images, train_targets),
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=(device.type == "cuda"),
    )
    history: List[Dict[str, Any]] = []
    best_val = math.inf
    best_epoch = 0
    wait = 0
    for epoch in range(1, spec.epochs + 1):
        model.train()
        running = 0.0
        running_absolute_error = 0.0
        seen = 0
        for batch_images, batch_targets in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                prediction = model(batch_images)
                loss = criterion(prediction, batch_targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().item()) * len(batch_images)
            running_absolute_error += float(torch.sum(torch.abs(prediction.detach() - batch_targets)).item())
            seen += len(batch_images)
        train_loss = running / max(1, seen)
        val_loss, val_predictions = _predict_model(model, val_images, val_targets, device, spec.batch_size, amp_enabled)
        scheduler.step(val_loss)
        val_coordinate_mae_px = float(np.mean(np.abs(val_predictions - val_targets)) * (IMAGE_SIZE - 1))
        # Training MAE is accumulated online in train mode to avoid a second
        # full 8,000-image pass every epoch. Validation remains a complete
        # eval-mode pass and is the only signal used for scheduling/selection.
        train_coordinate_mae_px = running_absolute_error / max(1, seen * COORDINATE_DIM) * (IMAGE_SIZE - 1)
        row = {
            "epoch": epoch,
            "train_smooth_l1": train_loss,
            "val_smooth_l1": val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_coordinate_mae_px": train_coordinate_mae_px,
            "train_coordinate_mae_scope": "online_train_mode",
            "val_coordinate_mae_px": val_coordinate_mae_px,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"[train:{spec.name}] epoch {epoch:03d}/{spec.epochs}: train={train_loss:.6g} val={val_loss:.6g}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            payload = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_smooth_l1": val_loss,
                "profile": spec.name,
                "fingerprint": config_fingerprint(spec),
                "target_names": list(TARGET_NAMES),
            }
            torch.save(payload, best_path)
            # The shorter alias is convenient for command-line users and is
            # intentionally identical, not a second independently trained run.
            torch.save(payload, dirs["checkpoints"] / "best.pt")
        else:
            wait += 1
            min_epochs = min(10, spec.epochs)
            if epoch >= min_epochs and wait >= spec.patience:
                print(f"[train:{spec.name}] early stop after {epoch} epochs (patience={spec.patience})")
                break
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": history[-1]["epoch"] if history else 0,
            "profile": spec.name,
            "fingerprint": config_fingerprint(spec),
        },
        dirs["checkpoints"] / "last.pt",
    )
    write_rows(dirs["tables"] / "history.csv", history)
    _plot_training_history(spec, history)
    summary = {
        "profile": spec.name,
        "fingerprint": config_fingerprint(spec),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_smooth_l1": best_val,
        "device": str(device),
        "amp": amp_enabled,
    }
    write_rows(dirs["tables"] / "training_summary.csv", [summary])
    write_run_metadata(spec, "train", summary)
    print(f"[train] best checkpoint: {best_path}")
    return summary


def _load_checkpoint(profile: str | Profile, device: torch.device) -> nn.Module:
    dirs = output_dirs(profile)
    path = dirs["checkpoints"] / "resnet18_best.pt"
    if not path.exists():
        path = dirs["checkpoints"] / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing best checkpoint under {dirs['checkpoints']}; run train first")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("fingerprint") != config_fingerprint(profile):
        raise RuntimeError("checkpoint fingerprint mismatch; retrain with --force")
    model = build_model().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics, predictions and figures
# ---------------------------------------------------------------------------


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    error = predictions - targets
    absolute = np.abs(error)
    result: Dict[str, float] = {
        "count": float(len(targets)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae_px": float(np.mean(absolute) * (IMAGE_SIZE - 1)),
        "rmse_px": float(np.sqrt(np.mean(error**2)) * (IMAGE_SIZE - 1)),
        "maxabs": float(np.max(absolute)),
        "maxabs_px": float(np.max(absolute) * (IMAGE_SIZE - 1)),
        "endpoint_mae": float(np.mean(absolute[:, ENDPOINT_INDICES])),
        "endpoint_rmse": float(np.sqrt(np.mean(error[:, ENDPOINT_INDICES] ** 2))),
        "endpoint_mae_px": float(np.mean(absolute[:, ENDPOINT_INDICES]) * (IMAGE_SIZE - 1)),
        "endpoint_rmse_px": float(np.sqrt(np.mean(error[:, ENDPOINT_INDICES] ** 2)) * (IMAGE_SIZE - 1)),
        "control_mae": float(np.mean(absolute[:, CONTROL_INDICES])),
        "control_rmse": float(np.sqrt(np.mean(error[:, CONTROL_INDICES] ** 2))),
        "control_mae_px": float(np.mean(absolute[:, CONTROL_INDICES]) * (IMAGE_SIZE - 1)),
        "control_rmse_px": float(np.sqrt(np.mean(error[:, CONTROL_INDICES] ** 2)) * (IMAGE_SIZE - 1)),
    }
    true_inside = (targets >= 0.0) & (targets <= 1.0)
    pred_inside = (predictions >= 0.0) & (predictions <= 1.0)
    inside_mask = true_inside
    outside_mask = ~inside_mask
    result.update(
        {
            "target_inside_fraction": float(np.mean(true_inside)),
            "prediction_inside_fraction": float(np.mean(pred_inside)),
            "inside_mae": float(np.mean(absolute[inside_mask])) if np.any(inside_mask) else float("nan"),
            "outside_mae": float(np.mean(absolute[outside_mask])) if np.any(outside_mask) else float("nan"),
            "inside_count": float(np.sum(inside_mask)),
            "outside_count": float(np.sum(outside_mask)),
            "pred_inside_when_target_inside": float(np.mean(pred_inside[inside_mask])) if np.any(inside_mask) else float("nan"),
            "pred_inside_when_target_outside": float(np.mean(pred_inside[outside_mask])) if np.any(outside_mask) else float("nan"),
        }
    )
    result["inside_mae_px"] = result["inside_mae"] * (IMAGE_SIZE - 1)
    result["outside_mae_px"] = result["outside_mae"] * (IMAGE_SIZE - 1)
    # Control-point inside/outside metrics are reported independently for P1
    # and P2, as these are the coordinates deliberately exercised by the
    # hole/extra splits.  Point-level groups keep the x/y errors together.
    control_targets = targets[:, CONTROL_INDICES].reshape(-1, 2, 2)
    control_errors = absolute[:, CONTROL_INDICES].reshape(-1, 2, 2)
    control_point_inside = np.all((control_targets >= 0.0) & (control_targets <= 1.0), axis=2)
    control_point_outside = ~control_point_inside
    result["control_inside_point_mae"] = float(np.mean(control_errors[control_point_inside])) if np.any(control_point_inside) else float("nan")
    result["control_outside_point_mae"] = float(np.mean(control_errors[control_point_outside])) if np.any(control_point_outside) else float("nan")
    result["control_inside_point_mae_px"] = result["control_inside_point_mae"] * (IMAGE_SIZE - 1)
    result["control_outside_point_mae_px"] = result["control_outside_point_mae"] * (IMAGE_SIZE - 1)
    result["control_inside_point_count"] = float(np.sum(control_point_inside))
    result["control_outside_point_count"] = float(np.sum(control_point_outside))
    result["samples_with_outside_control_count"] = float(np.sum(np.any(control_point_outside, axis=1)))
    for point_name, point_indices in (("p1", (2, 3)), ("p2", (4, 5))):
        point_target = targets[:, point_indices]
        point_error = absolute[:, point_indices]
        point_inside = (point_target >= 0.0) & (point_target <= 1.0)
        point_outside = ~point_inside
        point_inside_rows = np.all(point_inside, axis=1)
        point_outside_rows = np.any(point_outside, axis=1)
        result[f"{point_name}_inside_mae"] = float(np.mean(point_error[point_inside])) if np.any(point_inside) else float("nan")
        result[f"{point_name}_outside_mae"] = float(np.mean(point_error[point_outside])) if np.any(point_outside) else float("nan")
        result[f"{point_name}_inside_mae_px"] = result[f"{point_name}_inside_mae"] * (IMAGE_SIZE - 1)
        result[f"{point_name}_outside_mae_px"] = result[f"{point_name}_outside_mae"] * (IMAGE_SIZE - 1)
        result[f"{point_name}_inside_count"] = float(np.sum(point_inside))
        result[f"{point_name}_outside_count"] = float(np.sum(point_outside))
        result[f"{point_name}_point_inside_mae"] = float(np.mean(point_error[point_inside_rows])) if np.any(point_inside_rows) else float("nan")
        result[f"{point_name}_point_outside_mae"] = float(np.mean(point_error[point_outside_rows])) if np.any(point_outside_rows) else float("nan")
        result[f"{point_name}_point_inside_count"] = float(np.sum(point_inside_rows))
        result[f"{point_name}_point_outside_count"] = float(np.sum(point_outside_rows))
    outside_coordinates = ~((targets >= 0.0) & (targets <= 1.0))
    result["outside_coordinate_only_mae"] = float(np.mean(absolute[outside_coordinates])) if np.any(outside_coordinates) else float("nan")
    result["outside_coordinate_only_mae_px"] = result["outside_coordinate_only_mae"] * (IMAGE_SIZE - 1)
    result["outside_coordinate_only_count"] = float(np.sum(outside_coordinates))
    # Extrapolation-support coordinates are stricter than merely being outside
    # the image: only P1/P2 values beyond the training support [-0.25, 1.25]
    # are included.  Direction accuracy asks whether a prediction crosses the
    # same low/high support boundary as its target.
    extra_support_mask = np.zeros_like(targets, dtype=bool)
    extra_support_mask[:, CONTROL_INDICES] = (
        (targets[:, CONTROL_INDICES] < CONTROL_SUPPORT_MIN)
        | (targets[:, CONTROL_INDICES] > CONTROL_SUPPORT_MAX)
    )
    result["extra_support_coordinate_count"] = float(np.sum(extra_support_mask))
    result["extra_support_coordinate_mae"] = (
        float(np.mean(absolute[extra_support_mask])) if np.any(extra_support_mask) else float("nan")
    )
    result["extra_support_coordinate_mae_px"] = result["extra_support_coordinate_mae"] * (IMAGE_SIZE - 1)
    if np.any(extra_support_mask):
        extra_true = targets[extra_support_mask]
        extra_pred = predictions[extra_support_mask]
        same_side = np.where(
            extra_true < CONTROL_SUPPORT_MIN,
            extra_pred < CONTROL_SUPPORT_MIN,
            extra_pred > CONTROL_SUPPORT_MAX,
        )
        result["extra_support_correct_side_fraction"] = float(np.mean(same_side))
    else:
        result["extra_support_correct_side_fraction"] = float("nan")
    # Auxiliary geometry error at corresponding t values, in normalized units
    # and pixels.  This catches a geometrically poor curve even if one control
    # coordinate happens to cancel another.
    curve_target = np.stack([bezier_points(row, 64) for row in targets], axis=0)
    curve_prediction = np.stack([bezier_points(row, 64) for row in predictions], axis=0)
    curve_error = curve_prediction - curve_target
    result["curve_t_mae"] = float(np.mean(np.abs(curve_error)))
    result["curve_t_rmse"] = float(np.sqrt(np.mean(curve_error**2)))
    result["curve_t_mae_px"] = result["curve_t_mae"] * (IMAGE_SIZE - 1)
    result["curve_t_rmse_px"] = result["curve_t_rmse"] * (IMAGE_SIZE - 1)
    for index, name in enumerate(TARGET_NAMES):
        result[f"{name}_mae"] = float(np.mean(absolute[:, index]))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[:, index] ** 2)))
        result[f"{name}_rmse_px"] = result[f"{name}_rmse"] * (IMAGE_SIZE - 1)
        result[f"{name}_mae_px"] = float(np.mean(absolute[:, index]) * (IMAGE_SIZE - 1))
        result[f"{name}_maxabs"] = float(np.max(absolute[:, index]))
        result[f"{name}_maxabs_px"] = result[f"{name}_maxabs"] * (IMAGE_SIZE - 1)
    return result


def _prediction_rows(split: str, ids: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, sample_id in enumerate(ids.tolist()):
        row: Dict[str, Any] = {"id": str(sample_id), "split": split}
        for col, name in enumerate(TARGET_NAMES):
            row[f"true_{name}"] = float(targets[index, col])
            row[f"pred_{name}"] = float(predictions[index, col])
            row[f"error_{name}"] = float(predictions[index, col] - targets[index, col])
            row[f"abs_error_{name}"] = float(abs(predictions[index, col] - targets[index, col]))
        row["true_endpoint_inside"] = bool(np.all((targets[index, ENDPOINT_INDICES] >= 0) & (targets[index, ENDPOINT_INDICES] <= 1)))
        row["pred_endpoint_inside"] = bool(np.all((predictions[index, ENDPOINT_INDICES] >= 0) & (predictions[index, ENDPOINT_INDICES] <= 1)))
        rows.append(row)
    return rows


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _savefig_aliases(fig: Any, directory: Path, names: Sequence[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig.savefig(directory / name, dpi=160, bbox_inches="tight")


def _plot_analysis_figures(
    profile: str | Profile,
    image_map: Mapping[str, np.ndarray],
    id_map: Mapping[str, np.ndarray],
    target_map: Mapping[str, np.ndarray],
    pred_map: Mapping[str, np.ndarray],
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    plt = _import_matplotlib()
    dirs = output_dirs(profile)
    splits = list(target_map)
    labels = [str(row["split"]) for row in metric_rows]
    maes = [float(row["mae_px"]) for row in metric_rows]
    endpoint = [float(row["endpoint_mae_px"]) for row in metric_rows]
    control = [float(row["control_mae_px"]) for row in metric_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, maes, width, label="all")
    ax.bar(x, endpoint, width, label="endpoint")
    ax.bar(x + width, control, width, label="control")
    ax.set_xticks(x, labels, rotation=25)
    ax.set_ylabel("MAE (pixels)")
    ax.set_title("Bezier inverse metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("metrics_overview.png", "metrics.png", "id_hole_extra_mae.png"))
    plt.close(fig)

    # Explicit per-coordinate MAE (pixels) bar chart.
    fig, ax = plt.subplots(figsize=(12, 5))
    per_width = 0.8 / max(1, len(labels))
    for split_index, split in enumerate(splits):
        row = metric_rows[split_index]
        values = [float(row[f"{name}_mae_px"]) for name in TARGET_NAMES]
        ax.bar(np.arange(COORDINATE_DIM) - 0.4 + (split_index + 0.5) * per_width, values, per_width, label=split)
    ax.set_xticks(np.arange(COORDINATE_DIM), TARGET_NAMES, rotation=35)
    ax.set_ylabel("MAE (pixels)")
    ax.set_title("Per-coordinate MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("per_coordinate_mae.png",))
    plt.close(fig)

    # Per-coordinate true-vs-predicted scatter.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    for j, name in enumerate(TARGET_NAMES):
        ax = axes.flat[j]
        all_true = np.concatenate([target_map[s][:, j] for s in splits])
        all_pred = np.concatenate([pred_map[s][:, j] for s in splits])
        ax.scatter(all_true, all_pred, s=4, alpha=0.22)
        lo = float(min(all_true.min(), all_pred.min()))
        hi = float(max(all_true.max(), all_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.2)
    fig.suptitle("Per-coordinate inverse predictions")
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("coordinate_scatter.png", "per_coordinate.png"))
    plt.close(fig)

    # One scatter panel per control point (P0/P1/P2/P3), with x and y
    # coordinates together as requested by the inverse-geometry report.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
    merged_true = np.concatenate([target_map[s].reshape(-1, 4, 2) for s in splits], axis=0)
    merged_pred = np.concatenate([pred_map[s].reshape(-1, 4, 2) for s in splits], axis=0)
    for point_index, ax in enumerate(axes.flat):
        for coord, marker, label in ((0, "o", "x"), (1, "^", "y")):
            ax.scatter(merged_true[:, point_index, coord], merged_pred[:, point_index, coord], s=5, alpha=0.25, marker=marker, label=label)
        lo = float(min(merged_true[:, point_index].min(), merged_pred[:, point_index].min()))
        hi = float(max(merged_true[:, point_index].max(), merged_pred[:, point_index].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set_title(f"P{point_index}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Control-point scatter")
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("point_scatter.png", "control_point_scatter.png"))
    plt.close(fig)

    # Endpoint/control error distributions.
    fig, ax = plt.subplots(figsize=(10, 5))
    for split in splits:
        errors = pred_map[split] - target_map[split]
        ax.hist(np.mean(np.abs(errors[:, ENDPOINT_INDICES]), axis=1) * (IMAGE_SIZE - 1), bins=30, alpha=0.35, label=f"{split} endpoint")
        ax.hist(np.mean(np.abs(errors[:, CONTROL_INDICES]), axis=1) * (IMAGE_SIZE - 1), bins=30, histtype="step", linewidth=1.2, label=f"{split} control")
    ax.set_xlabel("per-sample MAE (pixels)")
    ax.set_ylabel("count")
    ax.set_title("Endpoint vs control errors")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("endpoint_control.png", "endpoint_vs_control.png"))
    plt.close(fig)

    # Inside/outside behavior is a separate MAE figure because extra controls
    # intentionally extend outside the canvas.
    fig, ax = plt.subplots(figsize=(9, 5))
    inside = [float(row["control_inside_point_mae_px"]) for row in metric_rows]
    outside = [float(row["control_outside_point_mae_px"]) for row in metric_rows]
    ax.bar(x - 0.18, inside, 0.36, label="target-inside MAE")
    ax.bar(x + 0.18, outside, 0.36, label="target-outside MAE")
    ax.set_xticks(x, labels, rotation=25)
    ax.set_ylabel("coordinate MAE (pixels)")
    ax.set_title("Inside / outside control-point MAE")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("inside_outside.png", "inside-outside.png", "inside_outside_mae.png"))
    plt.close(fig)

    # Deterministically random test images as backgrounds, with true/predicted
    # Bezier curves and control polygons.  Sampling two examples per split
    # makes both interpolation and extrapolation behavior auditable.
    rng = np.random.default_rng(_split_seed(profile, "overlay"))
    selections: List[Tuple[str, int]] = []
    for split in splits:
        count = min(2, len(target_map[split]))
        indices = rng.choice(len(target_map[split]), size=count, replace=False)
        selections.extend((split, int(index)) for index in indices.tolist())
    overlay_rows = [
        {"panel": panel, "split": split, "index": index, "id": str(id_map[split][index])}
        for panel, (split, index) in enumerate(selections)
    ]
    write_rows(dirs["tables"] / "overlay_samples.csv", overlay_rows)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
    for panel, (split, sample_index) in enumerate(selections):
        ax = axes.flat[panel]
        background = image_map[split][sample_index]
        ax.imshow(background, cmap="gray", origin="upper", extent=(0, 1, 1, 0), alpha=0.6)
        truth = target_map[split][sample_index]
        guess = pred_map[split][sample_index]
        truth_curve = bezier_points(truth, 128)
        pred_curve = bezier_points(guess, 128)
        truth_points = truth.reshape(4, 2)
        pred_points = guess.reshape(4, 2)
        ax.plot(truth_curve[:, 0], truth_curve[:, 1], "c-", linewidth=1.4, label="true curve")
        ax.plot(pred_curve[:, 0], pred_curve[:, 1], "r--", linewidth=1.2, label="pred curve")
        ax.plot(truth_points[:, 0], truth_points[:, 1], "co-", markersize=3, alpha=0.8, label="true controls")
        ax.plot(pred_points[:, 0], pred_points[:, 1], "rx--", markersize=3, alpha=0.8, label="pred controls")
        ax.set_title(f"{split}: {id_map[split][sample_index]}")
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.45, 1.40)
        ax.set_ylim(1.40, -0.45)
    for ax in axes.flat[len(selections):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Seeded random test images with curve/control overlays (expanded axes)")
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("curve_overlay.png", "curve_overlay_zoom.png", "prediction_overlay.png"))
    plt.close(fig)


def analyze(profile: str = "full", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    _load_manifest(spec)
    dirs = output_dirs(spec)
    device = choose_device()
    model = _load_checkpoint(spec, device)
    metric_rows: List[Dict[str, Any]] = []
    image_map: Dict[str, np.ndarray] = {}
    id_map: Dict[str, np.ndarray] = {}
    target_map: Dict[str, np.ndarray] = {}
    pred_map: Dict[str, np.ndarray] = {}
    _, train_targets, _ = _load_npz(spec, "train")
    train_mean_target = np.mean(train_targets.astype(np.float64), axis=0, keepdims=True)
    for split in TEST_SPLITS:
        images, targets, ids = _load_npz(spec, split)
        _, predictions = _predict_model(model, images, targets, device, spec.batch_size, amp_enabled=False)
        image_map[split] = images
        id_map[split] = ids
        target_map[split] = targets
        pred_map[split] = predictions
        write_rows(dirs["tables"] / f"predictions_{split}.csv", _prediction_rows(split, ids, targets, predictions))
        split_metrics = compute_metrics(targets, predictions)
        constant_mae_px = float(np.mean(np.abs(targets - train_mean_target)) * (IMAGE_SIZE - 1))
        split_metrics["train_mean_constant_mae_px"] = constant_mae_px
        split_metrics["model_to_constant_mae_ratio"] = float(split_metrics["mae_px"] / constant_mae_px)
        split_metrics["improvement_over_constant_fraction"] = 1.0 - split_metrics["model_to_constant_mae_ratio"]
        metric_rows.append({"split": split, **split_metrics})
    write_rows(dirs["tables"] / "metrics.csv", metric_rows)
    write_rows(dirs["tables"] / "metrics_long.csv", [{"split": row["split"], "metric": key, "value": value} for row in metric_rows for key, value in row.items() if key != "split"])
    _plot_analysis_figures(spec, image_map, id_map, target_map, pred_map, metric_rows)
    summary = {
        "profile": spec.name,
        "splits": [row["split"] for row in metric_rows],
        "best_checkpoint": "checkpoints/resnet18_best.pt",
    }
    (dirs["tables"] / "summary.json").write_text(
        json.dumps(_jsonable({"profile": spec.name, "metrics": metric_rows}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_rows(dirs["tables"] / "summary.csv", metric_rows)
    write_run_metadata(spec, "analyze", summary)
    print(f"[analyze:{spec.name}] metrics and PNGs written under {dirs['root']}")
    print("split       MAE(px)  endpoint  control  curve-RMSE  control-outside")
    for row in metric_rows:
        print(
            f"{str(row['split']):<11} {float(row['mae_px']):7.3f}  "
            f"{float(row['endpoint_mae_px']):8.3f}  {float(row['control_mae_px']):7.3f}  "
            f"{float(row['curve_t_rmse_px']):10.3f}  {float(row['control_outside_point_mae_px']):15.3f}"
        )
    return {"metrics": metric_rows, "summary": summary}


# ---------------------------------------------------------------------------
# Finite-difference identifiability
# ---------------------------------------------------------------------------


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-15 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def identifiability(profile: str = "full", force: bool = False) -> Dict[str, Any]:
    """Estimate rendering identifiability for ID, hole and extra samples.

    For each sample, ``J`` is the float64 finite-difference image Jacobian and
    ``J.T @ J`` is eigendecomposed directly.  Correlations are between log
    conditioning summaries and the model's per-sample prediction error; this
    is distinct from correlating derivative columns.
    """

    spec = profile_spec(profile)
    _load_manifest(spec)
    dirs = output_dirs(spec)
    device = choose_device()
    model = _load_checkpoint(spec, device)
    epsilon = 0.75 / 223.0
    pixel_normalization = float(IMAGE_SIZE * IMAGE_SIZE)
    all_rows: List[Dict[str, Any]] = []
    split_means: Dict[str, np.ndarray] = {}
    split_counts: Dict[str, int] = {}

    for split in TEST_SPLITS:
        images, targets, ids = _load_npz(spec, split)
        _, predictions = _predict_model(model, images, targets, device, spec.batch_size, amp_enabled=False)
        count = min(spec.identifiability_count, len(targets), 200)
        rng = np.random.default_rng(_split_seed(spec, f"identifiability:{split}"))
        selected = rng.choice(len(targets), size=count, replace=False) if count < len(targets) else np.arange(len(targets))
        jtj_sum = np.zeros((COORDINATE_DIM, COORDINATE_DIM), dtype=np.float64)
        split_counts[split] = int(len(selected))
        valid_count = 0
        for sample_index in selected.tolist():
            theta = targets[sample_index].astype(np.float64)
            jacobian = np.empty((IMAGE_SIZE * IMAGE_SIZE, COORDINATE_DIM), dtype=np.float64)
            for parameter_index in range(COORDINATE_DIM):
                plus = theta.copy()
                minus = theta.copy()
                plus[parameter_index] += epsilon
                minus[parameter_index] -= epsilon
                jacobian[:, parameter_index] = (
                    render_bezier(plus).astype(np.float64).ravel() - render_bezier(minus).astype(np.float64).ravel()
                ) / (2.0 * epsilon * 255.0)
            finite = bool(np.all(np.isfinite(jacobian)))
            jtj = (jacobian.T @ jacobian / pixel_normalization) if finite else np.full((COORDINATE_DIM, COORDINATE_DIM), np.nan)
            if finite:
                valid_count += 1
                jtj_sum += jtj
                eigvals = np.linalg.eigvalsh(jtj)
                eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
                sigma = np.sqrt(eigvals)[::-1]
                tolerance = max(float(sigma[0]) * 1e-8, 1e-12)
                rank = int(np.sum(sigma > tolerance))
                sigma_min = float(sigma[-1])
                sigma_max = float(sigma[0])
                condition = float(sigma_max / sigma_min) if sigma_min > tolerance else float("inf")
            else:
                sigma = np.full(COORDINATE_DIM, np.nan)
                rank, sigma_min, sigma_max, condition = 0, float("nan"), float("nan"), float("nan")
            prediction_error = float(np.mean(np.abs(predictions[sample_index] - targets[sample_index])) * (IMAGE_SIZE - 1))
            row: Dict[str, Any] = {
                "split": split,
                "id": str(ids[sample_index]),
                "prediction_mae_px": prediction_error,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rank": rank,
                "condition": condition,
                "valid": finite,
                "epsilon": epsilon,
                "jtj_pixel_normalization": pixel_normalization,
            }
            row.update({f"sigma_{i + 1}": float(value) for i, value in enumerate(sigma)})
            all_rows.append(row)
        split_means[split] = jtj_sum / max(1, valid_count)
        split_means[split] = split_means[split].astype(np.float64)

    pooled_rows = [row for row in all_rows if bool(row["valid"])]
    correlation_summaries: List[Dict[str, Any]] = []
    for split in (*TEST_SPLITS, "pooled"):
        rows = pooled_rows if split == "pooled" else [row for row in pooled_rows if row["split"] == split]
        if not rows:
            continue
        sigma_min = np.asarray([float(row["sigma_min"]) for row in rows], dtype=np.float64)
        condition = np.asarray([float(row["condition"]) for row in rows], dtype=np.float64)
        error = np.asarray([float(row["prediction_mae_px"]) for row in rows], dtype=np.float64)
        finite_cond = np.isfinite(condition) & (condition > 0) & (sigma_min > 0) & (error >= 0)
        log_sigma = np.log10(np.maximum(sigma_min[finite_cond], 1e-30))
        log_condition = np.log10(np.maximum(condition[finite_cond], 1e-30))
        err = error[finite_cond]
        all_rows_summary = {
            "split": split,
            "samples": len(rows),
            "valid_log_samples": int(np.sum(finite_cond)),
            "excluded_nonfinite_condition_samples": int(len(rows) - np.sum(finite_cond)),
            "rank_deficient_samples": int(sum(int(row["rank"]) < COORDINATE_DIM for row in rows)),
            "sigma_min_error_pearson": _pearson(log_sigma, err) if len(err) > 1 else float("nan"),
            "sigma_min_error_spearman": _spearman(log_sigma, err) if len(err) > 1 else float("nan"),
            "condition_error_pearson": _pearson(log_condition, err) if len(err) > 1 else float("nan"),
            "condition_error_spearman": _spearman(log_condition, err) if len(err) > 1 else float("nan"),
        }
        # Report Pearson/Spearman for every singular value, not only sigma_min;
        # sigma_min remains a convenient alias for the smallest-value row.
        for sigma_index in range(COORDINATE_DIM):
            values = np.asarray([float(row[f"sigma_{sigma_index + 1}"]) for row in rows], dtype=np.float64)
            valid_sigma = np.isfinite(values) & (values > 0) & (error >= 0)
            log_values = np.log10(np.maximum(values[valid_sigma], 1e-30))
            error_values = error[valid_sigma]
            all_rows_summary[f"sigma_{sigma_index + 1}_error_pearson"] = _pearson(log_values, error_values) if len(error_values) > 1 else float("nan")
            all_rows_summary[f"sigma_{sigma_index + 1}_error_spearman"] = _spearman(log_values, error_values) if len(error_values) > 1 else float("nan")
        all_rows_summary["rank_deficient_samples"] = int(sum(int(row["rank"]) < COORDINATE_DIM for row in rows))
        all_rows_summary["condition_infinite_samples"] = int(sum(not np.isfinite(float(row["condition"])) for row in rows))
        correlation_summaries.append(all_rows_summary)
        write_rows(dirs["tables"] / f"identifiability_correlation_{split}.csv", [all_rows_summary])
    write_rows(dirs["tables"] / "identifiability_correlations.csv", correlation_summaries)

    write_rows(dirs["tables"] / "identifiability_samples.csv", all_rows)
    # Mean JtJ matrices remain useful for a compact split comparison.
    jtj_rows: List[Dict[str, Any]] = []
    for split, matrix in split_means.items():
        for i, name_i in enumerate(TARGET_NAMES):
            row = {"split": split, "parameter": name_i}
            row.update({name_j: float(matrix[i, j]) for j, name_j in enumerate(TARGET_NAMES)})
            jtj_rows.append(row)
    write_rows(dirs["tables"] / "identifiability_jtj.csv", jtj_rows)

    valid_all = [row for row in all_rows if bool(row["valid"])]
    write_rows(
        dirs["tables"] / "identifiability_singular_values.csv",
        [
            {"split": split, "index": i + 1, "singular_value": float(np.nanmean([row[f"sigma_{i + 1}"] for row in valid_all if row["split"] == split])) if any(row["split"] == split for row in valid_all) else float("nan"), "samples": split_counts.get(split, len(valid_all)), "epsilon": epsilon}
            for split in TEST_SPLITS
            for i in range(COORDINATE_DIM)
        ],
    )

    plt = _import_matplotlib()
    # JtJ heatmaps per split.
    fig, axes = plt.subplots(1, len(TEST_SPLITS), figsize=(15, 4), squeeze=False)
    for ax, split in zip(axes.flat, TEST_SPLITS):
        matrix = split_means[split]
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_title(split)
        ax.set_xticks(range(COORDINATE_DIM), TARGET_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(COORDINATE_DIM), TARGET_NAMES)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Finite-difference JtJ (pixel-normalized float64)")
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("identifiability_jtj.png",))
    plt.close(fig)

    # Mean singular values by split (the per-sample values are in the table).
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in TEST_SPLITS:
        rows = [row for row in valid_all if row["split"] == split]
        if not rows:
            continue
        mean_sigma = [float(np.mean([row[f"sigma_{i + 1}"] for row in rows])) for i in range(COORDINATE_DIM)]
        ax.semilogy(np.arange(1, COORDINATE_DIM + 1), np.maximum(mean_sigma, 1e-12), "o-", label=split)
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("mean singular value (log scale)")
    ax.set_title("Finite-difference singular values")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _savefig_aliases(fig, dirs["figures"], ("identifiability_singular_values.png", "singular_values.png"))
    plt.close(fig)

    # Required correlation scatter plots, with per-split colors and pooled
    # trend context.  Correlations are written to split CSVs above.
    for metric, xlabel, filename in (("sigma_min", "log10 sigma_min", "identifiability_sigma_min_error.png"), ("condition", "log10 condition", "identifiability_condition_error.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for split in TEST_SPLITS:
            rows = [row for row in valid_all if row["split"] == split and np.isfinite(float(row[metric])) and float(row[metric]) > 0]
            if not rows:
                continue
            xx = np.log10(np.asarray([float(row[metric]) for row in rows]))
            yy = np.asarray([float(row["prediction_mae_px"]) for row in rows])
            ax.scatter(xx, yy, s=10, alpha=0.45, label=split)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sample prediction MAE (pixels)")
        ax.set_title(f"{xlabel} vs prediction error")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        alias = "identifiability_sigma_min_vs_error.png" if metric == "sigma_min" else "identifiability_condition_vs_error.png"
        _savefig_aliases(fig, dirs["figures"], (filename, alias))
        plt.close(fig)

    summary = {
        "profile": spec.name,
        "samples_per_split": split_counts,
        "valid_samples": len(valid_all),
        "invalid_samples": len(all_rows) - len(valid_all),
        "epsilon": epsilon,
        "jtj_pixel_normalization": pixel_normalization,
        "sigma_source": "sqrt(eigvalsh(J.T @ J))",
        "correlations": correlation_summaries,
    }
    (dirs["tables"] / "identifiability_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    write_run_metadata(spec, "identifiability", summary)
    print(f"[identifiability:{spec.name}] samples={split_counts} -> {dirs['tables']}")
    print("split       r(log sigma_min,error)  rho     r(log cond,error)  rho")
    for row in correlation_summaries:
        print(
            f"{str(row['split']):<11} {float(row['sigma_min_error_pearson']):22.3f}  "
            f"{float(row['sigma_min_error_spearman']):5.3f}  "
            f"{float(row['condition_error_pearson']):21.3f}  "
            f"{float(row['condition_error_spearman']):5.3f}"
        )
    return summary


# ---------------------------------------------------------------------------
# Read-only environment check and CLI
# ---------------------------------------------------------------------------


def check(profile: str = "full") -> int:
    """Check dependencies, renderer invariants and model architecture only."""

    spec = profile_spec(profile)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    try:
        import torchvision

        print(f"torchvision: {torchvision.__version__}")
    except Exception as exc:
        print(f"torchvision: IMPORT ERROR ({exc})")
        return 1
    try:
        import PIL

        print(f"Pillow: {PIL.__version__}")
        sample_parameters = np.asarray((0.15, 0.45, 0.35, 0.35, 0.58, 0.62, 0.85, 0.55), dtype=np.float32)
        image = render_bezier(sample_parameters)
        if image.dtype != np.uint8 or image.shape != (IMAGE_SIZE, IMAGE_SIZE) or image.ndim != 2:
            raise RuntimeError(f"renderer invariant failed: {image.shape} {image.dtype}")
        print(f"renderer: {IMAGE_SIZE}x{IMAGE_SIZE} uint8 L, supersample={SUPER_SAMPLE}, LANCZOS")
    except Exception as exc:
        print(f"renderer: ERROR ({exc})")
        return 1
    try:
        seed_everything()
        reversed_parameters = np.asarray((0.80, 0.30, 0.68, 0.25, 0.34, 0.75, 0.20, 0.65), dtype=np.float32)
        canonical_parameters = canonicalize_parameters(reversed_parameters)
        if not (canonical_parameters[0] < canonical_parameters[6] and np.allclose(canonical_parameters.reshape(4, 2)[1], reversed_parameters.reshape(4, 2)[2])):
            raise RuntimeError("whole-control reversal canonicalization failed")
        hole_boundary = np.asarray((0.20, 0.20, 0.35, 0.30, 0.60, 0.45, 0.80, 0.80), dtype=np.float32)
        if not is_hole(hole_boundary) or is_hole(hole_boundary + np.asarray((0, 0, 0.21, 0, 0, 0, 0, 0), dtype=np.float32)):
            raise RuntimeError("hole boundary predicate failed")
        extra_boundary = hole_boundary.copy()
        extra_boundary[2] = EXTRA_LOW_MAX
        if is_extra(extra_boundary):
            raise RuntimeError("extra lower-open boundary failed")
        extra_boundary[2] = EXTRA_LOW_MAX - 1e-4
        if not is_extra(extra_boundary):
            raise RuntimeError("extra lower band failed")
        model = build_model().eval()
        if not isinstance(model.avgpool, nn.AdaptiveAvgPool2d) or tuple(model.avgpool.output_size) != (1, 1):
            raise RuntimeError(f"avgpool was changed: {model.avgpool!r}")
        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float().div_(255.0)
        with torch.no_grad():
            output = model(image_tensor)
        if tuple(output.shape) != (1, COORDINATE_DIM) or isinstance(model.fc, nn.Sequential):
            raise RuntimeError(f"unexpected head/output: {model.fc!r} {tuple(output.shape)}")
        print(f"model: torchvision resnet18(weights=None), avgpool retained, fc={model.fc}, output={tuple(output.shape)}")
    except Exception as exc:
        print(f"model: ERROR ({exc})")
        return 1
    print(f"profile={spec.name}: train={spec.train_count}, val={spec.val_count}, tests={spec.test_count}x{len(TEST_SPLITS)}, epochs={spec.epochs}")
    print(f"fingerprint: {config_fingerprint(spec)}")
    print(f"output root (not created by check): {output_root(spec)}")
    print("check completed (no data generation and no training)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("check", "prepare", "train", "analyze", "identifiability", "all"), default="check")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="full")
    parser.add_argument("--force", action="store_true", help="overwrite this script's profile-specific cache/checkpoints")
    args = parser.parse_args(argv)
    if args.stage == "check":
        return check(args.profile)
    if args.stage == "prepare":
        seed_everything()
        prepare_data(args.profile, force=args.force)
        return 0
    if args.stage == "train":
        train(args.profile, force=args.force)
        return 0
    if args.stage == "analyze":
        analyze(args.profile, force=args.force)
        return 0
    if args.stage == "identifiability":
        identifiability(args.profile, force=args.force)
        return 0
    if args.stage == "all":
        seed_everything()
        prepare_data(args.profile, force=args.force)
        train(args.profile, force=args.force)
        analyze(args.profile, force=args.force)
        identifiability(args.profile, force=args.force)
        print(f"[all:{args.profile}] complete under {output_root(args.profile)}")
        return 0
    raise AssertionError(f"unhandled stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
