"""Shared constants, paths, renderer, ResNet18, and the 6x6 orthogonal map A."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

SEED = 20260810
IMAGE_SIZE = 224
SUPER_SAMPLE = 4
CURVE_SAMPLES = 256
STROKE_WIDTH = 3.0
COORDINATE_DIM = 6
COORD_SCALE = float(IMAGE_SIZE - 1)
MARGIN_PX = 8.0
MIN_CHORD_PX = 40.0
MIN_CURVE_PX = 12.0
MAX_Q_HALF_EXTENT_PX = 56.0
CURVE_FRAC = 0.30
ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)
CHORDS_PX = (48.0, 62.0, 76.0, 90.0)
CURVE_SIGNS = (-1.0, 1.0)
ALPHAS = (-0.40, 0.40)
TARGET_NAMES = ("p0x", "p0y", "p1x", "p1y", "p2x", "p2y")
SPLITS = ("train", "val", "test")

LR = 0.1
LR_MIN = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
BATCH_SIZE = 64
EARLY_STOP_MAE_PX = 0.2
PATIENCE = 10
ANALYSIS_GATE_MAE_PX = 1.0
L1_WEIGHT = 0.25
RANDOM_ERASURE_REPEATS = 5
ENERGY_RANK_FRACTIONS = (0.80, 0.90, 0.95)
READOUT_ENERGY_FRACTION = 0.90

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results" / "phase1_gap_rep"


@dataclass(frozen=True)
class Profile:
    name: str
    n_angle: int
    n_scale: int
    n_curve: int
    n_alpha: int
    n_tx: int
    n_ty: int

    @property
    def n_shapes(self) -> int:
        return self.n_angle * self.n_scale * self.n_curve * self.n_alpha

    @property
    def n_translations(self) -> int:
        return self.n_tx * self.n_ty

    @property
    def n_total(self) -> int:
        return self.n_shapes * self.n_translations


PROFILES: Dict[str, Profile] = {
    "factorial": Profile("factorial", 4, 4, 2, 2, 8, 8),
    "smoke": Profile("smoke", 2, 2, 2, 1, 4, 2),
}


@dataclass(frozen=True)
class TrainRun:
    name: str
    optimizer: str
    lr: float
    lr_min: float
    epochs: int
    patience: int
    schedule: str
    l1_weight: float = L1_WEIGHT
    momentum: float = 0.9
    resume_from: str | None = None
    amp: bool | None = None
    sgd_fallback_lr: float | None = None
    clip_grad_norm: float | None = None
    data_profile: str = "factorial"


TRAIN_RUNS: Dict[str, TrainRun] = {
    "adamw_l1_resume": TrainRun(
        name="adamw_l1_resume",
        optimizer="adamw",
        lr=1e-3,
        lr_min=1e-4,
        epochs=120,
        patience=50,
        schedule="cosine",
        resume_from="factorial",
    ),
    "adamw_l1_scratch": TrainRun(
        name="adamw_l1_scratch",
        optimizer="adamw",
        lr=1e-3,
        lr_min=1e-5,
        epochs=100,
        patience=50,
        schedule="cosine",
    ),
    "sgd_l1": TrainRun(
        name="sgd_l1",
        optimizer="sgd",
        lr=1.0,
        lr_min=0.1,
        epochs=50,
        patience=50,
        schedule="linear",
        amp=False,
        sgd_fallback_lr=0.1,
        clip_grad_norm=1.0,
    ),
    "sgd_l1_fixed": TrainRun(
        name="sgd_l1_fixed",
        optimizer="sgd",
        lr=0.1,
        lr_min=0.1,
        epochs=50,
        patience=50,
        schedule="constant",
        amp=False,
        clip_grad_norm=1.0,
    ),
}

TRAIN_RUN_ORDER = ("adamw_l1_resume", "adamw_l1_scratch", "sgd_l1")


def profile_spec(profile: str | Profile) -> Profile:
    if isinstance(profile, Profile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile!r}; choose {sorted(PROFILES)}") from exc


def dirs(profile: str | Profile) -> Dict[str, Path]:
    root = RESULTS_ROOT / profile_spec(profile).name
    names = ("config", "manifest", "data", "checkpoints", "tables", "figures", "features")
    return {name: root / name for name in names} | {"root": root}


def ensure_dirs(profile: str | Profile) -> Dict[str, Path]:
    mapping = dirs(profile)
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def run_dirs(run_name: str) -> Dict[str, Path]:
    if run_name in PROFILES:
        raise ValueError(f"{run_name!r} is a data profile; use dirs() so factorial wreckage is not overwritten")
    root = RESULTS_ROOT / run_name
    names = ("config", "checkpoints", "tables", "figures", "features")
    return {name: root / name for name in names} | {"root": root}


def ensure_run_dirs(run_name: str) -> Dict[str, Path]:
    mapping = run_dirs(run_name)
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def train_run_spec(run_name: str) -> TrainRun:
    try:
        return TRAIN_RUNS[run_name]
    except KeyError as exc:
        raise ValueError(f"unknown train run {run_name!r}; choose {sorted(TRAIN_RUNS)}") from exc


def train_run_config(run: TrainRun) -> Dict[str, Any]:
    return {
        "run": run.name,
        "data_profile": run.data_profile,
        "model": {
            "name": "torchvision.resnet18",
            "weights": None,
            "avgpool": "AdaptiveAvgPool2d((1,1))",
            "head": "Linear(512,6)",
            "sigmoid": False,
        },
        "loss": f"MSE + {run.l1_weight} * L1",
        "l1_weight": run.l1_weight,
        "optimizer": run.optimizer,
        "lr": run.lr,
        "lr_min": run.lr_min,
        "lr_schedule": run.schedule,
        "momentum": run.momentum if run.optimizer == "sgd" else None,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": run.epochs,
        "batch_size": BATCH_SIZE,
        "early_stop_mae_px": EARLY_STOP_MAE_PX,
        "patience": run.patience,
        "analysis_gate_mae_px": ANALYSIS_GATE_MAE_PX,
        "resume_from": run.resume_from,
        "amp": run.amp,
        "sgd_fallback_lr": run.sgd_fallback_lr,
        "clip_grad_norm": run.clip_grad_norm,
        "augmentation": None,
        "seed": SEED,
    }


def geometry_config(profile: str | Profile) -> Dict[str, Any]:
    spec = profile_spec(profile)
    return {
        "schema": 1,
        "profile": asdict(spec) | {"n_shapes": spec.n_shapes, "n_translations": spec.n_translations, "n_total": spec.n_total},
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "supersample": SUPER_SAMPLE,
        "resampling": "LANCZOS",
        "curve_samples": CURVE_SAMPLES,
        "stroke_width": STROKE_WIDTH,
        "coord_scale": COORD_SCALE,
        "margin_px": MARGIN_PX,
        "min_chord_px": MIN_CHORD_PX,
        "min_curve_px": MIN_CURVE_PX,
        "max_q_half_extent_px": MAX_Q_HALF_EXTENT_PX,
        "curve_frac": CURVE_FRAC,
        "angles_deg": list(ANGLES_DEG[: spec.n_angle]),
        "chords_px": list(CHORDS_PX[: spec.n_scale]),
        "curve_signs": list(CURVE_SIGNS[: spec.n_curve]),
        "alphas": list(ALPHAS[: spec.n_alpha]),
        "targets": list(TARGET_NAMES),
        "canonical": "lexicographic (Q0x,Q0y) < (Q2x,Q2y)",
        "renderer": "Pillow high-resolution polyline then LANCZOS",
        "augmentation": None,
        "label_normalization": "absolute P in [0,1]; pixel = normalized * 223; no z-score",
        "split": {
            "type": "recombination",
            "rule": "cyclic Latin (row_perm[s]+col_perm[t]) mod n; first n/8 val, next n/8 test, rest train",
            "note": "every shape and translation appears in train; test pairs are held-out combinations",
        },
    }


def training_config() -> Dict[str, Any]:
    return {
        "model": {
            "name": "torchvision.resnet18",
            "weights": None,
            "avgpool": "AdaptiveAvgPool2d((1,1))",
            "head": "Linear(512,6)",
            "sigmoid": False,
        },
        "loss": "MSELoss on 6 absolute normalized coordinates",
        "optimizer": "AdamW",
        "lr": LR,
        "lr_min": LR_MIN,
        "lr_schedule": f"CosineAnnealingLR(T_max={MAX_EPOCHS}, eta_min={LR_MIN})",
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stop_mae_px": EARLY_STOP_MAE_PX,
        "patience": PATIENCE,
        "analysis_gate_mae_px": ANALYSIS_GATE_MAE_PX,
        "amp": True,
        "seed": SEED,
    }


def fingerprint(profile: str | Profile) -> str:
    payload = geometry_config(profile)
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


def px_to_norm(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / COORD_SCALE


def norm_to_px(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * COORD_SCALE


def quadratic_points(control_xy: np.ndarray, count: int = CURVE_SAMPLES) -> np.ndarray:
    p = np.asarray(control_xy, dtype=np.float64).reshape(3, 2)
    t = np.linspace(0.0, 1.0, int(count), dtype=np.float64)[:, None]
    return (1.0 - t) ** 2 * p[0] + 2.0 * (1.0 - t) * t * p[1] + t**2 * p[2]


def render_quadratic(parameters_norm: Sequence[float]) -> np.ndarray:
    from PIL import Image, ImageDraw

    points = quadratic_points(np.asarray(parameters_norm, dtype=np.float64).reshape(3, 2))
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
    if out.shape != (IMAGE_SIZE, IMAGE_SIZE) or out.dtype != np.uint8 or out.ndim != 2:
        raise RuntimeError(f"renderer invariant: {out.shape} {out.dtype}")
    return out


def build_model() -> nn.Module:
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    if not isinstance(model.avgpool, nn.AdaptiveAvgPool2d) or tuple(model.avgpool.output_size) != (1, 1):
        raise RuntimeError("expected original AdaptiveAvgPool2d((1,1))")
    model.fc = nn.Linear(512, COORDINATE_DIM)
    return model


def gap_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(images)
    if array.ndim == 2:
        array = array[None, ...]
    tensor = torch.from_numpy(array).unsqueeze(1).repeat(1, 3, 1, 1).float().div_(255.0)
    return tensor


def orthonormal_1d() -> np.ndarray:
    basis = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, -2.0],
        ],
        dtype=np.float64,
    )
    basis[0] /= math.sqrt(3.0)
    basis[1] /= math.sqrt(2.0)
    basis[2] /= math.sqrt(6.0)
    return basis


def interleaved_to_xxx_yyy() -> np.ndarray:
    """Permute [P0x,P0y,P1x,P1y,P2x,P2y] -> [P0x,P1x,P2x,P0y,P1y,P2y]."""
    matrix = np.zeros((6, 6), dtype=np.float64)
    order = (0, 2, 4, 1, 3, 5)
    for row, col in enumerate(order):
        matrix[row, col] = 1.0
    return matrix


def orthogonal_matrix_a() -> np.ndarray:
    basis = orthonormal_1d()
    block = np.zeros((6, 6), dtype=np.float64)
    block[:3, :3] = basis
    block[3:, 3:] = basis
    return block @ interleaved_to_xxx_yyy()


def decompose_controls(p_interleaved: np.ndarray) -> Dict[str, np.ndarray]:
    """Split absolute P into translation t and relative Q, plus orthonormal coeffs."""
    p = np.asarray(p_interleaved, dtype=np.float64)
    original = p.shape
    flat = p.reshape(-1, 6)
    pts = flat.reshape(-1, 3, 2)
    t = pts.mean(axis=1)
    q = pts - t[:, None, :]
    coeffs = flat @ orthogonal_matrix_a().T
    translation_comp = coeffs[:, [0, 3]]
    geometry_comp = coeffs[:, [1, 2, 4, 5]]
    return {
        "t": t.reshape(original[:-1] + (2,)),
        "q": q.reshape(original[:-1] + (3, 2)),
        "translation_comp": translation_comp.reshape(original[:-1] + (2,)),
        "geometry_comp": geometry_comp.reshape(original[:-1] + (4,)),
        "t_from_comp": (translation_comp / math.sqrt(3.0)).reshape(original[:-1] + (2,)),
    }


def setup_matplotlib_chinese():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = (
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    )
    available = {item.name for item in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_metadata(profile: str | Profile, stage: str, extra: Mapping[str, Any] | None = None) -> None:
    spec = profile_spec(profile)
    payload: Dict[str, Any] = {
        "stage": stage,
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "fingerprint": fingerprint(spec),
        "geometry": geometry_config(spec),
        "training": training_config(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    if extra:
        payload.update(dict(extra))
    dump_json(dirs(spec)["config"] / "run_metadata.json", payload)


def write_run_metadata(run: TrainRun, stage: str, extra: Mapping[str, Any] | None = None) -> None:
    spec = profile_spec(run.data_profile)
    payload: Dict[str, Any] = {
        "stage": stage,
        "run": run.name,
        "data_profile": run.data_profile,
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "fingerprint": fingerprint(spec),
        "geometry": geometry_config(spec),
        "training": train_run_config(run),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    if extra:
        payload.update(dict(extra))
    dump_json(run_dirs(run.name)["config"] / "run_metadata.json", payload)


def environment_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception as exc:
        versions["torchvision"] = f"unavailable: {exc}"
    try:
        from PIL import Image

        versions["pillow"] = getattr(Image, "__version__", "unknown")
    except Exception as exc:
        versions["pillow"] = f"unavailable: {exc}"
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception as exc:
        versions["matplotlib"] = f"unavailable: {exc}"
    return versions
