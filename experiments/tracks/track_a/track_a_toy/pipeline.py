from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import (
    CONFIG,
    DATA_ROOT,
    FEATURE_ROOT,
    FIGURE_ROOT,
    MODEL_SPECS,
    OUTPUT_ROOT,
    PROBE_ROOT,
    TRAIN_ROOT,
    config_fingerprint,
    ensure_output_dirs,
)
from .models import (
    STAGE_NAMES,
    ToyResNet18,
    build_models_with_shared_state,
    extract_pooled_features,
    model_shape_report,
    seed_everything,
)
from .renderer import (
    finite_position_grid,
    grid_points_with_visibility,
    is_fully_visible,
    position_grid,
    render_many,
    render_one,
)
from .visuals import (
    chinese_font,
    write_concept_figure,
    write_input_grid_figure,
    write_summary_figure,
    write_training_curve_figure,
)


FEATURE_NAMES = ("stem_gap", "layer1_gap", "layer2_gap", "layer3_gap", "layer4_gap", "gap")
FEATURE_DIMS = {"stem_gap": 64, "layer1_gap": 64, "layer2_gap": 128, "layer3_gap": 256, "layer4_gap": 512, "gap": 512}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        value = str(requested).lower()
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _config_payload() -> dict[str, Any]:
    return {
        "config": CONFIG,
        "model_specs": MODEL_SPECS,
        "config_fingerprint": config_fingerprint(),
    }


def _dataset_path(dataset: str) -> Path:
    if dataset not in {"finite", "torus"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    return DATA_ROOT / f"{dataset}.npz"


def _load_dataset(dataset: str) -> dict[str, np.ndarray]:
    path = _dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare first")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _dataset_for_model(name: str) -> str:
    return str(MODEL_SPECS[name]["renderer"])


def _save_dataset(
    name: str,
    points: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    images: np.ndarray,
    *,
    fully_visible: np.ndarray | None = None,
) -> dict[str, Any]:
    path = _dataset_path(name)
    payload: dict[str, np.ndarray] = {
        "points": np.asarray(points, dtype=np.float32),
        "ix": np.asarray(ix, dtype=np.int64),
        "iy": np.asarray(iy, dtype=np.int64),
        "images": np.asarray(images, dtype=np.uint8),
    }
    if fully_visible is not None:
        visibility = np.asarray(fully_visible, dtype=bool)
        if visibility.shape != (len(points),):
            raise ValueError(f"fully_visible must have shape {(len(points),)}, got {visibility.shape}")
        payload["fully_visible"] = visibility
        # Keep the protocol vocabulary explicit: for this toy, interior means
        # the actual translated triangle is fully inside the finite canvas.
        payload["interior"] = visibility.copy()
    np.savez_compressed(path, **payload)
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "count": int(len(points)),
        "points_shape": list(points.shape),
        "images_shape": list(images.shape),
        "points_dtype": str(points.dtype),
        "images_dtype": str(images.dtype),
        "renderer": name,
    }
    if fully_visible is not None:
        record.update(
            {
                "fully_visible_count": int(np.asarray(fully_visible, dtype=bool).sum()),
                "clipped_count": int(len(points) - np.asarray(fully_visible, dtype=bool).sum()),
                "fully_visible_field": "fully_visible",
                "interior_field": "interior",
            }
        )
    return record


def _text_for_image(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str = "white",
    font: ImageFont.FreeTypeFont | None = None,
) -> None:
    draw.rectangle((xy[0], xy[1], xy[0] + max(60, 7 * len(text)), xy[1] + 14), fill="black")
    draw.text(xy, text, font=font, fill=fill)


def _cross(draw: ImageDraw.ImageDraw, x: float, y: float, *, color: str, radius: int = 3) -> None:
    cx, cy = int(round(x)), int(round(y))
    draw.line((cx - radius, cy, cx + radius, cy), fill=color, width=1)
    draw.line((cx, cy - radius, cx, cy + radius), fill=color, width=1)


def _make_montage(
    images: list[np.ndarray],
    labels: list[str],
    path: Path,
    *,
    title: str,
    points: list[tuple[float, float]] | None = None,
    predictions: list[tuple[float, float]] | None = None,
    prediction_labels: list[str] | None = None,
) -> None:
    if len(images) != len(labels) or not images:
        raise ValueError("montage requires equally sized non-empty images and labels")
    if prediction_labels is not None and len(prediction_labels) != len(images):
        raise ValueError("prediction_labels must match images")
    tile = 128 if prediction_labels is not None else 96
    cols = 5
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * (tile + 18) + 20), "#202020")
    draw_canvas = ImageDraw.Draw(canvas)
    title_font = chinese_font(13)
    label_font = chinese_font(12 if prediction_labels is not None else 10)
    _text_for_image(draw_canvas, (4, 3), title, fill="#f0f0f0", font=title_font)
    y_offset = 20
    for index, (array, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, cols)
        image = Image.fromarray(np.asarray(array)[0], mode="L").convert("RGB").resize((tile, tile))
        draw = ImageDraw.Draw(image)
        if points is not None:
            _cross(draw, float(points[index][0]) * tile / 64.0, float(points[index][1]) * tile / 64.0, color="#ff4040")
        if predictions is not None:
            _cross(draw, float(predictions[index][0]) * tile / 64.0, float(predictions[index][1]) * tile / 64.0, color="#40a0ff")
        left = col * tile
        top = y_offset + row * (tile + 18)
        canvas.paste(image, (left, top))
        display_label = prediction_labels[index] if prediction_labels is not None else label
        draw_canvas.text((left + 2, top + tile + 2), display_label, font=label_font, fill="#f0f0f0")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _visual_points(dataset: str, count: int = 10) -> np.ndarray:
    if count != len(CONFIG["visual_positions_xy"]):
        raise ValueError("the toy protocol fixes exactly the configured ten visualization positions")
    data = _load_dataset(dataset)
    available = {tuple(np.rint(point).astype(int).tolist()): index for index, point in enumerate(data["points"])}
    configured = np.asarray(CONFIG["visual_positions_xy"], dtype=np.float32)
    if configured.shape != (10, 2):
        raise ValueError(f"visual_positions_xy must be exactly [10,2], got {configured.shape}")
    chosen = [(int(round(point[0])), int(round(point[1]))) for point in configured]
    if len(set(chosen)) != 10:
        raise ValueError("visual_positions_xy must contain ten unique positions")
    missing = [point for point in chosen if point not in available]
    if missing:
        raise ValueError(f"configured visualization positions missing from {dataset}: {missing}")
    return np.asarray(chosen, dtype=np.float32)


def prepare_datasets() -> dict[str, Any]:
    ensure_output_dirs()
    seed_everything()
    all_points, all_ix, all_iy, visible = grid_points_with_visibility()
    # Both materialized datasets retain all 256 configured coordinates.  The
    # finite renderer intentionally keeps clipped shapes; the mask is the
    # separate interior/fully-visible analysis subset.
    finite_points = all_points.copy()
    finite_ix = all_ix.copy()
    finite_iy = all_iy.copy()
    finite_images = render_many(finite_points, torus=False)
    torus_points = all_points.copy()
    torus_ix = all_ix.copy()
    torus_iy = all_iy.copy()
    torus_images = render_many(torus_points, torus=True)
    records = {
        "schema_version": 1,
        "kind": "track_a_toy_materialized_data",
        "config_fingerprint": config_fingerprint(),
        "grid_count": int(len(all_points)),
        "grid_values": list(CONFIG["grid_values"]),
        "triangle_vertices_xy": np.asarray(CONFIG["triangle_vertices_xy"], dtype=np.float64),
        "finite_fully_visible_count": int(visible.sum()),
        "finite_clipped_grid_count": int((~visible).sum()),
        "datasets": {
            "finite": _save_dataset(
                "finite",
                finite_points,
                finite_ix,
                finite_iy,
                finite_images,
                fully_visible=visible,
            ),
            "torus": _save_dataset("torus", torus_points, torus_ix, torus_iy, torus_images),
        },
        "finite_visibility_rule": "all three actual translated triangle vertices lie in [0,image_size] on both axes; mask only, not a materialization filter",
        "torus_rule": "all configured grid positions, with periodic rendering modulo image_size",
        "analysis_scopes": {
            "all": "all 256 configured positions; primary random-feature/probe scope",
            "finite_interior": "finite fully_visible/interior mask (182 true, 74 clipped for this triangle); supplementary scope",
        },
    }
    write_json(DATA_ROOT / "manifest.json", records)
    write_json(OUTPUT_ROOT / "config.json", _config_payload())

    for dataset in ("finite", "torus"):
        data = _load_dataset(dataset)
        selected = _visual_points(dataset)
        lookup = {tuple(np.rint(point).astype(int).tolist()): index for index, point in enumerate(data["points"])}
        indices = [lookup[tuple(np.rint(point).astype(int).tolist())] for point in selected]
        images = [data["images"][index] for index in indices]
        labels = [f"{dataset} {int(point[0])},{int(point[1])}" for point in selected]
        _make_montage(images, labels, FIGURE_ROOT / f"inputs_{dataset}.png", title=f"Track A-Toy inputs: {dataset}")
    finite_data = _load_dataset("finite")
    torus_data = _load_dataset("torus")
    write_concept_figure(FIGURE_ROOT / "实验概念图.png")
    write_input_grid_figure(
        FIGURE_ROOT / "有限画布_固定输入_10张.png",
        dataset="finite",
        images=finite_data["images"],
        points=finite_data["points"],
        fully_visible=finite_data["fully_visible"],
    )
    write_input_grid_figure(
        FIGURE_ROOT / "环面画布_固定输入_10张.png",
        dataset="torus",
        images=torus_data["images"],
        points=torus_data["points"],
        fully_visible=None,
    )
    return records


def _batch_feature_vectors(model: ToyResNet18, images: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    for start in range(0, len(images), int(batch_size)):
        batch = torch.from_numpy(np.asarray(images[start : start + batch_size])).to(device=device, dtype=torch.float32).div_(255.0)
        values = extract_pooled_features(model, batch)
        for name in FEATURE_NAMES:
            chunks[name].append(values[name].detach().cpu().numpy().astype(np.float32, copy=False))
        del batch, values
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {name: np.concatenate(chunks[name], axis=0) for name in FEATURE_NAMES}


def _load_models(device: torch.device) -> tuple[dict[str, ToyResNet18], str]:
    models, _state, shared_hash = build_models_with_shared_state(device=device)
    for model in models.values():
        model.eval()
    return models, shared_hash


def random_features(device_name: str | None = None) -> dict[str, Any]:
    ensure_output_dirs()
    device = resolve_device(device_name)
    if not _dataset_path("finite").exists() or not _dataset_path("torus").exists():
        prepare_datasets()
    models, shared_hash = _load_models(device)
    batch_size = int(CONFIG["feature_batch_size"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "track_a_toy_random_features",
        "config_fingerprint": config_fingerprint(),
        "shared_state_hash": shared_hash,
        "device": str(device),
        "models": {},
    }
    for name, model in models.items():
        dataset = _dataset_for_model(name)
        data = _load_dataset(dataset)
        features = _batch_feature_vectors(model, data["images"], device, batch_size)
        path = FEATURE_ROOT / f"{name}.npz"
        feature_payload: dict[str, np.ndarray] = {
            "points": data["points"],
            "ix": data["ix"],
            "iy": data["iy"],
            **features,
        }
        for mask_name in ("fully_visible", "interior"):
            if mask_name in data:
                feature_payload[mask_name] = data[mask_name]
        np.savez_compressed(path, **feature_payload)
        result["models"][name] = {
            "dataset": dataset,
            "path": str(path),
            "sha256": sha256_file(path),
            "count": int(len(data["points"])),
            "scope": "all_256",
            "fully_visible_count": int(data.get("fully_visible", np.ones(len(data["points"]), dtype=bool)).sum()),
            "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        }
        print(f"[random-features] {name}: {dataset} n={len(data['points'])} device={device}")

    shift_metrics = compute_shift_metrics(models, device=device)
    result["shift_metrics_path"] = str(FEATURE_ROOT / "shift_metrics.json")
    write_json(FEATURE_ROOT / "manifest.json", result)
    write_json(FEATURE_ROOT / "shift_metrics.json", shift_metrics)
    return result


def _on_demand_pair_features(
    model: ToyResNet18,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    torus: bool,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not pairs:
        return (
            {name: np.empty((0, FEATURE_DIMS[name]), dtype=np.float32) for name in FEATURE_NAMES},
            {name: np.empty((0, FEATURE_DIMS[name]), dtype=np.float32) for name in FEATURE_NAMES},
        )
    left_chunks: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    right_chunks: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    for start in range(0, len(pairs), int(batch_size)):
        chunk = pairs[start : start + batch_size]
        images = render_many([point for point, _ in chunk] + [point for _, point in chunk], torus=torus)
        left_images = images[: len(chunk)]
        right_images = images[len(chunk) :]
        left = _batch_feature_vectors(model, left_images, device, len(chunk))
        right = _batch_feature_vectors(model, right_images, device, len(chunk))
        for name in FEATURE_NAMES:
            left_chunks[name].append(left[name])
            right_chunks[name].append(right[name])
    return (
        {name: np.concatenate(left_chunks[name], axis=0) for name in FEATURE_NAMES},
        {name: np.concatenate(right_chunks[name], axis=0) for name in FEATURE_NAMES},
    )


def compute_shift_metrics(models: dict[str, ToyResNet18], *, device: torch.device | None = None) -> dict[str, Any]:
    device = resolve_device(None) if device is None else device
    all_points, _all_ix, _all_iy = position_grid()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "track_a_toy_on_demand_shift_diagnostics",
        "config_fingerprint": config_fingerprint(),
        "axis": "x",
        "shifts_px": list(CONFIG["shifts_px"]),
        "models": {},
        "source": "each p and p+delta rendered immediately; no sparse prepared-grid lookup",
    }
    for name, model in models.items():
        torus = _dataset_for_model(name) == "torus"
        model_result: dict[str, Any] = {"dataset": "torus" if torus else "finite", "shifts": {}}
        for delta in CONFIG["shifts_px"]:
            shift = int(delta)
            pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for point in all_points:
                target = point + np.asarray([shift, 0.0], dtype=np.float32)
                if torus:
                    target = target % float(CONFIG["image_size"])
                elif np.any(target < 0.0) or np.any(target >= float(CONFIG["image_size"])):
                    # Finite p and p+delta are rendered immediately whenever
                    # both centers are on the canvas, including cropped shapes.
                    continue
                pairs.append((point, target))
            left, right = _on_demand_pair_features(
                model,
                pairs,
                torus=torus,
                device=device,
                batch_size=int(CONFIG["feature_batch_size"]),
            )
            metrics: dict[str, Any] = {
                "pair_count": len(pairs),
                "source_scope": "all_256",
                "target_center_rule": "finite centers must lie in [0,image_size); torus target wraps",
                "stages": {},
            }
            for feature_name in FEATURE_NAMES:
                difference = right[feature_name].astype(np.float64) - left[feature_name].astype(np.float64)
                norms = np.linalg.norm(difference, axis=1) if len(difference) else np.empty(0, dtype=np.float64)
                metrics["stages"][feature_name] = {
                    "mean_l2": float(np.mean(norms)) if len(norms) else None,
                    "median_l2": float(np.median(norms)) if len(norms) else None,
                    "max_l2": float(np.max(norms)) if len(norms) else None,
                }
            model_result["shifts"][str(shift)] = metrics
        result["models"][name] = model_result
    return result


def _split_mask(ix: np.ndarray, iy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test = ((ix.astype(np.int64) + 2 * iy.astype(np.int64)) % 4) == 0
    return ~test, test


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    design = np.column_stack((np.ones(len(x64), dtype=np.float64), x64))
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    system = design.T @ design + float(alpha) * regularizer
    return np.linalg.solve(system, design.T @ y64)


def _predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x), dtype=np.float64), np.asarray(x, dtype=np.float64)))
    return design @ coefficients


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    denominator = float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2))
    if denominator <= 1e-15:
        return 0.0
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def _circular_phase_targets(points: np.ndarray, period: float = 32.0) -> np.ndarray:
    phase = (np.asarray(points, dtype=np.float64) % period) / period * (2.0 * math.pi)
    return np.column_stack((np.sin(phase[:, 0]), np.cos(phase[:, 0]), np.sin(phase[:, 1]), np.cos(phase[:, 1])))


def _coarse_cell_targets(points: np.ndarray, period: float = 32.0) -> np.ndarray:
    """Return direct coarse-cell labels floor(x/period), floor(y/period)."""

    values = np.asarray(points, dtype=np.float64)
    return np.floor(values / float(period)).astype(np.float64)


def _decode_phase(values: np.ndarray, period: float = 32.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    x = np.mod(np.arctan2(array[:, 0], array[:, 1]) / (2.0 * math.pi) * period, period)
    y = np.mod(np.arctan2(array[:, 2], array[:, 3]) / (2.0 * math.pi) * period, period)
    return np.column_stack((x, y))


def _circular_mae(prediction: np.ndarray, truth: np.ndarray, period: float = 32.0) -> float:
    delta = (np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64) + period / 2.0) % period - period / 2.0
    return float(np.mean(np.abs(delta)))


def _select_alpha(x: np.ndarray, y: np.ndarray, alphas: list[float]) -> float:
    if len(x) < 8:
        return float(alphas[0])
    folds = np.arange(len(x), dtype=np.int64) % 4
    scores: list[tuple[float, float]] = []
    for alpha in alphas:
        errors: list[float] = []
        for fold in range(4):
            train = folds != fold
            valid = ~train
            if not valid.any() or not train.any():
                continue
            coefficients = _fit_ridge(x[train], y[train], float(alpha))
            prediction = _predict_ridge(x[valid], coefficients)
            errors.append(float(np.mean((prediction - y[valid]) ** 2)))
        scores.append((float(np.mean(errors)), float(alpha)))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def _fit_standard_scaler(x: np.ndarray) -> dict[str, np.ndarray | int]:
    """Fit a train-only scaler and remove numerical near-constant columns.

    The GAP features are stored as float32.  A mathematically invariant
    feature can therefore contain tiny storage noise, which must not be
    amplified by standardization.  The allclose test is deliberately fitted
    on the train split only and its tolerances are recorded in the receipt.
    """

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"scaler input must be non-empty [N,D], got {values.shape}")
    mean = values.mean(axis=0)
    observed_scale = values.std(axis=0)
    near_constant_rtol = 1e-5
    near_constant_atol = 1e-6
    near_constant = np.all(
        np.isclose(
            values,
            mean[None, :],
            rtol=near_constant_rtol,
            atol=near_constant_atol,
        ),
        axis=0,
    )
    zero = observed_scale <= 1e-12
    scale = observed_scale.copy()
    scale = scale.copy()
    scale[near_constant] = 1.0
    return {
        "mean": mean,
        "scale": scale,
        "observed_scale": observed_scale,
        "near_constant": near_constant,
        "near_constant_rtol": near_constant_rtol,
        "near_constant_atol": near_constant_atol,
        "zero_variance_count": int(zero.sum()),
        "near_constant_count": int(near_constant.sum()),
        "active_feature_count": int((~near_constant).sum()),
    }


def _transform_standard_scaler(x: np.ndarray, scaler: dict[str, np.ndarray | int]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(mean):
        raise ValueError(f"scaler feature dimension mismatch: values={values.shape}, mean={mean.shape}")
    transformed = (values - mean[None, :]) / scale[None, :]
    near_constant = np.asarray(scaler["near_constant"], dtype=bool)
    if near_constant.shape != (len(mean),):
        raise ValueError(f"scaler near-constant mask mismatch: {near_constant.shape} vs {(len(mean),)}")
    # Exact zero is important: test/all rows must not reintroduce float32
    # storage noise into a feature identified as invariant on train.
    transformed[:, near_constant] = 0.0
    return transformed


def _scaler_record(scaler: dict[str, np.ndarray | int]) -> dict[str, Any]:
    return {
        "fit_scope": "train_split_only",
        "feature_dim": int(len(np.asarray(scaler["mean"]))),
        "mean": np.asarray(scaler["mean"], dtype=np.float64),
        "scale": np.asarray(scaler["scale"], dtype=np.float64),
        "observed_scale": np.asarray(scaler["observed_scale"], dtype=np.float64),
        "near_constant_mask": np.asarray(scaler["near_constant"], dtype=bool),
        "near_constant_rtol": float(scaler["near_constant_rtol"]),
        "near_constant_atol": float(scaler["near_constant_atol"]),
        "zero_variance_count": int(scaler["zero_variance_count"]),
        "near_constant_count": int(scaler["near_constant_count"]),
        "active_feature_count": int(scaler["active_feature_count"]),
    }


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if not key.startswith("_")}


def _fit_probe_scope(
    *,
    features: np.ndarray,
    points: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    scope: str,
) -> dict[str, Any]:
    """Fit all three probe targets from one train-only standardized feature view."""

    feature_values = np.asarray(features, dtype=np.float64)
    point_values = np.asarray(points, dtype=np.float64)
    train_mask, test_mask = _split_mask(ix, iy)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"probe scope {scope!r} needs non-empty train and test partitions")
    scaler = _fit_standard_scaler(feature_values[train_mask])
    scaled_all = _transform_standard_scaler(feature_values, scaler)
    scaled_train = scaled_all[train_mask]
    scaled_test = scaled_all[test_mask]
    train_points = point_values[train_mask]
    test_points = point_values[test_mask]
    alphas = [float(value) for value in CONFIG["ridge_alphas"]]
    targets = {
        "raw_xy": point_values,
        "phase_sincos": _circular_phase_targets(point_values),
        "quotient_cell": _coarse_cell_targets(point_values),
    }
    tasks: dict[str, dict[str, Any]] = {}
    active_feature_count = int(scaler["active_feature_count"])
    for task_name, target_all in targets.items():
        target_train = target_all[train_mask]
        target_test = target_all[test_mask]
        constant_prediction = np.mean(target_train, axis=0, keepdims=True)
        constant_baseline = active_feature_count == 0
        if constant_baseline:
            # Do not select alpha or call ridge when every feature is
            # mathematically invariant.  The only permitted prediction is
            # the train-target mean, applied unchanged to every split.
            alpha = None
            train_prediction = np.repeat(constant_prediction, len(target_train), axis=0)
            test_prediction = np.repeat(constant_prediction, len(target_test), axis=0)
            all_prediction = np.repeat(constant_prediction, len(target_all), axis=0)
        else:
            # Each target gets its own CV selection.  The standardized feature
            # transform remains train-only and is applied to train/test/all alike.
            alpha = _select_alpha(scaled_train, target_train, alphas)
            coefficients = _fit_ridge(scaled_train, target_train, alpha)
            train_prediction = _predict_ridge(scaled_train, coefficients)
            test_prediction = _predict_ridge(scaled_test, coefficients)
            all_prediction = _predict_ridge(scaled_all, coefficients)
        task: dict[str, Any] = {
            "task": task_name,
            "alpha": alpha,
            "fit_scope": "train_split_only",
            "train_count": int(train_mask.sum()),
            "test_count": int(test_mask.sum()),
            "constant_baseline": constant_baseline,
            "active_feature_count": active_feature_count,
            "ridge_fit": not constant_baseline,
            "alpha_selection": (
                "not_applicable_constant_baseline"
                if constant_baseline
                else "independent_train_only_cv"
            ),
            "_train_prediction": train_prediction,
            "_test_prediction": test_prediction,
            "_all_prediction": all_prediction,
        }
        if constant_baseline:
            task["constant_prediction"] = constant_prediction
        if task_name == "raw_xy":
            constant = np.mean(train_points, axis=0, keepdims=True)
            task.update(
                {
                    "train_mae_px": float(np.mean(np.abs(train_prediction - train_points))),
                    "test_mae_px": float(np.mean(np.abs(test_prediction - test_points))),
                    "test_axis_mae_px": np.mean(np.abs(test_prediction - test_points), axis=0),
                    "test_r2": _r2(test_points, test_prediction),
                    "constant_test_mae_px": float(np.mean(np.abs(constant - test_points))),
                }
            )
        elif task_name == "phase_sincos":
            task.update(
                {
                    "train_circular_mae_px": _circular_mae(_decode_phase(train_prediction), train_points % 32.0),
                    "test_circular_mae_px": _circular_mae(_decode_phase(test_prediction), test_points % 32.0),
                    "test_sincos_r2": _r2(target_test, test_prediction),
                    "interpretation": "phase modulo 32 only; not a global absolute-position claim",
                }
            )
        else:
            train_labels = (train_prediction >= 0.5).astype(np.int64)
            test_labels = (test_prediction >= 0.5).astype(np.int64)
            train_truth = target_train.astype(np.int64)
            test_truth = target_test.astype(np.int64)
            task.update(
                {
                    "target_definition": "floor(points_xy / 32.0), direct coarse-cell labels (0/1)",
                    "classification_threshold": 0.5,
                    "label_values": [0, 1],
                    "train_axis_accuracy": np.mean(train_labels == train_truth, axis=0),
                    "test_axis_accuracy": np.mean(test_labels == test_truth, axis=0),
                    "train_average_accuracy": float(np.mean(train_labels == train_truth)),
                    "test_average_accuracy": float(np.mean(test_labels == test_truth)),
                    "train_direct_mae_cells": float(np.mean(np.abs(train_prediction - target_train))),
                    "test_direct_mae_cells": float(np.mean(np.abs(test_prediction - target_test))),
                    "test_direct_r2": _r2(target_test, test_prediction),
                    "interpretation": "direct coarse-cell prediction; no modulo/circular metric",
                }
            )
        tasks[task_name] = task
    return {
        "scope": scope,
        "count": int(len(point_values)),
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "split_rule": str(CONFIG["probe_test_rule"]),
        "gap_feature_dim": int(feature_values.shape[1]),
        "active_feature_count": active_feature_count,
        "constant_baseline": active_feature_count == 0,
        "feature_scaler": _scaler_record(scaler),
        "tasks": tasks,
        "_train_mask": train_mask,
        "_test_mask": test_mask,
    }


def _probe_one(name: str, path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["points"], dtype=np.float64)
        ix = np.asarray(payload["ix"], dtype=np.int64)
        iy = np.asarray(payload["iy"], dtype=np.int64)
        gap = np.asarray(payload["gap"], dtype=np.float32)
        if "fully_visible" in payload:
            fully_visible = np.asarray(payload["fully_visible"], dtype=bool)
        else:
            fully_visible = np.ones(len(points), dtype=bool)
    all_scope = _fit_probe_scope(features=gap, points=points, ix=ix, iy=iy, scope="all_256")
    raw = all_scope["tasks"]["raw_xy"]
    phase = all_scope["tasks"]["phase_sincos"]
    quotient = all_scope["tasks"]["quotient_cell"]
    result: dict[str, Any] = {
        "model": name,
        "dataset": _dataset_for_model(name),
        "scope": "all_256",
        "count": all_scope["count"],
        "train_count": all_scope["train_count"],
        "test_count": all_scope["test_count"],
        "split_rule": all_scope["split_rule"],
        "gap_feature_dim": all_scope["gap_feature_dim"],
        "active_feature_count": all_scope["active_feature_count"],
        "constant_baseline": all_scope["constant_baseline"],
        "feature_scaler": all_scope["feature_scaler"],
        "alphas": {
            "raw_xy": raw["alpha"],
            "phase_sincos": phase["alpha"],
            "quotient_cell": quotient["alpha"],
        },
        # Keep the historical short field as the raw-xy alpha, while the
        # explicit mapping above prevents accidental alpha reuse.
        "alpha": raw["alpha"],
        "global_xy": _public_task(raw),
        "modulo32_phase": _public_task(phase),
        "coarse_cell": _public_task(quotient),
        "raw_xy_boundary_caution": _dataset_for_model(name) == "torus",
        "analysis_scope": "primary all 256 configured positions",
        "predictions": {
            "test_points": points[all_scope["_test_mask"]],
            "test_prediction": raw["_test_prediction"],
        },
        "all_prediction": raw["_all_prediction"],
    }
    if _dataset_for_model(name) == "finite":
        if fully_visible.shape != (len(points),):
            raise ValueError(f"fully_visible mask shape mismatch for {name}: {fully_visible.shape}")
        interior_scope = _fit_probe_scope(
            features=gap[fully_visible],
            points=points[fully_visible],
            ix=ix[fully_visible],
            iy=iy[fully_visible],
            scope="finite_interior_182",
        )
        interior_raw = interior_scope["tasks"]["raw_xy"]
        interior_phase = interior_scope["tasks"]["phase_sincos"]
        interior_quotient = interior_scope["tasks"]["quotient_cell"]
        result["interior_supplementary"] = {
            "scope": "finite_interior_182",
            "count": interior_scope["count"],
            "train_count": interior_scope["train_count"],
            "test_count": interior_scope["test_count"],
            "split_rule": interior_scope["split_rule"],
            "active_feature_count": interior_scope["active_feature_count"],
            "constant_baseline": interior_scope["constant_baseline"],
            "feature_scaler": interior_scope["feature_scaler"],
            "alphas": {
                "raw_xy": interior_raw["alpha"],
                "phase_sincos": interior_phase["alpha"],
                "quotient_cell": interior_quotient["alpha"],
            },
            "raw_xy": _public_task(interior_raw),
            "phase_sincos": _public_task(interior_phase),
            "coarse_cell": _public_task(interior_quotient),
            "interpretation": "supplementary finite fully-visible/interior subset; primary metrics remain all_256",
        }
    return result


def _write_prediction_visual(name: str, result: dict[str, Any]) -> str:
    dataset = str(result["dataset"])
    data = _load_dataset(dataset)
    visual = _visual_points(dataset)
    lookup = {tuple(np.rint(point).astype(int).tolist()): index for index, point in enumerate(data["points"])}
    all_prediction = np.asarray(result["all_prediction"], dtype=np.float64)
    images: list[np.ndarray] = []
    labels: list[str] = []
    points: list[tuple[float, float]] = []
    predictions: list[tuple[float, float]] = []
    short_names = {
        "a0_standard": "A0（有限/零填充/S32，基线）",
        "a1_zero_s1": "A1（有限/零填充/S1）",
        "a2_torus_s32": "A2（环面/循环/S32）",
        "a3_torus_s1": "A3（环面/循环/S1）",
        "a0_reflect_s32": "A0-R（有限/反射/S32）",
        "a0_finite_circular_s32": "A0-C（有限/循环/S32）",
    }
    for point in visual:
        key = tuple(np.rint(point).astype(int).tolist())
        index = lookup[key]
        images.append(data["images"][index])
        predicted = all_prediction[index]
        labels.append(f"真实中心 ({int(point[0])},{int(point[1])})")
        points.append((float(point[0]), float(point[1])))
        predictions.append((float(predicted[0]), float(predicted[1])))
    path = FIGURE_ROOT / f"predictions_{name}_{dataset}.png"
    _make_montage(
        images,
        labels,
        path,
        title=f"{short_names.get(name, name)}：{('有限' if dataset == 'finite' else '环面')}全256预测（红=真实，蓝=预测）",
        points=points,
        predictions=predictions,
    )
    return str(path)


def _write_trained_prediction_visual(
    name: str,
    model: ToyResNet18,
    data: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> str:
    """Render the same ten configured positions using a best checkpoint."""

    was_training = model.training
    model.eval()
    images_array = np.asarray(data["images"], dtype=np.uint8)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images_array), int(batch_size)):
            batch = torch.from_numpy(images_array[start : start + batch_size]).to(device=device, dtype=torch.float32).div_(255.0)
            predictions.append(model(batch).detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    all_prediction = np.concatenate(predictions, axis=0).astype(np.float64, copy=False)
    if was_training:
        model.train()

    dataset = str(data.get("renderer", _dataset_for_model(name)))
    visual = _visual_points(dataset)
    lookup = {tuple(np.rint(point).astype(int).tolist()): index for index, point in enumerate(data["points"])}
    images: list[np.ndarray] = []
    labels: list[str] = []
    prediction_labels: list[str] = []
    points: list[tuple[float, float]] = []
    selected_predictions: list[tuple[float, float]] = []
    short_names = {
        "a0_standard": "A0（有限/零填充/S32，基线）",
        "a1_zero_s1": "A1（有限/零填充/S1）",
        "a2_torus_s32": "A2（环面/循环/S32）",
        "a3_torus_s1": "A3（环面/循环/S1）",
    }
    for point in visual:
        key = tuple(np.rint(point).astype(int).tolist())
        index = lookup[key]
        images.append(images_array[index])
        predicted = all_prediction[index]
        labels.append(f"真实中心 ({int(point[0])},{int(point[1])})")
        prediction_labels.append(
            f"真({int(point[0])},{int(point[1])}) 预({predicted[0]:.1f},{predicted[1]:.1f})"
        )
        points.append((float(point[0]), float(point[1])))
        selected_predictions.append((float(predicted[0]), float(predicted[1])))
    path = FIGURE_ROOT / f"trained_predictions_{name}_{dataset}.png"
    _make_montage(
        images,
        labels,
        path,
        title=f"{short_names.get(name, name)}：最佳 checkpoint 预测（红=真实，蓝=预测）",
        points=points,
        predictions=selected_predictions,
        prediction_labels=prediction_labels,
    )
    return str(path)


def probe_features() -> dict[str, Any]:
    ensure_output_dirs()
    manifest_path = FEATURE_ROOT / "manifest.json"
    if not manifest_path.exists():
        random_features()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "track_a_toy_fixed_split_ridge_probe",
        "config_fingerprint": config_fingerprint(),
        "shared_state_hash": manifest["shared_state_hash"],
        "models": {},
    }
    for name in MODEL_SPECS:
        result = _probe_one(name, Path(manifest["models"][name]["path"]))
        result["prediction_visualization"] = _write_prediction_visual(name, result)
        result.pop("all_prediction", None)
        summary["models"][name] = result
        write_json(PROBE_ROOT / f"{name}.json", result)
        print(
            f"[probe] {name}: test raw xy MAE={result['global_xy']['test_mae_px']:.3f}px "
            f"phase MAE={result['modulo32_phase']['test_circular_mae_px']:.3f}px "
            f"coarse-cell acc={result['coarse_cell']['test_average_accuracy']:.3f}"
        )
    write_json(PROBE_ROOT / "summary.json", summary)
    return summary


def check_environment(device_name: str | None = None) -> dict[str, Any]:
    ensure_output_dirs()
    device = resolve_device(device_name)
    models, shared_hash = _load_models(device)
    shapes = {name: model_shape_report(model) for name, model in models.items()}
    with torch.no_grad():
        sample = torch.randn(1, 3, int(CONFIG["image_size"]), int(CONFIG["image_size"]), device=device)
        finite = torch.from_numpy(render_one((32.0, 32.0), torus=False)).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)
        torus = torch.from_numpy(render_one((0.0, 0.0), torus=True)).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)
        sample_out = {name: list(model(sample).shape) for name, model in models.items()}
        finite_out = {name: list(models[name](finite).shape) for name in ("a0_standard", "a1_zero_s1", "a0_reflect_s32", "a0_finite_circular_s32")}
        torus_out = {name: list(models[name](torus).shape) for name in ("a2_torus_s32", "a3_torus_s1")}
    all_points, ix, iy = position_grid()
    finite_points, _fix, _fiy = finite_position_grid()
    result = {
        "schema_version": 1,
        "kind": "track_a_toy_check",
        "config_fingerprint": config_fingerprint(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "shared_state_hash": shared_hash,
        "model_shapes": shapes,
        "forward_output_shapes": {"all": sample_out, "finite": finite_out, "torus": torus_out},
        "grid_count": int(len(all_points)),
        "finite_fully_visible_count": int(len(finite_points)),
        "finite_clipped_count": int(len(all_points) - len(finite_points)),
        "grid_ix_shape": list(ix.shape),
        "grid_iy_shape": list(iy.shape),
    }
    write_json(OUTPUT_ROOT / "check.json", result)
    print(f"[check] device={device} cuda={torch.cuda.is_available()} shared_state={shared_hash[:16]}...")
    print(f"[check] grid={len(all_points)} finite_fully_visible={len(finite_points)}")
    return result


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def _format_seconds_cn(value: float) -> str:
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    return f"{minutes / 60.0:.2f} 小时"


def _format_alpha_label(value: Any) -> str:
    """Format an independently selected alpha or an explicit constant baseline."""

    if value is None:
        return "常数基线"
    return f"{float(value):g}"


def _legacy_write_report() -> dict[str, Any]:
    check_path = OUTPUT_ROOT / "check.json"
    data_manifest_path = DATA_ROOT / "manifest.json"
    feature_manifest_path = FEATURE_ROOT / "manifest.json"
    probe_path = PROBE_ROOT / "summary.json"
    for path in (check_path, data_manifest_path, feature_manifest_path, probe_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run check, prepare, random-features and probe first")
    check = json.loads(check_path.read_text(encoding="utf-8"))
    data = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    features = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    rows = []
    for name, result in probe["models"].items():
        rows.append(
            {
                "model": name,
                "scope": "all_256",
                "dataset": result["dataset"],
                "n": result["count"],
                "test_raw_xy_MAE_px": f"{result['global_xy']['test_mae_px']:.3f}",
                "test_raw_xy_R2": f"{result['global_xy']['test_r2']:.3f}",
                "test_phase32_circular_MAE_px": f"{result['modulo32_phase']['test_circular_mae_px']:.3f}",
                "test_coarse_cell_accuracy": f"{result['coarse_cell']['test_average_accuracy']:.3f}",
                "alphas(raw/phase/quotient)": "/".join(
                    _format_alpha_label(result["alphas"][key]) for key in ("raw_xy", "phase_sincos", "quotient_cell")
                ),
                "raw_xy_boundary_caution": result["raw_xy_boundary_caution"],
            }
        )
        interior = result.get("interior_supplementary")
        if interior is not None:
            rows.append(
                {
                    "model": name,
                    "scope": "finite_interior",
                    "dataset": result["dataset"],
                    "n": interior["count"],
                    "test_raw_xy_MAE_px": f"{interior['raw_xy']['test_mae_px']:.3f}",
                    "test_raw_xy_R2": f"{interior['raw_xy']['test_r2']:.3f}",
                    "test_phase32_circular_MAE_px": f"{interior['phase_sincos']['test_circular_mae_px']:.3f}",
                    "test_coarse_cell_accuracy": f"{interior['coarse_cell']['test_average_accuracy']:.3f}",
                    "alphas(raw/phase/quotient)": "/".join(
                        _format_alpha_label(interior["alphas"][key]) for key in ("raw_xy", "phase_sincos", "quotient_cell")
                    ),
                    "raw_xy_boundary_caution": False,
                }
            )
    report = {
        "schema_version": 1,
        "kind": "track_a_toy_report",
        "config_fingerprint": config_fingerprint(),
        "check": check,
        "data": data,
        "features": features,
        "probe": probe,
        "limitations": [
            "This is a local teaching/mechanism reproduction, not publication-grade evidence.",
            "Both finite and torus datasets contain all 256 configured positions; finite fully_visible/interior is a mask (182 true, 74 clipped), and the finite interior probe is supplementary.",
            "Torus raw xy probe metrics are diagnostic only because global coordinates on a periodic domain have a seam; the modulo-32 metric is phase-only, while coarse-cell uses direct labels.",
            "Each raw-xy, phase-sincos, and coarse-cell quotient target has an independently CV-selected ridge alpha; all feature scalers are fit on train split only.",
            "No short training was started in this run; train is implemented separately and remains an explicit user action.",
        ],
    }
    markdown = "\n".join(
        [
            "# Track A-Toy v1 report",
            "",
            f"- config fingerprint: `{config_fingerprint()}`",
            f"- device used for check: `{check['device']}` ({check.get('cuda_name')})",
            f"- shared initialization hash: `{features['shared_state_hash']}`",
            f"- configured grid: `{check['grid_count']}`; finite fully-visible/interior mask: `{check['finite_fully_visible_count']}`; clipped mask entries: `{check['finite_clipped_count']}`",
            "",
            "## Frozen random feature ridge probe",
            "",
            _markdown_table(rows),
            "",
            "## Interpretation guardrails",
            "",
            "- Main rows are `all_256`: A0/A1 and optional finite conditions include clipped finite renders; finite `interior` rows are supplementary mask-only analyses.",
            "- A0/A1 prediction figures use finite all-256 renders; A2/A3 prediction figures use torus all-256 renders.",
            "- Every probe task fits its StandardScaler on the train split only and applies it unchanged to train/test/all.",
            "- Raw xy, phase-sincos, and coarse-cell quotient each select ridge alpha independently by deterministic train-only CV; the table shows raw/phase/quotient alphas.",
            "- `test_phase32_circular_MAE_px` is a modulo-32 phase diagnostic and must not be called global absolute-position recovery.",
            "- `test_coarse_cell_accuracy` uses direct floor(x/32), floor(y/32) labels with regression threshold 0.5; it is not a phase or circular metric.",
            "- Torus raw xy is reported with a seam caution; A2 may retain phase sensitivity while losing coarse-cell/global position.",
            "- The shift diagnostics render each `p` and `p+delta` pair on demand, including deltas absent from the 4-pixel training/probe grid.",
            "",
            "## Output visualizations",
            "",
            "- `figures/inputs_finite.png` and `figures/inputs_torus.png` contain exactly the same ten config-fixed positions, including center, interior, edges/corners, and clipped finite cases.",
            "- `figures/predictions_a0_standard_finite.png` and `figures/predictions_a1_zero_s1_finite.png` are finite all-256 prediction views.",
            "- `figures/predictions_a2_torus_s32_torus.png` and `figures/predictions_a3_torus_s1_torus.png` are torus all-256 views with explicit torus labels.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    write_json(OUTPUT_ROOT / "report.json", report)
    (OUTPUT_ROOT / "report.md").write_text(markdown, encoding="utf-8")
    print(f"[report] wrote {OUTPUT_ROOT / 'report.md'}")
    return report


def write_report() -> dict[str, Any]:
    """Write the final Chinese report and the report-facing figures."""

    check_path = OUTPUT_ROOT / "check.json"
    data_manifest_path = DATA_ROOT / "manifest.json"
    feature_manifest_path = FEATURE_ROOT / "manifest.json"
    probe_path = PROBE_ROOT / "summary.json"
    for path in (check_path, data_manifest_path, feature_manifest_path, probe_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run check, prepare, random-features and probe first")
    check = json.loads(check_path.read_text(encoding="utf-8"))
    data = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    features = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    training_path = TRAIN_ROOT / "summary.json"
    training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else None

    finite_data = _load_dataset("finite")
    torus_data = _load_dataset("torus")
    concept_path = write_concept_figure(FIGURE_ROOT / "实验概念图.png")
    finite_figure = write_input_grid_figure(
        FIGURE_ROOT / "有限画布_固定输入_10张.png",
        dataset="finite",
        images=finite_data["images"],
        points=finite_data["points"],
        fully_visible=finite_data["fully_visible"],
    )
    torus_figure = write_input_grid_figure(
        FIGURE_ROOT / "环面画布_固定输入_10张.png",
        dataset="torus",
        images=torus_data["images"],
        points=torus_data["points"],
        fully_visible=None,
    )
    summary_figure = write_summary_figure(FIGURE_ROOT / "结果总览.png", probe)

    model_labels = {
        "a0_standard": "A0（有限/零填充/S32，基线）",
        "a1_zero_s1": "A1（有限/零填充/S1）",
        "a2_torus_s32": "A2（环面/循环/S32）",
        "a3_torus_s1": "A3（环面/循环/S1）",
    }
    rows: list[dict[str, Any]] = []
    for name, result in probe["models"].items():
        rows.append(
            {
                "条件": name,
                "口径": "all-256 主结果",
                "输入": "有限画布" if result["dataset"] == "finite" else "环面画布",
                "绝对坐标误差（像素）": f"{result['global_xy']['test_mae_px']:.3f}",
                "32像素周期误差（像素）": f"{result['modulo32_phase']['test_circular_mae_px']:.3f}",
                "粗网格准确率": f"{result['coarse_cell']['test_average_accuracy']:.3f}",
                "有效特征数": result["active_feature_count"],
                "probe处理": "常数基线（无有效特征）" if result["constant_baseline"] else "独立ridge",
                "alpha（绝对/周期/粗网格）": "/".join(
                    _format_alpha_label(result["alphas"][key]) for key in ("raw_xy", "phase_sincos", "quotient_cell")
                ),
            }
        )
        interior = result.get("interior_supplementary")
        if interior is not None:
            rows.append(
                {
                    "条件": name,
                    "口径": "finite interior 补充",
                    "输入": "有限画布完整三角形",
                    "绝对坐标误差（像素）": f"{interior['raw_xy']['test_mae_px']:.3f}",
                    "32像素周期误差（像素）": f"{interior['phase_sincos']['test_circular_mae_px']:.3f}",
                    "粗网格准确率": f"{interior['coarse_cell']['test_average_accuracy']:.3f}",
                    "有效特征数": interior["active_feature_count"],
                    "probe处理": "常数基线（无有效特征）" if interior["constant_baseline"] else "独立ridge",
                    "alpha（绝对/周期/粗网格）": "/".join(
                        _format_alpha_label(interior["alphas"][key]) for key in ("raw_xy", "phase_sincos", "quotient_cell")
                    ),
                }
            )

    model_rows = [
        {"条件": "A0：有限/零填充/S32（基线）", "含义": "有限画布的主基线；边界外是零填充，网络总 stride 为 32。"},
        {"条件": "A1：有限/零填充/S1", "含义": "仍是有限画布和零填充，但所有 stride 为 1，保留更多空间分辨率。"},
        {"条件": "A2：环面/循环/S32", "含义": "输入和卷积/池化边界都按环面连接，总 stride 为 32。"},
        {"条件": "A3：环面/循环/S1", "含义": "环面边界且所有 stride 为 1，用来观察没有下采样时的周期不变性。"},
    ]
    wallclock_path = OUTPUT_ROOT / "wallclock_receipt.json"
    wallclock = json.loads(wallclock_path.read_text(encoding="utf-8")) if wallclock_path.exists() else None
    actual_training_seconds = (
        float(training["training_elapsed_seconds"]) if training is not None else None
    )
    actual_nontraining_seconds = (
        float(wallclock["observed_nontraining_seconds"]) if wallclock is not None else None
    )
    actual_clean_seconds = (
        actual_training_seconds + actual_nontraining_seconds
        if actual_training_seconds is not None and actual_nontraining_seconds is not None
        else None
    )
    wallclock_lines = [
        "## 墙钟怎么看",
        "",
        "全过程口径是 check、prepare、random-features、probe、A0–A3 短训练、report 和 tests；不含人工开机/上传等待，也不含复核返工。",
    ]
    if wallclock is not None and actual_clean_seconds is not None:
        wallclock_lines += [
            f"本次纯训练实际耗时 **{_format_seconds_cn(actual_training_seconds)}（{actual_training_seconds:.3f} 秒）**。",
            f"已完成非训练阶段实测 **{_format_seconds_cn(actual_nontraining_seconds)}（{actual_nontraining_seconds:.3f} 秒）**。",
            f"因此，本次干净实验流程实测 **{_format_seconds_cn(actual_clean_seconds)}（{actual_clean_seconds:.3f} 秒）**。",
            "今后本机操作建议预留 **50–60 分钟**；若包括复核或重画图，建议预留 **60–90 分钟**。",
            "详细计算见 [墙钟估算](墙钟估算.md)，机器可读数据见 [wallclock_receipt.json](wallclock_receipt.json)。",
            "### 被实跑推翻的低估对照（不作当前承诺）",
            "正式训练前的 5 步短 profile 只保留作历史对照；由它得到的约 21.7/25.6 分钟已被本次实际训练推翻，不是当前承诺或上限。",
        ]
    elif wallclock is not None:
        wallclock_lines += [
            f"已完成的非训练阶段实测合计 **{_format_seconds_cn(wallclock['observed_nontraining_seconds'])}**。",
            "正式短训练尚未有实际记录；profile 外推只能作为临时规划，不能当作正式全过程承诺。",
            "详细计算见 [墙钟估算](墙钟估算.md)，机器可读数据见 [wallclock_receipt.json](wallclock_receipt.json)。",
        ]
    else:
        wallclock_lines.append("墙钟 receipt 尚未生成；运行 wallclock 命令后会自动补上实测和外推。")

    constant_baseline_lines = [
        "标准化参数只在 train split 拟合；列若满足 `np.isclose` 的 near-constant 规则（rtol=1e-5、atol=1e-6），在 train/test/all 变换后都精确置零。",
        "当有效特征数为 0 时，三个 target 不调用 ridge，而是分别使用各自 train target 的常数均值作为 baseline；因此 alpha 表中的‘常数基线’不是一次 alpha 选择。",
    ]
    constant_models = [
        name for name, result in probe["models"].items() if result.get("constant_baseline")
    ]
    if constant_models:
        constant_baseline_lines.append(
            "本次检测到理论不变特征的条件："
            + "、".join(constant_models)
            + "（active_feature_count=0）；其 raw/phase/coarse 数值均应按常数 baseline 解读，不是被微小 float32 噪声放大的位置证据。"
        )

    training_rows: list[dict[str, Any]] = []
    training_lines: list[str] = ["## 正式短训练结果", ""]
    if training is None:
        training_lines += [
            "本轮尚未执行正式短训练；下面的随机特征 probe 仍是未训练网络的结果。",
            "",
        ]
    else:
        training_lines += [
            "已按冻结规格执行 A0–A3 正式短训练：batch16、FP32、AdamW、全 256 位置每 50 步评估；训练后图像均使用各模型的最佳 checkpoint。",
            f"四个模型正式短训练实际耗时 **{_format_seconds_cn(training['training_elapsed_seconds'])}**；没有记录到 OOM。",
            "机器可读训练记录见 [training/summary.json](training/summary.json)，最佳 checkpoint 和逐次 history 位于 `training/`。",
            "",
            "| 模型 | 实际步数 | 最佳全域 MAE（像素） | 最佳步 | 停止原因 | 模型耗时 |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for name, row in training["models"].items():
            training_rows.append(
                {
                    "模型": model_labels.get(name, name),
                    "实际步数": row["steps_completed"],
                    "最佳全域 MAE（像素）": f"{row['best_mae_px']:.3f}",
                    "最佳步": row["best_step"],
                    "停止原因": "连续两次 MAE<2 像素" if row["stopped_early"] else "达到步数上限",
                    "模型耗时": _format_seconds_cn(row["elapsed_seconds"]),
                }
            )
            training_lines.append(
                f"| {model_labels.get(name, name)} | {row['steps_completed']} | {row['best_mae_px']:.3f} | {row['best_step']} | "
                f"{'连续两次 MAE<2 像素' if row['stopped_early'] else '达到步数上限'} | {_format_seconds_cn(row['elapsed_seconds'])} |"
            )
        training_lines += [
            "",
            "学习曲线和预测图均固定使用同一组 10 个中心位置；红色标记是真实中心，蓝色标记是最佳 checkpoint 的预测中心。",
        ]
        for name, row in training["models"].items():
            curve_name = Path(row["training_curve"]).name
            prediction_name = Path(row["prediction_visualization"]).name
            training_lines += [
                f"### {model_labels.get(name, name)}",
                "",
                f"![{model_labels.get(name, name)}学习曲线](figures/{curve_name})",
                "",
                f"![{model_labels.get(name, name)}最佳checkpoint预测](figures/{prediction_name})",
                "",
            ]

    if training is not None:
        wallclock_lines += [
            "详细步数、停止原因和曲线见上方正式短训练结果。",
        ]

    limitations = [
        (
            "这是本机单 seed 的教学/机制复现实验，不是论文级证据。"
            if training is not None
            else "这是本机单 seed、未训练的教学/机制复现实验，不是论文级证据。"
        ),
        "有限画布和环面画布都保留 256 个位置；有限画布的完整三角形 mask 为 182 个，另外 74 个位置保留为裁切输入。",
        "环面上的绝对坐标误差要谨慎看，因为周期边界有 seam；32 像素指标只描述周期位置，粗网格指标是直接的 cell 标签准确率。",
        "三个 probe 任务各自用 train split 拟合标准化参数并独立选择 alpha。",
    ]
    if training is None:
        limitations.append("本轮没有正式短训练；是否进行短训练仍需用户确认。")
    else:
        limitations.append("正式短训练只有一个 seed，结果用于本机机制观察，不是论文级泛化结论。")
    report = {
        "schema_version": 1,
        "kind": "track_a_toy_report",
        "config_fingerprint": config_fingerprint(),
        "check": check,
        "data": data,
        "features": features,
        "probe": probe,
        "training": training,
        "figures": {
            "concept": concept_path,
            "finite_inputs": finite_figure,
            "torus_inputs": torus_figure,
            "summary": summary_figure,
        },
        "wallclock_receipt": str(wallclock_path) if wallclock is not None else None,
        "actual_wallclock": (
            {
                "training_seconds": actual_training_seconds,
                "nontraining_seconds": actual_nontraining_seconds,
                "clean_process_seconds": actual_clean_seconds,
                "planning_budget_without_review_minutes": "50-60",
                "planning_budget_with_review_or_redraw_minutes": "60-90",
            }
            if actual_clean_seconds is not None
            else None
        ),
        "limitations": limitations,
    }
    markdown = "\n".join(
        [
            "# Track A-Toy v1：边界与位置信息小实验",
            "",
            "## 本次墙钟结论（先看这里）",
            "",
            (
                f"本次纯训练实际耗时约 **{_format_seconds_cn(actual_training_seconds)}（{actual_training_seconds:.3f} 秒）**。"
                if actual_training_seconds is not None
                else "本次尚无正式短训练实测。"
            ),
            (
                f"加上已完成非训练阶段 **{actual_nontraining_seconds:.3f} 秒** 后，本次干净实验流程为 **{actual_clean_seconds:.3f} 秒**，约 **{_format_seconds_cn(actual_clean_seconds)}**。"
                if actual_clean_seconds is not None
                else "正式训练完成后，才能给出本次干净实验流程的实际总时间。"
            ),
            (
                "今后本机操作建议预留 **50–60 分钟**；若包括复核或重画图，建议预留 **60–90 分钟**。"
                if actual_clean_seconds is not None
                else ""
            ),
            "",
            "## 先说结论",
            "",
            (
                "这是一个单 seed 的本机小实验。它先用冻结随机特征观察位置线索，再记录短训练后的拟合结果；不能据此推出普遍结论。"
                if training is not None
                else "这是一个单 seed、未训练的本机小实验。它只回答一个很窄的问题：同一个三角形在画布上移动时，网络的随机特征里还剩多少位置线索。"
            ),
            (
                "正式短训练已按冻结规格完成；随机特征表和训练表是两个不同阶段，不能把训练后的误差当作未训练 probe 证据。"
                if training is not None
                else "本报告的数值不能证明某个模型普遍具有或不具有位置能力；正式短训练尚未进行，是否运行要单独确认。"
            ),
            "",
            "## 模型条件",
            "",
            _markdown_table(model_rows),
            "",
            "术语先用白话说明：**绝对坐标误差**是预测中心和真实中心逐坐标比较后的平均绝对像素差；**32 像素周期位置误差**把坐标按 32 像素周期比较，只看周期内相位；**随机特征**是没有训练过的网络在 head 之前输出的固定向量；**线性 probe**是用一个简单的 ridge 线性回归从这个向量预测坐标。**粗网格 quotient**不是 phase，而是直接预测 `floor(x/32), floor(y/32)` 得到的 0/1 粗网格 cell，本报告用输出阈值 0.5 计算准确率。",
            "",
            "## 先看懂实验",
            "",
            "![有限画布和环面画布概念图](figures/实验概念图.png)",
            "",
            "看图时先记住：有限画布的边界会裁掉三角形；环面画布把越界部分从另一边接回来。每张输入图上的坐标都是三角形中心。",
            "",
            "## 输入图怎么看",
            "",
            "下面两张图各有严格 10 个固定位置，且位置完全相同。有限画布中绿色标签是完整三角形，红色标签是裁切三角形；环面画布中蓝色标签表示图形碰到边界并从另一边绕回。白色区域就是输入三角形。",
            "",
            "![有限画布固定十个输入](figures/有限画布_固定输入_10张.png)",
            "",
            "![环面画布固定十个输入](figures/环面画布_固定输入_10张.png)",
            "",
            "## 结果总览：只先看三块",
            "",
            "先看下面的三块图：左边是绝对坐标误差，中间是 32 像素周期误差，右边是粗网格准确率。前两块越低越好，最后一块越高越好。A0 是有限画布基线；A1 只改 stride；A2/A3 改成环面边界。图中展示的是 all-256 主口径，且全部来自未训练的随机特征线性 probe。",
            "",
            "![三类指标结果总览](figures/结果总览.png)",
            "",
            "## 数值表（主结果和有限画布补充）",
            "",
            _markdown_table(rows),
            "",
            "表中第一行口径是 all-256 主结果；finite interior 是只保留真实三角形完整可见位置的补充，不应与主结果混在一起解读。周期误差只属于 phase 指标；粗网格准确率使用直接的 0/1 cell 标签，没有 circular MAE。",
            *constant_baseline_lines,
            "",
            *training_lines,
            *wallclock_lines,
            "",
            "## 数据和复现记录",
            "",
            f"- 配置指纹：`{config_fingerprint()}`。",
            f"- check 设备：`{check['device']}`（{check.get('cuda_name')}）。",
            f"- 共享初始化指纹：`{features['shared_state_hash']}`；所有条件从同一个基准 state_dict 复制。",
            f"- 网格：256 个中心位置；finite 完整可见 182 个，裁切 74 个。",
            "- shift 诊断会即时渲染每个 `p` 和 `p+Δ`，Δ 为 1、2、4、8、16、32，不从稀疏网格找目标图。",
            "- 旧 Track A 资产没有被导入、移动或删除。",
            "",
            "## 限制",
            "",
            *[f"- {item}" for item in limitations],
            "",
        ]
    )
    write_json(OUTPUT_ROOT / "report.json", report)
    (OUTPUT_ROOT / "report.md").write_text(markdown, encoding="utf-8")
    print(f"[report] wrote {OUTPUT_ROOT / 'report.md'}")
    return report


def _evaluate_model_mae(
    model: ToyResNet18,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    """Evaluate raw xy MAE over the complete materialized domain."""

    was_training = model.training
    model.eval()
    total_abs = 0.0
    total_values = 0
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + batch_size].to(device=device, non_blocking=False)
            target = targets[start : start + batch_size].to(device=device, non_blocking=False)
            prediction = model(batch)
            total_abs += float(torch.abs(prediction - target).sum().detach().cpu())
            total_values += int(target.numel())
    if was_training:
        model.train()
    return total_abs / max(total_values, 1)


def train_models(device_name: str | None = None) -> dict[str, Any]:
    """Run the frozen short training with full-domain evaluation and two-hit early stop."""

    ensure_output_dirs()
    device = resolve_device(device_name)
    if not _dataset_path("finite").exists() or not _dataset_path("torus").exists():
        prepare_datasets()
    models, shared_hash = _load_models(device)
    eval_interval = int(CONFIG["train_eval_interval"])
    batch_size = int(CONFIG["train_batch_size"])
    training_started_utc = datetime.now(timezone.utc).isoformat()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_start = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "track_a_toy_formal_short_training",
        "device": str(device),
        "shared_state_hash": shared_hash,
        "config_fingerprint": config_fingerprint(),
        "training_started_utc": training_started_utc,
        "batch_size": batch_size,
        "optimizer": dict(CONFIG["optimizer"]),
        "evaluation_scope": "all materialized 256 positions, including finite clipped renders",
        "early_stop_rule": "stop after two consecutive full-domain evaluations with MAE < 2 px",
        "formal_training_started": True,
        "oom": False,
        "models": {},
    }
    for name, model in models.items():
        if not bool(MODEL_SPECS[name]["train"]):
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        model_start = time.perf_counter()
        dataset = _dataset_for_model(name)
        data = _load_dataset(dataset)
        if len(data["points"]) != 256:
            raise AssertionError(f"training requires the full 256-position dataset, got {len(data['points'])} for {name}")
        targets = torch.from_numpy(data["points"].astype(np.float32)).to(device=device)
        images = torch.from_numpy(data["images"]).to(device=device, dtype=torch.float32).div_(255.0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(CONFIG["optimizer"]["lr"]),
            weight_decay=float(CONFIG["optimizer"]["weight_decay"]),
        )
        steps = int(CONFIG["train_max_steps"][name])
        max_allowed = 300 if name == "a3_torus_s1" else 1000
        if steps > max_allowed:
            raise AssertionError(f"{name} exceeds the frozen short-training cap: {steps}>{max_allowed}")
        generator = torch.Generator(device=device).manual_seed(int(CONFIG["seed"]))
        model.train()
        history: list[dict[str, float | int | bool]] = []
        best_mae = float("inf")
        best_step = 0
        below_streak = 0
        stopped_early = False
        best_checkpoint = TRAIN_ROOT / f"{name}_best.pt"
        history_path = TRAIN_ROOT / f"{name}_history.json"
        for step in range(1, steps + 1):
            indices = torch.randint(
                0,
                len(images),
                (batch_size,),
                generator=generator,
                device=device,
            )
            prediction = model(images[indices])
            loss = torch.nn.functional.mse_loss(prediction, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % eval_interval == 0 or step == steps:
                full_mae = _evaluate_model_mae(
                    model,
                    images,
                    targets,
                    device=device,
                    batch_size=batch_size,
                )
                below_streak = below_streak + 1 if full_mae < 2.0 else 0
                record: dict[str, float | int | bool] = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "full_domain_mae_px": full_mae,
                    "below_2px_streak": below_streak,
                    "is_best": full_mae < best_mae,
                }
                history.append(record)
                if full_mae < best_mae:
                    best_mae = full_mae
                    best_step = step
                    checkpoint_payload = {
                        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "optimizer_state": optimizer.state_dict(),
                        "shared_state_hash": shared_hash,
                        "config_fingerprint": config_fingerprint(),
                        "model": name,
                        "dataset": dataset,
                        "step": step,
                        "full_domain_mae_px": full_mae,
                    }
                    torch.save(checkpoint_payload, best_checkpoint)
                if below_streak >= 2:
                    stopped_early = True
                    break
        model.eval()
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
        curve_path = write_training_curve_figure(
            FIGURE_ROOT / f"training_curve_{name}.png",
            name,
            history,
        )
        prediction_path = _write_trained_prediction_visual(
            name,
            model,
            data,
            device=device,
            batch_size=batch_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_elapsed = time.perf_counter() - model_start
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        stop_reason = (
            "two_consecutive_full_domain_mae_below_2px"
            if stopped_early
            else "max_steps_reached"
        )
        write_json(history_path, {"model": name, "dataset": dataset, "history": history})
        result["models"][name] = {
            "dataset": dataset,
            "steps_requested": steps,
            "steps_completed": int(history[-1]["step"]) if history else 0,
            "best_mae_px": best_mae,
            "best_step": best_step,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "elapsed_seconds": model_elapsed,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "best_checkpoint": str(best_checkpoint),
            "history_path": str(history_path),
            "training_curve": curve_path,
            "prediction_visualization": prediction_path,
            "history": history,
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["training_elapsed_seconds"] = time.perf_counter() - training_start
    result["training_finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(TRAIN_ROOT / "summary.json", result)
    return result
