from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from track_a_toy.renderer import grid_points_with_visibility, is_fully_visible, render_one
from track_a_toy.visuals import chinese_font, chinese_font_path, write_concept_figure

from .config import (
    ANCHOR_JSON_PATH,
    BATCH_SIZE,
    CONDITIONS,
    EVAL_INTERVAL,
    FIGURE_ROOT,
    GRID_VALUES,
    IMAGE_SIZE,
    MAX_STEPS,
    OPTIMIZER,
    OUTPUT_ROOT,
    POOL_DENOMINATOR,
    PROBE_JSON_PATH,
    REPORT_JSON_PATH,
    REPORT_PATH,
    RIDGE_ALPHAS,
    SAFE_INPUTS_XY,
    SEED,
    SHIFT_ANCHORS_XY,
    SOURCE_DATA_ROOT,
    SPLIT_RULE,
    TRAINING_SUMMARY_PATH,
    TRAIN_ROOT,
    TRIANGLE_VERTICES_XY,
    VISIBLE_INPUTS_XY,
    ensure_output_dirs,
)
from .model import MiniCNN, build_shared_models, seed_everything


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = str(requested).lower()
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _load_source_dataset(dataset: str) -> dict[str, np.ndarray]:
    path = SOURCE_DATA_ROOT / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing reusable source dataset: {path}")
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    if result.get("images", np.empty((0,))).shape[:1] != (256,):
        raise ValueError(f"source {path} must contain all 256 positions")
    return result


def load_source_data() -> dict[str, dict[str, np.ndarray]]:
    return {"finite": _load_source_dataset("finite"), "torus": _load_source_dataset("torus")}


def _raster_bounds(image: np.ndarray) -> tuple[int, int, int, int]:
    foreground = np.asarray(image)[0] > 0
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        raise ValueError("a triangle raster unexpectedly has no foreground pixels")
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def build_masks(finite: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    points = np.asarray(finite["points"], dtype=np.float32)
    ix = np.asarray(finite["ix"], dtype=np.int64)
    iy = np.asarray(finite["iy"], dtype=np.int64)
    visible = np.asarray([is_fully_visible(point) for point in points], dtype=bool)
    if "fully_visible" in finite and not np.array_equal(visible, np.asarray(finite["fully_visible"], dtype=bool)):
        raise AssertionError("recomputed visible mask disagrees with reusable finite data")
    bounds = np.asarray([_raster_bounds(image) for image in finite["images"]], dtype=np.int64)
    safe = (
        visible
        & (points[:, 0] >= 16)
        & (points[:, 0] <= 48)
        & (points[:, 1] >= 12)
        & (points[:, 1] <= 48)
        & (bounds[:, 0] >= 8)
        & (bounds[:, 1] <= 55)
        & (bounds[:, 2] >= 8)
        & (bounds[:, 3] <= 55)
    )
    all_mask = np.ones(len(points), dtype=bool)
    clipped = ~visible
    visible_outer = visible & ~safe
    if (int(visible.sum()), int(safe.sum()), int(visible_outer.sum()), int(clipped.sum())) != (182, 90, 92, 74):
        raise AssertionError(
            "unexpected scope counts: "
            f"visible={int(visible.sum())}, safe={int(safe.sum())}, "
            f"visible_outer={int(visible_outer.sum())}, clipped={int(clipped.sum())}"
        )
    # Keep grid arrays in the result so the split rule is directly auditable.
    return {
        "all": all_mask,
        "visible": visible,
        "safe": safe,
        "visible_outer": visible_outer,
        "clipped": clipped,
        "raster_bounds": bounds,
        "ix": ix,
        "iy": iy,
        "points": points,
    }


def scope_splits(masks: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    ix = np.asarray(masks["ix"], dtype=np.int64)
    iy = np.asarray(masks["iy"], dtype=np.int64)
    held_out = ((ix + 2 * iy) % 4) == 0
    result: dict[str, dict[str, np.ndarray]] = {}
    for scope in ("visible", "safe"):
        scope_mask = np.asarray(masks[scope], dtype=bool)
        test = scope_mask & held_out
        train = scope_mask & ~held_out
        expected = (133, 49) if scope == "visible" else (65, 25)
        if (int(train.sum()), int(test.sum())) != expected:
            raise AssertionError(f"unexpected {scope} split counts: {int(train.sum())}/{int(test.sum())}")
        if np.any(train & test) or np.any((train | test) & ~scope_mask):
            raise AssertionError(f"split masks cross the {scope} scope")
        result[scope] = {"scope": scope_mask, "train": train, "test": test}
    return result


def scope_summary(masks: dict[str, np.ndarray], splits: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    return {
        "all": int(np.asarray(masks["all"]).sum()),
        "visible": {
            "count": int(np.asarray(masks["visible"]).sum()),
            "train_count": int(splits["visible"]["train"].sum()),
            "test_count": int(splits["visible"]["test"].sum()),
        },
        "safe": {
            "count": int(np.asarray(masks["safe"]).sum()),
            "train_count": int(splits["safe"]["train"].sum()),
            "test_count": int(splits["safe"]["test"].sum()),
        },
        "visible_outer": int(np.asarray(masks["visible_outer"]).sum()),
        "clipped": int(np.asarray(masks["clipped"]).sum()),
        "safe_rule": "actual finite raster foreground min>=8 and max<=55, with grid x=16..48 and y=12..48",
        "split_rule": SPLIT_RULE,
    }


def _phase_targets(points: np.ndarray) -> np.ndarray:
    angles = 2.0 * np.pi * np.asarray(points, dtype=np.float64) / float(IMAGE_SIZE)
    return np.column_stack(
        [
            np.sin(angles[:, 0]),
            np.cos(angles[:, 0]),
            np.sin(angles[:, 1]),
            np.cos(angles[:, 1]),
        ]
    ).astype(np.float64)


def _decode_phase(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x = np.mod(np.arctan2(values[:, 0], values[:, 1]) * IMAGE_SIZE / (2.0 * np.pi), IMAGE_SIZE)
    y = np.mod(np.arctan2(values[:, 2], values[:, 3]) * IMAGE_SIZE / (2.0 * np.pi), IMAGE_SIZE)
    return np.column_stack([x, y])


def _circular64_mae(prediction: np.ndarray, truth: np.ndarray) -> float:
    delta = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64))
    delta = np.minimum(delta, IMAGE_SIZE - delta)
    return float(np.mean(delta))


def _raw_xy_mae(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64))))


def _fit_standardizer(train_values: np.ndarray) -> dict[str, np.ndarray | int | float | str]:
    values = np.asarray(train_values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    near = np.isclose(std, 0.0, rtol=1e-5, atol=1e-6)
    safe_std = np.where(near, 1.0, std)
    return {
        "mean": mean,
        "std": safe_std,
        "near_constant": near,
        "active_feature_count": int((~near).sum()),
        "near_constant_tolerance": "np.isclose(std, 0, rtol=1e-5, atol=1e-6), fit on train only",
    }


def _transform_standardizer(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float64) - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    near = np.asarray(scaler["near_constant"], dtype=bool)
    if near.any():
        result[:, near] = 0.0
    return result


def _fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape[1] == 0:
        return np.empty((0, y.shape[1]), dtype=np.float64), y.mean(axis=0)
    y_mean = y.mean(axis=0)
    gram = x.T @ x + float(alpha) * np.eye(x.shape[1], dtype=np.float64)
    coef = np.linalg.solve(gram, x.T @ (y - y_mean))
    return coef, y_mean


def _predict_ridge(features: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return x @ np.asarray(coef, dtype=np.float64) + np.asarray(intercept, dtype=np.float64)


def _select_alpha(features: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if len(values) < 8 or values.shape[1] == 0:
        return float(RIDGE_ALPHAS[0])
    folds = np.arange(len(values), dtype=np.int64) % 4
    scores: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        fold_scores: list[float] = []
        for fold in range(4):
            train = folds != fold
            valid = folds == fold
            if not train.any() or not valid.any():
                continue
            coef, intercept = _fit_ridge(values[train], labels[train], float(alpha))
            prediction = _predict_ridge(values[valid], coef, intercept)
            fold_scores.append(float(np.mean((prediction - labels[valid]) ** 2)))
        scores.append((float(np.mean(fold_scores)), float(alpha)))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def _fit_probe_task(
    features: np.ndarray,
    points: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    target_kind: str,
) -> dict[str, Any]:
    all_features = np.asarray(features, dtype=np.float64)
    all_points = np.asarray(points, dtype=np.float64)
    targets = _phase_targets(all_points) if target_kind == "phase_sincos" else all_points
    scaler = _fit_standardizer(all_features[train_mask])
    scaled_all = _transform_standardizer(all_features, scaler)
    active = np.flatnonzero(~np.asarray(scaler["near_constant"], dtype=bool))
    train_x = scaled_all[train_mask][:, active]
    test_x = scaled_all[test_mask][:, active]
    train_y = targets[train_mask]
    test_y = targets[test_mask]
    if len(active) == 0:
        alpha = None
        coef = np.empty((0, train_y.shape[1]), dtype=np.float64)
        intercept = train_y.mean(axis=0)
        constant_baseline = True
    else:
        alpha = _select_alpha(train_x, train_y)
        coef, intercept = _fit_ridge(train_x, train_y, alpha)
        constant_baseline = False
    test_prediction = _predict_ridge(test_x, coef, intercept)
    baseline_prediction = np.repeat(intercept[None, :], len(test_y), axis=0)
    if target_kind == "phase_sincos":
        decoded = _decode_phase(test_prediction)
        baseline_decoded = _decode_phase(baseline_prediction)
        test_metric = _circular64_mae(decoded, all_points[test_mask])
        baseline_metric = _circular64_mae(baseline_decoded, all_points[test_mask])
    else:
        test_metric = _raw_xy_mae(test_prediction, all_points[test_mask])
        baseline_metric = _raw_xy_mae(baseline_prediction, all_points[test_mask])
    return {
        "target": target_kind,
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "active_feature_count": int(len(active)),
        "constant_baseline": constant_baseline,
        "alpha": alpha,
        "scaler": {
            "fit_scope": "train_only",
            "active_feature_count": int(len(active)),
            "near_constant_count": int((~np.isin(np.arange(all_features.shape[1]), active)).sum()),
            "near_constant_tolerance": scaler["near_constant_tolerance"],
        },
        "test_metric": float(test_metric),
        "baseline_metric": float(baseline_metric),
        "relative_improvement": float((baseline_metric - test_metric) / baseline_metric) if baseline_metric else 0.0,
    }


def _images_to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(images)).to(device=device, dtype=torch.float32).div_(255.0)


def _extract_features(model: MiniCNN, images: np.ndarray, *, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = _images_to_tensor(images, device)
        values = model.forward_features(tensor).detach().cpu().numpy().astype(np.float64, copy=False)
    return values


def run_untrained_probe(
    data: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    models = build_shared_models(device="cpu")
    result: dict[str, Any] = {"seed": SEED, "pool_denominator": POOL_DENOMINATOR, "scopes": {}}
    feature_cache: dict[str, np.ndarray] = {}
    for condition, specification in CONDITIONS.items():
        dataset = str(specification["dataset"])
        feature_cache[condition] = _extract_features(models[condition], data[dataset]["images"], device=torch.device("cpu"))
    for scope in ("visible", "safe"):
        scope_result: dict[str, Any] = {"scope_count": int(masks[scope].sum()), "conditions": {}}
        for condition, specification in CONDITIONS.items():
            dataset = str(specification["dataset"])
            points = np.asarray(data[dataset]["points"], dtype=np.float64)
            split = splits[scope]
            phase = _fit_probe_task(
                feature_cache[condition],
                points,
                split["train"],
                split["test"],
                target_kind="phase_sincos",
            )
            row: dict[str, Any] = {
                "label": specification["label"],
                "dataset": dataset,
                "phase_sincos": phase,
            }
            if dataset == "finite":
                row["raw_xy"] = _fit_probe_task(
                    feature_cache[condition],
                    points,
                    split["train"],
                    split["test"],
                    target_kind="raw_xy",
                )
            scope_result["conditions"][condition] = row
        result["scopes"][scope] = scope_result
    return result


def _make_batch_schedule(train_indices: np.ndarray) -> np.ndarray:
    """Make one deterministic replacement schedule shared by Z/V/C in a scope."""

    rng = np.random.default_rng(SEED)
    indices = np.asarray(train_indices, dtype=np.int64)
    schedule = np.empty((MAX_STEPS, BATCH_SIZE), dtype=np.int64)
    for step in range(MAX_STEPS):
        schedule[step] = rng.choice(indices, size=BATCH_SIZE, replace=True)
    return schedule


def _phase_prediction(model: MiniCNN, images: torch.Tensor, indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = torch.from_numpy(np.asarray(indices[start : start + BATCH_SIZE], dtype=np.int64)).to(device=device)
            predictions.append(model(images[batch_indices]).detach().cpu().numpy().astype(np.float64, copy=False))
    if not predictions:
        return np.empty((0, 4), dtype=np.float64)
    return np.concatenate(predictions, axis=0)


def _phase_metric_for_indices(
    model: MiniCNN,
    images: torch.Tensor,
    points: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    prediction = _phase_prediction(model, images, indices, device)
    return _circular64_mae(_decode_phase(prediction), np.asarray(points, dtype=np.float64)[indices])


def _save_final_checkpoint(path: Path, model: MiniCNN, optimizer: torch.optim.Optimizer, *, scope: str, condition: str) -> None:
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": cpu_state,
            "optimizer_state": optimizer_state,
            "scope": scope,
            "condition": condition,
            "step": MAX_STEPS,
            "seed": SEED,
        },
        path,
    )


def _train_scope(
    scope: str,
    data: dict[str, dict[str, np.ndarray]],
    splits: dict[str, dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    models = build_shared_models(device=device)
    schedule = _make_batch_schedule(np.flatnonzero(splits[scope]["train"]))
    result: dict[str, Any] = {
        "scope": scope,
        "scope_count": int(splits[scope]["scope"].sum()),
        "train_count": int(splits[scope]["train"].sum()),
        "test_count": int(splits[scope]["test"].sum()),
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "optimizer": dict(OPTIMIZER),
        "conditions": {},
    }
    for condition, specification in CONDITIONS.items():
        model = models[condition]
        dataset = str(specification["dataset"])
        source = data[dataset]
        points = np.asarray(source["points"], dtype=np.float64)
        images = _images_to_tensor(source["images"], device)
        targets = torch.from_numpy(_phase_targets(points).astype(np.float32)).to(device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(OPTIMIZER["lr"]),
            weight_decay=float(OPTIMIZER["weight_decay"]),
        )
        model_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        history: list[dict[str, float | int]] = []
        model.train()
        for step in range(1, MAX_STEPS + 1):
            batch_indices = torch.from_numpy(schedule[step - 1]).to(device=device)
            prediction = model(images[batch_indices])
            loss = torch.nn.functional.mse_loss(prediction, targets[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                train_metric = _phase_metric_for_indices(
                    model,
                    images,
                    points,
                    np.flatnonzero(splits[scope]["train"]),
                    device,
                )
                test_metric = _phase_metric_for_indices(
                    model,
                    images,
                    points,
                    np.flatnonzero(splits[scope]["test"]),
                    device,
                )
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "train_circular64_mae_px": train_metric,
                        "test_circular64_mae_px": test_metric,
                    }
                )
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - model_start
        checkpoint_path = TRAIN_ROOT / f"{scope}_{condition}_final.pt"
        history_path = TRAIN_ROOT / f"{scope}_{condition}_history.json"
        _save_final_checkpoint(checkpoint_path, model, optimizer, scope=scope, condition=condition)
        write_json(history_path, {"scope": scope, "condition": condition, "history": history})
        result["conditions"][condition] = {
            "label": specification["label"],
            "dataset": dataset,
            "steps_completed": MAX_STEPS,
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "elapsed_seconds": float(elapsed),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "final_test_circular64_mae_px": float(history[-1]["test_circular64_mae_px"]),
        }
    return result


def run_formal_training(device_name: str | None = None) -> dict[str, Any]:
    """Run the six approved runs; this function is only called by the run CLI."""

    ensure_output_dirs()
    device = resolve_device(device_name)
    seed_everything(SEED)
    data = load_source_data()
    masks = build_masks(data["finite"])
    splits = scope_splits(masks)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scopes = {
        scope: _train_scope(scope, data, splits, device=device)
        for scope in ("visible", "safe")
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "seed": SEED,
        "device": str(device),
        "scope_counts": scope_summary(masks, splits),
        "steps_are_fixed": True,
        "early_stop": False,
        "checkpoint_selection": "final step500 only; no test-based checkpoint selection",
        "clipped_training": False,
        "scopes": scopes,
        "training_elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(TRAINING_SUMMARY_PATH, result)
    return result


def _render_pair_features(model: MiniCNN, left: np.ndarray, right: np.ndarray) -> float:
    left_tensor = _images_to_tensor(np.asarray(left)[None, ...], torch.device("cpu"))
    right_tensor = _images_to_tensor(np.asarray(right)[None, ...], torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        left_feature = model.forward_features(left_tensor).numpy()
        right_feature = model.forward_features(right_tensor).numpy()
    return float(np.mean(np.abs(left_feature - right_feature)))


def anchor_shift_diagnostics() -> dict[str, Any]:
    models = build_shared_models(device="cpu")
    result: dict[str, Any] = {
        "axis": "x",
        "deltas_px": [1, 4, 16],
        "anchor_count": len(SHIFT_ANCHORS_XY),
        "anchors_xy": [list(point) for point in SHIFT_ANCHORS_XY],
        "models": {},
        "rule": "render p and p+delta immediately; finite pairs stay on canvas, torus target wraps modulo64",
    }
    for condition, specification in CONDITIONS.items():
        torus = str(specification["dataset"]) == "torus"
        rows: list[dict[str, Any]] = []
        for delta in (1, 4, 16):
            for anchor in SHIFT_ANCHORS_XY:
                point = np.asarray(anchor, dtype=np.float32)
                target = point + np.asarray([delta, 0], dtype=np.float32)
                if torus:
                    target = target % float(IMAGE_SIZE)
                if not torus and (np.any(target < 0) or np.any(target >= IMAGE_SIZE)):
                    rows.append({"anchor": list(anchor), "delta": delta, "valid": False, "gap_mean_abs": None})
                    continue
                left = render_one(point, torus=torus)
                right = render_one(target, torus=torus)
                rows.append(
                    {
                        "anchor": list(anchor),
                        "delta": delta,
                        "valid": True,
                        "gap_mean_abs": _render_pair_features(models[condition], left, right),
                    }
                )
        result["models"][condition] = {"label": specification["label"], "rows": rows}
    return result


def _center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.FreeTypeFont, *, fill: str = "#172033") -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=3)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=3)


def _cross(draw: ImageDraw.ImageDraw, x: float, y: float, *, color: str, radius: int = 5, width: int = 3) -> None:
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def write_mask_figure(path: Path, masks: dict[str, np.ndarray]) -> str:
    size = 16
    cell = 42
    left, top = 100, 100
    width, height = left + size * cell + 610, top + size * cell + 140
    canvas = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    title = chinese_font(30, bold=True)
    body = chinese_font(18)
    _center_text(draw, (0, 12, width, 55), "位置范围：完整、safe、可见外圈与裁切", title)
    colors = {"safe": "#2878b5", "visible_outer": "#2c9b67", "clipped": "#bd4a45"}
    points = np.asarray(masks["points"], dtype=np.float64)
    for index, point in enumerate(points):
        x = left + int(round(point[0] / 4.0)) * cell
        y = top + int(round(point[1] / 4.0)) * cell
        if bool(masks["safe"][index]):
            fill = colors["safe"]
        elif bool(masks["visible_outer"][index]):
            fill = colors["visible_outer"]
        else:
            fill = colors["clipped"]
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=fill, outline="#172033", width=1)
    draw.rectangle((left, top, left + (size - 1) * cell, top + (size - 1) * cell), outline="#75869a", width=2)
    legend = [
        ("#2878b5", "safe 90：实际前景 min≥8、max≤55"),
        ("#2c9b67", "visible 外圈 92"),
        ("#bd4a45", "clipped 74：完全不训练"),
    ]
    for row, (color, label) in enumerate(legend):
        y = top + row * 48
        draw.rounded_rectangle((left + size * cell + 28, y, left + size * cell + 52, y + 24), radius=5, fill=color)
        draw.text((left + size * cell + 66, y - 2), label, font=body, fill="#172033")
    draw.text((left, height - 72), "每个格点间隔4像素；visible=182，safe=90，visible\u2011safe=92，clipped=74。", font=body, fill="#455468")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _source_image_for_point(data: dict[str, np.ndarray], point: tuple[int, int]) -> np.ndarray:
    lookup = {tuple(np.rint(value).astype(int).tolist()): index for index, value in enumerate(data["points"])}
    key = tuple(int(value) for value in point)
    if key not in lookup:
        raise ValueError(f"input point {key} is not in the reusable grid")
    return np.asarray(data["images"][lookup[key]])


def write_scope_inputs(path: Path, scope: str, data: dict[str, dict[str, np.ndarray]]) -> str:
    points = VISIBLE_INPUTS_XY if scope == "visible" else SAFE_INPUTS_XY
    if len(points) != 10:
        raise AssertionError("each scope input figure must have ten fixed points")
    tile = 160
    cell_w, cell_h = 250, 225
    width, height = cell_w * 5, 110 + cell_h * 2
    canvas = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    title = chinese_font(30, bold=True)
    body = chinese_font(19)
    small = chinese_font(16)
    title_text = "visible=182：固定十个输入" if scope == "visible" else "safe=90：固定十个输入"
    _center_text(draw, (0, 8, width, 52), title_text, title)
    _center_text(draw, (0, 57, width, 100), "白色区域=三角形；坐标=中心；Z/V用finite，C用torus", body, fill="#455468")
    for order, point in enumerate(points):
        row, col = divmod(order, 5)
        left = col * cell_w
        top = 110 + row * cell_h
        image = _source_image_for_point(data["finite"], point)
        tile_image = Image.fromarray(image[0].astype(np.uint8), mode="L").convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
        image_left = left + (cell_w - tile) // 2
        image_top = top + 42
        canvas.paste(tile_image, (image_left, image_top))
        draw.rectangle((image_left, image_top, image_left + tile, image_top + tile), outline="#75869a", width=2)
        _center_text(draw, (left + 2, top + 4, left + cell_w - 2, top + 35), f"中心 ({point[0]},{point[1]})", body)
        status = "完整且safe" if scope == "safe" else "完整可见"
        _center_text(draw, (left + 20, top + 202, left + cell_w - 20, top + 222), status, small, fill="#2878b5" if scope == "safe" else "#2c9b67")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _write_placeholder(path: Path, title_text: str, body_text: str) -> str:
    canvas = Image.new("RGB", (1200, 500), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    title = chinese_font(32, bold=True)
    body = chinese_font(23)
    _center_text(draw, (40, 60, 1160, 150), title_text, title)
    _center_text(draw, (80, 190, 1120, 390), body_text, body, fill="#455468")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def write_probe_result_figure(path: Path, scope: str, probe: dict[str, Any]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_prop = FontProperties(fname=str(chinese_font_path()))
    labels = ["Z\nfinite-zero", "V\nfinite-valid", "C\ntorus-circular"]
    rows = [probe["scopes"][scope]["conditions"][condition] for condition in ("Z", "V", "C")]
    metrics = [
        [float(row["phase_sincos"]["test_metric"]) for row in rows],
        [float(row["phase_sincos"]["baseline_metric"]) for row in rows],
        [100.0 * float(row["phase_sincos"]["relative_improvement"]) for row in rows],
    ]
    titles = ["未训练GAP probe：circular64 MAE（越低越好）", "训练集常数基线 MAE", "相对基线改善（%）"]
    colors = ["#3b6ea8", "#d08a35", "#3b9b78"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8), dpi=170)
    figure.patch.set_facecolor("#f5f7fb")
    for axis, values, title, color in zip(axes, metrics, titles, colors):
        bars = axis.bar(np.arange(3), values, color=color, edgecolor="#243448")
        axis.set_title(title, fontproperties=font_prop, fontsize=13)
        axis.set_xticks(np.arange(3), labels)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontproperties(font_prop)
            tick.set_fontsize(10)
        axis.grid(axis="y", color="#dbe3ec")
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontproperties=font_prop, fontsize=10)
    figure.suptitle(f"{scope} scope：未训练GAP probe结果（phase sin/cos）", fontproperties=font_prop, fontsize=19)
    figure.text(0.5, 0.015, "probe只用scope内train split拟合标准化与alpha；clipped=74从未进入训练或排名。", ha="center", fontproperties=font_prop, fontsize=11, color="#455468")
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return str(path)


def _load_trained_model(path: Path, condition: str) -> MiniCNN:
    mode = str(CONDITIONS[condition]["mode"])
    model = MiniCNN(mode=mode)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _predict_all_phase(model: MiniCNN, images: np.ndarray) -> np.ndarray:
    tensor = _images_to_tensor(images, torch.device("cpu"))
    indices = np.arange(len(images), dtype=np.int64)
    return _decode_phase(_phase_prediction(model, tensor, indices, torch.device("cpu")))


def write_prediction_figure(path: Path, scope: str, data: dict[str, dict[str, np.ndarray]], training: dict[str, Any] | None) -> str:
    required = [TRAIN_ROOT / f"{scope}_{condition}_final.pt" for condition in ("Z", "V", "C")]
    if training is None or not all(item.exists() for item in required):
        return _write_placeholder(path, f"{scope} scope：三条件预测图", "正式六run尚未执行；运行 python -m track_a_minicnn.cli run 后生成最终checkpoint预测。")
    selected = VISIBLE_INPUTS_XY if scope == "visible" else SAFE_INPUTS_XY
    tile = 150
    cell_w, cell_h = 205, 210
    panel_w = cell_w * 5
    width, height = panel_w * 3, 125 + cell_h * 2
    canvas = Image.new("RGB", (width, height), "#f5f7fb")
    draw_canvas = ImageDraw.Draw(canvas)
    title = chinese_font(28, bold=True)
    body = chinese_font(16)
    small = chinese_font(14)
    _center_text(draw_canvas, (0, 8, width, 52), f"{scope} scope：step500 三条件预测（红=真实，蓝=预测）", title)
    prediction_cache: dict[str, np.ndarray] = {}
    for condition in ("Z", "V", "C"):
        model = _load_trained_model(required[("Z", "V", "C").index(condition)], condition)
        prediction_cache[condition] = _predict_all_phase(model, data[str(CONDITIONS[condition]["dataset"])] ["images"])
    for panel_index, condition in enumerate(("Z", "V", "C")):
        panel_left = panel_index * panel_w
        _center_text(draw_canvas, (panel_left, 60, panel_left + panel_w, 105), str(CONDITIONS[condition]["label"]), body, fill="#254e70")
        source = data[str(CONDITIONS[condition]["dataset"])]
        lookup = {tuple(np.rint(value).astype(int).tolist()): index for index, value in enumerate(source["points"])}
        for order, point in enumerate(selected):
            row, col = divmod(order, 5)
            left = panel_left + col * cell_w
            top = 125 + row * cell_h
            index = lookup[tuple(point)]
            image = Image.fromarray(source["images"][index][0].astype(np.uint8), mode="L").convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
            image_left = left + (cell_w - tile) // 2
            image_top = top + 3
            canvas.paste(image, (image_left, image_top))
            _cross(draw_canvas, image_left + point[0] * tile / IMAGE_SIZE, image_top + point[1] * tile / IMAGE_SIZE, color="#d13f38")
            prediction = prediction_cache[condition][index]
            _cross(draw_canvas, image_left + prediction[0] * tile / IMAGE_SIZE, image_top + prediction[1] * tile / IMAGE_SIZE, color="#2c78c4")
            draw_canvas.rectangle((image_left, image_top, image_left + tile, image_top + tile), outline="#75869a", width=2)
            _center_text(draw_canvas, (left + 2, top + 158, left + cell_w - 2, top + 184), f"真({point[0]},{point[1]}) 预({prediction[0]:.1f},{prediction[1]:.1f})", small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _training_from_disk() -> dict[str, Any] | None:
    if not TRAINING_SUMMARY_PATH.exists():
        return None
    return json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))


def _anchor_summary(anchor_data: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition, model_data in anchor_data["models"].items():
        rows = model_data["rows"]
        by_delta: dict[str, Any] = {}
        for delta in (1, 4, 16):
            values = [row["gap_mean_abs"] for row in rows if row["delta"] == delta and row["valid"]]
            by_delta[str(delta)] = {
                "valid_anchor_count": len(values),
                "mean_gap_mean_abs": float(np.mean(values)) if values else None,
            }
        summary[condition] = by_delta
    return summary


def write_report() -> dict[str, Any]:
    """Write the Chinese report and all phase-1 figures without starting training."""

    ensure_output_dirs()
    data = load_source_data()
    masks = build_masks(data["finite"])
    splits = scope_splits(masks)
    counts = scope_summary(masks, splits)
    probe = run_untrained_probe(data, masks, splits)
    write_json(PROBE_JSON_PATH, probe)
    anchors = anchor_shift_diagnostics()
    write_json(ANCHOR_JSON_PATH, anchors)
    training = _training_from_disk()

    figures = {
        "concept": write_concept_figure(FIGURE_ROOT / "实验概念图.png"),
        "position_masks": write_mask_figure(FIGURE_ROOT / "位置范围_mask.png", masks),
        "visible_inputs": write_scope_inputs(FIGURE_ROOT / "visible_固定输入10张.png", "visible", data),
        "safe_inputs": write_scope_inputs(FIGURE_ROOT / "safe_固定输入10张.png", "safe", data),
        "visible_probe": write_probe_result_figure(FIGURE_ROOT / "visible_scope_probe结果.png", "visible", probe),
        "safe_probe": write_probe_result_figure(FIGURE_ROOT / "safe_scope_probe结果.png", "safe", probe),
        "visible_predictions": write_prediction_figure(FIGURE_ROOT / "visible_三条件预测.png", "visible", data, training),
        "safe_predictions": write_prediction_figure(FIGURE_ROOT / "safe_三条件预测.png", "safe", data, training),
    }
    report: dict[str, Any] = {
        "experiment": "Track A MiniCNN Valid-Boundary v1",
        "seed": SEED,
        "geometry": {
            "image_size": IMAGE_SIZE,
            "grid_values": list(GRID_VALUES),
            "triangle_vertices_xy": [list(point) for point in TRIANGLE_VERTICES_XY],
        },
        "architecture": {
            "layers": ["conv3x3_s1 3->16", "conv3x3_s1 16->32", "conv3x3_s1 32->32", "conv3x3_s1 32->32"],
            "activation": "ReLU",
            "batch_norm": False,
            "residual": False,
            "pool": "sum / 4096",
            "head": "32 -> 4 [sin_x, cos_x, sin_y, cos_y]",
            "conditions": CONDITIONS,
        },
        "scope_counts": counts,
        "probe": probe,
        "anchor_shift_diagnostics": anchors,
        "anchor_summary": _anchor_summary(anchors),
        "training": training,
        "figures": figures,
        "training_protocol": {
            "batch_size": BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "eval_interval": EVAL_INTERVAL,
            "early_stop": False,
            "checkpoint_selection": "final step500 only; no test-based selection",
            "clipped_training": False,
        },
    }
    write_json(REPORT_JSON_PATH, report)

    lines: list[str] = [
        "# Track A MiniCNN Valid-Boundary v1",
        "",
        "## 先看结论",
        "",
        "这是一个独立的小型边界条件实验：Z 使用 finite+zero，V 使用 finite+真正的 valid padding0，C 使用 torus+circular；三者都保持 stride=1。",
        "正式协议固定为 visible=182 和 safe=90 两个 scope；clipped=74 只保留作边界说明，完全不训练、不参与排名。",
        (
            "阶段1报告已生成；六个正式训练 run 尚未执行，预测图当前是待训练占位图。"
            if training is None
            else "六个正式 run 已完成；最终结论只使用 step500 的 held-out test，checkpoint 不按 test 误差挑选。"
        ),
        "",
        "## 模型和空间尺寸",
        "",
        _markdown_table(
            ["条件", "输入边界", "网络边界", "四层空间尺寸", "pool/head"],
            [
                ["Z", "finite", "每层显式 zero", "64→64→64→64", "sum/4096→32→4"],
                ["V", "finite", "每层不补边（padding0）", "62→60→58→56", "sum/4096→32→4"],
                ["C", "torus", "每层显式 circular", "64→64→64→64", "sum/4096→32→4"],
            ],
        ),
        "",
        "四层 3×3、stride=1 的理论感受野边长为 9，半径为 4 像素。三条件直接复制同一个 state_dict；卷积均为 bias=False，无 BN、残差、池化或 Dropout。",
        "",
        "## 位置范围和输入图",
        "",
        "visible 按真实三角形顶点和有限画布边界计算，共182个；safe 进一步要求实际 raster 前景 min≥8、max≤55，并固定中心 x=16..48、y=12..48，共90个。",
        "",
        f"![位置范围](figures/{Path(figures['position_masks']).name})",
        "",
        f"![visible固定十个输入](figures/{Path(figures['visible_inputs']).name})",
        "",
        f"![safe固定十个输入](figures/{Path(figures['safe_inputs']).name})",
        "",
        "## 未训练 GAP ridge probe",
        "",
        "probe 只在各 scope 的 train split 拟合标准化和 alpha；near-constant 特征按 train-only 的 `np.isclose(std, 0, rtol=1e-5, atol=1e-6)` 精确置零，不能放大浮点噪声。主指标是 64 像素周期 circular MAE；baseline 是 train target 的常数圆周均值。",
        "",
    ]
    for scope in ("visible", "safe"):
        lines.extend(
            [
                f"### {scope} scope（{counts[scope]['count']}；train={counts[scope]['train_count']}，test={counts[scope]['test_count']}）",
                "",
                f"![{scope} probe结果](figures/{Path(figures[f'{scope}_probe']).name})",
                "",
                _markdown_table(
                    ["条件", "phase test MAE(px)", "baseline(px)", "相对改善", "active特征", "finite raw xy补充"],
                    [
                        [
                            condition,
                            f"{probe['scopes'][scope]['conditions'][condition]['phase_sincos']['test_metric']:.3f}",
                            f"{probe['scopes'][scope]['conditions'][condition]['phase_sincos']['baseline_metric']:.3f}",
                            f"{100.0 * probe['scopes'][scope]['conditions'][condition]['phase_sincos']['relative_improvement']:.1f}%",
                            probe["scopes"][scope]["conditions"][condition]["phase_sincos"]["active_feature_count"],
                            (
                                f"{probe['scopes'][scope]['conditions'][condition]['raw_xy']['test_metric']:.3f}px"
                                if "raw_xy" in probe["scopes"][scope]["conditions"][condition]
                                else "—（torus seam）"
                            ),
                        ]
                        for condition in ("Z", "V", "C")
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 十个锚点的即时平移诊断",
            "",
            "对同一组10个锚点即时渲染 p 和 p+Δ，Δ=1、4、16，比较未训练 GAP 特征的平均绝对变化；finite 不查稀疏网格，torus 对目标坐标取 modulo64。",
            "",
            _markdown_table(
                ["条件", "Δ=1", "Δ=4", "Δ=16"],
                [
                    [
                        condition,
                        *[
                            (
                                f"{report['anchor_summary'][condition][str(delta)]['mean_gap_mean_abs']:.6f}"
                                if report["anchor_summary"][condition][str(delta)]["mean_gap_mean_abs"] is not None
                                else "无有效对"
                            )
                            for delta in (1, 4, 16)
                        ],
                    ]
                    for condition in ("Z", "V", "C")
                ],
            ),
            "",
            "## 正式训练和最终预测",
            "",
            (
                "当前尚未执行正式训练；运行 `python -m track_a_minicnn.cli run --device cuda` 后会生成六个 scope×condition 的 step500 checkpoint、history、耗时和显存记录。"
                if training is None
                else f"六个 run 已完成；总训练墙钟为 {training['training_elapsed_seconds']:.3f} 秒。每个 run 固定500步、每25步记录，不早停、不重试、不增加seed。"
            ),
        ]
    )
    if training is not None:
        training_rows: list[list[Any]] = []
        for scope in ("visible", "safe"):
            for condition in ("Z", "V", "C"):
                row = training["scopes"][scope]["conditions"][condition]
                training_rows.append(
                    [
                        scope,
                        condition,
                        row["steps_completed"],
                        f"{row['final_test_circular64_mae_px']:.3f}",
                        f"{row['elapsed_seconds']:.3f}",
                        row["peak_memory_reserved_bytes"] if row["peak_memory_reserved_bytes"] is not None else "CPU",
                        Path(row["history"]).name,
                    ]
                )
        lines.extend(
            [
                "### step500 训练收据",
                "",
                _markdown_table(["scope", "条件", "steps", "held-out circular64 MAE", "秒", "peak reserved bytes", "history"], training_rows),
                "",
            ]
        )
    lines.extend(
        [
            f"![visible三条件预测](figures/{Path(figures['visible_predictions']).name})",
            "",
            f"![safe三条件预测](figures/{Path(figures['safe_predictions']).name})",
            "",
            "## 口径限制",
            "",
            "- clipped=74 从未进入训练、checkpoint选择或模型排名。",
            "- visible 和 safe 的 test split 分别为 49 和 25，固定规则为 `(ix + 2*iy) mod 4 == 0`；train/test 不交叉。",
            "- C 的 raw xy 只作 seam 敏感诊断，主要结论使用 circular64 MAE。",
            "- 这是单 seed、固定小数据集的机制实验，不是泛化或论文级证据。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_all(device_name: str | None = None) -> dict[str, Any]:
    training = run_formal_training(device_name)
    report = write_report()
    return {"training": training, "report": report}


__all__ = [
    "build_masks",
    "load_source_data",
    "run_all",
    "run_formal_training",
    "run_untrained_probe",
    "scope_splits",
    "write_report",
]
