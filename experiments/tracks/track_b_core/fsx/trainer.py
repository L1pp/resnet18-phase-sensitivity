from __future__ import annotations

import json
import os
import platform
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoints import (
    load_checkpoint_payload,
    optimizer_parameter_names,
    restore_checkpoint,
    resume_identity_from_cell,
    save_checkpoint,
    validate_resume_checkpoint,
)
from .config import PACKAGE_ROOT, load_protocol, resolve_cell, resolved_scientific_config
from .data import (
    b7_endpoint_occurrences,
    canonical_b7_pair_graph,
    coordinate_target,
    coordinate_target_f64,
    preprocess_rgb,
    render_u8,
    resolve_query_target,
)
from .models import (
    build_explicit_xy_mlp,
    build_resnet18,
    build_smoke_cnn,
    calibrate_and_freeze_bn,
    forward_gap_features,
    require_torch,
    seed_all,
    set_trainable_regime,
    torch,
    unfreeze_lp_ft,
)
from .formal_provenance import validate_formal_attempt
from .records import (
    FORMAL_ROOT_MARKER_NAME,
    FormalRootError,
    append_run_catalog,
    create_attempt,
    initialize_project_records,
    path_within_root,
    require_formal_execution_root,
    write_json,
)
from .schedules import b4_causal_lr_for_update, common_lr_for_update
from .science import b6_ridge, metric_summary, support_gate


@dataclass(frozen=True)
class SmokeOptions:
    steps: int = 1
    query_limit: int = 8
    image_side: int = 32
    lp_switch_step: int = 1

    def validate(self) -> None:
        if not 1 <= int(self.steps) <= 4:
            raise ValueError("smoke steps must stay in [1,4]")
        if not 1 <= int(self.query_limit) <= 32:
            raise ValueError("smoke query limit must stay in [1,32]")
        if not 16 <= int(self.image_side) <= 64:
            raise ValueError("smoke image side must stay in [16,64]")
        if int(self.lp_switch_step) != 1:
            raise ValueError("the tiny LPFT smoke transition is fixed at actual step 1")


def _initialize_project_root(root: Path, mode: str) -> None:
    if mode == "formal":
        require_formal_execution_root(root)
    else:
        if (root / FORMAL_ROOT_MARKER_NAME).exists():
            raise FormalRootError("smoke/nonformal runs cannot use a formal execution root")
        initialize_project_records(root)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _asset_path(asset_root: Path, kind: str, artifact_id: str, extension: str) -> Path:
    path = asset_root / kind / f"{artifact_id}.{extension}"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_index(asset_root: Path, index_id: str) -> dict[str, Any]:
    return _load_json(_asset_path(asset_root, "indices", index_id, "json"))


def _load_query(asset_root: Path, query_id: str) -> dict[str, np.ndarray]:
    with np.load(_asset_path(asset_root, "queries", query_id, "npz"), allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _stream_path(asset_root: Path, sampler_id: str) -> Path:
    extension = "npz" if sampler_id.startswith("S_B7_DOUBLE_DRAW") else "npy"
    return _asset_path(asset_root, "streams", sampler_id, extension)


def _load_stream(asset_root: Path, sampler_id: str) -> np.ndarray | dict[str, np.ndarray]:
    path = _stream_path(asset_root, sampler_id)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _index_arrays(index: Mapping[str, Any], target_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = index.get("rows")
    if not isinstance(rows, list) or len(rows) != int(index.get("row_count", -1)):
        raise ValueError("training index row structure is invalid")
    xy = np.asarray([row["position_px"] for row in rows], dtype=np.float64)
    truth = coordinate_target(xy, target_kind)
    appearances = None
    if index.get("renderer") is not None:
        appearances = np.asarray([row["appearance_id"] for row in rows], dtype=np.int64)
        for row in rows:
            if row["target_kind"] != target_kind:
                raise ValueError("index target_kind does not match resolved cell")
    return xy, truth, appearances


def _render_index_images(index: Mapping[str, Any]) -> np.ndarray:
    renderer = index.get("renderer")
    if not isinstance(renderer, str):
        raise ValueError("coordinate-only shard cannot be consumed as images")
    rows = index["rows"]
    return np.stack(
        [
            render_u8(
                row["position_px"],
                int(row["appearance_id"]),
                renderer,
                int(row["appearance_seed"]),
            )
            for row in rows
        ],
        axis=0,
    )


def _image_tensor(images_u8: np.ndarray, device: Any, *, smoke_side: int | None) -> Any:
    value = torch.from_numpy(preprocess_rgb(np.asarray(images_u8))).to(device=device, dtype=torch.float32)
    if smoke_side is not None and tuple(value.shape[-2:]) != (smoke_side, smoke_side):
        value = torch.nn.functional.interpolate(value, size=(smoke_side, smoke_side), mode="bilinear", align_corners=False)
    return value


def _torch_common_loss(prediction: Any, truth: Any) -> Any:
    return torch.mean((prediction - truth) ** 2) + 0.25 * torch.mean(torch.abs(prediction - truth))


def _set_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _adamw(parameters: Sequence[Any], lr: float = 1e-3) -> Any:
    return torch.optim.AdamW(list(parameters), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)


def _optimizer_from_named_groups(model: Any, groups: Sequence[Sequence[str]], *, kind: str) -> Any:
    named = dict(model.named_parameters())
    parameter_groups = []
    for names in groups:
        missing = [name for name in names if name not in named]
        if missing:
            raise ValueError(f"checkpoint parameter names absent from model: {missing}")
        parameter_groups.append({"params": [named[name] for name in names]})
    if kind == "adamw_common":
        return torch.optim.AdamW(parameter_groups, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    if kind == "adamw_b4":
        return torch.optim.AdamW(parameter_groups, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    if kind == "sgd_b4":
        return torch.optim.SGD(parameter_groups, lr=1e-3, momentum=0.0, weight_decay=0.0)
    raise ValueError(kind)


def _build_image_model(cell: Mapping[str, Any], mode: str, device: Any) -> Any:
    normalization = str(cell.get("normalization", "gn"))
    model = (
        build_smoke_cnn(normalization, int(cell["run_seed"]))
        if mode == "smoke"
        else build_resnet18(normalization, int(cell["run_seed"]))
    )
    return model.to(device)


def _query_spec(query_id: str, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [spec for spec in protocol["prepared_inputs"]["query_specs"] if spec["id"] == query_id]
    if len(matches) != 1:
        raise KeyError(query_id)
    return matches[0]


def _query_rows(
    query_id: str,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    *,
    limit: int | None,
) -> tuple[dict[str, np.ndarray], str | None]:
    protocol = load_protocol()
    arrays = _load_query(asset_root, query_id)
    if query_id == "Q_DENSE41_CANONICAL":
        arrays = resolve_query_target(arrays, str(cell["target_kind"]), float(protocol["data"]["coordinate_scale"]))
        renderer = index.get("renderer")
    else:
        expected = coordinate_target(arrays["xy_px"], str(cell["target_kind"]))
        if "truth_u" not in arrays or not np.array_equal(arrays["truth_u"], expected):
            raise ValueError(f"query truth does not match resolved target: {query_id}")
        renderer = _query_spec(query_id, protocol).get("renderer")
        if index.get("renderer") is not None and renderer != index.get("renderer"):
            raise ValueError("dedicated query renderer differs from resolved index renderer")
    if limit is not None:
        arrays = {name: value[:limit] for name, value in arrays.items()}
    return arrays, renderer


def _predict_image_coordinates(
    model: Any,
    arrays: Mapping[str, np.ndarray],
    renderer: str,
    index: Mapping[str, Any],
    device: Any,
    *,
    smoke_side: int | None,
    batch_rows: int = 64,
) -> np.ndarray:
    xy = np.asarray(arrays["xy_px"], dtype=np.float64)
    if "appearance_id" in arrays:
        appearance_ids = np.asarray(arrays["appearance_id"], dtype=np.int64)
        appearance_seeds = np.asarray(arrays["appearance_seed"], dtype=np.int64)
    else:
        row0 = next(row for row in index["rows"] if int(row.get("appearance_id", -1)) == 0)
        appearance_ids = np.zeros(len(xy), dtype=np.int64)
        appearance_seeds = np.full(len(xy), int(row0["appearance_seed"]), dtype=np.int64)
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xy), batch_rows):
            stop = min(start + batch_rows, len(xy))
            images = np.stack(
                [
                    render_u8(xy[row], int(appearance_ids[row]), renderer, int(appearance_seeds[row]))
                    for row in range(start, stop)
                ]
            )
            prediction = model(_image_tensor(images, device, smoke_side=smoke_side))
            output.append(prediction.detach().cpu().numpy().astype(np.float32, copy=False))
    model.train(True)
    return np.concatenate(output, axis=0)


def _predict_coordinate_model(model: Any, xy_px: np.ndarray) -> np.ndarray:
    xy_u = np.asarray(xy_px, dtype=np.float32) / np.float32(223.0)
    model.eval()
    with torch.no_grad():
        result = model(torch.from_numpy(xy_u)).detach().cpu().numpy()
    model.train(True)
    return result.astype(np.float32, copy=False)


def _support_rows(cell: Mapping[str, Any], index: Mapping[str, Any]) -> np.ndarray:
    xy = np.asarray([row["position_px"] for row in index["rows"]], dtype=np.float64)
    unique = np.unique(xy, axis=0)
    if cell["family"] == "B7" and cell["training_path"] == "b7_relative":
        return unique[np.asarray(cell["anchors"], dtype=np.int64)]
    return unique


def _evaluate_and_save(
    run_path: Path,
    model: Any | None,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    device: Any,
    *,
    mode: str,
    query_limit: int | None,
    smoke_side: int | None,
    direct_predictor: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrays_to_save: dict[str, np.ndarray] = {}
    query_metrics: dict[str, Any] = {}
    appearance_breakdown: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    scale = float(load_protocol()["data"]["coordinate_scale"])
    analytic_float64 = str(cell["training_path"]) in {"b6_ridge", "b8_degree2"}
    for q_index, query_id in enumerate(cell["query_ids"]):
        arrays, renderer = _query_rows(query_id, cell, index, asset_root, limit=query_limit)
        if direct_predictor is not None:
            prediction_u = np.asarray(direct_predictor(arrays["xy_px"]))
            if analytic_float64 and prediction_u.dtype != np.float64:
                raise ValueError("analytic baseline prediction must remain float64")
        else:
            if model is None or not isinstance(renderer, str):
                raise ValueError("image evaluator requires a model and renderer")
            prediction_u = _predict_image_coordinates(
                model, arrays, renderer, index, device, smoke_side=smoke_side
            )
        truth_u = (
            coordinate_target_f64(np.asarray(arrays["xy_px"]), str(cell["target_kind"]), scale)
            if analytic_float64
            else np.asarray(arrays["truth_u"], dtype=np.float32)
        )
        prediction_px = prediction_u * (scale if analytic_float64 else np.float32(scale))
        truth_px = truth_u * (scale if analytic_float64 else np.float32(scale))
        prefix = f"q{q_index}"
        arrays_to_save[f"{prefix}_xy_px"] = np.asarray(arrays["xy_px"], dtype=np.float64)
        arrays_to_save[f"{prefix}_pred_u"] = prediction_u
        arrays_to_save[f"{prefix}_truth_u"] = truth_u
        arrays_to_save[f"{prefix}_pred_px"] = prediction_px
        arrays_to_save[f"{prefix}_truth_px"] = truth_px
        for optional in ("appearance_id", "appearance_seed", "field_index", "position_index"):
            if optional in arrays:
                arrays_to_save[f"{prefix}_{optional}"] = np.asarray(arrays[optional])
        query_metrics[query_id] = metric_summary(prediction_px, truth_px)
        if "field_index" in arrays and "appearance_id" in arrays:
            per_field: list[dict[str, Any]] = []
            for field_id in np.unique(np.asarray(arrays["field_index"], dtype=np.int64)):
                mask = np.asarray(arrays["field_index"], dtype=np.int64) == field_id
                field_metrics = metric_summary(prediction_px[mask], truth_px[mask])
                per_field.append(
                    {
                        "field_index": int(field_id),
                        "appearance_id": int(np.asarray(arrays["appearance_id"])[mask][0]),
                        **field_metrics,
                    }
                )
            appearance_breakdown[query_id] = {
                "field_count": len(per_field),
                "per_field": per_field,
                "macro_full_box_raw_mae_px": float(np.mean([row["full_box_raw_mae_px"] for row in per_field])),
                "macro_full_box_u_mae": float(np.mean([row["full_box_u_mae"] for row in per_field])),
                "pooled": query_metrics[query_id],
            }
        manifest.append({"query_id": query_id, "prefix": prefix, "rows": int(len(truth_u))})

    support_xy = _support_rows(cell, index)
    support_truth_u = (
        coordinate_target_f64(support_xy, str(cell["target_kind"]), scale)
        if analytic_float64
        else coordinate_target(support_xy, str(cell["target_kind"]), scale)
    )
    if direct_predictor is not None:
        support_prediction_u = np.asarray(direct_predictor(support_xy))
        if analytic_float64:
            if support_prediction_u.dtype != np.float64:
                raise ValueError("analytic baseline support prediction must remain float64")
        else:
            support_prediction_u = support_prediction_u.astype(np.float32, copy=False)
    else:
        row0 = next(row for row in index["rows"] if int(row.get("appearance_id", -1)) == 0)
        support_arrays = {
            "xy_px": support_xy,
            "appearance_id": np.zeros(len(support_xy), dtype=np.int64),
            "appearance_seed": np.full(len(support_xy), int(row0["appearance_seed"]), dtype=np.int64),
        }
        support_prediction_u = _predict_image_coordinates(
            model,
            support_arrays,
            str(index["renderer"]),
            index,
            device,
            smoke_side=smoke_side,
        )
    arrays_to_save["support_xy_px"] = support_xy
    arrays_to_save["support_pred_u"] = support_prediction_u
    arrays_to_save["support_truth_u"] = support_truth_u
    arrays_to_save["support_pred_px"] = support_prediction_u * (scale if analytic_float64 else np.float32(scale))
    arrays_to_save["support_truth_px"] = support_truth_u * (scale if analytic_float64 else np.float32(scale))
    support_mae = float(
        np.mean(
            np.abs(
                arrays_to_save["support_pred_px"].astype(np.float64)
                - arrays_to_save["support_truth_px"].astype(np.float64)
            )
        )
    )
    np.savez_compressed(run_path / "predictions.npz", **arrays_to_save)
    primary_id = str(cell["query_ids"][0])
    metrics = {
        "primary_query_id": primary_id,
        "primary": query_metrics[primary_id],
        "queries": query_metrics,
        "appearance_breakdown": appearance_breakdown,
        "support_anchor_raw_mae_px": support_mae,
        "support_eligible": support_gate(support_mae),
        "mode": mode,
        "formal_eligible": False if mode == "smoke" else True,
    }
    write_json(run_path / "producer_metrics.json", metrics)
    return metrics, manifest


def _environment(started: float, device: Any) -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    peak = None
    if cuda:
        peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "peak_cuda_bytes": peak,
        "wall_seconds": float(time.perf_counter() - started),
        "cpu_count_visible": os.cpu_count(),
    }


def _resolved_run_config(
    cell_id: str,
    attempt: Mapping[str, str],
    project_root: Path,
    asset_root: Path,
    mode: str,
    device: Any,
    smoke: SmokeOptions | None,
    resume_from: Path | None,
    upstream_run: Path | None,
    paired_initialization: Path | None,
) -> dict[str, Any]:
    resolved = resolved_scientific_config(cell_id)
    cell = resolved["cell"]
    return {
        **resolved,
        "execution": {
            "mode": mode,
            "formal_eligible": mode == "formal",
            "attempt_id": attempt["attempt_id"],
            "run_path": attempt["run_path"],
            "review_path": attempt["review_path"],
            "project_root": str(project_root.resolve()),
            "asset_root": str(asset_root.resolve()),
            "device": str(device),
            "model_variant": (
                "analytic_degree2"
                if cell["training_path"] == "b8_degree2"
                else "explicit_xy_mlp"
                if cell["training_path"] == "b8_explicit_xy_mlp"
                else "tiny_smoke_cnn"
                if mode == "smoke"
                else "resnet18"
            ),
            "smoke_overrides": None if smoke is None else asdict(smoke),
            "resume_from": None if resume_from is None else str(resume_from.resolve()),
            "upstream_run": None if upstream_run is None else str(upstream_run.resolve()),
            "paired_initialization": None if paired_initialization is None else str(paired_initialization.resolve()),
            "input_modalities": (
                ["coordinate", "target"]
                if cell["training_path"] in {"b8_degree2", "b8_explicit_xy_mlp"}
                else ["image", "target"]
            ),
            "init_seed": int(cell["run_seed"]),
        },
    }


def _common_train(
    model: Any,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    device: Any,
    run_path: Path,
    *,
    mode: str,
    smoke: SmokeOptions | None,
    resume_from: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resume_identity = resume_identity_from_cell(cell)
    images = _render_index_images(index)
    _, truth, _ = _index_arrays(index, str(cell["target_kind"]))
    stream = _load_stream(asset_root, str(cell["sampler_id"]))
    if not isinstance(stream, np.ndarray) or stream.shape[1] != 64:
        raise ValueError("common stream must be [T,64]")
    total_actual = int(smoke.steps) if smoke is not None else 3000
    smoke_side = int(smoke.image_side) if smoke is not None else None
    optimizer = _adamw([parameter for parameter in model.parameters() if parameter.requires_grad])
    completed = 0
    resume_runtime: dict[str, Any] = {}
    if resume_from is not None:
        payload = load_checkpoint_payload(resume_from, map_location=device)
        groups = payload["optimizer_parameter_names"]
        optimizer = _optimizer_from_named_groups(model, groups, kind="adamw_common")
        payload = restore_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_schedule_kind="common_cosine_after_optimizer",
            expected_resume_identity=resume_identity,
            expected_parameter_names=groups,
            map_location=device,
        )
        completed = int(payload["completed_step"])
        if int(payload["stream_cursor"]) != completed:
            raise ValueError("common resume stream cursor drift")
        resume_runtime = dict(payload.get("runtime_state", {}))
    trace: list[dict[str, Any]] = [dict(row) for row in resume_runtime.get("trace", [])]
    if resume_from is not None and len(trace) != completed:
        raise ValueError("common checkpoint does not contain a complete trace prefix")
    if cell["training_path"] == "common_frozen_bn" and completed == 0:
        row_ids = np.tile(np.arange(4, dtype=np.int64), 64)
        calibration = _image_tensor(images[row_ids], device, smoke_side=smoke_side)
        calibration_report = calibrate_and_freeze_bn(model, calibration)
    elif cell["training_path"] == "common_frozen_bn" and completed > 0:
        frozen_layers = [module for module in model.modules() if hasattr(module, "_stats_frozen")]
        if not frozen_layers:
            raise ValueError("FrozenBN resume model has no frozen-stat layers")
        for layer in frozen_layers:
            layer.freeze_running_stats()
        calibration_report = resume_runtime.get("calibration_report")
    else:
        calibration_report = None
    for actual_step in range(completed + 1, total_actual + 1):
        protocol_step = actual_step
        ids = np.asarray(stream[actual_step - 1], dtype=np.int64)
        if ids.shape != (64,):
            raise ValueError("fixed common batch lost width 64")
        lr = common_lr_for_update(protocol_step)
        _set_lr(optimizer, lr)
        batch = _image_tensor(images[ids], device, smoke_side=smoke_side)
        target = torch.from_numpy(truth[ids]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = _torch_common_loss(prediction, target)
        loss.backward()
        optimizer.step()
        trace.append({"actual_step": actual_step, "protocol_step": protocol_step, "lr": lr, "loss": float(loss.detach().cpu()), "batch_occurrences": 64})
    save_checkpoint(
        run_path / "checkpoints" / "final.pt",
        model,
        optimizer,
        completed_step=total_actual,
        stream_cursor=total_actual,
        schedule_kind="common_cosine_after_optimizer",
        resume_identity=resume_identity,
        stage_state={"stage": "complete", "formal_completed_step": total_actual if mode == "formal" else None},
        runtime_state={"trace": trace, "calibration_report": calibration_report},
    )
    return trace, {"path_kind": "common", "loss_semantics": "mean_mse_plus_0.25_mean_l1", "calibration": calibration_report}


def _b4_train(
    model: Any,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    device: Any,
    run_path: Path,
    *,
    mode: str,
    smoke: SmokeOptions | None,
    resume_from: Path | None,
    paired_initialization: Path,
    gate_threshold_px: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resume_identity = resume_identity_from_cell(cell)
    if not paired_initialization.is_file():
        raise FileNotFoundError(paired_initialization)
    initial = torch.load(paired_initialization, map_location=device, weights_only=False)
    if initial.get("run_seed") != int(cell["run_seed"]):
        raise ValueError("paired initialization seed mismatch")
    model.load_state_dict(initial["model_state"], strict=True)
    images = _render_index_images(index)
    _, truth, _ = _index_arrays(index, str(cell["target_kind"]))
    stream = _load_stream(asset_root, str(cell["sampler_id"]))
    if not isinstance(stream, np.ndarray) or stream.shape[1] != 64:
        raise ValueError("B4 stream must be [T,64]")
    optimizer_kind = str(cell["optimizer"])
    if optimizer_kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
        restore_kind = "adamw_b4"
    elif optimizer_kind == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0, weight_decay=0.0)
        restore_kind = "sgd_b4"
    else:
        raise ValueError(optimizer_kind)
    completed = 0
    resume_runtime: dict[str, Any] = {}
    if resume_from is not None:
        payload = load_checkpoint_payload(resume_from, map_location=device)
        groups = payload["optimizer_parameter_names"]
        optimizer = _optimizer_from_named_groups(model, groups, kind=restore_kind)
        payload = restore_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_schedule_kind="b4_warmup_then_cosine",
            expected_resume_identity=resume_identity,
            expected_parameter_names=groups,
            map_location=device,
        )
        completed = int(payload["completed_step"])
        if int(payload["stream_cursor"]) != completed:
            raise ValueError("B4 resume stream cursor drift")
        resume_runtime = dict(payload.get("runtime_state", {}))
    total_actual = int(smoke.steps) if smoke is not None else 20000
    smoke_side = int(smoke.image_side) if smoke is not None else None
    effective_gate_threshold = 0.25 if gate_threshold_px is None else float(gate_threshold_px)
    if completed > total_actual:
        raise ValueError("B4 resume checkpoint is past the requested final step")
    trace: list[dict[str, Any]] = [dict(row) for row in resume_runtime.get("trace", [])]
    gate_history: list[dict[str, Any]] = [dict(row) for row in resume_runtime.get("gate_history", [])]
    restored_predictions = np.asarray(resume_runtime.get("gate_predictions", np.empty((0, 4, 2), dtype=np.float32)))
    gate_predictions: list[np.ndarray] = [row.astype(np.float32, copy=True) for row in restored_predictions]
    gate_truth = np.asarray(
        resume_runtime.get(
            "gate_truth",
            coordinate_target(np.asarray([row["position_px"] for row in index["rows"]]), str(cell["target_kind"])),
        ),
        dtype=np.float32,
    )
    restored_first_gate = resume_runtime.get("first_gate")
    first_gate: dict[str, Any] | None = None if restored_first_gate is None else dict(restored_first_gate)
    snapshot_metrics: dict[str, Any] = {str(key): dict(value) for key, value in resume_runtime.get("snapshot_metrics", {}).items()}
    snapshot_payloads: dict[str, dict[str, np.ndarray]] = {
        str(label): {str(key): np.asarray(value) for key, value in payload_value.items()}
        for label, payload_value in resume_runtime.get("snapshot_payloads", {}).items()
    }
    if resume_from is not None:
        if not resume_runtime or len(trace) != completed or len(gate_history) != completed or len(gate_predictions) != completed:
            raise ValueError("B4 checkpoint lacks a complete trace/gate/support prefix")
        if float(resume_runtime.get("gate_threshold_px", effective_gate_threshold)) != effective_gate_threshold:
            raise ValueError("B4 gate threshold changed across resume")
        if first_gate is not None and int(first_gate["selection_step"]) > completed:
            raise ValueError("B4 checkpoint first-gate state is inconsistent with completed step")

    def materialize_snapshot(label: str, payload_value: Mapping[str, np.ndarray]) -> None:
        snapshot_root = run_path / "snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        target = snapshot_root / f"{label}_predictions.npz"
        if target.exists():
            raise FileExistsError(target)
        np.savez_compressed(target, **{key: np.asarray(value) for key, value in payload_value.items()})

    for restored_label, restored_payload in snapshot_payloads.items():
        materialize_snapshot(restored_label, restored_payload)

    def save_raw_snapshot(label: str) -> None:
        if label in snapshot_payloads:
            raise ValueError(f"B4 snapshot label already exists in resumed history: {label}")
        arrays, renderer = _query_rows(
            str(cell["query_ids"][0]),
            cell,
            index,
            asset_root,
            limit=int(smoke.query_limit) if smoke is not None else None,
        )
        prediction_u = _predict_image_coordinates(
            model,
            arrays,
            str(renderer),
            index,
            device,
            smoke_side=smoke_side,
        )
        truth_u = np.asarray(arrays["truth_u"], dtype=np.float32)
        prediction_px = prediction_u * np.float32(223.0)
        truth_px = truth_u * np.float32(223.0)
        payload_value = {
            "xy_px": np.asarray(arrays["xy_px"]),
            "pred_u": prediction_u,
            "truth_u": truth_u,
            "pred_px": prediction_px,
            "truth_px": truth_px,
        }
        snapshot_payloads[label] = payload_value
        materialize_snapshot(label, payload_value)
        snapshot_metrics[label] = metric_summary(prediction_px, truth_px)

    def runtime_state() -> dict[str, Any]:
        return {
            "trace": trace,
            "gate_history": gate_history,
            "gate_predictions": np.stack(gate_predictions) if gate_predictions else np.empty((0, 4, 2), dtype=np.float32),
            "gate_truth": gate_truth,
            "first_gate": first_gate,
            "snapshot_metrics": snapshot_metrics,
            "snapshot_payloads": snapshot_payloads,
            "gate_threshold_px": effective_gate_threshold,
        }
    for step in range(completed + 1, total_actual + 1):
        ids = np.asarray(stream[step - 1], dtype=np.int64)
        lr = b4_causal_lr_for_update(step)
        _set_lr(optimizer, lr)
        batch = _image_tensor(images[ids], device, smoke_side=smoke_side)
        target = torch.from_numpy(truth[ids]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        support_arrays = {"xy_px": np.asarray([row["position_px"] for row in index["rows"]]), "appearance_id": np.zeros(4, dtype=np.int64), "appearance_seed": np.full(4, int(index["rows"][0]["appearance_seed"]), dtype=np.int64)}
        support_prediction = _predict_image_coordinates(model, support_arrays, str(index["renderer"]), index, device, smoke_side=smoke_side)
        support_truth = coordinate_target(support_arrays["xy_px"], str(cell["target_kind"]))
        gate_mae = float(np.mean(np.abs((support_prediction - support_truth) * np.float32(223.0))))
        gate_predictions.append(support_prediction.astype(np.float32, copy=True))
        gate_row = {
            "step": step,
            "support_anchor_raw_mae_px": gate_mae,
            "passes": support_gate(gate_mae, effective_gate_threshold),
        }
        gate_history.append(gate_row)
        selected_this_step = False
        if first_gate is None and gate_row["passes"]:
            first_gate = {"selection_step": step, "before": gate_history[-2] if len(gate_history) > 1 else None, "at": gate_row, "after": None}
            selected_this_step = True
        elif first_gate is not None and first_gate["after"] is None and step == int(first_gate["selection_step"]) + 1:
            first_gate["after"] = gate_row
        trace.append({"actual_step": step, "protocol_step": step, "lr": lr, "loss": float(loss.detach().cpu()), "batch_occurrences": 64, **gate_row})
        if selected_this_step:
            save_raw_snapshot("first_gate")
            save_checkpoint(
                run_path / "checkpoints" / "first_gate.pt",
                model,
                optimizer,
                completed_step=step,
                stream_cursor=step,
                schedule_kind="b4_warmup_then_cosine",
                resume_identity=resume_identity,
                stage_state={"stage": "first_gate", "optimizer": optimizer_kind},
                runtime_state=runtime_state(),
            )
        if mode == "formal" and step in {3000, 20000}:
            save_raw_snapshot(f"step_{step}")
            save_checkpoint(
                run_path / "checkpoints" / f"step_{step}.pt",
                model,
                optimizer,
                completed_step=step,
                stream_cursor=step,
                schedule_kind="b4_warmup_then_cosine",
                resume_identity=resume_identity,
                stage_state={"stage": f"step_{step}", "optimizer": optimizer_kind},
                runtime_state=runtime_state(),
            )
    save_checkpoint(
        run_path / "checkpoints" / "final.pt",
        model,
        optimizer,
        completed_step=total_actual,
        stream_cursor=total_actual,
        schedule_kind="b4_warmup_then_cosine",
        resume_identity=resume_identity,
        stage_state={"stage": "complete", "optimizer": optimizer_kind},
        runtime_state=runtime_state(),
    )
    np.savez_compressed(
        run_path / "b4_support_trajectory.npz",
        support_pred_u=np.stack(gate_predictions),
        support_truth_u=gate_truth.astype(np.float32),
        support_xy_px=np.asarray([row["position_px"] for row in index["rows"]], dtype=np.float64),
    )
    return trace, {
        "path_kind": "b4_causal",
        "optimizer": optimizer_kind,
        "loss_semantics": "pure_mean_mse",
        "paired_initialization": str(paired_initialization.resolve()),
        "first_gate": first_gate or {"selection_step": None, "before": gate_history[-1] if gate_history else None, "at": None, "after": None},
        "support_trajectory_path": str((run_path / "b4_support_trajectory.npz").resolve()),
        "snapshot_metrics": snapshot_metrics,
        "required_formal_snapshots": {"first_gate_when_reached": first_gate is not None, "step_3000": mode == "formal", "step_20000": mode == "formal"},
        "resume_source_checkpoint": None if resume_from is None else str(resume_from.resolve()),
        "gate_threshold_px": effective_gate_threshold,
    }


def _validate_upstream(upstream_run: Path, seed: int) -> tuple[dict[str, Any], Path]:
    status = _load_json(upstream_run / "status.json")
    config = _load_json(upstream_run / "run_config.json")
    if status.get("state") != "COMPLETE":
        raise ValueError("B6 upstream is not complete")
    upstream_cell = config.get("cell", {})
    if upstream_cell.get("family") != "B6" or upstream_cell.get("condition_id") != "g64_upstream" or int(upstream_cell.get("run_seed", -1)) != int(seed):
        raise ValueError("B6 downstream must read the same-seed new g64_upstream run")
    checkpoint = upstream_run / "checkpoints" / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return config, checkpoint


def _b6_ridge_run(
    model: Any,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    device: Any,
    run_path: Path,
    upstream_checkpoint: Path,
    *,
    smoke: SmokeOptions | None,
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    payload = load_checkpoint_payload(upstream_checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    images = _render_index_images(index)
    xy, _, _ = _index_arrays(index, str(cell["target_kind"]))
    truth = coordinate_target_f64(xy, str(cell["target_kind"]))
    smoke_side = int(smoke.image_side) if smoke is not None else None
    with torch.no_grad():
        train_features = forward_gap_features(model, _image_tensor(images, device, smoke_side=smoke_side)).cpu().numpy().astype(np.float64)
    primary_query, renderer = _query_rows("Q_DENSE41_CANONICAL", cell, index, asset_root, limit=None if smoke is None else int(smoke.query_limit))
    query_images = np.stack([render_u8(point, 0, str(renderer), int(index["rows"][0]["appearance_seed"])) for point in primary_query["xy_px"]])
    with torch.no_grad():
        query_features = forward_gap_features(model, _image_tensor(query_images, device, smoke_side=smoke_side)).cpu().numpy().astype(np.float64)
    multipliers = [float(value) for value in load_protocol()["family_specific"]["B6"]["ridge_multipliers"]]
    sensitivity: dict[str, Any] = {}
    primary = None
    for multiplier in multipliers:
        result = b6_ridge(train_features, truth, query_features, multiplier=multiplier)
        sensitivity[str(multiplier)] = {"alpha": result["alpha"], "rank": result["rank"], "prediction_u": result["prediction"].tolist()}
        if multiplier == 1e-4:
            primary = result
    if primary is None:
        raise AssertionError("B6 primary ridge multiplier missing")
    support_result = b6_ridge(train_features, truth, train_features, multiplier=1e-4)
    np.savez_compressed(
        run_path / "ridge_artifact.npz",
        train_features=train_features,
        train_truth_u=truth,
        query_features=query_features,
        beta=primary["beta"],
        singular_values=primary["singular_values"],
    )
    facts = {
        "path_kind": "b6_ridge",
        "input_rows": 4,
        "feature_source": "same_seed_new_g64_upstream_gap",
        "fit_dtype": "float64",
        "fit_heldout_data_used": False,
        "primary_multiplier": 1e-4,
        "rcond": 1e-6,
        "sensitivity": sensitivity,
    }
    direct = {
        "primary_xy": np.asarray(primary_query["xy_px"]),
        "primary_prediction": np.asarray(primary["prediction"], dtype=np.float64),
        "support_xy": xy,
        "support_prediction": np.asarray(support_result["prediction"], dtype=np.float64),
    }
    return model, facts, direct


def _b6_train(
    model: Any,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    device: Any,
    run_path: Path,
    upstream_checkpoint: Path,
    *,
    mode: str,
    smoke: SmokeOptions | None,
    resume_from: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resume_identity = resume_identity_from_cell(cell)
    upstream = load_checkpoint_payload(upstream_checkpoint, map_location=device)
    model.load_state_dict(upstream["model_state"], strict=True)
    regime = str(cell["condition_id"])
    trainable = set_trainable_regime(model, regime)
    images = _render_index_images(index)
    _, truth, _ = _index_arrays(index, str(cell["target_kind"]))
    stream = _load_stream(asset_root, str(cell["sampler_id"]))
    if not isinstance(stream, np.ndarray) or stream.shape[1] != 64:
        raise ValueError("B6 downstream stream must be [T,64]")
    optimizer = _adamw([parameter for parameter in model.parameters() if parameter.requires_grad])
    completed = 0
    stage = "lp_only" if regime == "lp_ft" else regime
    resume_runtime: dict[str, Any] = {}
    if resume_from is not None:
        checkpoint = load_checkpoint_payload(resume_from, map_location=device)
        stage = str(checkpoint["stage_state"].get("stage", stage))
        if regime == "lp_ft" and stage in {"full", "full_pre_update"}:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        groups = checkpoint["optimizer_parameter_names"]
        optimizer = _optimizer_from_named_groups(model, groups, kind="adamw_common")
        checkpoint = restore_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_schedule_kind="common_cosine_after_optimizer",
            expected_resume_identity=resume_identity,
            expected_parameter_names=groups,
            map_location=device,
        )
        completed = int(checkpoint["completed_step"])
        if int(checkpoint["stream_cursor"]) != completed:
            raise ValueError("B6 resume stream cursor drift")
        resume_runtime = dict(checkpoint.get("runtime_state", {}))
    total_actual = (2 if regime == "lp_ft" else int(smoke.steps)) if smoke is not None else 3000
    switch_actual = int(smoke.lp_switch_step) if smoke is not None else 500
    smoke_side = int(smoke.image_side) if smoke is not None else None
    if completed > total_actual:
        raise ValueError("B6 resume checkpoint is past the requested final step")
    trace: list[dict[str, Any]] = [dict(row) for row in resume_runtime.get("trace", [])]
    restored_transition = resume_runtime.get("transition")
    transition: dict[str, Any] | None = None if restored_transition is None else dict(restored_transition)
    boundary_metrics: dict[str, Any] = {str(key): dict(value) for key, value in resume_runtime.get("boundary_metrics", {}).items()}
    boundary_payloads: dict[str, dict[str, np.ndarray]] = {
        str(label): {str(key): np.asarray(value) for key, value in payload_value.items()}
        for label, payload_value in resume_runtime.get("boundary_payloads", {}).items()
    }
    if resume_from is not None and (not resume_runtime or len(trace) != completed):
        raise ValueError("B6 checkpoint does not contain a complete trace prefix")

    def materialize_lpft_snapshot(label: str, payload_value: Mapping[str, np.ndarray]) -> None:
        snapshot_root = run_path / "snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        target = snapshot_root / f"{label}_predictions.npz"
        if target.exists():
            raise FileExistsError(target)
        np.savez_compressed(target, **{key: np.asarray(value) for key, value in payload_value.items()})

    for restored_label, restored_payload in boundary_payloads.items():
        materialize_lpft_snapshot(restored_label, restored_payload)

    def save_lpft_raw_snapshot(label: str) -> None:
        if label in boundary_payloads:
            raise ValueError(f"LPFT boundary snapshot label already exists: {label}")
        query, renderer = _query_rows(
            str(cell["query_ids"][0]),
            cell,
            index,
            asset_root,
            limit=int(smoke.query_limit) if smoke is not None else None,
        )
        prediction_u = _predict_image_coordinates(
            model,
            query,
            str(renderer),
            index,
            device,
            smoke_side=smoke_side,
        )
        truth_u = np.asarray(query["truth_u"], dtype=np.float32)
        prediction_px = prediction_u * np.float32(223.0)
        truth_px = truth_u * np.float32(223.0)
        payload_value = {
            "xy_px": np.asarray(query["xy_px"]),
            "pred_u": prediction_u,
            "truth_u": truth_u,
            "pred_px": prediction_px,
            "truth_px": truth_px,
        }
        boundary_payloads[label] = payload_value
        materialize_lpft_snapshot(label, payload_value)
        boundary_metrics[label] = metric_summary(prediction_px, truth_px)

    def b6_runtime_state() -> dict[str, Any]:
        return {
            "trace": trace,
            "transition": transition,
            "boundary_metrics": boundary_metrics,
            "boundary_payloads": boundary_payloads,
        }

    if resume_from is not None and regime == "lp_ft" and stage == "full_pre_update":
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("LPFT pre-step501 resume did not restore all parameters as trainable")
        save_checkpoint(
            run_path / "checkpoints" / "pre_step501.pt",
            model,
            optimizer,
            completed_step=completed,
            stream_cursor=completed,
            schedule_kind="common_cosine_after_optimizer",
            resume_identity=resume_identity,
            schedule_position=500,
            stage_state=dict(checkpoint["stage_state"]),
            runtime_state=b6_runtime_state(),
        )
        stage = "full"
    for actual_step in range(completed + 1, total_actual + 1):
        protocol_step = actual_step
        if smoke is not None and regime == "lp_ft":
            protocol_step = 500 if actual_step == 1 else 501
        if regime == "lp_ft" and actual_step == switch_actual + 1 and stage == "lp_only":
            fc_state_ids = {id(parameter): len(optimizer.state.get(parameter, {})) for parameter in model.fc.parameters()}
            newly_names = unfreeze_lp_ft(model)
            newly_parameters = [parameter for name, parameter in model.named_parameters() if name in set(newly_names)]
            new_state_entries_before = sum(len(optimizer.state.get(parameter, {})) for parameter in newly_parameters)
            lr = common_lr_for_update(protocol_step)
            optimizer.add_param_group({"params": newly_parameters, "lr": lr})
            fc_state_preserved = all(len(optimizer.state.get(parameter, {})) == fc_state_ids[id(parameter)] for parameter in model.fc.parameters())
            stage = "full"
            transition = {
                "selection_step": 500,
                "first_full_update_step": 501,
                "actual_smoke_selection_step": switch_actual if smoke is not None else None,
                "actual_smoke_first_full_step": actual_step if smoke is not None else None,
                "weights_changed_at_unfreeze": False,
                "fc_optimizer_state_preserved": fc_state_preserved,
                "new_parameter_optimizer_state_entries_before_first_full": new_state_entries_before,
                "new_parameter_names": newly_names,
                "stream_cursor_before_first_full": actual_step - 1,
                "first_full_lr": lr,
            }
            save_checkpoint(
                run_path / "checkpoints" / "pre_step501.pt",
                model,
                optimizer,
                completed_step=actual_step - 1,
                stream_cursor=actual_step - 1,
                schedule_kind="common_cosine_after_optimizer",
                resume_identity=resume_identity,
                schedule_position=500,
                stage_state={
                    "stage": "full_pre_update",
                    "protocol_completed_step": 500,
                    "next_protocol_step": 501,
                    "new_parameter_names": newly_names,
                },
                runtime_state=b6_runtime_state(),
            )
        ids = np.asarray(stream[actual_step - 1], dtype=np.int64)
        lr = common_lr_for_update(protocol_step)
        _set_lr(optimizer, lr)
        batch = _image_tensor(images[ids], device, smoke_side=smoke_side)
        target = torch.from_numpy(truth[ids]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = _torch_common_loss(prediction, target)
        loss.backward()
        optimizer.step()
        trace.append({
            "actual_step": actual_step,
            "protocol_step": protocol_step,
            "lr": lr,
            "loss": float(loss.detach().cpu()),
            "batch_occurrences": 64,
            "stage": "lp_only" if regime == "lp_ft" and actual_step <= switch_actual else stage,
            "trainable_parameter_count": sum(parameter.requires_grad for parameter in model.parameters()),
        })
        if regime == "lp_ft" and actual_step == switch_actual:
            save_lpft_raw_snapshot("lp_end_step500")
            save_checkpoint(
                run_path / "checkpoints" / "lp_end.pt",
                model,
                optimizer,
                completed_step=actual_step,
                stream_cursor=actual_step,
                schedule_kind="common_cosine_after_optimizer",
                resume_identity=resume_identity,
                schedule_position=500,
                stage_state={"stage": "lp_only", "protocol_completed_step": 500, "next_protocol_step": 501},
                runtime_state=b6_runtime_state(),
            )
        if regime == "lp_ft" and actual_step == switch_actual + 1:
            save_lpft_raw_snapshot("post_first_full_step501")
            save_checkpoint(
                run_path / "checkpoints" / "post_step501.pt",
                model,
                optimizer,
                completed_step=actual_step,
                stream_cursor=actual_step,
                schedule_kind="common_cosine_after_optimizer",
                resume_identity=resume_identity,
                schedule_position=501,
                stage_state={"stage": "full", "protocol_completed_step": 501, "first_full_update_completed": True},
                runtime_state=b6_runtime_state(),
            )
    save_checkpoint(
        run_path / "checkpoints" / "final.pt",
        model,
        optimizer,
        completed_step=total_actual,
        stream_cursor=total_actual,
        schedule_kind="common_cosine_after_optimizer",
        resume_identity=resume_identity,
        schedule_position=trace[-1]["protocol_step"],
        stage_state={"stage": stage, "regime": regime, "protocol_completed_step": trace[-1]["protocol_step"]},
        runtime_state=b6_runtime_state(),
    )
    facts = {
        "path_kind": f"b6_{regime}",
        "upstream_checkpoint": str(upstream_checkpoint.resolve()),
        "same_seed_upstream_required": True,
        "initial_trainable_names": trainable,
        "loss_semantics": "mean_mse_plus_0.25_mean_l1",
        "lpft_transition": transition,
        "lpft_boundary_metrics": boundary_metrics,
        "resume_source_checkpoint": None if resume_from is None else str(resume_from.resolve()),
    }
    return trace, facts


def _b7_train(
    model: Any,
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    asset_root: Path,
    device: Any,
    run_path: Path,
    *,
    mode: str,
    smoke: SmokeOptions | None,
    resume_from: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resume_identity = resume_identity_from_cell(cell)
    images = _render_index_images(index)
    _, truth, _ = _index_arrays(index, str(cell["target_kind"]))
    optimizer = _adamw(list(model.parameters()))
    total = int(smoke.steps) if smoke is not None else 3000
    smoke_side = int(smoke.image_side) if smoke is not None else None
    completed = 0
    resume_runtime: dict[str, Any] = {}
    if resume_from is not None:
        payload = load_checkpoint_payload(resume_from, map_location=device)
        stage_state = dict(payload.get("stage_state", {}))
        if stage_state.get("b7_path") != str(cell["training_path"]):
            raise ValueError("B7 resume checkpoint belongs to a different execution path")
        groups = payload["optimizer_parameter_names"]
        optimizer = _optimizer_from_named_groups(model, groups, kind="adamw_common")
        payload = restore_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_schedule_kind="common_cosine_after_optimizer",
            expected_resume_identity=resume_identity,
            expected_parameter_names=groups,
            map_location=device,
        )
        completed = int(payload["completed_step"])
        if int(payload["stream_cursor"]) != completed:
            raise ValueError("B7 resume stream cursor drift")
        resume_runtime = dict(payload.get("runtime_state", {}))
    if completed > total:
        raise ValueError("B7 resume checkpoint is past the requested final step")
    trace: list[dict[str, Any]] = [dict(row) for row in resume_runtime.get("trace", [])]
    if resume_from is not None and len(trace) != completed:
        raise ValueError("B7 checkpoint does not contain a complete trace prefix")
    if cell["training_path"] == "b7_context":
        stream = _load_stream(asset_root, str(cell["sampler_id"]))
        if not isinstance(stream, np.ndarray) or stream.shape[1] != 64:
            raise ValueError("B7 context must use fixed common N4 batch64 stream")
        raw_batch_ids = [np.asarray(value, dtype=np.int64) for value in resume_runtime.get("batch_row_ids", [])]
        raw_predictions = [np.asarray(value, dtype=np.float32) for value in resume_runtime.get("prediction_u", [])]
        raw_truth = [np.asarray(value, dtype=np.float32) for value in resume_runtime.get("truth_u", [])]
        if resume_from is not None and not all(len(values) == completed for values in (raw_batch_ids, raw_predictions, raw_truth)):
            raise ValueError("B7 context checkpoint has incomplete raw training history")
        for step in range(completed + 1, total + 1):
            ids = np.asarray(stream[step - 1], dtype=np.int64)
            lr = common_lr_for_update(step)
            _set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            pred = model(_image_tensor(images[ids], device, smoke_side=smoke_side))
            target = torch.from_numpy(truth[ids]).to(device=device)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            raw_batch_ids.append(ids.copy())
            raw_predictions.append(pred.detach().cpu().numpy().astype(np.float32, copy=True))
            raw_truth.append(target.detach().cpu().numpy().astype(np.float32, copy=True))
            trace.append({"actual_step": step, "protocol_step": step, "lr": lr, "loss": float(loss.detach().cpu()), "batch_occurrences": 64})
        exposure = None
        training_raw = run_path / "b7_training_raw.npz"
        np.savez_compressed(
            training_raw,
            batch_row_ids=np.stack(raw_batch_ids),
            prediction_u=np.stack(raw_predictions),
            truth_u=np.stack(raw_truth),
        )
        runtime_state = {
            "trace": trace,
            "batch_row_ids": raw_batch_ids,
            "prediction_u": raw_predictions,
            "truth_u": raw_truth,
        }
        facts = {
            "path_kind": "b7_context",
            "loss_semantics": "pure_mean_mse_over_64_common_stream_occurrences",
            "training_raw_path": str(training_raw.resolve()),
            "resume_source_checkpoint": None if resume_from is None else str(resume_from.resolve()),
        }
    else:
        stream = _load_stream(asset_root, str(cell["sampler_id"]))
        if not isinstance(stream, dict) or set(stream) != {"image_ids", "pair_ids"}:
            raise ValueError("B7 dense path requires the frozen double-draw stream")
        pair_graph = np.asarray(index["b7_pair_graph"]["pairs"], dtype=np.int64)
        if not np.array_equal(pair_graph, canonical_b7_pair_graph()):
            raise ValueError("B7 index pair graph drift")
        used_pairs = [np.asarray(value, dtype=np.int64) for value in resume_runtime.get("pair_ids", [])]
        used_endpoints = [np.asarray(value, dtype=np.int64) for value in resume_runtime.get("endpoint_row_ids", [])]
        used_image_draws = [np.asarray(value, dtype=np.int64) for value in resume_runtime.get("image_draw_ids", [])]
        raw_endpoint_predictions = [np.asarray(value, dtype=np.float32) for value in resume_runtime.get("endpoint_prediction_u", [])]
        raw_endpoint_truth = [np.asarray(value, dtype=np.float32) for value in resume_runtime.get("endpoint_truth_u", [])]
        raw_anchor_predictions = [np.asarray(value, dtype=np.float32) for value in resume_runtime.get("anchor_prediction_u", [])]
        raw_anchor_truth = [np.asarray(value, dtype=np.float32) for value in resume_runtime.get("anchor_truth_u", [])]
        anchors = np.asarray(cell.get("anchors", []), dtype=np.int64)
        dense_prefixes = (used_pairs, used_endpoints, used_image_draws, raw_endpoint_predictions, raw_endpoint_truth)
        if resume_from is not None and not all(len(values) == completed for values in dense_prefixes):
            raise ValueError("B7 dense checkpoint has incomplete exposure or raw training history")
        expected_anchor_rows = completed if cell["training_path"] == "b7_relative" else 0
        if resume_from is not None and (len(raw_anchor_predictions) != expected_anchor_rows or len(raw_anchor_truth) != expected_anchor_rows):
            raise ValueError("B7 checkpoint anchor history does not match the execution path")
        for step in range(completed + 1, total + 1):
            pair_ids = np.asarray(stream["pair_ids"][step - 1], dtype=np.int64)
            image_draw_ids = np.asarray(stream["image_ids"][step - 1], dtype=np.int64)
            endpoints = b7_endpoint_occurrences(pair_ids, pair_graph)
            if endpoints.shape != (128,):
                raise ValueError("B7 endpoint exposure must be ordered [left64,right64]")
            lr = common_lr_for_update(step)
            _set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            endpoint_pred = model(_image_tensor(images[endpoints], device, smoke_side=smoke_side))
            endpoint_truth = torch.from_numpy(truth[endpoints]).to(device=device)
            if cell["training_path"] == "b7_dense_absolute":
                loss = torch.mean((endpoint_pred - endpoint_truth) ** 2)
                components = {"endpoint_mean_mse": float(loss.detach().cpu())}
            else:
                displacement_pred = endpoint_pred[64:] - endpoint_pred[:64]
                displacement_truth = endpoint_truth[64:] - endpoint_truth[:64]
                pair_loss = torch.mean((displacement_pred - displacement_truth) ** 2)
                anchor_pred = model(_image_tensor(images[anchors], device, smoke_side=smoke_side))
                anchor_truth = torch.from_numpy(truth[anchors]).to(device=device)
                anchor_loss = torch.mean((anchor_pred - anchor_truth) ** 2)
                loss = pair_loss + anchor_loss
                components = {"pair_mean_mse": float(pair_loss.detach().cpu()), "anchor_mean_mse": float(anchor_loss.detach().cpu())}
            loss.backward()
            optimizer.step()
            used_pairs.append(pair_ids.copy())
            used_endpoints.append(endpoints.copy())
            used_image_draws.append(image_draw_ids.copy())
            raw_endpoint_predictions.append(endpoint_pred.detach().cpu().numpy().astype(np.float32, copy=True))
            raw_endpoint_truth.append(endpoint_truth.detach().cpu().numpy().astype(np.float32, copy=True))
            if cell["training_path"] == "b7_relative":
                raw_anchor_predictions.append(anchor_pred.detach().cpu().numpy().astype(np.float32, copy=True))
                raw_anchor_truth.append(anchor_truth.detach().cpu().numpy().astype(np.float32, copy=True))
            trace.append({"actual_step": step, "protocol_step": step, "lr": lr, "loss": float(loss.detach().cpu()), "pair_occurrences": 64, "endpoint_occurrences": 128, **components})
        exposure = run_path / "b7_exposure.npz"
        np.savez_compressed(
            exposure,
            pair_ids=np.stack(used_pairs),
            endpoint_row_ids=np.stack(used_endpoints),
            image_draw_ids=np.stack(used_image_draws),
        )
        training_raw = run_path / "b7_training_raw.npz"
        training_payload = {
            "endpoint_prediction_u": np.stack(raw_endpoint_predictions),
            "endpoint_truth_u": np.stack(raw_endpoint_truth),
        }
        if raw_anchor_predictions:
            training_payload["anchor_prediction_u"] = np.stack(raw_anchor_predictions)
            training_payload["anchor_truth_u"] = np.stack(raw_anchor_truth)
        np.savez_compressed(training_raw, **training_payload)
        runtime_state = {
            "trace": trace,
            "pair_ids": used_pairs,
            "endpoint_row_ids": used_endpoints,
            "image_draw_ids": used_image_draws,
            "endpoint_prediction_u": raw_endpoint_predictions,
            "endpoint_truth_u": raw_endpoint_truth,
            "anchor_prediction_u": raw_anchor_predictions,
            "anchor_truth_u": raw_anchor_truth,
        }
        facts = {
            "path_kind": str(cell["training_path"]),
            "loss_semantics": (
                "pure_mean_mse_over_128_endpoint_occurrences"
                if cell["training_path"] == "b7_dense_absolute"
                else "ordered_right_minus_left_displacement_mean_mse_plus_anchor_mean_mse_1_to_1"
            ),
            "endpoint_order": "left64_then_right64",
            "image_draw_role": "consumed_first_to_preserve_frozen_double_draw_rng_but_not_used_for_matched_dense_loss",
            "pair_occurrences": 64,
            "endpoint_occurrences": 128,
            "anchors": anchors.tolist(),
            "exposure_path": str(exposure.resolve()),
            "training_raw_path": str(training_raw.resolve()),
            "resume_source_checkpoint": None if resume_from is None else str(resume_from.resolve()),
        }
    save_checkpoint(
        run_path / "checkpoints" / "final.pt",
        model,
        optimizer,
        completed_step=total,
        stream_cursor=total,
        schedule_kind="common_cosine_after_optimizer",
        resume_identity=resume_identity,
        stage_state={"stage": "complete", "b7_path": str(cell["training_path"])},
        runtime_state=runtime_state,
    )
    return trace, facts


def _b8_degree2(index: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    xy, _, _ = _index_arrays(index, "quadratic_b8")
    truth = coordinate_target_f64(xy, "quadratic_b8")
    u = xy / 223.0
    design = np.stack((np.ones(len(u)), u[:, 0], u[:, 1], u[:, 0] ** 2, u[:, 0] * u[:, 1], u[:, 1] ** 2), axis=1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, truth, rcond=None)
    if int(rank) != 6:
        raise ValueError("B8 degree2 design is not full rank")

    def predict(query_xy: np.ndarray) -> np.ndarray:
        q = np.asarray(query_xy, dtype=np.float64) / 223.0
        q_design = np.stack((np.ones(len(q)), q[:, 0], q[:, 1], q[:, 0] ** 2, q[:, 0] * q[:, 1], q[:, 1] ** 2), axis=1)
        return (q_design @ coefficients).astype(np.float64, copy=False)

    facts = {
        "path_kind": "b8_degree2",
        "input_modalities": ["coordinate", "target"],
        "image_data_read": False,
        "activation_data_read": False,
        "heldout_fit_data_read": False,
        "basis_order": ["1", "x", "y", "x2", "xy", "y2"],
        "fit_rows": 9,
        "rank": int(rank),
        "coefficients": coefficients.tolist(),
        "singular_values": singular_values.tolist(),
    }
    return predict, facts


def _b8_mlp(
    cell: Mapping[str, Any],
    index: Mapping[str, Any],
    run_path: Path,
    *,
    smoke: SmokeOptions | None,
    resume_from: Path | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    resume_identity = resume_identity_from_cell(cell)
    model = build_explicit_xy_mlp(int(cell["run_seed"]))
    xy, truth, _ = _index_arrays(index, str(cell["target_kind"]))
    x = torch.from_numpy((xy / 223.0).astype(np.float32))
    y = torch.from_numpy(truth.astype(np.float32))
    optimizer = _adamw(list(model.parameters()))
    completed = 0
    resume_runtime: dict[str, Any] = {}
    if resume_from is not None:
        payload = load_checkpoint_payload(resume_from)
        groups = payload["optimizer_parameter_names"]
        optimizer = _optimizer_from_named_groups(model, groups, kind="adamw_common")
        payload = restore_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_schedule_kind="b8_mlp_fixed_lr_no_scheduler",
            expected_resume_identity=resume_identity,
            expected_parameter_names=groups,
        )
        completed = int(payload["completed_step"])
        if int(payload["stream_cursor"]) != completed:
            raise ValueError("B8 MLP resume stream cursor drift")
        resume_runtime = dict(payload.get("runtime_state", {}))
    total = int(smoke.steps) if smoke is not None else 3000
    if completed > total:
        raise ValueError("B8 MLP resume checkpoint is past the requested final step")
    trace: list[dict[str, Any]] = [dict(row) for row in resume_runtime.get("trace", [])]
    if resume_from is not None and len(trace) != completed:
        raise ValueError("B8 MLP checkpoint does not contain a complete trace prefix")
    for step in range(completed + 1, total + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = _torch_common_loss(prediction, y)
        loss.backward()
        optimizer.step()
        trace.append({"actual_step": step, "protocol_step": step, "lr": 1e-3, "loss": float(loss.detach()), "full_batch_rows": 9})
    save_checkpoint(
        run_path / "checkpoints" / "final.pt",
        model,
        optimizer,
        completed_step=total,
        stream_cursor=total,
        schedule_kind="b8_mlp_fixed_lr_no_scheduler",
        resume_identity=resume_identity,
        stage_state={"stage": "complete"},
        runtime_state={"trace": trace},
    )
    return model, trace, {
        "path_kind": "b8_explicit_xy_mlp",
        "input_modalities": ["coordinate", "target"],
        "image_data_read": False,
        "activation_data_read": False,
        "heldout_fit_data_read": False,
        "fit_rows": 9,
        "fixed_lr": 1e-3,
        "scheduler": "none",
        "loss_semantics": "mean_mse_plus_0.25_mean_l1",
    }


def run_cell(
    cell_id: str,
    *,
    project_root: str | Path = PACKAGE_ROOT,
    asset_root: str | Path = PACKAGE_ROOT / "assets",
    mode: str = "smoke",
    device: str | None = None,
    smoke_options: SmokeOptions | None = None,
    resume_from: str | Path | None = None,
    upstream_run: str | Path | None = None,
    paired_initialization: str | Path | None = None,
) -> dict[str, Any]:
    """Run one cell without ever silently promoting a local smoke to formal.

    Formal semantics are implemented, but the frozen protocol currently keeps
    formal training locked.  A later reviewed execution release must explicitly
    authorize it before this function will accept ``mode='formal'``.
    """
    protocol = load_protocol()
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    if mode == "formal" and protocol.get("formal_training_authorized") is not True:
        raise PermissionError("formal training is not authorized by this local implementation release")
    root = Path(project_root).resolve()
    if mode == "formal":
        require_formal_execution_root(root)
    elif (root / FORMAL_ROOT_MARKER_NAME).exists():
        raise FormalRootError("smoke/nonformal runs cannot use a formal execution root")
    require_torch()
    smoke = smoke_options or SmokeOptions()
    if mode == "smoke":
        smoke.validate()
    else:
        smoke = None
    cell = resolve_cell(cell_id)
    resume_path = None if resume_from is None else Path(resume_from)
    if resume_path is not None:
        training_path = str(cell["training_path"])
        if training_path == "b8_degree2":
            raise ValueError("B8 analytic degree2 does not support checkpoint resume")
        if training_path == "b6_ridge":
            raise ValueError("B6 analytic ridge does not support checkpoint resume")
        validate_resume_checkpoint(
            resume_path,
            resume_identity_from_cell(cell),
            map_location="cpu",
        )
        if mode == "formal":
            validate_formal_attempt(
                resume_path.resolve().parent.parent,
                execution_root=root,
                expected_cell_id=cell_id,
                require_complete=False,
            )
    upstream_path = None if upstream_run is None else Path(upstream_run).resolve()
    paired_path = None if paired_initialization is None else Path(paired_initialization).resolve()
    if mode == "formal" and upstream_path is not None:
        validate_formal_attempt(
            upstream_path,
            execution_root=root,
            expected_cell_id=f"B6.g64_upstream.s{int(cell['run_seed'])}",
            require_complete=True,
        )
    if mode == "formal" and str(cell["training_path"]) == "b4_causal":
        if paired_path is None:
            raise ValueError("B4 causal formal runs require the matched initialization before attempt creation")
        path_within_root(paired_path, root / "paired", label="B4 paired initialization")
        expected_pair = (root / "paired" / "B4" / f"seed_{int(cell['run_seed'])}" / "initialization.pt").resolve()
        if paired_path != expected_pair:
            raise ValueError("B4 paired initialization is not the matching formal seed artifact")
    assets = Path(asset_root)
    _initialize_project_root(root, mode)
    attempt = create_attempt(str(cell["family"]), str(cell["condition_id"]), int(cell["run_seed"]), project_root=root)
    run_path = Path(attempt["run_path"])
    review_path = Path(attempt["review_path"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device("cpu" if cell["training_path"] == "b8_explicit_xy_mlp" else device)
    if str(torch_device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch_device)
    started = time.perf_counter()
    config = _resolved_run_config(cell_id, attempt, root, assets, mode, torch_device, smoke, resume_path, upstream_path, paired_path)
    write_json(run_path / "run_config.json", config)
    write_json(run_path / "status.json", {"state": "RUNNING", "formal_eligible": mode == "formal", "cell_id": cell_id})
    try:
        index = _load_index(assets, str(cell["index_id"]))
        path_kind = str(cell["training_path"])
        trace: list[dict[str, Any]] = []
        direct_predictor = None
        model = None
        if path_kind == "b8_degree2":
            direct_predictor, facts = _b8_degree2(index)
            facts["shared_physical_source"] = str(run_path.resolve())
            facts["logical_record_role"] = "seed_labelled_reference_to_shared_analytic_artifact"
            config["execution"]["shared_physical_source"] = str(run_path.resolve())
            write_json(run_path / "run_config.json", config, overwrite=True)
            np.savez_compressed(run_path / "analytic_degree2.npz", coefficients=np.asarray(facts["coefficients"], dtype=np.float64))
        elif path_kind == "b8_explicit_xy_mlp":
            model, trace, facts = _b8_mlp(cell, index, run_path, smoke=smoke, resume_from=resume_path)
            direct_predictor = lambda xy: _predict_coordinate_model(model, xy)
        else:
            model = _build_image_model(cell, mode, torch_device)
            if path_kind in {"common", "common_frozen_bn"}:
                trace, facts = _common_train(model, cell, index, assets, torch_device, run_path, mode=mode, smoke=smoke, resume_from=resume_path)
            elif path_kind == "b4_causal":
                if paired_path is None:
                    raise ValueError("B4 causal arms require an explicit matched initialization artifact")
                trace, facts = _b4_train(model, cell, index, assets, torch_device, run_path, mode=mode, smoke=smoke, resume_from=resume_path, paired_initialization=paired_path)
            elif path_kind == "b6_ridge":
                if upstream_path is None:
                    raise ValueError("B6 ridge requires same-seed upstream_run")
                _, upstream_checkpoint = _validate_upstream(upstream_path, int(cell["run_seed"]))
                model, facts, direct = _b6_ridge_run(model, cell, index, assets, torch_device, run_path, upstream_checkpoint, smoke=smoke)
                def direct_predictor(xy: np.ndarray) -> np.ndarray:
                    query = np.asarray(xy)
                    if np.array_equal(query, direct["primary_xy"]):
                        return direct["primary_prediction"]
                    if np.array_equal(query, direct["support_xy"]):
                        return direct["support_prediction"]
                    raise ValueError("ridge predictor is defined only for saved primary/support feature rows")
            elif path_kind.startswith("b6_"):
                if upstream_path is None:
                    raise ValueError("B6 downstream requires same-seed upstream_run")
                _, upstream_checkpoint = _validate_upstream(upstream_path, int(cell["run_seed"]))
                trace, facts = _b6_train(model, cell, index, assets, torch_device, run_path, upstream_checkpoint, mode=mode, smoke=smoke, resume_from=resume_path)
            elif path_kind.startswith("b7_"):
                trace, facts = _b7_train(
                    model,
                    cell,
                    index,
                    assets,
                    torch_device,
                    run_path,
                    mode=mode,
                    smoke=smoke,
                    resume_from=resume_path,
                )
            else:
                raise ValueError(f"unimplemented training path {path_kind}")

        metrics, query_manifest = _evaluate_and_save(
            run_path,
            model,
            cell,
            index,
            assets,
            torch_device,
            mode=mode,
            query_limit=None if smoke is None else int(smoke.query_limit),
            smoke_side=None if smoke is None else int(smoke.image_side),
            direct_predictor=direct_predictor,
        )
        write_json(run_path / "training_trace.json", {"steps": trace})
        facts = {**facts, "query_manifest": query_manifest, "completed_actual_steps": len(trace), "mode": mode, "formal_eligible": mode == "formal"}
        write_json(run_path / "run_facts.json", facts)
        environment = _environment(started, torch_device)
        write_json(run_path / "environment.json", environment)
        status = {
            "state": "COMPLETE",
            "cell_id": cell_id,
            "formal_eligible": mode == "formal",
            "producer_metrics_path": str((run_path / "producer_metrics.json").resolve()),
            "review_state": "AWAITING_INDEPENDENT_REVIEW",
        }
        write_json(run_path / "status.json", status, overwrite=True)
        append_run_catalog(
            {
                "cell_id": cell_id,
                "family": cell["family"],
                "condition": cell["condition_id"],
                "seed": cell["run_seed"],
                "run_path": str(run_path.resolve()),
                "review_path": str(review_path.resolve()),
                "status": "COMPLETE",
                "mode": mode,
                "formal_eligible": mode == "formal",
            },
            project_root=root,
        )
        return {"status": "COMPLETE", "cell_id": cell_id, "run_path": str(run_path), "review_path": str(review_path), "metrics": metrics, "environment": environment}
    except Exception as exc:
        write_json(
            run_path / "status.json",
            {"state": "FAILED", "cell_id": cell_id, "formal_eligible": False, "error_type": type(exc).__name__, "error": str(exc)},
            overwrite=True,
        )
        write_json(run_path / "environment.json", _environment(started, torch_device))
        append_run_catalog(
            {
                "cell_id": cell_id,
                "family": cell["family"],
                "condition": cell["condition_id"],
                "seed": cell["run_seed"],
                "run_path": str(run_path.resolve()),
                "review_path": str(review_path.resolve()),
                "status": "FAILED",
                "mode": mode,
                "formal_eligible": False,
            },
            project_root=root,
        )
        raise


def create_b4_paired_initialization(
    seed: int,
    *,
    project_root: str | Path,
    mode: str = "smoke",
) -> Path:
    require_torch()
    root = Path(project_root).resolve()
    if mode == "formal":
        require_formal_execution_root(root)
    elif (root / FORMAL_ROOT_MARKER_NAME).exists():
        raise FormalRootError("smoke B4 initialization cannot use a formal execution root")
    pair_root = root / "paired" / "B4" / f"seed_{int(seed)}"
    target = pair_root / "initialization.pt"
    if target.exists():
        raise FileExistsError(target)
    pair_root.mkdir(parents=True, exist_ok=True)
    model = build_smoke_cnn("gn", int(seed)) if mode == "smoke" else build_resnet18("gn", int(seed))
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(
            {
                "format": "trackb_b4_paired_initialization_v1",
                "run_seed": int(seed),
                "mode": mode,
                "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            },
            temporary,
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def ensure_b4_paired_initialization(
    seed: int,
    *,
    project_root: str | Path,
    mode: str = "formal",
) -> Path:
    root = Path(project_root).resolve()
    if mode == "formal":
        require_formal_execution_root(root)
    elif (root / FORMAL_ROOT_MARKER_NAME).exists():
        raise FormalRootError("smoke B4 initialization cannot use a formal execution root")
    target = root / "paired" / "B4" / f"seed_{int(seed)}" / "initialization.pt"
    if not target.exists():
        create_b4_paired_initialization(seed, project_root=project_root, mode=mode)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "trackb_b4_paired_initialization_v1"
        or int(payload.get("run_seed", -1)) != int(seed)
        or payload.get("mode") != mode
    ):
        raise ValueError("existing B4 paired initialization does not match the requested seed/mode")
    return target


def run_b4_pair(
    seed: int,
    *,
    project_root: str | Path,
    asset_root: str | Path = PACKAGE_ROOT / "assets",
    mode: str = "smoke",
    device: str | None = None,
    smoke_options: SmokeOptions | None = None,
) -> list[dict[str, Any]]:
    initialization = create_b4_paired_initialization(seed, project_root=project_root, mode=mode)
    return [
        run_cell(
            f"B4.{condition}.s{int(seed)}",
            project_root=project_root,
            asset_root=asset_root,
            mode=mode,
            device=device,
            smoke_options=smoke_options,
            paired_initialization=initialization,
        )
        for condition in ("causal_adamw", "causal_sgd")
    ]


def run_b8_degree2_shared(
    *,
    project_root: str | Path,
    asset_root: str | Path = PACKAGE_ROOT / "assets",
    mode: str = "smoke",
    smoke_options: SmokeOptions | None = None,
) -> list[dict[str, Any]]:
    """Materialize one analytic fit and three seed-labelled logical records."""
    root = Path(project_root).resolve()
    if mode == "formal":
        require_formal_execution_root(root)
    elif (root / FORMAL_ROOT_MARKER_NAME).exists():
        raise FormalRootError("smoke B8 shared run cannot use a formal execution root")
    seeds = [int(value) for value in load_protocol()["discovery_seeds"]]
    first = run_cell(
        f"B8.degree2.s{seeds[0]}",
        project_root=project_root,
        asset_root=asset_root,
        mode=mode,
        device="cpu",
        smoke_options=smoke_options,
    )
    results = [first]
    source_run = Path(first["run_path"])
    assets = Path(asset_root)
    for seed in seeds[1:]:
        cell_id = f"B8.degree2.s{seed}"
        cell = resolve_cell(cell_id)
        attempt = create_attempt("B8", "degree2", seed, project_root=root)
        run_path = Path(attempt["run_path"])
        review_path = Path(attempt["review_path"])
        smoke = smoke_options or SmokeOptions()
        config = _resolved_run_config(cell_id, attempt, root, assets, mode, torch.device("cpu"), smoke if mode == "smoke" else None, None, None, None)
        config["execution"]["shared_physical_source"] = str(source_run.resolve())
        write_json(run_path / "run_config.json", config)
        for name in ("predictions.npz", "producer_metrics.json", "training_trace.json", "analytic_degree2.npz"):
            shutil.copy2(source_run / name, run_path / name)
        facts = _load_json(source_run / "run_facts.json")
        facts["shared_physical_source"] = str(source_run.resolve())
        facts["logical_record_role"] = "seed_labelled_reference_to_shared_analytic_artifact"
        write_json(run_path / "run_facts.json", facts)
        source_environment = _load_json(source_run / "environment.json")
        source_environment["logical_reference_only"] = True
        source_environment["wall_seconds"] = 0.0
        write_json(run_path / "environment.json", source_environment)
        write_json(
            run_path / "status.json",
            {
                "state": "COMPLETE",
                "cell_id": cell_id,
                "formal_eligible": mode == "formal",
                "producer_metrics_path": str((run_path / "producer_metrics.json").resolve()),
                "review_state": "AWAITING_INDEPENDENT_REVIEW",
                "shared_physical_source": str(source_run.resolve()),
            },
        )
        append_run_catalog(
            {
                "cell_id": cell_id,
                "family": "B8",
                "condition": "degree2",
                "seed": seed,
                "run_path": str(run_path.resolve()),
                "review_path": str(review_path.resolve()),
                "status": "COMPLETE",
                "mode": mode,
                "formal_eligible": mode == "formal",
                "shared_physical_source": str(source_run.resolve()),
            },
            project_root=root,
        )
        metrics = _load_json(run_path / "producer_metrics.json")
        results.append(
            {
                "status": "COMPLETE",
                "cell_id": cell_id,
                "run_path": str(run_path),
                "review_path": str(review_path),
                "metrics": metrics,
                "environment": source_environment,
            }
        )
    return results
