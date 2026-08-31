"""Thin adapter from the independent oracle API to runtime scientific functions."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import iter_cells, load_matrix
from .data import (
    b7_double_draw_stream,
    b7_endpoint_occurrences as _endpoint_occurrences,
    canonical_b7_pair_graph,
    common_stream,
    coordinate_target_f64,
    positions,
)
from .schedules import b4_causal_lr_for_update, common_lr_after_update, mean_mse
from .science import b6_ridge, metric_summary as _metric_summary, support_gate as _support_gate


def matrix_contract():
    matrix = load_matrix()
    cells = list(iter_cells(matrix))
    return {
        "logical_cells": len(cells),
        "physical_jobs": len({cell["physical_job_id"] for cell in cells}),
        "optimizer_jobs": sum(bool(cell["optimizer_job"]) for cell in cells),
        "seeds": tuple(matrix["seeds"]),
        "family_conditions": {family: tuple(item["condition_id"] for item in conditions) for family, conditions in matrix["families"].items()},
    }


def canonical_grid41() -> np.ndarray:
    return positions("dense41")


def normalized_xy(xy_px: np.ndarray) -> np.ndarray:
    return np.asarray(xy_px, dtype=np.float64) / 223.0


def b8_quadratic(xy_norm: np.ndarray) -> np.ndarray:
    return coordinate_target_f64(np.asarray(xy_norm, dtype=np.float64) * 223.0, "quadratic_b8")


def b2_identity_sets():
    return {
        "A1": {"train": (0,), "heldout": (1, 2, 3, 4)},
        "A8": {"train": tuple(range(8)), "heldout": (8, 9, 10, 11)},
        "A64": {"train": tuple(range(64)), "heldout": (64, 65, 66, 67)},
    }


def fixed_batch_stream(n_rows: int, run_seed: int, steps: int = 3) -> np.ndarray:
    return common_stream(n_rows, steps, run_seed, 64)


def b7_canonical_edges() -> np.ndarray:
    return canonical_b7_pair_graph()


def b7_interleaved_stream(run_seed: int, image_count: int = 64, steps: int = 3):
    if image_count != 64:
        raise ValueError("runtime B7 image population is frozen at 64")
    return b7_double_draw_stream(steps, run_seed, 64)


def b7_endpoint_occurrences(pair_ids: Iterable[int], edges: np.ndarray | None = None) -> np.ndarray:
    ids = np.asarray(list(pair_ids), dtype=np.int64)
    if ids.shape != (64,):
        # The oracle also uses shorter lists in direct loss tests.
        graph = canonical_b7_pair_graph() if edges is None else np.asarray(edges, dtype=np.int64)
        selected = graph[ids]
        return np.concatenate((selected[:, 0], selected[:, 1]))
    return _endpoint_occurrences(ids, edges)


def b7_matched_exposure(run_seed: int, step: int = 0):
    stream = b7_double_draw_stream(step + 1, run_seed, 64)
    pair_ids = stream["pair_ids"][step]
    return {"image_ids": stream["image_ids"][step], "pair_ids": pair_ids, "endpoint_ids": _endpoint_occurrences(pair_ids)}


def b7_dense_absolute_loss(pred: np.ndarray, truth: np.ndarray, pair_ids: Iterable[int]) -> float:
    endpoints = b7_endpoint_occurrences(pair_ids)
    return mean_mse(np.asarray(pred)[endpoints], np.asarray(truth)[endpoints])


def b7_relative_loss(pred: np.ndarray, truth: np.ndarray, pair_ids: Iterable[int], anchor_ids: Iterable[int]):
    prediction = np.asarray(pred, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    selected = canonical_b7_pair_graph()[np.asarray(list(pair_ids), dtype=np.int64)]
    pair_mse = mean_mse(prediction[selected[:, 1]] - prediction[selected[:, 0]], target[selected[:, 1]] - target[selected[:, 0]])
    anchors = np.asarray(list(anchor_ids), dtype=np.int64)
    anchor_mse = mean_mse(prediction[anchors], target[anchors])
    return {"pair_mse": pair_mse, "anchor_mse": anchor_mse, "total": pair_mse + anchor_mse}


def b6_ridge_reference(features, targets, queries, multiplier=1e-4, rcond=1e-6):
    return b6_ridge(features, targets, queries, multiplier, rcond)


def common_lr(step: int, total_steps: int = 3000, base_lr: float = 1e-3, eta_min: float = 1e-5) -> float:
    return common_lr_after_update(step, total_steps=total_steps, base_lr=base_lr, eta_min=eta_min)


def b4_causal_lr(step: int, total_steps: int = 20000, warmup_steps: int = 500, base_lr: float = 1e-3, eta_min: float = 1e-5) -> float:
    return b4_causal_lr_for_update(step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=base_lr, eta_min=eta_min)


def metric_summary(prediction_px, truth_px, anchor_prediction_px=None, anchor_truth_px=None):
    return _metric_summary(prediction_px, truth_px, anchor_prediction_px, anchor_truth_px)


def support_gate(anchor_raw_mae_px: float, threshold_px: float = 0.25) -> bool:
    return _support_gate(anchor_raw_mae_px, threshold_px)
