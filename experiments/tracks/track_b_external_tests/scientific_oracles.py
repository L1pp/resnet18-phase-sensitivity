"""Small, independent scientific reference functions for Track B V3.

This module is deliberately not an implementation of the Track B runtime.  It
contains only the numerical contracts that can change the scientific meaning of
an experiment.  The accompanying unittest module can dispatch these calls to a
future implementation adapter through ``TRACK_B_ORACLE_TARGET``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


SEEDS = (20260816, 20260817, 20260818)
MATRIX_CONDITIONS = {
    "B1": ("bn", "gn", "frozen_bn"),
    "B2": ("A1", "A8", "A64"),
    "B3": ("4x16", "16x4", "64x1"),
    "B4": ("reference_adamw", "causal_adamw", "causal_sgd"),
    "B5": ("blob_sigma6", "line_fixed", "scalene_triangle"),
    "B6": ("g64_upstream", "ridge", "head_only", "lp_ft", "layer4", "full"),
    "B7": (
        "dense_absolute",
        "dense_relative_4_neighbor_plus_1_anchor",
        "dense_relative_4_neighbor_plus_4_corners",
        "corners4_absolute_context",
    ),
    "B8": ("degree2", "explicit_xy_mlp", "cnn"),
}


def matrix_contract() -> dict[str, Any]:
    """Return the compact matrix contract used by the oracle tests."""

    return {
        "logical_cells": 84,
        "physical_jobs": 82,
        "optimizer_jobs": 78,
        "seeds": SEEDS,
        "family_conditions": MATRIX_CONDITIONS,
    }


def canonical_grid41() -> np.ndarray:
    """41x41 query grid, with x as the outer/slow axis and y as inner/fast."""

    axis = np.linspace(59.0, 164.0, 41, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((x, y), axis=-1).reshape(-1, 2)


def normalized_xy(xy_px: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates to the protocol's normalized target units."""

    return np.asarray(xy_px, dtype=np.float64) / 223.0


def b8_quadratic(xy_norm: np.ndarray) -> np.ndarray:
    """The frozen B8 degree-2 target in normalized coordinates."""

    xy = np.asarray(xy_norm, dtype=np.float64)
    x = xy[..., 0]
    y = xy[..., 1]
    qx = 0.70 * x + 0.08 * y + 0.06 * x * x - 0.05 * x * y + 0.04 * y * y
    qy = -0.06 * x + 0.72 * y - 0.04 * x * x + 0.06 * x * y + 0.05 * y * y
    return np.stack((qx, qy), axis=-1)


def b2_identity_sets() -> dict[str, dict[str, tuple[int, ...]]]:
    """Return the frozen B2 train/held-out appearance identities."""

    return {
        "A1": {"train": (0,), "heldout": (1, 2, 3, 4)},
        "A8": {"train": tuple(range(8)), "heldout": (8, 9, 10, 11)},
        "A64": {"train": tuple(range(64)), "heldout": (64, 65, 66, 67)},
    }


def fixed_batch_stream(n_rows: int, run_seed: int, steps: int = 3) -> np.ndarray:
    """Draw a fixed [steps, 64] with-replacement stream from PCG64."""

    if n_rows <= 0 or steps < 0:
        raise ValueError("n_rows must be positive and steps must be nonnegative")
    rng = np.random.Generator(np.random.PCG64(int(run_seed) + 17))
    return rng.integers(0, int(n_rows), size=(int(steps), 64), dtype=np.int64)


def b7_canonical_edges() -> np.ndarray:
    """Return the 112 +x/+y edges of a row-major 8x8 grid.

    The index is ``x_index * 8 + y_index``: x is the outer axis, y the inner
    axis.  For each node the +x edge is emitted before the +y edge.
    """

    pairs: list[tuple[int, int]] = []
    for x_index in range(8):
        for y_index in range(8):
            left = x_index * 8 + y_index
            if x_index < 7:
                pairs.append((left, (x_index + 1) * 8 + y_index))
            if y_index < 7:
                pairs.append((left, x_index * 8 + y_index + 1))
    return np.asarray(pairs, dtype=np.int64)


def b7_interleaved_stream(
    run_seed: int, image_count: int = 64, steps: int = 3
) -> dict[str, np.ndarray]:
    """Draw image ids then pair ids for every step from one PCG64 state."""

    if image_count <= 0 or steps < 0:
        raise ValueError("image_count must be positive and steps must be nonnegative")
    rng = np.random.Generator(np.random.PCG64(int(run_seed) + 17))
    images = np.empty((int(steps), 64), dtype=np.int64)
    pairs = np.empty((int(steps), 64), dtype=np.int64)
    for step in range(int(steps)):
        images[step] = rng.integers(0, int(image_count), size=64, dtype=np.int64)
        pairs[step] = rng.integers(0, 112, size=64, dtype=np.int64)
    return {"image_ids": images, "pair_ids": pairs}


def b7_endpoint_occurrences(pair_ids: Iterable[int], edges: np.ndarray | None = None) -> np.ndarray:
    """Return [left_0..left_63, right_0..right_63], retaining duplicates."""

    edge_table = b7_canonical_edges() if edges is None else np.asarray(edges, dtype=np.int64)
    selected = edge_table[np.asarray(list(pair_ids), dtype=np.int64)]
    return np.concatenate((selected[:, 0], selected[:, 1])).astype(np.int64, copy=False)


def b7_matched_exposure(run_seed: int, step: int = 0) -> dict[str, np.ndarray]:
    """Return one step's shared image/pair/endpoints exposure for all dense arms."""

    if step < 0:
        raise ValueError("step must be nonnegative")
    stream = b7_interleaved_stream(run_seed, image_count=64, steps=int(step) + 1)
    pair_ids = stream["pair_ids"][int(step)]
    return {
        "image_ids": stream["image_ids"][int(step)],
        "pair_ids": pair_ids,
        "endpoint_ids": b7_endpoint_occurrences(pair_ids),
    }


def b7_dense_absolute_loss(
    pred: np.ndarray, truth: np.ndarray, pair_ids: Iterable[int]
) -> float:
    """Pure coordinate MSE on the 128 matched endpoint occurrences."""

    endpoint_ids = b7_endpoint_occurrences(pair_ids)
    error = np.asarray(pred, dtype=np.float64)[endpoint_ids] - np.asarray(truth, dtype=np.float64)[endpoint_ids]
    return float(np.mean(error * error))


def b7_relative_loss(
    pred: np.ndarray,
    truth: np.ndarray,
    pair_ids: Iterable[int],
    anchor_ids: Iterable[int],
) -> dict[str, float]:
    """Pair mean MSE plus anchor mean MSE, with equal unit weights."""

    pred_arr = np.asarray(pred, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    selected = b7_canonical_edges()[np.asarray(list(pair_ids), dtype=np.int64)]
    left, right = selected[:, 0], selected[:, 1]
    pair_error = (pred_arr[right] - pred_arr[left]) - (truth_arr[right] - truth_arr[left])
    anchors = np.asarray(list(anchor_ids), dtype=np.int64)
    anchor_error = pred_arr[anchors] - truth_arr[anchors]
    pair_mse = float(np.mean(pair_error * pair_error))
    anchor_mse = float(np.mean(anchor_error * anchor_error))
    return {"pair_mse": pair_mse, "anchor_mse": anchor_mse, "total": pair_mse + anchor_mse}


def b6_ridge_reference(
    features: np.ndarray,
    targets: np.ndarray,
    queries: np.ndarray,
    multiplier: float = 1e-4,
    rcond: float = 1e-6,
) -> dict[str, Any]:
    """Float64 centered SVD ridge specified by the B6 V6 decision."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or q.ndim != 2 or len(x) != len(y) or x.shape[1] != q.shape[1]:
        raise ValueError("B6 ridge shapes are inconsistent")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("B6 ridge inputs must be finite")
    n = x.shape[0]
    alpha_scale = float(np.trace(x @ x.T) / n)
    if not np.isfinite(alpha_scale) or alpha_scale <= 0.0:
        raise ValueError("B6 alpha scale must be finite and positive")
    alpha = float(multiplier) * alpha_scale
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    keep = s > float(rcond) * (float(s[0]) if len(s) else 0.0)
    if not np.any(keep):
        beta = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    else:
        uk = u[:, keep]
        sk = s[keep]
        vtk = vt[keep]
        beta = vtk.T @ ((sk / (sk * sk + alpha))[:, None] * (uk.T @ yc))
    prediction = (q - x_mean) @ beta + y_mean
    return {
        "prediction": prediction,
        "beta": beta,
        "alpha_scale": alpha_scale,
        "alpha": alpha,
        "rank": int(np.count_nonzero(keep)),
    }


def common_lr(step: int, total_steps: int = 3000, base_lr: float = 1e-3, eta_min: float = 1e-5) -> float:
    """Common cosine schedule after the optimizer update (step 0 is base)."""

    if not 0 <= int(step) <= int(total_steps) or total_steps <= 0:
        raise ValueError("step must be within the common schedule")
    progress = float(step) / float(total_steps)
    return float(eta_min + 0.5 * (base_lr - eta_min) * (1.0 + np.cos(np.pi * progress)))


def b4_causal_lr(
    step: int, total_steps: int = 20000, warmup_steps: int = 500, base_lr: float = 1e-3, eta_min: float = 1e-5
) -> float:
    """B4 linear warmup through step 500, then cosine through step 20000."""

    if not 1 <= int(step) <= int(total_steps) or not 0 < warmup_steps < total_steps:
        raise ValueError("step or causal schedule bounds are invalid")
    if step <= warmup_steps:
        return float(base_lr * step / warmup_steps)
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    return float(eta_min + 0.5 * (base_lr - eta_min) * (1.0 + np.cos(np.pi * progress)))


def metric_summary(
    prediction_px: np.ndarray,
    truth_px: np.ndarray,
    anchor_prediction_px: np.ndarray | None = None,
    anchor_truth_px: np.ndarray | None = None,
) -> dict[str, float]:
    """Recompute raw MAE from fields, with the protocol's two-coordinate mean."""

    pred = np.asarray(prediction_px, dtype=np.float64)
    truth = np.asarray(truth_px, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError("prediction and truth fields must have shape [N,2]")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise ValueError("prediction and truth fields must be finite")
    result = {
        "full_box_raw_mae_px": float(np.mean(np.abs(pred - truth))),
        "full_box_u_mae": float(np.mean(np.abs(pred - truth)) / 223.0),
    }
    if anchor_prediction_px is not None or anchor_truth_px is not None:
        if anchor_prediction_px is None or anchor_truth_px is None:
            raise ValueError("anchor prediction and truth must be supplied together")
        ap = np.asarray(anchor_prediction_px, dtype=np.float64)
        at = np.asarray(anchor_truth_px, dtype=np.float64)
        if ap.shape != at.shape or ap.ndim != 2 or ap.shape[-1] != 2:
            raise ValueError("anchor fields must have shape [N,2]")
        if not (np.isfinite(ap).all() and np.isfinite(at).all()):
            raise ValueError("anchor fields must be finite")
        result["anchor_raw_mae_px"] = float(np.mean(np.abs(ap - at)))
    return result


def support_gate(anchor_raw_mae_px: float, threshold_px: float = 0.25) -> bool:
    """Inclusive support gate; nonfinite values do not qualify."""

    value = float(anchor_raw_mae_px)
    return bool(np.isfinite(value) and value <= float(threshold_px))

