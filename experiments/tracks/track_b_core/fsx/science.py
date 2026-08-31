from __future__ import annotations

from typing import Any

import numpy as np


def b6_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    queries: np.ndarray,
    multiplier: float = 1e-4,
    rcond: float = 1e-6,
) -> dict[str, Any]:
    """Approved centered float64 SVD ridge, including the rank-zero rule."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or q.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != q.shape[1]:
        raise ValueError("B6 ridge shapes are inconsistent")
    if x.shape[0] != 4 or y.shape[1] != 2:
        raise ValueError("B6 ridge requires four sparse rows and two targets")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("B6 ridge inputs must be finite")
    if not np.isfinite(multiplier) or float(multiplier) <= 0.0:
        raise ValueError("B6 ridge multiplier must be finite and positive")
    alpha_scale = float(np.trace(x @ x.T) / x.shape[0])
    if not np.isfinite(alpha_scale) or alpha_scale <= 0.0:
        raise ValueError("B6_RIDGE_DEGENERATE_SCALE")
    alpha = float(multiplier) * alpha_scale
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("B6_RIDGE_ALPHA_INVALID")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    try:
        u, singular_values, vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("B6_RIDGE_SVD_FAILURE") from exc
    maximum = float(singular_values[0]) if len(singular_values) else 0.0
    keep = singular_values > float(rcond) * maximum if maximum > 0.0 else np.zeros_like(singular_values, dtype=bool)
    if np.any(keep):
        uk = u[:, keep]
        sk = singular_values[keep]
        vtk = vt[keep]
        beta = vtk.T @ ((sk / (sk * sk + alpha))[:, None] * (uk.T @ yc))
    else:
        beta = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    prediction = (q - x_mean) @ beta + y_mean
    return {
        "prediction": prediction,
        "beta": beta,
        "alpha_scale": alpha_scale,
        "alpha": alpha,
        "rank": int(np.count_nonzero(keep)),
        "singular_values": singular_values,
    }


def metric_summary(
    prediction_px: np.ndarray,
    truth_px: np.ndarray,
    anchor_prediction_px: np.ndarray | None = None,
    anchor_truth_px: np.ndarray | None = None,
) -> dict[str, float]:
    prediction = np.asarray(prediction_px, dtype=np.float64)
    truth = np.asarray(truth_px, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("prediction and truth must be [N,2]")
    if not (np.isfinite(prediction).all() and np.isfinite(truth).all()):
        raise ValueError("prediction and truth must be finite")
    mae = float(np.mean(np.abs(prediction - truth)))
    result = {"full_box_raw_mae_px": mae, "full_box_u_mae": mae / 223.0}
    if anchor_prediction_px is not None or anchor_truth_px is not None:
        if anchor_prediction_px is None or anchor_truth_px is None:
            raise ValueError("anchor prediction and truth must be supplied together")
        anchor_prediction = np.asarray(anchor_prediction_px, dtype=np.float64)
        anchor_truth = np.asarray(anchor_truth_px, dtype=np.float64)
        if anchor_prediction.shape != anchor_truth.shape or anchor_prediction.ndim != 2 or anchor_prediction.shape[1] != 2:
            raise ValueError("anchor prediction and truth must be [N,2]")
        if not (np.isfinite(anchor_prediction).all() and np.isfinite(anchor_truth).all()):
            raise ValueError("anchor prediction and truth must be finite")
        result["anchor_raw_mae_px"] = float(np.mean(np.abs(anchor_prediction - anchor_truth)))
    return result


def support_gate(anchor_raw_mae_px: float, threshold_px: float = 0.25) -> bool:
    value = float(anchor_raw_mae_px)
    return bool(np.isfinite(value) and value <= float(threshold_px))
