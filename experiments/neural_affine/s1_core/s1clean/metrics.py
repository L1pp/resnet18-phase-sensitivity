"""Numerically explicit S1 field metrics.

All affine fitting is performed in float64.  The public MAE helper expects
normalized coordinate fields by default (the S1 protocol scale is 223 px),
which prevents accidental mixing of normalized and pixel units in the runner.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


DEFAULT_COORD_SCALE = 223.0


def _points(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 2)
    if array.shape[-1:] != (2,):
        raise ValueError(f"coordinate field must end in 2, got {array.shape}")
    return array.reshape(-1, 2)


def _scale(coord_scale: float, image_size: int | None) -> float:
    if image_size is not None:
        return float(int(image_size) - 1)
    value = float(coord_scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"coord_scale must be positive and finite, got {coord_scale!r}")
    return value


def mae_px(
    pred: np.ndarray,
    true: np.ndarray,
    coord_scale: float = DEFAULT_COORD_SCALE,
    *,
    image_size: int | None = None,
) -> float:
    """Mean Euclidean point error in pixels.

    ``pred`` and ``true`` are normalized coordinates unless the caller passes
    ``coord_scale=1``.  ``image_size=224`` is accepted as a convenience and
    resolves to the protocol scale 223.
    """

    p = _points(pred)
    t = _points(true)
    if p.shape != t.shape:
        raise ValueError(f"prediction/target shape mismatch: {p.shape} vs {t.shape}")
    if len(p) == 0:
        return float("nan")
    return float(np.linalg.norm((p - t) * _scale(coord_scale, image_size), axis=-1).mean())


def fit_affine(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``true ~= pred @ A.T + b`` using float64 least squares."""

    p = _points(pred)
    target = _points(true)
    if p.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {p.shape} vs {target.shape}")
    if len(p) == 0:
        raise ValueError("cannot fit an affine map to an empty field")
    design = np.column_stack((p, np.ones(len(p), dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = np.asarray(coefficients[:2].T, dtype=np.float64)
    bias = np.asarray(coefficients[2], dtype=np.float64)
    return matrix, bias


def apply_affine(pred: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply a fitted affine map while preserving the input field shape."""

    original = np.asarray(pred)
    points = _points(original)
    matrix64 = np.asarray(matrix, dtype=np.float64).reshape(2, 2)
    bias64 = np.asarray(bias, dtype=np.float64).reshape(2)
    mapped = points @ matrix64.T + bias64
    return mapped.reshape(original.shape)


def affine_residual(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return the pointwise residual after the best affine correction."""

    matrix, bias = fit_affine(pred, true)
    corrected = apply_affine(pred, matrix, bias)
    return np.asarray(corrected, dtype=np.float64) - np.asarray(true, dtype=np.float64)


def affine_residual_mae_px(
    pred: np.ndarray,
    true: np.ndarray,
    coord_scale: float = DEFAULT_COORD_SCALE,
    *,
    image_size: int | None = None,
) -> float:
    """Mean Euclidean pixel error after fitting a float64 affine correction."""

    corrected = np.asarray(true, dtype=np.float64) + affine_residual(pred, true)
    return mae_px(corrected, true, coord_scale, image_size=image_size)


# Descriptive aliases keep the evaluator API readable without duplicating the
# implementation or introducing a second numerical convention.
affine_fit = fit_affine
affine_apply = apply_affine
residual_mae_px = affine_residual_mae_px


__all__ = [
    "DEFAULT_COORD_SCALE",
    "affine_apply",
    "affine_fit",
    "affine_residual",
    "affine_residual_mae_px",
    "apply_affine",
    "fit_affine",
    "mae_px",
    "residual_mae_px",
]
