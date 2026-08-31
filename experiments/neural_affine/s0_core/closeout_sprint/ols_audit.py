"""Numerically explicit clean-room Frozen-G64 OLS/ridge stability audit.

The formal input contract is ``frozen_g64_gap_to_p6_v1``: separate GAP
features and normalized six-dimensional P6 targets, train/eval geometry,
semantic row IDs, and JSON provenance.  The formal comparable path fixes the
materializer's 3584 train pairs and 32x41x41 eval Cartesian rows. Historical
``train_features/train_targets`` and ``ols_head.npz`` W/b-only artifacts are
intentionally insufficient and are rejected before any output bundle exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .assets import write_run_bundle
from .contracts import SCHEMA_VERSION, SchemaError, write_json


PRECISIONS = ("float32", "float64")
OLS_SCHEMA_KIND = "frozen_g64_gap_to_p6_v1"
OLS_TARGET_SEMANTICS = "p6_normalized"
OLS_COORD_SCALE = 223.0
OLS_HEAD_DIM = 6
OLS_FORMAL_TRAIN_ROWS = 3584
OLS_FORMAL_EVAL_ROWS = 32 * 41 * 41
OLS_FORMAL_TRAIN_PER_TRANSLATION = 56
OLS_FORMAL_TRANSLATION_COUNT = 64
OLS_FORMAL_DENSE_SIDE = 41
OLS_PROVENANCE_REQUIRED_KEYS = {
    "protocol_id",
    "protocol_sha256",
    "source_assets",
    "checkpoint",
    "train_pair_split",
    "eval",
}
OLS_REQUIRED_KEYS = {
    "schema_kind",
    "target_semantics",
    "coord_scale",
    "head_dim",
    "train_gap_features",
    "eval_gap_features",
    "train_p6_norm",
    "train_q_px",
    "train_t_px",
    "eval_p6_norm",
    "eval_q_px",
    "eval_t_px",
    "train_shape_id",
    "train_translation_id",
    "eval_shape_id",
    "eval_dense_index",
    "provenance_json",
}
OLS_ID_KEY_PAIRS = (("train_ids", "eval_ids"), ("train_support_ids", "eval_support_ids"))


def _scalar(value: Any, name: str) -> Any:
    arr = np.asarray(value)
    if arr.size != 1:
        raise SchemaError(f"{name} must be a scalar, got shape {arr.shape}")
    return arr.reshape(-1)[0].item()


def _scalar_text(value: Any, name: str) -> str:
    result = _scalar(value, name)
    if not isinstance(result, str):
        raise SchemaError(f"{name} must be a string scalar, got {type(result).__name__}")
    return result


def _ids(value: Any, name: str, expected: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or result.size != expected:
        raise SchemaError(f"{name} must be a 1D array with {expected} rows, got {result.shape}")
    if result.dtype.kind not in {"b", "i", "u", "f", "U", "S"}:
        raise SchemaError(f"{name} has unsupported dtype {result.dtype}")
    if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
        raise SchemaError(f"{name} contains non-finite values")
    # IDs identify rows; duplicates or cross-split reuse make row-level
    # comparisons unverifiable even if the numerical arrays have valid shapes.
    values = [_id_token(item.item() if isinstance(item, np.generic) else item) for item in result]
    if len(set(values)) != len(values):
        raise SchemaError(f"{name} contains duplicate row IDs")
    return np.array(result, copy=True)


def _id_token(value: Any) -> tuple[str, Any]:
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("number", int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return ("number", int(number) if number.is_integer() else number)
    return ("text", str(value))


def _integer_vector(value: Any, name: str, expected: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size != expected:
        raise SchemaError(f"{name} must be a 1D integer array with {expected} rows, got {arr.shape}")
    if arr.dtype.kind not in {"i", "u"}:
        raise SchemaError(f"{name} must have an integer dtype, got {arr.dtype}")
    return np.asarray(arr, dtype=np.int64).copy()


def _validate_historical_formal_contract(
    train_p6: np.ndarray,
    eval_p6: np.ndarray,
    train_q_px: Any,
    train_t_px: Any,
    eval_q_px: np.ndarray,
    eval_t_px: np.ndarray,
    train_shape_id: Any,
    train_translation_id: Any,
    eval_shape_id: Any,
    eval_dense_index: Any,
    *,
    coord_scale: float,
    provenance: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the materializer's historical-comparable row contract."""

    n_train, n_eval = train_p6.shape[0], eval_p6.shape[0]
    if n_train != OLS_FORMAL_TRAIN_ROWS or n_eval != OLS_FORMAL_EVAL_ROWS:
        raise SchemaError(
            "formal OLS completion requires "
            f"n_train={OLS_FORMAL_TRAIN_ROWS}, n_eval={OLS_FORMAL_EVAL_ROWS}; "
            f"got {n_train}, {n_eval}"
        )
    train_q = np.asarray(train_q_px)
    train_t = np.asarray(train_t_px)
    for name, arr, shape in (
        ("train_q_px", train_q, (n_train, 3, 2)),
        ("train_t_px", train_t, (n_train, 2)),
    ):
        if arr.shape != shape:
            raise SchemaError(f"{name} must have shape {shape}, got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
            raise SchemaError(f"{name} must be finite numeric data")

    train_shape = _integer_vector(train_shape_id, "train_shape_id", n_train)
    train_tid = _integer_vector(train_translation_id, "train_translation_id", n_train)
    eval_shape = _integer_vector(eval_shape_id, "eval_shape_id", n_eval)
    eval_dense = _integer_vector(eval_dense_index, "eval_dense_index", n_eval)
    if np.any((train_shape < 0) | (train_shape >= 64)) or np.any((train_tid < 0) | (train_tid >= OLS_FORMAL_TRANSLATION_COUNT)):
        raise SchemaError("train_shape_id/train_translation_id are outside the frozen 64x64 factorial domain")
    tid_counts = np.bincount(train_tid, minlength=OLS_FORMAL_TRANSLATION_COUNT)
    if not np.array_equal(tid_counts, np.full(OLS_FORMAL_TRANSLATION_COUNT, OLS_FORMAL_TRAIN_PER_TRANSLATION)):
        raise SchemaError("each train_translation_id must have exactly 56 rows")
    train_pairs = list(zip(train_shape.tolist(), train_tid.tolist()))
    if len(set(train_pairs)) != n_train:
        raise SchemaError("train_shape_id/train_translation_id pair IDs are duplicated")
    if len(np.unique(train_shape)) != 64:
        raise SchemaError("formal train pair split must cover all 64 shape IDs")
    for tid in np.unique(train_tid):
        rows = train_t[train_tid == tid]
        if not np.allclose(rows, rows[0], atol=0.0, rtol=0.0):
            raise SchemaError(f"train_translation_id={int(tid)} maps to multiple train_t_px rows")
    for sid in np.unique(train_shape):
        rows = train_q[train_shape == sid]
        if not np.allclose(rows, rows[0], atol=0.0, rtol=0.0):
            raise SchemaError(f"train_shape_id={int(sid)} maps to multiple train_q_px rows")

    unique_eval_shapes = np.unique(eval_shape)
    if unique_eval_shapes.size != 32:
        raise SchemaError("formal eval must contain exactly 32 unique shape IDs")
    if np.any((unique_eval_shapes < 0) | (unique_eval_shapes >= 64)):
        raise SchemaError("eval_shape_id is outside the frozen 64-shape factorial domain")
    if np.any(eval_dense < 0) or np.any(eval_dense >= OLS_FORMAL_DENSE_SIDE**2):
        raise SchemaError("eval_dense_index is outside the 41x41 dense grid")
    for sid in unique_eval_shapes:
        rows = eval_dense[eval_shape == sid]
        if rows.size != OLS_FORMAL_DENSE_SIDE**2 or not np.array_equal(np.sort(rows), np.arange(OLS_FORMAL_DENSE_SIDE**2)):
            raise SchemaError(f"eval_shape_id={int(sid)} is not paired with the complete dense index grid")
    expected_eval_shape = np.repeat(np.sort(unique_eval_shapes), OLS_FORMAL_DENSE_SIDE**2)
    expected_eval_dense = np.tile(np.arange(OLS_FORMAL_DENSE_SIDE**2), unique_eval_shapes.size)
    if not np.array_equal(eval_shape, expected_eval_shape) or not np.array_equal(eval_dense, expected_eval_dense):
        raise SchemaError("eval_shape_id/eval_dense_index rows are not in the frozen Cartesian order")
    for sid in unique_eval_shapes:
        rows = eval_q_px[eval_shape == sid]
        if not np.allclose(rows, rows[0], atol=0.0, rtol=0.0):
            raise SchemaError(f"eval_shape_id={int(sid)} maps to multiple eval_q_px rows")
    for dense_index in np.unique(eval_dense):
        rows = eval_t_px[eval_dense == dense_index]
        if not np.allclose(rows, rows[0], atol=0.0, rtol=0.0):
            raise SchemaError(f"eval_dense_index={int(dense_index)} maps to multiple eval_t_px rows")

    expected_train_p6 = ((train_q + train_t[:, None, :]) / float(coord_scale)).reshape(n_train, OLS_HEAD_DIM)
    expected_eval_p6 = ((eval_q_px + eval_t_px[:, None, :]) / float(coord_scale)).reshape(n_eval, OLS_HEAD_DIM)
    if not np.allclose(train_p6, expected_train_p6, atol=2e-5, rtol=0.0):
        raise SchemaError("train_p6_norm does not equal (train_q_px + train_t_px) / 223")
    if not np.allclose(eval_p6, expected_eval_p6, atol=2e-5, rtol=0.0):
        raise SchemaError("eval_p6_norm does not equal (eval_q_px + eval_t_px) / 223")

    if not isinstance(provenance, Mapping) or not OLS_PROVENANCE_REQUIRED_KEYS.issubset(provenance):
        missing = sorted(OLS_PROVENANCE_REQUIRED_KEYS.difference(provenance) if isinstance(provenance, Mapping) else OLS_PROVENANCE_REQUIRED_KEYS)
        raise SchemaError(f"provenance_json missing formal keys {missing}")
    if provenance.get("protocol_id") != "a10_materialize_v1":
        raise SchemaError("provenance_json.protocol_id must be a10_materialize_v1")
    protocol_sha = str(provenance.get("protocol_sha256", ""))
    if len(protocol_sha) != 64 or any(char not in "0123456789abcdefABCDEF" for char in protocol_sha):
        raise SchemaError("provenance_json.protocol_sha256 must be a SHA-256 hex digest")
    if not isinstance(provenance["source_assets"], Mapping) or not isinstance(provenance["checkpoint"], Mapping):
        raise SchemaError("provenance_json source_assets/checkpoint must be objects")
    checkpoint = provenance["checkpoint"]
    checkpoint_sha = str(checkpoint.get("sha256", ""))
    if checkpoint.get("id") != "G64" or len(checkpoint_sha) != 64 or any(char not in "0123456789abcdefABCDEF" for char in checkpoint_sha):
        raise SchemaError("provenance_json.checkpoint must identify G64 with a SHA-256 hex digest")
    pair_meta = provenance["train_pair_split"]
    eval_meta = provenance["eval"]
    if not isinstance(pair_meta, Mapping) or not isinstance(eval_meta, Mapping):
        raise SchemaError("provenance_json train_pair_split/eval must be objects")
    try:
        pair_train_count = int(pair_meta.get("train_count", -1))
        pair_train_per_tid = int(pair_meta.get("train_per_translation", -1))
        eval_row_count = int(eval_meta.get("row_count", -1))
    except (TypeError, ValueError) as exc:
        raise SchemaError("provenance_json row counts must be integers") from exc
    if pair_train_count != n_train or pair_train_per_tid != OLS_FORMAL_TRAIN_PER_TRANSLATION:
        raise SchemaError("provenance_json train_pair_split does not match formal rows")
    try:
        eval_shape_meta = np.asarray(eval_meta.get("shape_ids", []), dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise SchemaError("provenance_json eval.shape_ids must be integer IDs") from exc
    if eval_shape_meta.shape != (32,) or not np.array_equal(eval_shape_meta, np.sort(unique_eval_shapes)):
        raise SchemaError("provenance_json eval.shape_ids does not match eval rows")
    if eval_row_count != n_eval:
        raise SchemaError("provenance_json eval.row_count does not match eval rows")
    return train_q, train_t, train_shape, train_tid, eval_shape, eval_dense


def _validate_formal_arrays(
    train_gap_features: Any,
    eval_gap_features: Any,
    train_p6_norm: Any,
    eval_p6_norm: Any,
    eval_q_px: Any,
    eval_t_px: Any,
    *,
    train_ids: Any = None,
    eval_ids: Any = None,
    schema_kind: str = OLS_SCHEMA_KIND,
    target_semantics: str = OLS_TARGET_SEMANTICS,
    coord_scale: float = OLS_COORD_SCALE,
    head_dim: int = OLS_HEAD_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if schema_kind != OLS_SCHEMA_KIND:
        raise SchemaError(f"schema_kind must be {OLS_SCHEMA_KIND!r}, got {schema_kind!r}")
    if target_semantics != OLS_TARGET_SEMANTICS:
        raise SchemaError(f"target_semantics must be {OLS_TARGET_SEMANTICS!r}, got {target_semantics!r}")
    try:
        scale = float(coord_scale)
    except (TypeError, ValueError) as exc:
        raise SchemaError("coord_scale must be numeric 223") from exc
    if not np.isfinite(scale) or not np.isclose(scale, OLS_COORD_SCALE, rtol=0.0, atol=1e-12):
        raise SchemaError(f"coord_scale must be {OLS_COORD_SCALE:g}, got {coord_scale!r}")
    if isinstance(head_dim, bool) or not isinstance(head_dim, (int, np.integer)):
        raise SchemaError("head_dim must be integer 6")
    dim = int(head_dim)
    if dim != OLS_HEAD_DIM:
        raise SchemaError(f"head_dim must be {OLS_HEAD_DIM}, got {head_dim!r}")

    train_gap = _matrix(train_gap_features, "train_gap_features")
    eval_gap = _matrix(eval_gap_features, "eval_gap_features")
    train_p6 = _matrix(train_p6_norm, "train_p6_norm")
    eval_p6 = _matrix(eval_p6_norm, "eval_p6_norm")
    q_px = np.asarray(eval_q_px)
    t_px = np.asarray(eval_t_px)
    for name, arr in (("eval_q_px", q_px), ("eval_t_px", t_px)):
        if not np.issubdtype(arr.dtype, np.number):
            raise SchemaError(f"{name} must be numeric, got {arr.dtype}")
        if not np.all(np.isfinite(arr)):
            raise SchemaError(f"{name} contains non-finite values")
    if train_gap.shape[0] != train_p6.shape[0]:
        raise SchemaError("train_gap_features and train_p6_norm row counts differ")
    if eval_gap.shape[0] != eval_p6.shape[0]:
        raise SchemaError("eval_gap_features and eval_p6_norm row counts differ")
    if train_gap.shape[1] != eval_gap.shape[1]:
        raise SchemaError("train/eval GAP feature dimensions differ")
    if train_p6.shape[1] != OLS_HEAD_DIM or eval_p6.shape[1] != OLS_HEAD_DIM:
        raise SchemaError("train_p6_norm/eval_p6_norm must have shape [N, 6]")
    if q_px.ndim != 3 or q_px.shape[1:] != (3, 2):
        raise SchemaError(f"eval_q_px must have shape [N, 3, 2], got {q_px.shape}")
    if t_px.ndim != 2 or t_px.shape[1:] != (2,):
        raise SchemaError(f"eval_t_px must have shape [N, 2], got {t_px.shape}")
    if not train_gap.shape[0] or not eval_gap.shape[0]:
        raise SchemaError("formal OLS train/eval arrays cannot be empty")
    if q_px.shape[0] != eval_gap.shape[0] or t_px.shape[0] != eval_gap.shape[0]:
        raise SchemaError("eval GAP/P6/Q/t row counts differ")
    train_row_ids = _ids(train_ids, "train_ids", train_gap.shape[0])
    eval_row_ids = _ids(eval_ids, "eval_ids", eval_gap.shape[0])
    train_values = {_id_token(item.item() if isinstance(item, np.generic) else item) for item in train_row_ids}
    eval_values = {_id_token(item.item() if isinstance(item, np.generic) else item) for item in eval_row_ids}
    overlap = sorted(train_values.intersection(eval_values))
    if overlap:
        raise SchemaError(f"train_ids/eval_ids overlap ({overlap[:3]})")
    return train_gap, train_p6, eval_gap, eval_p6, q_px, t_px, train_row_ids, eval_row_ids


def _matrix(value: Any, name: str) -> np.ndarray:
    """Validate without changing the caller's dtype.

    Precision conversion happens inside each explicitly requested solver path,
    so the audit can compare the float32 and float64 calculations.
    """

    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise SchemaError(f"{name} must be a 2D matrix, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise SchemaError(f"{name} must be numeric, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise SchemaError(f"{name} contains non-finite values")
    return arr


def _check_xy(features: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    x, y = _matrix(features, "features"), _matrix(targets, "targets")
    if x.shape[0] != y.shape[0]:
        raise SchemaError(f"features rows {x.shape[0]} != targets rows {y.shape[0]}")
    if not x.shape[0]:
        raise SchemaError("empty design matrix")
    return x, y


def _dtype(precision: str) -> np.dtype:
    value = str(precision).lower()
    if value not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return np.dtype(value)


def _svd_solve(design: np.ndarray, target: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    if len(singular) == 0:
        return np.zeros((design.shape[1], target.shape[1]), dtype=design.dtype), singular, 0, None
    cutoff = float(cutoff)
    threshold = cutoff * float(singular[0]) if 0 < cutoff < 1 else cutoff
    keep = singular > threshold
    rank = int(np.count_nonzero(keep))
    inv = np.zeros_like(singular)
    inv[keep] = 1.0 / singular[keep]
    coef = (vt.T * inv) @ (u.T @ target)
    condition = float(singular[0] / singular[keep][-1]) if rank else None
    return coef, singular, rank, condition


def _prediction_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(prediction) - np.asarray(target)
    if diff.shape[1] == 1:
        return float(np.mean(np.abs(diff)))
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _support_mask(
    train_support_ids: Any,
    eval_support_ids: Any,
    n_eval: int,
) -> np.ndarray | None:
    if train_support_ids is None or eval_support_ids is None:
        return None
    train = np.asarray(train_support_ids).reshape(-1)
    eval_ids = np.asarray(eval_support_ids).reshape(-1)
    if eval_ids.size != n_eval:
        raise SchemaError(f"eval_support_ids has {eval_ids.size} entries, expected {n_eval}")
    return np.isin(eval_ids, train)


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    train_support_ids: Any = None,
    eval_support_ids: Any = None,
    precision: str = "float64",
    eval_q_px: Any = None,
    eval_t_px: Any = None,
    coord_scale: float = OLS_COORD_SCALE,
) -> dict[str, Any]:
    dtype = _dtype(precision)
    pred = np.asarray(prediction, dtype=dtype)
    true = np.asarray(target, dtype=dtype)
    raw = pred - true
    design = np.column_stack([pred, np.ones(pred.shape[0], dtype=dtype)])
    coef, _, _, _ = _svd_solve(design, true, 1e-12)
    affine_pred = design @ coef
    u = affine_pred - true
    support = _support_mask(train_support_ids, eval_support_ids, len(pred))
    out: dict[str, Any] = {
        "precision": precision,
        "n_eval": int(len(pred)),
        "box_mae_px": _prediction_mae(pred, true),
        "raw_mae_px": _prediction_mae(pred, true),
        "u_mae_px": _prediction_mae(affine_pred, true),
        "affine_removed_mae_px": _prediction_mae(affine_pred, true),
        "u_rms_px": float(np.sqrt(np.mean(np.square(u)))),
        "affine_A_pred_to_true": coef[:-1].T,
        "affine_b_px": coef[-1],
        "anchor_mae_px": None,
        "anchor_count": 0,
        "prediction": pred,
        "true": true,
        "u": u,
    }
    if eval_q_px is not None or eval_t_px is not None:
        if eval_q_px is None or eval_t_px is None:
            raise SchemaError("eval_q_px and eval_t_px must be supplied together")
        q_px = np.asarray(eval_q_px, dtype=dtype)
        t_px = np.asarray(eval_t_px, dtype=dtype)
        if pred.ndim != 2 or pred.shape[1] != OLS_HEAD_DIM:
            raise SchemaError("formal solver predictions must have shape [N, 6]")
        if q_px.shape != (len(pred), 3, 2) or t_px.shape != (len(pred), 2):
            raise SchemaError("eval_q_px/eval_t_px are not aligned with solver predictions")
        if not np.all(np.isfinite(q_px)) or not np.all(np.isfinite(t_px)):
            raise SchemaError("eval_q_px/eval_t_px contain non-finite values")
        try:
            scale_value = float(coord_scale)
        except (TypeError, ValueError) as exc:
            raise SchemaError("coord_scale must be numeric") from exc
        if not np.isfinite(scale_value) or not np.isclose(scale_value, OLS_COORD_SCALE, rtol=0.0, atol=1e-12):
            raise SchemaError(f"coord_scale must be {OLS_COORD_SCALE:g}, got {coord_scale!r}")
        scale = dtype.type(scale_value)
        pred_px = pred.reshape(len(pred), 3, 2) * scale
        t_hat = np.mean(pred_px - q_px, axis=1)
        corrected = pred_px - t_hat[:, None, :]
        # The protocol gate is the translation estimate itself against the
        # held-out translation target.  The centered point cloud below is an
        # optional shape-residual diagnostic only; it must not become the
        # headline metric (a common [3,4] translation must report 5 px).
        translation_error = t_hat - t_px
        shape_residual = corrected - q_px
        translation_norm = np.linalg.norm(translation_error, axis=-1)
        out.update(
            {
                # This is the protocol metric: compare the estimated
                # translation with eval_t_px using row-wise Euclidean error.
                "eval_box_mae_px": float(np.mean(translation_norm)),
                "t_hat_px": t_hat,
                "translation_error_px": translation_error,
                "eval_translation_mae_px": float(np.mean(translation_norm)),
                "eval_translation_mae_px_elementwise": float(np.mean(np.abs(translation_error))),
                "shape_residual_px": shape_residual,
                "shape_residual_mae_px": float(np.mean(np.linalg.norm(shape_residual, axis=-1))),
                "p6_mae_norm": float(np.mean(np.abs(pred - np.asarray(target, dtype=dtype)))),
                "p6_mae_px_elementwise": float(np.mean(np.abs(pred_px - np.asarray(target, dtype=dtype).reshape(len(pred), 3, 2) * scale))),
                "eval_q_px": q_px,
                "eval_t_px": t_px,
            }
        )
        # Keep the legacy key populated, but make it mean the formal metric
        # whenever this is a formal GAP->P6 evaluation.
        out["box_mae_px"] = out["eval_box_mae_px"]
    if support is not None:
        out["anchor_count"] = int(np.count_nonzero(support))
        if np.any(support):
            out["anchor_mae_px"] = _prediction_mae(pred[support], true[support])
    return out


def _fit_transform(
    features: np.ndarray,
    *,
    standardize: bool,
    whiten: bool,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=dtype)
    mean = np.zeros(x.shape[1], dtype=dtype)
    scale = np.ones(x.shape[1], dtype=dtype)
    if standardize or whiten:
        mean = x.mean(axis=0, dtype=dtype)
        x = x - mean
    if standardize:
        scale = x.std(axis=0, dtype=dtype)
        scale = np.where(scale <= np.finfo(dtype).eps, dtype.type(1.0), scale).astype(dtype)
        x = x / scale
    whitener = np.eye(x.shape[1], dtype=dtype)
    if whiten:
        cov = (x.T @ x) / dtype.type(max(1, x.shape[0] - 1))
        eigval, eigvec = np.linalg.eigh(cov)
        floor = max(float(np.max(eigval)) * 1e-6 if eigval.size else 0.0, float(np.finfo(dtype).eps))
        invsqrt = np.where(eigval > floor, 1.0 / np.sqrt(eigval), 0.0).astype(dtype)
        whitener = (eigvec * invsqrt) @ eigvec.T
        x = x @ whitener
    return {"features": x, "mean": mean, "scale": scale, "whitener": whitener}


def _apply_transform(features: np.ndarray, transform: Mapping[str, np.ndarray], *, dtype: np.dtype) -> np.ndarray:
    x = np.asarray(features, dtype=dtype)
    x = (x - transform["mean"]) / transform["scale"]
    return x @ transform["whitener"]


def _design(features: np.ndarray, *, fit_intercept: bool, dtype: np.dtype) -> np.ndarray:
    return np.column_stack([features, np.ones(features.shape[0], dtype=dtype)]) if fit_intercept else features


def fit_ols_svd(
    features: Any,
    targets: Any,
    *,
    svd_cutoff: float = 1e-12,
    fit_intercept: bool = True,
    standardize: bool = False,
    whiten: bool = False,
    precision: str = "float64",
    eval_features: Any = None,
    eval_targets: Any = None,
    train_support_ids: Any = None,
    eval_support_ids: Any = None,
    eval_q_px: Any = None,
    eval_t_px: Any = None,
    coord_scale: float = OLS_COORD_SCALE,
) -> dict[str, Any]:
    """Fit on train support and evaluate on the same dense eval grid."""

    x, y = _check_xy(features, targets)
    ex, ey = _check_xy(eval_features if eval_features is not None else x, eval_targets if eval_targets is not None else y)
    dtype = _dtype(precision)
    transform = _fit_transform(x, standardize=standardize, whiten=whiten, dtype=dtype)
    z_train = transform["features"]
    z_eval = _apply_transform(ex, transform, dtype=dtype)
    train_design, eval_design = _design(z_train, fit_intercept=fit_intercept, dtype=dtype), _design(z_eval, fit_intercept=fit_intercept, dtype=dtype)
    target_train, target_eval = np.asarray(y, dtype=dtype), np.asarray(ey, dtype=dtype)
    coef, singular, rank, condition = _svd_solve(train_design, target_train, svd_cutoff)
    prediction = train_design @ coef
    eval_prediction = eval_design @ coef
    eval_metrics = _metrics(
        eval_prediction,
        target_eval,
        train_support_ids=train_support_ids,
        eval_support_ids=eval_support_ids,
        precision=precision,
        eval_q_px=eval_q_px,
        eval_t_px=eval_t_px,
        coord_scale=coord_scale,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "svd",
        "precision": precision,
        "dtype": str(dtype),
        "svd_cutoff": float(svd_cutoff),
        "fit_intercept": bool(fit_intercept),
        "standardize": bool(standardize),
        "whiten": bool(whiten),
        "n_train": int(x.shape[0]),
        "n_eval": int(ex.shape[0]),
        "n_features": int(x.shape[1]),
        "rank": rank,
        "effective_rank": rank,
        "condition_number": condition,
        "singular_values": singular,
        "coef": coef,
        "prediction": prediction,
        "eval_prediction": eval_prediction,
        "prediction_mae": _prediction_mae(prediction, target_train),
        "prediction_rmse": float(np.sqrt(np.mean(np.square(prediction - target_train)))),
        "eval_prediction_mae": _prediction_mae(eval_prediction, target_eval),
        "eval_metrics": eval_metrics,
        "feature_mean": transform["mean"],
        "feature_scale": transform["scale"],
    }


def ridge_path(
    features: Any,
    targets: Any,
    *,
    lambdas: Iterable[float] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
    standardize: bool = False,
    whiten: bool = False,
    fit_intercept: bool = True,
    precision: str = "float64",
    eval_features: Any = None,
    eval_targets: Any = None,
    train_support_ids: Any = None,
    eval_support_ids: Any = None,
    eval_q_px: Any = None,
    eval_t_px: Any = None,
    coord_scale: float = OLS_COORD_SCALE,
) -> list[dict[str, Any]]:
    x, y = _check_xy(features, targets)
    ex, ey = _check_xy(eval_features if eval_features is not None else x, eval_targets if eval_targets is not None else y)
    dtype = _dtype(precision)
    transform = _fit_transform(x, standardize=standardize, whiten=whiten, dtype=dtype)
    z_train, z_eval = transform["features"], _apply_transform(ex, transform, dtype=dtype)
    train_design, eval_design = _design(z_train, fit_intercept=fit_intercept, dtype=dtype), _design(z_eval, fit_intercept=fit_intercept, dtype=dtype)
    target_train, target_eval = np.asarray(y, dtype=dtype), np.asarray(ey, dtype=dtype)
    singular = np.linalg.svd(train_design, compute_uv=False)
    rank_threshold = max(float(singular[0]) * 1e-6, float(np.finfo(dtype).eps)) if len(singular) else 0.0
    effective_rank = int(np.count_nonzero(singular > rank_threshold))
    rows: list[dict[str, Any]] = []
    for value in lambdas:
        lam = float(value)
        if lam < 0:
            raise ValueError("ridge lambda must be non-negative")
        penalty = np.eye(train_design.shape[1], dtype=dtype) * dtype.type(lam)
        if fit_intercept:
            penalty[-1, -1] = dtype.type(0.0)
        if lam == 0.0:
            coef, _, rank, condition = _svd_solve(train_design, target_train, 1e-12)
        else:
            coef = np.linalg.solve(train_design.T @ train_design + penalty, train_design.T @ target_train)
            rank = effective_rank
            condition = float(np.linalg.cond(train_design.T @ train_design + penalty))
        prediction = train_design @ coef
        eval_prediction = eval_design @ coef
        eval_metrics = _metrics(
            eval_prediction,
            target_eval,
            train_support_ids=train_support_ids,
            eval_support_ids=eval_support_ids,
            precision=precision,
            eval_q_px=eval_q_px,
            eval_t_px=eval_t_px,
            coord_scale=coord_scale,
        )
        rows.append(
            {
                "lambda": lam,
                "precision": precision,
                "standardize": bool(standardize),
                "whiten": bool(whiten),
                "fit_intercept": bool(fit_intercept),
                "effective_rank": rank,
                "condition_number": condition,
                "coef": coef,
                "prediction": prediction,
                "eval_prediction": eval_prediction,
                "prediction_mae": _prediction_mae(prediction, target_train),
                "prediction_rmse": float(np.sqrt(np.mean(np.square(prediction - target_train)))),
                "eval_prediction_mae": _prediction_mae(eval_prediction, target_eval),
                "eval_metrics": eval_metrics,
            }
        )
    return rows


def _relative_change(left: np.ndarray, right: np.ndarray) -> float:
    den = max(float(np.linalg.norm(right)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)) / den)


def _relative_metric_change(left: float, right: float) -> float:
    """Relative change of the protocol's corrected Euclidean box metric."""

    den = max(abs(float(right)), np.finfo(np.float64).eps)
    return float(abs(float(left) - float(right)) / den)


def _one_precision(
    x: np.ndarray,
    y: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    *,
    precision: str,
    svd_cutoffs: Sequence[float],
    ridge_lambdas: Sequence[float],
    perturbation_scales: Sequence[float],
    standardize: bool,
    whiten: bool,
    seed: int,
    fit_intercept: bool,
    train_support_ids: Any,
    eval_support_ids: Any,
    eval_q_px: np.ndarray,
    eval_t_px: np.ndarray,
    coord_scale: float,
) -> dict[str, Any]:
    svd_rows = [
        fit_ols_svd(
            x,
            y,
            svd_cutoff=c,
            fit_intercept=fit_intercept,
            precision=precision,
            eval_features=ex,
            eval_targets=ey,
            train_support_ids=train_support_ids,
            eval_support_ids=eval_support_ids,
            eval_q_px=eval_q_px,
            eval_t_px=eval_t_px,
            coord_scale=coord_scale,
        )
        for c in svd_cutoffs
    ]
    ridge_rows = ridge_path(
        x,
        y,
        lambdas=ridge_lambdas,
        standardize=standardize,
        whiten=whiten,
        fit_intercept=fit_intercept,
        precision=precision,
        eval_features=ex,
        eval_targets=ey,
        train_support_ids=train_support_ids,
        eval_support_ids=eval_support_ids,
        eval_q_px=eval_q_px,
        eval_t_px=eval_t_px,
        coord_scale=coord_scale,
    )
    rng = np.random.default_rng(int(seed))
    scale_base = np.asarray(x, dtype=np.float64).std(axis=0)
    scale_base[scale_base <= np.finfo(np.float64).eps] = 1.0
    perturb_rows = []
    baseline = None
    for value in perturbation_scales:
        scale = float(value)
        if scale < 0:
            raise ValueError("perturbation scale must be non-negative")
        perturbed = np.asarray(x, dtype=np.float64) + rng.normal(size=x.shape) * scale_base[None, :] * scale
        row = fit_ols_svd(
            perturbed,
            y,
            svd_cutoff=1e-12,
            fit_intercept=fit_intercept,
            precision=precision,
            eval_features=ex,
            eval_targets=ey,
            train_support_ids=train_support_ids,
            eval_support_ids=eval_support_ids,
            eval_q_px=eval_q_px,
            eval_t_px=eval_t_px,
            coord_scale=coord_scale,
        )
        if baseline is None:
            baseline = row
            prediction_change = 0.0
            coefficient_change = 0.0
            eval_change = 0.0
            eval_box_change = 0.0
        else:
            prediction_change = _relative_change(row["prediction"], baseline["prediction"])
            coefficient_change = _relative_change(row["coef"], baseline["coef"])
            eval_change = _relative_change(row["eval_prediction"], baseline["eval_prediction"])
            eval_box_change = _relative_metric_change(
                row["eval_metrics"]["eval_box_mae_px"],
                baseline["eval_metrics"]["eval_box_mae_px"],
            )
        perturb_rows.append(
            {
                "scale": scale,
                "precision": precision,
                "rank": row["rank"],
                "condition_number": row["condition_number"],
                "prediction_mae": row["prediction_mae"],
                "eval_prediction_mae": row["eval_prediction_mae"],
                "prediction_change_vs_first": prediction_change,
                "eval_prediction_change_vs_first": eval_change,
                "eval_box_mae_px_change_vs_first": eval_box_change,
                "coefficient_change_vs_first": coefficient_change,
                "eval_metrics": row["eval_metrics"],
                "eval_prediction": row["eval_prediction"],
            }
        )
    cutoff_changes = []
    for left, right in zip(svd_rows, svd_rows[1:]):
        cutoff_changes.append(
            {
                "left_cutoff": left["svd_cutoff"],
                "right_cutoff": right["svd_cutoff"],
                "prediction_change": _relative_change(left["prediction"], right["prediction"]),
                "eval_prediction_change": _relative_change(left["eval_prediction"], right["eval_prediction"]),
                "eval_box_mae_px_change": _relative_metric_change(
                    left["eval_metrics"]["eval_box_mae_px"],
                    right["eval_metrics"]["eval_box_mae_px"],
                ),
                "coefficient_change": _relative_change(left["coef"], right["coef"]),
            }
        )
    return {
        "precision": precision,
        "svd_path": svd_rows,
        "ridge_path": ridge_rows,
        "perturbation_path": perturb_rows,
        "cutoff_pairwise_changes": cutoff_changes,
    }


def audit_ols(
    features: Any,
    targets: Any,
    *,
    eval_features: Any = None,
    eval_targets: Any = None,
    train_support_ids: Any = None,
    eval_support_ids: Any = None,
    eval_q_px: Any = None,
    eval_t_px: Any = None,
    schema_kind: str = OLS_SCHEMA_KIND,
    target_semantics: str = OLS_TARGET_SEMANTICS,
    coord_scale: float = OLS_COORD_SCALE,
    head_dim: int = OLS_HEAD_DIM,
    provenance: Any = None,
    precision_modes: Sequence[str] = PRECISIONS,
    svd_cutoffs: Sequence[float] = (1e-14, 1e-12, 1e-10, 1e-8),
    ridge_lambdas: Sequence[float] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
    perturbation_scales: Sequence[float] = (0.0, 1e-8, 1e-6),
    standardize: bool = True,
    whiten: bool = True,
    seed: int = 20260821,
    fit_intercept: bool = True,
) -> dict[str, Any]:
    x, y = _check_xy(features, targets)
    ex, ey = _check_xy(eval_features if eval_features is not None else x, eval_targets if eval_targets is not None else y)
    if eval_q_px is None or eval_t_px is None:
        raise SchemaError("formal GAP->P6 audit requires eval_q_px and eval_t_px")
    train_ids = np.arange(x.shape[0], dtype=np.int64) if train_support_ids is None else train_support_ids
    eval_ids = (
        np.arange(x.shape[0], x.shape[0] + ex.shape[0], dtype=np.int64)
        if eval_support_ids is None
        else eval_support_ids
    )
    x, y, ex, ey, q_px, t_px, train_ids, eval_ids = _validate_formal_arrays(
        x,
        ex,
        y,
        ey,
        eval_q_px,
        eval_t_px,
        train_ids=train_ids,
        eval_ids=eval_ids,
        schema_kind=schema_kind,
        target_semantics=target_semantics,
        coord_scale=coord_scale,
        head_dim=head_dim,
    )
    modes = tuple(dict.fromkeys(str(mode).lower() for mode in precision_modes))
    if not modes:
        raise ValueError("precision_modes cannot be empty")
    paths = {
        mode: _one_precision(
            x,
            y,
            ex,
            ey,
            precision=mode,
            svd_cutoffs=svd_cutoffs,
            ridge_lambdas=ridge_lambdas,
            perturbation_scales=perturbation_scales,
            standardize=standardize,
            whiten=whiten,
            seed=seed,
            fit_intercept=fit_intercept,
            train_support_ids=train_ids,
            eval_support_ids=eval_ids,
            eval_q_px=q_px,
            eval_t_px=t_px,
            coord_scale=coord_scale,
        )
        for mode in modes
    }
    reference = paths.get("float64", next(iter(paths.values())))
    cutoff_changes = reference["cutoff_pairwise_changes"]
    max_cutoff = max((row["eval_box_mae_px_change"] for row in cutoff_changes), default=0.0)
    max_perturb = max((row["eval_box_mae_px_change_vs_first"] for row in reference["perturbation_path"]), default=0.0)
    ref_svd = reference["svd_path"][0] if reference["svd_path"] else None
    condition = ref_svd.get("condition_number") if ref_svd else None
    rank = ref_svd.get("rank") if ref_svd else None
    rank_deficient = rank is not None and int(rank) < int(x.shape[1] + (1 if fit_intercept else 0))
    ill_conditioned = condition is not None and float(condition) > 1e10
    if max(max_cutoff, max_perturb) > 0.10 or rank_deficient or ill_conditioned:
        verdict = "unstable"
    elif reference["svd_path"] and reference["ridge_path"]:
        verdict = "stable"
    else:
        verdict = "inconclusive"
    precision_comparison: dict[str, Any] = {}
    if "float32" in paths and "float64" in paths and paths["float32"]["svd_path"] and paths["float64"]["svd_path"]:
        f32, f64 = paths["float32"]["svd_path"][0], paths["float64"]["svd_path"][0]
        precision_comparison = {
            "svd_eval_box_mae_px_change": _relative_metric_change(
                f32["eval_metrics"]["eval_box_mae_px"], f64["eval_metrics"]["eval_box_mae_px"]
            ),
            "svd_eval_prediction_change": _relative_change(f32["eval_prediction"], f64["eval_prediction"]),
            "svd_coefficient_change": _relative_change(f32["coef"], f64["coef"]),
            "float32_eval_metrics": f32["eval_metrics"],
            "float64_eval_metrics": f64["eval_metrics"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "clean_room_frozen_g64_gap_to_p6_ols_audit",
        "schema_kind": schema_kind,
        "target_semantics": target_semantics,
        "coord_scale": float(coord_scale),
        "head_dim": int(head_dim),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(x.shape[0]),
        "n_eval": int(ex.shape[0]),
        "n_features": int(x.shape[1]),
        "p_over_n": float(x.shape[1] / x.shape[0]),
        "fit_intercept": bool(fit_intercept),
        "standardize": bool(standardize),
        "whiten": bool(whiten),
        "seed": int(seed),
        "precision_modes": list(modes),
        "precision_path": paths,
        # Float64 aliases keep the first S0 reports compact while the full
        # precision_path remains available for independent review.
        "svd_path": reference["svd_path"],
        "ridge_path": reference["ridge_path"],
        "perturbation_path": reference["perturbation_path"],
        "cutoff_pairwise_changes": cutoff_changes,
        "precision_comparison": precision_comparison,
        "eval_targets": np.asarray(ey),
        "eval_q_px": np.asarray(q_px),
        "eval_t_px": np.asarray(t_px),
        "train_ids": np.asarray(train_ids),
        "eval_ids": np.asarray(eval_ids),
        "provenance": {} if provenance is None else provenance,
        # Compatibility aliases are retained in the JSON, but the formal
        # schema and all stability decisions use the explicit IDs above.
        "eval_support_ids": np.asarray(eval_ids),
        "stability": {
            "verdict": verdict,
            "max_cutoff_eval_box_mae_px_change": max_cutoff,
            "max_perturbation_eval_box_mae_px_change": max_perturb,
            "relative_change_threshold": 0.10,
            "condition_threshold": 1e10,
            "rank_deficient": bool(rank_deficient),
            "ill_conditioned": bool(ill_conditioned),
            "note": "Each solver is fit on train support and evaluated on the same dense eval grid; stable means numerical agreement only.",
        },
    }


def load_design_target_npz(
    path: Path | str,
    *,
    feature_key: str | None = None,
    target_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate the formal frozen-G64 GAP->P6 NPZ contract.

    The historical ``train_features/train_targets`` contract is deliberately
    rejected.  A solver result is not comparable unless the target semantics,
    coordinate scale, held-out geometry, row IDs, and provenance are all
    present in the same artifact.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as blob:
        if feature_key or target_key:
            raise SchemaError("feature_key/target_key shortcuts are not allowed; use formal GAP/P6 keys")
        missing = sorted(OLS_REQUIRED_KEYS.difference(blob.files))
        has_train_ids = any(train_key in blob.files for train_key, _ in OLS_ID_KEY_PAIRS)
        has_eval_ids = any(eval_key in blob.files for _, eval_key in OLS_ID_KEY_PAIRS)
        if has_train_ids != has_eval_ids:
            missing.append("train_ids/eval_ids must be supplied together")
        if missing:
            raise SchemaError(
                f"{path} is missing formal GAP->P6 OLS fields {missing}; found {blob.files}. "
                "The historical train_features/train_targets or W/b schemas are not comparable."
            )
        schema_kind = _scalar_text(blob["schema_kind"], "schema_kind")
        target_semantics = _scalar_text(blob["target_semantics"], "target_semantics")
        coord_scale = _scalar(blob["coord_scale"], "coord_scale")
        head_dim = _scalar(blob["head_dim"], "head_dim")
        provenance_text = _scalar_text(blob["provenance_json"], "provenance_json")
        try:
            provenance = json.loads(provenance_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SchemaError(f"provenance_json is not valid JSON: {path}") from exc
        if provenance is None:
            raise SchemaError("provenance_json must not be null")
        train_shape = _integer_vector(blob["train_shape_id"], "train_shape_id", blob["train_gap_features"].shape[0])
        train_tid = _integer_vector(blob["train_translation_id"], "train_translation_id", blob["train_gap_features"].shape[0])
        eval_shape = _integer_vector(blob["eval_shape_id"], "eval_shape_id", blob["eval_gap_features"].shape[0])
        eval_dense = _integer_vector(blob["eval_dense_index"], "eval_dense_index", blob["eval_gap_features"].shape[0])
        generated_train_ids = np.asarray([f"s{int(sid):02d}_t{int(tid):02d}" for sid, tid in zip(train_shape, train_tid)])
        generated_eval_ids = np.asarray([f"s{int(sid):02d}_d{int(did):04d}" for sid, did in zip(eval_shape, eval_dense)])
        id_pair = next((pair for pair in OLS_ID_KEY_PAIRS if pair[0] in blob.files and pair[1] in blob.files), None)
        train_id_key, eval_id_key = id_pair if id_pair is not None else (None, None)
        source_train_ids = blob[train_id_key] if train_id_key is not None else generated_train_ids
        source_eval_ids = blob[eval_id_key] if eval_id_key is not None else generated_eval_ids
        x, y, ex, ey, q_px, t_px, train_ids, eval_ids = _validate_formal_arrays(
            blob["train_gap_features"],
            blob["eval_gap_features"],
            blob["train_p6_norm"],
            blob["eval_p6_norm"],
            blob["eval_q_px"],
            blob["eval_t_px"],
            train_ids=source_train_ids,
            eval_ids=source_eval_ids,
            schema_kind=schema_kind,
            target_semantics=target_semantics,
            coord_scale=coord_scale,
            head_dim=head_dim,
        )
        if [_id_token(v.item() if isinstance(v, np.generic) else v) for v in train_ids] != [
            _id_token(v.item() if isinstance(v, np.generic) else v) for v in generated_train_ids
        ] or [_id_token(v.item() if isinstance(v, np.generic) else v) for v in eval_ids] != [
            _id_token(v.item() if isinstance(v, np.generic) else v) for v in generated_eval_ids
        ]:
            raise SchemaError("explicit train_ids/eval_ids do not match semantic row IDs")
        train_q_px, train_t_px, train_shape, train_tid, eval_shape, eval_dense = _validate_historical_formal_contract(
            y,
            ey,
            blob["train_q_px"],
            blob["train_t_px"],
            q_px,
            t_px,
            train_shape,
            train_tid,
            eval_shape,
            eval_dense,
            coord_scale=float(coord_scale),
            provenance=provenance,
        )
        for alt_train_key, alt_eval_key in OLS_ID_KEY_PAIRS:
            if (alt_train_key, alt_eval_key) == (train_id_key, eval_id_key):
                continue
            if alt_train_key in blob.files and alt_eval_key in blob.files:
                alt_train = _ids(blob[alt_train_key], alt_train_key, x.shape[0])
                alt_eval = _ids(blob[alt_eval_key], alt_eval_key, ex.shape[0])
                if [_id_token(v.item() if isinstance(v, np.generic) else v) for v in alt_train] != [
                    _id_token(v.item() if isinstance(v, np.generic) else v) for v in train_ids
                ] or [_id_token(v.item() if isinstance(v, np.generic) else v) for v in alt_eval] != [
                    _id_token(v.item() if isinstance(v, np.generic) else v) for v in eval_ids
                ]:
                    raise SchemaError("train/eval ID aliases disagree row-for-row")
        meta = {
            "source": str(path.resolve()),
            "schema_keys": list(blob.files),
            "schema_kind": schema_kind,
            "target_semantics": target_semantics,
            "coord_scale": float(coord_scale),
            "head_dim": int(head_dim),
            "provenance": provenance,
            "provenance_json": provenance_text,
            "train_ids_present": id_pair is not None,
            "eval_ids_present": id_pair is not None,
            "train_support_ids_present": id_pair == OLS_ID_KEY_PAIRS[1],
            "eval_support_ids_present": id_pair == OLS_ID_KEY_PAIRS[1],
        }
    return x, y, ex, ey, {
        **meta,
        "eval_q_px": q_px,
        "eval_t_px": t_px,
        "train_q_px": train_q_px,
        "train_t_px": train_t_px,
        "train_shape_id": train_shape,
        "train_translation_id": train_tid,
        "eval_shape_id": eval_shape,
        "eval_dense_index": eval_dense,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        # Runner compatibility aliases; they are the same validated arrays.
        "train_support_ids": train_ids,
        "eval_support_ids": eval_ids,
    }


def _strip_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): _strip_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_arrays(v) for v in value]
    return value


def audit_ols_npz(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    feature_key: str | None = None,
    target_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    x, y, ex, ey, meta = load_design_target_npz(input_path, feature_key=feature_key, target_key=target_key)
    source_machine = str(kwargs.pop("source_machine", "local"))
    bundle = write_run_bundle(
        output_dir,
        [input_path],
        command=["audit-ols", str(input_path)],
        source_machine=source_machine,
        role_map={str(input_path): "ols"},
        extra_manifest={"audit": "svd_ridge_whiten_perturbation", **_strip_arrays(meta)},
    )
    result = audit_ols(
        x,
        y,
        eval_features=ex,
        eval_targets=ey,
        train_support_ids=meta.get("train_support_ids"),
        eval_support_ids=meta.get("eval_support_ids"),
        eval_q_px=meta["eval_q_px"],
        eval_t_px=meta["eval_t_px"],
        schema_kind=meta["schema_kind"],
        target_semantics=meta["target_semantics"],
        coord_scale=meta["coord_scale"],
        head_dim=meta["head_dim"],
        provenance=meta["provenance"],
        **kwargs,
    )
    result["input"] = _strip_arrays(meta)
    result["provenance_files"] = {key: str(value) for key, value in bundle.items()}
    pred_payload: dict[str, Any] = {
        "eval_true": ey,
        "eval_q_px": meta["eval_q_px"],
        "eval_t_px": meta["eval_t_px"],
        "eval_ids": meta["eval_ids"],
    }
    for precision, branch in result["precision_path"].items():
        for index, row in enumerate(branch["svd_path"]):
            key = f"{precision}_svd_{index}"
            pred_payload[f"pred_{key}"] = row["eval_prediction"]
            pred_payload[f"error_{key}"] = row["eval_prediction"] - ey
            pred_payload[f"u_{key}"] = row["eval_metrics"]["u"]
            pred_payload[f"translation_error_px_{key}"] = row["eval_metrics"]["translation_error_px"]
            pred_payload[f"shape_residual_px_{key}"] = row["eval_metrics"]["shape_residual_px"]
        for index, row in enumerate(branch["ridge_path"]):
            key = f"{precision}_ridge_{index}"
            pred_payload[f"pred_{key}"] = row["eval_prediction"]
            pred_payload[f"error_{key}"] = row["eval_prediction"] - ey
            pred_payload[f"u_{key}"] = row["eval_metrics"]["u"]
            pred_payload[f"translation_error_px_{key}"] = row["eval_metrics"]["translation_error_px"]
            pred_payload[f"shape_residual_px_{key}"] = row["eval_metrics"]["shape_residual_px"]
    np.savez_compressed(Path(output_dir) / "predictions.npz", **pred_payload)
    write_json(Path(output_dir) / "ols_audit.json", _strip_arrays(result))
    return result
