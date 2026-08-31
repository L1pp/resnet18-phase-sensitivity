"""Independent headline recomputation from ``pred`` and ``true`` only."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .assets import sha256_file, write_run_bundle
from .contracts import SCHEMA_VERSION, SchemaError, write_json


def _field_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise SchemaError(f"{name} must have shape [..., 2], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise SchemaError(f"{name} contains non-finite values")
    return arr


def _broadcast_fields(pred: Any, true: Any) -> tuple[np.ndarray, np.ndarray]:
    p = _field_array(pred, "pred")
    t = _field_array(true, "true")
    try:
        p, t = np.broadcast_arrays(p, t)
    except ValueError as exc:
        raise SchemaError(f"pred shape {p.shape} and true shape {t.shape} are incompatible") from exc
    return p, t


def load_field_npz(path: Path | str) -> dict[str, Any]:
    """Load only pred/true; stale err/u/metrics keys are intentionally ignored."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as blob:
        if "pred" not in blob.files or "true" not in blob.files:
            raise SchemaError(f"{path} must contain pred and true arrays; found {blob.files}")
        pred = np.array(blob["pred"], copy=True)
        true = np.array(blob["true"], copy=True)
        coords = np.array(blob["coords"], copy=True) if "coords" in blob.files else None
        support_ids = np.array(blob["support_ids"], copy=True) if "support_ids" in blob.files else None
        metadata: dict[str, Any] = {}
        for key in ("n_eval", "n_grid", "image_size", "coord_scale", "code_rev", "machine"):
            if key in blob.files:
                value = blob[key]
                metadata[key] = value.item() if np.asarray(value).ndim == 0 else np.asarray(value).tolist()
    return {
        "pred": pred,
        "true": true,
        "coords": coords,
        "support_ids": support_ids,
        "metadata": metadata,
        "source": str(path.resolve()),
    }


def fit_affine(pred: Any, true: Any) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``true ~= A @ pred + b`` in float64 and return ``A, b``."""

    p, t = _broadcast_fields(pred, true)
    x = np.column_stack([p.reshape(-1, 2), np.ones(p.size // 2, dtype=np.float64)])
    coeff, _, _, _ = np.linalg.lstsq(x, t.reshape(-1, 2), rcond=None)
    return coeff[:2].T, coeff[2]


def apply_affine(pred: Any, matrix: Any, bias: Any) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    return p @ np.asarray(matrix, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)


def _mae(value: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(value, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.linalg.norm(diff.reshape(-1, 2), axis=1)))


def _support_mask(support_indices: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if support_indices is None:
        return None
    n = int(np.prod(shape[:-1]))
    support = np.asarray(support_indices)
    if support.dtype == bool:
        mask = support.reshape(-1)
        if mask.size != n:
            raise ValueError(f"boolean support has {mask.size} entries, expected {n}")
        return mask
    if support.ndim == 2 and support.shape[1] == 2:
        # Coordinate pairs are interpreted as (row, column) on a shared grid.
        if len(shape) < 4:
            raise ValueError("coordinate support requires a field with eval and 2D grid axes")
        ny, nx = int(shape[-3]), int(shape[-2])
        rows, cols = support[:, 0].astype(int), support[:, 1].astype(int)
        if np.any(rows < 0) or np.any(rows >= ny) or np.any(cols < 0) or np.any(cols >= nx):
            raise ValueError("support coordinates are outside the grid")
        one = np.zeros((ny, nx), dtype=bool)
        one[rows, cols] = True
        return np.broadcast_to(one, shape[:-3] + one.shape).reshape(-1)
    indices = support.astype(np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("support index is outside flattened field")
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return mask


def _default_coords(shape: tuple[int, ...]) -> np.ndarray:
    """Create explicit index coordinates when an old field omitted them."""

    if len(shape) >= 4:
        ny, nx = int(shape[-3]), int(shape[-2])
        yy, xx = np.meshgrid(np.arange(ny, dtype=np.int64), np.arange(nx, dtype=np.int64), indexing="ij")
        one = np.stack([yy, xx], axis=-1)
        return np.broadcast_to(one, shape[:-3] + one.shape).copy()
    n = int(np.prod(shape[:-1]))
    return np.arange(n, dtype=np.int64)[:, None]


def _align_metadata_array(value: Any, shape: tuple[int, ...], name: str, default: np.ndarray) -> tuple[np.ndarray, bool]:
    if value is None:
        return default, True
    arr = np.asarray(value)
    target_shape = shape if name == "coords" or (arr.ndim == len(shape) and arr.shape[-1:] == (2,)) else shape[:-1]
    if name == "support_ids" and arr.ndim == 1 and len(target_shape) > 1 and arr.size == target_shape[0]:
        arr = arr.reshape((arr.size,) + (1,) * (len(target_shape) - 1))
    try:
        arr = np.broadcast_to(arr, target_shape)
    except ValueError as exc:
        raise SchemaError(f"{name} shape {arr.shape} cannot align with field {shape}") from exc
    return np.array(arr, copy=True), False


def _condition_number(matrix: np.ndarray) -> float | None:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return None
    positive = singular[singular > np.finfo(np.float64).eps * singular[0]]
    return float(singular[0] / positive[-1]) if len(positive) else None


def evaluate_arrays(
    pred: Any,
    true: Any,
    *,
    support_indices: Any = None,
    metadata: Mapping[str, Any] | None = None,
    source: str = "arrays",
    coords: Any = None,
    support_ids: Any = None,
) -> dict[str, Any]:
    """Recompute raw error, affine fit, and residual ``u`` from first principles."""

    p, t = _broadcast_fields(pred, true)
    coords_arr, coords_generated = _align_metadata_array(coords, p.shape, "coords", _default_coords(p.shape))
    support_default = np.arange(int(np.prod(p.shape[:-1])), dtype=np.int64).reshape(p.shape[:-1])
    support_arr, support_generated = _align_metadata_array(support_ids, p.shape, "support_ids", support_default)
    raw_error = p - t
    matrix, bias = fit_affine(p, t)
    affine_pred = apply_affine(p, matrix, bias)
    u = affine_pred - t
    support_mask = _support_mask(support_indices, p.shape)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "clean_room_headline_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "shape": list(p.shape),
        "n_vectors": int(p.size // 2),
        "pred": p,
        "true": t,
        "coords": coords_arr,
        "support_ids": support_arr,
        "coordinates_generated": coords_generated,
        "support_ids_generated": support_generated,
        "raw_mae_px": _mae(p, t),
        "affine_removed_mae_px": _mae(affine_pred, t),
        "u_rms_px": float(np.sqrt(np.mean(np.square(u)))),
        "u_max_norm_px": float(np.max(np.linalg.norm(u.reshape(-1, 2), axis=1))),
        "affine": {
            "A_pred_to_true": matrix,
            "b_px": bias,
            "condition_number": _condition_number(np.column_stack([p.reshape(-1, 2), np.ones(p.size // 2)])),
        },
        "raw_error": raw_error,
        "u": u,
        "affine_pred": affine_pred,
    }
    if support_mask is not None:
        flat_p, flat_t = p.reshape(-1, 2), t.reshape(-1, 2)
        result["support_count"] = int(np.count_nonzero(support_mask))
        result["support_mae_px"] = float(np.mean(np.linalg.norm(flat_p[support_mask] - flat_t[support_mask], axis=1)))
        result["anchor_mae_px"] = result["support_mae_px"]
    per_eval = []
    if p.ndim >= 3:
        # The first axis is the appearance/evaluation axis for the standard
        # [n_eval, ny, nx, 2] field schema.  Keep this diagnostic separate from
        # the headline aggregate used by the historical reports.
        for index in range(p.shape[0]):
            a, b = fit_affine(p[index], t[index])
            q = apply_affine(p[index], a, b)
            per_eval.append(
                {
                    "index": index,
                    "raw_mae_px": _mae(p[index], t[index]),
                    "affine_removed_mae_px": _mae(q, t[index]),
                }
            )
        result["per_eval"] = per_eval
    if metadata:
        result["input_metadata"] = dict(metadata)
    return result


def evaluate_npz(path: Path | str, *, support_indices: Any = None) -> dict[str, Any]:
    loaded = load_field_npz(path)
    return evaluate_arrays(
        loaded["pred"],
        loaded["true"],
        support_indices=support_indices,
        metadata=loaded["metadata"],
        source=loaded["source"],
        coords=loaded["coords"],
        support_ids=loaded["support_ids"],
    )


def _safe_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return stem or "field"


def _run_id(path: Path, explicit: str | None = None) -> str:
    if explicit:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", explicit).strip("._")
        if not value:
            raise ValueError(f"run_id is empty after sanitization: {explicit!r}")
        return value
    # The hash prevents two old directories with the same stem from
    # overwriting one another while remaining stable across working dirs.
    return f"{_safe_name(path)}__{sha256_file(path)[:12]}"


def audit_headlines(
    input_paths: Iterable[Path | str],
    output_dir: Path | str,
    *,
    support_indices: Any = None,
    source_machine: str = "local",
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit one or more old field files into a new result directory."""

    paths = [Path(p) for p in input_paths]
    if run_ids is not None and len(run_ids) != len(paths):
        raise ValueError(f"run_ids has {len(run_ids)} entries for {len(paths)} inputs")
    ids = [_run_id(path, run_ids[index] if run_ids is not None else None) for index, path in enumerate(paths)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"run_id collision detected: {ids}")
    bundle = write_run_bundle(
        output_dir,
        paths,
        command=["audit-headlines", *[str(p) for p in paths]],
        source_machine=source_machine,
        role_map={str(path): "field_rendered_xy" for path in paths},
        extra_manifest={
            "evaluator": "clean_room",
            "uses_old_u": False,
            "uses_old_metrics": False,
            "run_ids": ids,
        },
    )
    out = Path(output_dir).resolve()
    rows = []
    for path, run_id in zip(paths, ids):
        result = evaluate_npz(path, support_indices=support_indices)
        summary = {
            key: value
            for key, value in result.items()
            if key not in {"pred", "true", "coords", "support_ids", "raw_error", "u", "affine_pred"}
        }
        summary["run_id"] = run_id
        write_json(out / f"{run_id}.json", summary)
        np.savez_compressed(
            out / f"{run_id}_recomputed.npz",
            pred=result["pred"],
            true=result["true"],
            coords=result["coords"],
            support_ids=result["support_ids"],
            raw_error=result["raw_error"],
            u=result["u"],
            affine_pred=result["affine_pred"],
        )
        rows.append(summary)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "kind": "clean_room_headline_audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "provenance_files": {key: str(value) for key, value in bundle.items()},
    }
    write_json(out / "summary.json", aggregate)
    return aggregate
