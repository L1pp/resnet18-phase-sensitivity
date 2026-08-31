"""Spatial block-CV partial-out audit for residual ``u`` fields.

The implementation is task-group aware: callers pass one field matrix per
task (for example Neural Affine 2D, AMD 2D, or A10 6D), and no rows are merged
across groups.  The nuisance basis is frozen and deliberately modest:
nearest-anchor distance and its square, boundary distance, a second-order 2D
polynomial, anchor-centred RBFs, and stride-32 sine/cosine terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_RIDGE_GRID: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_BLOCK_SIZE: float = 32.0
DEFAULT_RBF_BANDWIDTH: float = 0.25
DEFAULT_STRIDE: float = 32.0


def _coords(value: Any, *, name: str = "coords") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n, d>=2), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _signals(value: Any, *, n_rows: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 1:
        raise ValueError(f"signals must have shape (n, k>=1), got {result.shape}")
    if n_rows is not None and result.shape[0] != n_rows:
        raise ValueError(f"signals row count {result.shape[0]} != coords row count {n_rows}")
    if not np.all(np.isfinite(result)):
        raise ValueError("signals contain non-finite values")
    return result


def _spatial_bounds(coords: np.ndarray, domain: Any = None) -> tuple[np.ndarray, np.ndarray]:
    if domain is None:
        lower = np.min(coords[:, :2], axis=0)
        upper = np.max(coords[:, :2], axis=0)
    else:
        domain_array = np.asarray(domain, dtype=np.float64)
        if domain_array.shape != (2, 2):
            raise ValueError("domain must have shape ((xmin, xmax), (ymin, ymax))")
        lower = domain_array[:, 0]
        upper = domain_array[:, 1]
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper <= lower):
        raise ValueError("domain bounds must be finite with upper > lower")
    return lower, upper


def _normalised_spatial(coords: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (coords[:, :2] - lower[None, :]) / (upper - lower)[None, :]


def _default_anchors(coords: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    corners = np.asarray(
        [
            [lower[0], lower[1]],
            [lower[0], upper[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
        ],
        dtype=np.float64,
    )
    if coords.shape[1] == 2:
        return corners
    # For 6D groups the anchor-distance nuisance uses all task coordinates;
    # use the observed mean for non-spatial dimensions to avoid inventing a
    # cross-task geometry while retaining the same four 2D anchors.
    extension = np.repeat(np.mean(coords[:, 2:], axis=0, keepdims=True), 4, axis=0)
    return np.column_stack([corners, extension])


def nuisance_basis(
    coords: Any,
    *,
    anchors: Any = None,
    domain: Any = None,
    rbf_bandwidth: float = DEFAULT_RBF_BANDWIDTH,
    stride: float = DEFAULT_STRIDE,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct the frozen nuisance design matrix and column names.

    The polynomial and trigonometric terms use the first two spatial
    coordinates.  Anchor distances use all available coordinates, preserving
    a meaningful distance for the A10 6D group while keeping the basis shared.
    RBF bandwidth is in the normalised [0, 1] spatial coordinate system.
    """

    points = _coords(coords)
    lower, upper = _spatial_bounds(points, domain)
    spatial = _normalised_spatial(points, lower, upper)
    if anchors is None:
        anchor_points = _default_anchors(points, lower, upper)
    else:
        anchor_points = np.asarray(anchors, dtype=np.float64)
        if anchor_points.ndim != 2 or anchor_points.shape[0] == 0:
            raise ValueError("anchors must have shape (m, d)")
        if anchor_points.shape[1] != points.shape[1]:
            raise ValueError("anchors and coords must have the same dimension")
        if not np.all(np.isfinite(anchor_points)):
            raise ValueError("anchors contain non-finite values")
    rbf_bandwidth = float(rbf_bandwidth)
    stride = float(stride)
    if not np.isfinite(rbf_bandwidth) or rbf_bandwidth <= 0:
        raise ValueError("rbf_bandwidth must be finite and positive")
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError("stride must be finite and positive")

    delta = points[:, None, :] - anchor_points[None, :, :]
    nearest = np.sqrt(np.min(np.einsum("...i,...i->...", delta, delta), axis=1))
    boundary = np.min(
        np.column_stack(
            [
                points[:, 0] - lower[0],
                upper[0] - points[:, 0],
                points[:, 1] - lower[1],
                upper[1] - points[:, 1],
            ]
        ),
        axis=1,
    )
    anchor_spatial = (anchor_points[:, :2] - lower[None, :]) / (upper - lower)[None, :]
    rbf_delta = spatial[:, None, :] - anchor_spatial[None, :, :]
    rbf = np.exp(-0.5 * np.sum(rbf_delta * rbf_delta, axis=2) / (rbf_bandwidth**2))
    phase = 2.0 * np.pi * points[:, :2] / stride
    columns = [
        np.ones(points.shape[0], dtype=np.float64),
        nearest,
        nearest * nearest,
        boundary,
        spatial[:, 0],
        spatial[:, 1],
        spatial[:, 0] ** 2,
        spatial[:, 0] * spatial[:, 1],
        spatial[:, 1] ** 2,
    ]
    names = [
        "intercept",
        "nearest_anchor_distance",
        "nearest_anchor_distance_sq",
        "boundary_distance",
        "poly_x",
        "poly_y",
        "poly_x2",
        "poly_xy",
        "poly_y2",
    ]
    columns.extend(rbf[:, index] for index in range(rbf.shape[1]))
    names.extend(f"anchor_rbf_{index}" for index in range(rbf.shape[1]))
    columns.extend(
        [
            np.sin(phase[:, 0]),
            np.cos(phase[:, 0]),
            np.sin(phase[:, 1]),
            np.cos(phase[:, 1]),
        ]
    )
    names.extend(
        (
            f"sin_x_stride{int(stride) if stride.is_integer() else stride:g}",
            f"cos_x_stride{int(stride) if stride.is_integer() else stride:g}",
            f"sin_y_stride{int(stride) if stride.is_integer() else stride:g}",
            f"cos_y_stride{int(stride) if stride.is_integer() else stride:g}",
        )
    )
    matrix = np.column_stack(columns)
    return matrix, tuple(names)


def spatial_block_labels(coords: Any, *, block_size: float = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    """Assign deterministic rectangular spatial blocks using x/y coordinates."""

    points = _coords(coords)
    block_size = float(block_size)
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError("block_size must be finite and positive")
    origin = np.min(points[:, :2], axis=0)
    labels = np.floor((points[:, :2] - origin[None, :]) / block_size).astype(np.int64)
    return labels[:, 0] * (int(np.max(labels[:, 1])) + 1) + labels[:, 1]


def _ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = float(ridge) * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs, rcond=1e-12) @ rhs


@dataclass(frozen=True)
class RidgeCVResult:
    selected_ridge: float
    scores: dict[float, float]
    block_count: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scores"] = {str(key): float(value) for key, value in self.scores.items()}
        return result


def spatial_block_cv_ridge(
    nuisance: Any,
    target: Any,
    coords: Any,
    *,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    block_size: float = DEFAULT_BLOCK_SIZE,
) -> RidgeCVResult:
    """Select ridge by leave-one-spatial-block-out mean squared error."""

    design = _signals(nuisance)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    points = _coords(coords)
    if design.shape[0] != y.size or y.size != points.shape[0]:
        raise ValueError("nuisance, target, and coords row counts must agree")
    if not np.all(np.isfinite(y)):
        raise ValueError("target contains non-finite values")
    if not np.allclose(design[:, 0], 1.0, rtol=0.0, atol=1e-8):
        design = np.column_stack([np.ones(design.shape[0]), design])
    labels = spatial_block_labels(points, block_size=block_size)
    unique_blocks = np.unique(labels)
    if unique_blocks.size < 3:
        raise ValueError(
            "spatial block CV requires at least 3 distinct blocks; "
            "a single-block/in-sample score cannot support a scientific gate"
        )
    candidates = tuple(sorted({float(value) for value in ridge_grid}))
    if not candidates or any(not np.isfinite(value) or value < 0 for value in candidates):
        raise ValueError("ridge_grid must contain finite non-negative values")
    scores: dict[float, float] = {}
    for ridge in candidates:
        losses: list[float] = []
        for block in unique_blocks:
            test = labels == block
            train = ~test
            coefficients = _ridge_fit(design[train], y[train], ridge)
            prediction = design[test] @ coefficients
            losses.extend((prediction - y[test]) ** 2)
        scores[ridge] = float(np.mean(losses))
    selected = min(scores, key=lambda value: (scores[value], value))
    return RidgeCVResult(selected_ridge=selected, scores=scores, block_count=int(unique_blocks.size))


def residualize(
    signals: Any,
    nuisance: Any,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit nuisance ridge and return residuals plus fitted values."""

    values = _signals(signals)
    design = _signals(nuisance, n_rows=values.shape[0])
    if not np.allclose(design[:, 0], 1.0, rtol=0.0, atol=1e-8):
        design = np.column_stack([np.ones(design.shape[0]), design])
    fitted = np.empty_like(values)
    for index in range(values.shape[1]):
        fitted[:, index] = design @ _ridge_fit(design, values[:, index], ridge)
    return values - fitted, fitted


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def pairwise_correlations(signals: Any) -> np.ndarray:
    """Return a symmetric seed-correlation matrix."""

    values = _signals(signals)
    result = np.eye(values.shape[1], dtype=np.float64)
    for left in range(values.shape[1]):
        for right in range(left):
            result[left, right] = result[right, left] = _pearson(
                values[:, left], values[:, right]
            )
    return result


@dataclass(frozen=True)
class PartialOutGate:
    status: str
    passed: bool
    median_original_correlation: float
    median_partial_correlation: float
    retained_fraction: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_partial_out_gate(
    original_correlations: Any,
    partial_correlations: Any,
    *,
    min_median_partial: float = 0.50,
    min_retained_fraction: float = 0.70,
    close_median_partial: float = 0.20,
    close_retained_fraction: float = 0.30,
) -> PartialOutGate:
    """Apply the preregistered survived/closed/inconclusive thresholds."""

    original = np.asarray(original_correlations, dtype=np.float64)
    partial = np.asarray(partial_correlations, dtype=np.float64)
    if original.shape != partial.shape or original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError("correlation matrices must be square and shape-matched")
    upper = np.triu_indices(original.shape[0], k=1)
    original_values = original[upper]
    partial_values = partial[upper]
    mask = np.isfinite(original_values) & np.isfinite(partial_values)
    if not np.any(mask):
        raise ValueError("correlation matrices contain no finite off-diagonal pairs")
    original_values = original_values[mask]
    partial_values = partial_values[mask]
    median_original = float(np.median(original_values))
    median_partial = float(np.median(partial_values))
    if abs(median_original) <= 1e-15:
        retained = float("nan")
    else:
        retained = float(median_partial / median_original)
    if median_partial >= min_median_partial and retained >= min_retained_fraction:
        status = "survived"
        passed = True
        reason = "partial correlation remains strong after fixed nuisance removal"
    elif median_partial < close_median_partial or retained <= close_retained_fraction:
        status = "closed"
        passed = False
        reason = "partial correlation is weak or retains too little original signal"
    else:
        status = "inconclusive"
        passed = False
        reason = "partial correlation falls between preregistered thresholds"
    return PartialOutGate(
        status=status,
        passed=passed,
        median_original_correlation=median_original,
        median_partial_correlation=median_partial,
        retained_fraction=retained,
        reason=reason,
    )


@dataclass(frozen=True)
class PartialOutResult:
    original_correlations: np.ndarray
    partial_correlations: np.ndarray
    residual_signals: np.ndarray
    nuisance_feature_names: tuple[str, ...]
    ridge_cv: RidgeCVResult
    gate: PartialOutGate

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_correlations": self.original_correlations.tolist(),
            "partial_correlations": self.partial_correlations.tolist(),
            "residual_signals": self.residual_signals.tolist(),
            "nuisance_feature_names": list(self.nuisance_feature_names),
            "ridge_cv": self.ridge_cv.to_dict(),
            "gate": self.gate.to_dict(),
        }


def _vector_values(value: Any, coords: Any) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalise vector fields to [appearance, spatial, seed, component]."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 2:
        raise ValueError(
            "signals_vector must end in component dimension 2 and have "
            "[appearance, spatial, seed, 2] or [appearance, ny, nx, seed, 2] shape"
        )
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim == 4 and points.shape[-1] == 2:
        if array.ndim != 5 or tuple(array.shape[:3]) != tuple(points.shape[:3]):
            raise ValueError("grid-shaped signals_vector and coords are not shape-aligned")
        if not np.allclose(points, points[0][None, ...], rtol=0.0, atol=0.0):
            raise ValueError("appearance-dependent coords are not allowed in vector partial-out")
        points = points[0].reshape(-1, 2)
        array = array.reshape(array.shape[0], -1, array.shape[-2], 2)
    elif points.ndim == 3 and points.shape[-1] == 2:
        if array.ndim != 4 or tuple(array.shape[:2]) != tuple(points.shape[:2]):
            raise ValueError("appearance/spatial signals_vector and coords are not shape-aligned")
        if not np.allclose(points, points[0][None, ...], rtol=0.0, atol=0.0):
            raise ValueError("appearance-dependent coords are not allowed in vector partial-out")
        points = points[0]
    elif points.ndim == 2 and points.shape[1] >= 2:
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[1] != points.shape[0]:
            raise ValueError("flat signals_vector and coords are not row-aligned")
    else:
        raise ValueError("coords must have shape [spatial,2] or [appearance, spatial, 2]")
    if array.shape[1] != points.shape[0]:
        raise ValueError("signals_vector spatial rows and coords rows differ")
    if array.shape[2] < 2:
        raise ValueError("vector partial-out requires at least two seeds")
    if not np.all(np.isfinite(array)) or not np.all(np.isfinite(points)):
        raise ValueError("signals_vector/coords contain non-finite values")
    return array, points


@dataclass(frozen=True)
class VectorPartialOutResult:
    """Partial-out report for appearance x spatial x seed x (x,y) fields."""

    pooled_original_correlations: np.ndarray
    pooled_partial_correlations: np.ndarray
    x_original_correlations: np.ndarray
    x_partial_correlations: np.ndarray
    y_original_correlations: np.ndarray
    y_partial_correlations: np.ndarray
    residual_vectors: np.ndarray
    nuisance_feature_names: tuple[str, ...]
    ridge_cv: RidgeCVResult
    gate: PartialOutGate
    appearance_count: int
    spatial_count: int
    seed_count: int

    @property
    def original_correlations(self) -> np.ndarray:
        """Compatibility alias: the pooled-vector matrix is the scientific gate."""

        return self.pooled_original_correlations

    @property
    def partial_correlations(self) -> np.ndarray:
        return self.pooled_partial_correlations

    @property
    def residual_signals(self) -> np.ndarray:
        """Compatibility alias containing pooled vector rows x seed."""

        return self.residual_vectors.transpose(0, 1, 3, 2).reshape(
            self.appearance_count * self.spatial_count * 2, self.seed_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            # Keep the old keys mapped to the preregistered pooled-vector gate.
            "original_correlations": self.pooled_original_correlations.tolist(),
            "partial_correlations": self.pooled_partial_correlations.tolist(),
            "residual_signals": self.residual_signals.tolist(),
            "pooled_vector_original_correlations": self.pooled_original_correlations.tolist(),
            "pooled_vector_partial_correlations": self.pooled_partial_correlations.tolist(),
            "x_original_correlations": self.x_original_correlations.tolist(),
            "x_partial_correlations": self.x_partial_correlations.tolist(),
            "y_original_correlations": self.y_original_correlations.tolist(),
            "y_partial_correlations": self.y_partial_correlations.tolist(),
            "residual_vectors_shape": list(self.residual_vectors.shape),
            "nuisance_feature_names": list(self.nuisance_feature_names),
            "ridge_cv": self.ridge_cv.to_dict(),
            "gate": self.gate.to_dict(),
            "appearance_count": self.appearance_count,
            "spatial_count": self.spatial_count,
            "seed_count": self.seed_count,
            "gate_signal": "pooled_vector_flatten",
            "u_norm_is_supplementary": True,
        }


def partial_out_vector_group(
    signals_vector: Any,
    coords: Any,
    *,
    anchors: Any = None,
    domain: Any = None,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    block_size: float = DEFAULT_BLOCK_SIZE,
    rbf_bandwidth: float = DEFAULT_RBF_BANDWIDTH,
    stride: float = DEFAULT_STRIDE,
) -> VectorPartialOutResult:
    """Fit the same spatial nuisance per seed/component and gate pooled vectors."""

    values, points = _vector_values(signals_vector, coords)
    appearance_count, spatial_count, seed_count, _ = values.shape
    nuisance, names = nuisance_basis(
        points,
        anchors=anchors,
        domain=domain,
        rbf_bandwidth=rbf_bandwidth,
        stride=stride,
    )
    design_rep = np.tile(nuisance, (appearance_count, 1))
    coords_rep = np.tile(points, (appearance_count, 1))
    # CV target order is appearance, spatial, component, seed; repeat the
    # same spatial design for every seed/component instead of allowing
    # appearance or component to alter nuisance complexity.
    pooled_target = values.transpose(0, 1, 3, 2).reshape(-1)
    pooled_nuisance = np.repeat(design_rep, 2 * seed_count, axis=0)
    pooled_coords = np.repeat(coords_rep, 2 * seed_count, axis=0)
    ridge_cv = spatial_block_cv_ridge(
        pooled_nuisance,
        pooled_target,
        pooled_coords,
        ridge_grid=ridge_grid,
        block_size=block_size,
    )
    residual = np.empty_like(values)
    for component in range(2):
        component_values = values[..., component].reshape(appearance_count * spatial_count, seed_count)
        component_residual, _ = residualize(component_values, design_rep, ridge=ridge_cv.selected_ridge)
        residual[..., component] = component_residual.reshape(appearance_count, spatial_count, seed_count)
    original_x = pairwise_correlations(values[..., 0].reshape(appearance_count * spatial_count, seed_count))
    partial_x = pairwise_correlations(residual[..., 0].reshape(appearance_count * spatial_count, seed_count))
    original_y = pairwise_correlations(values[..., 1].reshape(appearance_count * spatial_count, seed_count))
    partial_y = pairwise_correlations(residual[..., 1].reshape(appearance_count * spatial_count, seed_count))
    pooled_original = pairwise_correlations(values.transpose(0, 1, 3, 2).reshape(appearance_count * spatial_count * 2, seed_count))
    pooled_partial = pairwise_correlations(residual.transpose(0, 1, 3, 2).reshape(appearance_count * spatial_count * 2, seed_count))
    gate = evaluate_partial_out_gate(pooled_original, pooled_partial)
    return VectorPartialOutResult(
        pooled_original_correlations=pooled_original,
        pooled_partial_correlations=pooled_partial,
        x_original_correlations=original_x,
        x_partial_correlations=partial_x,
        y_original_correlations=original_y,
        y_partial_correlations=partial_y,
        residual_vectors=residual,
        nuisance_feature_names=names,
        ridge_cv=ridge_cv,
        gate=gate,
        appearance_count=appearance_count,
        spatial_count=spatial_count,
        seed_count=seed_count,
    )


def partial_out_group(
    signals: Any,
    coords: Any,
    *,
    anchors: Any = None,
    domain: Any = None,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    block_size: float = DEFAULT_BLOCK_SIZE,
    rbf_bandwidth: float = DEFAULT_RBF_BANDWIDTH,
    stride: float = DEFAULT_STRIDE,
) -> PartialOutResult:
    """Run the complete partial-out audit for one task group only."""

    points = _coords(coords)
    values = _signals(signals, n_rows=points.shape[0])
    if values.shape[1] < 2:
        raise ValueError("partial-out gate requires at least two seed/field columns")
    nuisance, names = nuisance_basis(
        points,
        anchors=anchors,
        domain=domain,
        rbf_bandwidth=rbf_bandwidth,
        stride=stride,
    )
    # Select one ridge from pooled seed prediction error, then apply that fixed
    # value to every seed. This avoids using a different nuisance complexity for
    # each pairwise correlation.
    pooled_target = values.reshape(-1)
    pooled_nuisance = np.repeat(nuisance, values.shape[1], axis=0)
    pooled_coords = np.repeat(points, values.shape[1], axis=0)
    ridge_cv = spatial_block_cv_ridge(
        pooled_nuisance,
        pooled_target,
        pooled_coords,
        ridge_grid=ridge_grid,
        block_size=block_size,
    )
    residuals, _ = residualize(values, nuisance, ridge=ridge_cv.selected_ridge)
    original = pairwise_correlations(values)
    partial = pairwise_correlations(residuals)
    gate = evaluate_partial_out_gate(original, partial)
    return PartialOutResult(
        original_correlations=original,
        partial_correlations=partial,
        residual_signals=residuals,
        nuisance_feature_names=names,
        ridge_cv=ridge_cv,
        gate=gate,
    )


def partial_out_groups(
    groups: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, PartialOutResult]:
    """Run groups independently; no cross-task rows or correlations are mixed."""

    results: dict[str, PartialOutResult] = {}
    for group_name, payload in groups.items():
        if isinstance(payload, Mapping):
            if "coords" not in payload:
                raise KeyError(f"group {group_name!r} requires coords")
            local_kwargs = dict(kwargs)
            for key in ("anchors", "domain"):
                if key in payload:
                    local_kwargs[key] = payload[key]
            if "signals_vector" in payload:
                results[group_name] = partial_out_vector_group(
                    payload["signals_vector"], payload["coords"], **local_kwargs
                )
            elif "signals" in payload:
                results[group_name] = partial_out_group(
                    payload["signals"], payload["coords"], **local_kwargs
                )
            else:
                raise KeyError(f"group {group_name!r} requires signals or signals_vector")
        elif isinstance(payload, (tuple, list)) and len(payload) == 2:
            results[group_name] = partial_out_group(payload[1], payload[0], **kwargs)
        else:
            raise TypeError(
                f"group {group_name!r} must be mapping {{coords, signals}} or (coords, signals)"
            )
    return results


audit_partial_out = partial_out_group
audit_partial_out_vector = partial_out_vector_group


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_RBF_BANDWIDTH",
    "DEFAULT_RIDGE_GRID",
    "DEFAULT_STRIDE",
    "PartialOutGate",
    "PartialOutResult",
    "VectorPartialOutResult",
    "RidgeCVResult",
    "audit_partial_out",
    "audit_partial_out_vector",
    "evaluate_partial_out_gate",
    "nuisance_basis",
    "pairwise_correlations",
    "partial_out_group",
    "partial_out_vector_group",
    "partial_out_groups",
    "residualize",
    "spatial_block_cv_ridge",
    "spatial_block_labels",
]
