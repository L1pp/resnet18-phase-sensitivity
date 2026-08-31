"""Leakage-safe support-geometry and position-kernel audit helpers.

This module deliberately contains no project-specific imports.  It operates on
materialised coordinate arrays and scalar error summaries, which makes the
held-out split auditable before any model is run.  The fit/held-out support
names below are protocol constants; callers must provide the actual point sets.

The functions are small NumPy building blocks rather than an experiment runner:
the CLI can use them to build a frozen feature table from an appearance-aligned
GAP grid, fit the log-error model on the six fit geometries, and evaluate the
three held-out geometries exactly once.  Coordinates are used only for the
Euclidean covering feature; the kernel is computed in the frozen GAP feature
space, never in coordinate space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FIT_SUPPORT_NAMES: tuple[str, ...] = (
    "G64",
    "G32",
    "G16",
    "G9",
    "maximin9",
    "boundary8_center",
)
HELD_OUT_SUPPORT_NAMES: tuple[str, ...] = ("random9", "cross5", "diagonal8")
PREDICTOR_FEATURE_NAMES: tuple[str, ...] = (
    "euclidean_covering_radius",
    "gap_kernel_posterior_variance",
)
DEFAULT_RIDGE_GRID: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)


def _points(points: Any, *, name: str = "points") -> np.ndarray:
    """Return a finite two-dimensional float array without changing the input."""

    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d), got {result.shape}")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if result.shape[1] == 0:
        raise ValueError(f"{name} must have at least one coordinate")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return result


def _same_dimension(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"point dimensions differ: {left.shape[1]} versus {right.shape[1]}"
        )


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Numerically stable pairwise squared Euclidean distances."""

    delta = left[:, None, :] - right[None, :, :]
    result = np.einsum("...i,...i->...", delta, delta)
    return np.maximum(result, 0.0)


def rbf_kernel(
    left: Any,
    right: Any,
    *,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute an isotropic RBF kernel matrix.

    ``bandwidth`` is the kernel correlation length in the same coordinate
    units as the supplied points.  It is intentionally explicit so a frozen
    protocol can record it in its manifest.
    """

    x = _points(left, name="left")
    y = _points(right, name="right")
    _same_dimension(x, y)
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be a finite positive number")
    return np.exp(-0.5 * _squared_distances(x, y) / (bandwidth * bandwidth))


def nearest_euclidean_distances(
    support_points: Any,
    query_points: Any,
) -> np.ndarray:
    """Distance from each query point to its nearest support point."""

    support = _points(support_points, name="support_points")
    query = _points(query_points, name="query_points")
    _same_dimension(support, query)
    return np.sqrt(np.min(_squared_distances(query, support), axis=1))


def euclidean_covering_radius(
    support_points: Any,
    evaluation_points: Any,
) -> float:
    """Return the Euclidean covering radius of ``evaluation_points`` by support.

    The evaluation grid is supplied separately from the support set.  This is
    important for the diagonal8 case, whose in-hull metric can be deceptively
    small while its full-box covering radius is large.
    """

    distances = nearest_euclidean_distances(support_points, evaluation_points)
    return float(np.max(distances))


def empirical_kernel_distance(
    query_points: Any,
    support_points: Any,
    *,
    gap_manifest: Mapping[str, Any] | None = None,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Distance to the empirical support mean in the RBF RKHS.

    For a support empirical measure ``mu_S = mean_s k(s, .)``, this computes
    ``sqrt(k(q,q) - 2 mean_s k(q,s) + mean_{s,t} k(s,t))``.  It is an
    empirical, support-set-level quantity; no held-out target errors enter it.
    The arrays are feature vectors. In the closeout protocol they must be
    appearance-aligned GAP vectors, not raw support coordinates.
    """

    if gap_manifest is None:
        raise ValueError("gap_manifest is required for empirical kernel distance")
    query, _ = _validate_gap_grid(query_points, gap_manifest)
    support = _points(support_points, name="support_points")
    _same_dimension(query, support)
    q_s = rbf_kernel(query, support, bandwidth=bandwidth)
    s_s = rbf_kernel(support, support, bandwidth=bandwidth)
    squared = 1.0 - 2.0 * np.mean(q_s, axis=1) + float(np.mean(s_s))
    return np.sqrt(np.maximum(squared, 0.0))


def empirical_kernel_covering_radius(
    support_points: Any,
    evaluation_points: Any,
    *,
    gap_manifest: Mapping[str, Any] | None = None,
    bandwidth: float = 1.0,
) -> float:
    """Maximum empirical RKHS distance over an evaluation grid."""

    distances = empirical_kernel_distance(
        evaluation_points,
        support_points,
        gap_manifest=gap_manifest,
        bandwidth=bandwidth,
    )
    return float(np.max(distances))


def kernel_posterior_variance(
    query_points: Any,
    support_points: Any,
    *,
    gap_manifest: Mapping[str, Any] | None = None,
    bandwidth: float = 1.0,
    noise: float = 1e-6,
) -> np.ndarray:
    """Return a stable RBF GP posterior variance at query feature vectors.

    The function never observes target errors, which makes it safe to use for a
    held-out geometry prediction. In this audit the query and support arrays
    are rows from the same appearance-aligned GAP feature grid. A tiny positive
    ``noise`` also handles duplicate or nearly collinear feature rows.
    """

    if gap_manifest is None:
        raise ValueError("gap_manifest is required for GAP-kernel posterior variance")
    query, _ = _validate_gap_grid(query_points, gap_manifest)
    support = _points(support_points, name="support_points")
    _same_dimension(query, support)
    noise = float(noise)
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and non-negative")
    k_ss = rbf_kernel(support, support, bandwidth=bandwidth)
    k_qs = rbf_kernel(query, support, bandwidth=bandwidth)
    system = k_ss + noise * np.eye(support.shape[0], dtype=np.float64)
    # Symmetrising avoids a tiny numerical asymmetry changing a near-zero
    # variance into a negative value on different BLAS implementations.
    system = 0.5 * (system + system.T)
    try:
        solved = np.linalg.solve(system, k_qs.T)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(system, rcond=1e-12) @ k_qs.T
    variance = 1.0 - np.einsum("ij,ji->i", k_qs, solved)
    return np.maximum(variance, 0.0)


def kernel_posterior_std(*args: Any, **kwargs: Any) -> np.ndarray:
    """Square root of :func:`kernel_posterior_variance`."""

    return np.sqrt(kernel_posterior_variance(*args, **kwargs))


def gap_kernel_posterior_variance(
    query_gap_features: Any,
    support_gap_features: Any,
    *,
    gap_manifest: Mapping[str, Any],
    bandwidth: float = 1.0,
    noise: float = 1e-6,
) -> np.ndarray:
    """Compute posterior variance explicitly in a manifest-backed GAP space.

    This wrapper is the public kernel entry point for the closeout audit. The
    manifest requirement makes it difficult to accidentally pass coordinate
    arrays or a non-appearance-aligned feature dump.
    """

    query, _ = _validate_gap_grid(query_gap_features, gap_manifest)
    support = _points(support_gap_features, name="support_gap_features")
    if query.shape[1] != support.shape[1]:
        raise ValueError("query and support GAP feature dimensions differ")
    return kernel_posterior_variance(
        query,
        support,
        gap_manifest=gap_manifest,
        bandwidth=bandwidth,
        noise=noise,
    )


def empirical_gap_kernel_distance(
    query_gap_features: Any,
    support_gap_features: Any,
    *,
    gap_manifest: Mapping[str, Any],
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Empirical RKHS distance for manifest-backed GAP feature rows."""

    query, _ = _validate_gap_grid(query_gap_features, gap_manifest)
    support = _points(support_gap_features, name="support_gap_features")
    if query.shape[1] != support.shape[1]:
        raise ValueError("query and support GAP feature dimensions differ")
    return empirical_kernel_distance(
        query,
        support,
        gap_manifest=gap_manifest,
        bandwidth=bandwidth,
    )


def _manifest_grid_coords(manifest: Mapping[str, Any], n_rows: int) -> np.ndarray | None:
    """Extract and validate optional grid coordinates from an appearance manifest."""

    for key in ("grid_coords", "coordinates", "coords"):
        if key in manifest:
            coords = _points(manifest[key], name=f"gap_manifest[{key}]")
            if coords.shape[0] != n_rows:
                raise ValueError(
                    f"GAP manifest coordinate count {coords.shape[0]} does not match "
                    f"feature grid rows {n_rows}"
                )
            return coords
    return None


def _validate_gap_grid(
    gap_feature_grid: Any,
    gap_manifest: Any,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Require a materialised, appearance-aligned GAP grid and manifest."""

    if gap_feature_grid is None:
        raise ValueError("gap_feature_grid is required for the kernel audit")
    if not isinstance(gap_manifest, Mapping):
        raise ValueError("gap_manifest is required and must be a mapping")
    grid = _points(gap_feature_grid, name="gap_feature_grid")
    if "feature_shape" in gap_manifest:
        shape = tuple(int(value) for value in gap_manifest["feature_shape"])
        if shape and shape[-1] != grid.shape[1]:
            raise ValueError(
                f"GAP manifest feature_shape {shape} disagrees with grid dimension {grid.shape[1]}"
            )
    if "n_grid" in gap_manifest and int(gap_manifest["n_grid"]) != grid.shape[0]:
        raise ValueError("GAP manifest n_grid disagrees with feature grid row count")
    _manifest_grid_coords(gap_manifest, grid.shape[0])
    if "appearance_aligned" in gap_manifest and not bool(gap_manifest["appearance_aligned"]):
        raise ValueError("gap_manifest explicitly marks the feature grid as unaligned")
    return grid, gap_manifest


def _support_gap_rows(
    support_name: str,
    support_points: Any,
    *,
    gap_grid: np.ndarray,
    gap_manifest: Mapping[str, Any],
    support_indices: Mapping[str, Sequence[int]] | None,
) -> np.ndarray:
    if support_indices is not None and support_name in support_indices:
        indices = np.asarray(support_indices[support_name], dtype=np.int64).reshape(-1)
    else:
        grid_coords = _manifest_grid_coords(gap_manifest, gap_grid.shape[0])
        if grid_coords is None:
            raise ValueError(
                f"support_indices[{support_name!r}] is required because gap_manifest "
                "does not contain grid coordinates"
            )
        points = _points(support_points, name=support_name)
        _same_dimension(points, grid_coords)
        indices_list: list[int] = []
        for point in points:
            matches = np.flatnonzero(
                np.all(np.isclose(grid_coords, point, rtol=0.0, atol=1e-8), axis=1)
            )
            if matches.size == 0:
                raise ValueError(f"support point {point.tolist()} is absent from GAP manifest grid")
            indices_list.append(int(matches[0]))
        indices = np.asarray(indices_list, dtype=np.int64)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= gap_grid.shape[0]):
        raise ValueError(f"invalid GAP support indices for {support_name}")
    return gap_grid[indices]


def geometry_features(
    support_points: Any,
    evaluation_points: Any,
    *,
    gap_feature_grid: Any,
    gap_manifest: Any,
    support_name: str = "support",
    support_indices: Mapping[str, Sequence[int]] | None = None,
    bandwidth: float = 1.0,
) -> dict[str, float]:
    """Build the two frozen scalar features used by the low-DoF predictor.

    ``gap_feature_grid`` and ``gap_manifest`` are mandatory. The Euclidean
    feature is computed from coordinate support/evaluation arrays; the kernel
    posterior variance is computed from the aligned GAP feature rows indexed by
    the support geometry. Consequently no coordinate RBF can accidentally be
    mistaken for the empirical position kernel.
    """

    gap_grid, manifest = _validate_gap_grid(gap_feature_grid, gap_manifest)
    support = _points(support_points, name="support_points")
    evaluation = _points(evaluation_points, name="evaluation_points")
    _same_dimension(support, evaluation)
    support_gap = _support_gap_rows(
        support_name,
        support,
        gap_grid=gap_grid,
        gap_manifest=manifest,
        support_indices=support_indices,
    )
    posterior = gap_kernel_posterior_variance(
        gap_grid,
        support_gap,
        gap_manifest=manifest,
        bandwidth=bandwidth,
    )
    return {
        "euclidean_covering_radius": euclidean_covering_radius(support, evaluation),
        "gap_kernel_posterior_variance": float(np.max(posterior)),
    }


def validate_support_partition(
    supports: Mapping[str, Any],
    *,
    fit_names: Sequence[str] = FIT_SUPPORT_NAMES,
    held_out_names: Sequence[str] = HELD_OUT_SUPPORT_NAMES,
    geometry_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate and return a named fit/held-out partition.

    Coordinate rows may overlap across support geometries: all geometries are
    drawn from the same dense coordinate grid, and overlap is not target
    leakage. Leakage is defined by geometry identity/provenance and by using
    held-out target errors during fitting. If an explicit ``geometry_provenance``
    mapping is supplied, the same provenance token may not occur in both splits.
    """

    fit_names = tuple(fit_names)
    held_out_names = tuple(held_out_names)
    overlap = set(fit_names).intersection(held_out_names)
    if overlap:
        raise ValueError(f"fit and held-out names overlap: {sorted(overlap)}")
    missing = [name for name in (*fit_names, *held_out_names) if name not in supports]
    if missing:
        raise KeyError(f"missing support geometries: {missing}")
    fit = {name: _points(supports[name], name=name) for name in fit_names}
    held = {name: _points(supports[name], name=name) for name in held_out_names}
    dimensions = {points.shape[1] for points in (*fit.values(), *held.values())}
    if len(dimensions) != 1:
        raise ValueError(f"all support geometries must share one dimension: {dimensions}")
    if geometry_provenance is not None:
        missing_provenance = [
            name for name in (*fit_names, *held_out_names) if name not in geometry_provenance
        ]
        if missing_provenance:
            raise KeyError(f"missing geometry provenance: {missing_provenance}")
        fit_tokens = {str(geometry_provenance[name]) for name in fit_names}
        held_tokens = {str(geometry_provenance[name]) for name in held_out_names}
        duplicated_tokens = fit_tokens.intersection(held_tokens)
        if duplicated_tokens:
            raise ValueError(
                "fit and held-out geometry provenance overlap: "
                + ", ".join(sorted(duplicated_tokens))
            )
    return fit, held


def build_geometry_feature_table(
    supports: Mapping[str, Any],
    evaluation_points: Any,
    *,
    gap_feature_grid: Any,
    gap_manifest: Any,
    support_indices: Mapping[str, Sequence[int]] | None = None,
    bandwidth: float = 1.0,
    names: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return names, the two feature names, and a deterministic feature matrix.

    The GAP feature grid and manifest are intentionally required even when the
    caller only wants to inspect the Euclidean column. This prevents a
    coordinate-only implementation from being mistaken for the preregistered
    appearance-kernel audit.
    """

    if names is None:
        names = tuple(supports.keys())
    names = tuple(names)
    if not names:
        raise ValueError("names must contain at least one support")
    features = [
        geometry_features(
            supports[name],
            evaluation_points,
            gap_feature_grid=gap_feature_grid,
            gap_manifest=gap_manifest,
            support_name=name,
            support_indices=support_indices,
            bandwidth=bandwidth,
        )
        for name in names
    ]
    feature_names = tuple(features[0].keys())
    matrix = np.asarray(
        [[row[name] for name in feature_names] for row in features],
        dtype=np.float64,
    )
    return names, feature_names, matrix


def _feature_matrix(features: Any) -> tuple[np.ndarray, tuple[str, ...] | None]:
    if isinstance(features, Mapping):
        names = tuple(features.keys())
        if not names:
            raise ValueError("features mapping must not be empty")
        first = features[names[0]]
        if isinstance(first, Mapping):
            feature_names = tuple(first.keys())
            matrix = np.asarray(
                [[features[name][key] for key in feature_names] for name in names],
                dtype=np.float64,
            )
        else:
            feature_names = None
            matrix = np.asarray([features[name] for name in names], dtype=np.float64)
    else:
        names = None
        feature_names = None
        matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"features must have shape (n, p), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features contain non-finite values")
    return matrix, feature_names


def _error_vector(errors: Any, n: int, *, names: Sequence[str] | None = None) -> np.ndarray:
    if isinstance(errors, Mapping):
        if names is None:
            names = tuple(errors.keys())
        values = [errors[name] for name in names]
    else:
        values = errors
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape[0] != n:
        raise ValueError(f"error count {result.shape[0]} does not match feature rows {n}")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("errors must be finite and strictly positive")
    return result


@dataclass(frozen=True)
class FixedLogErrorPredictor:
    """A predictor fit only on the declared fit support geometries."""

    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: tuple[str, ...] | None = None
    ridge: float = 1e-6

    def predict_log(self, features: Any) -> np.ndarray:
        matrix, feature_names = _feature_matrix(features)
        if self.feature_names is not None and feature_names is not None:
            if tuple(feature_names) != tuple(self.feature_names):
                raise ValueError(
                    f"feature names {tuple(feature_names)} do not match frozen "
                    f"predictor names {tuple(self.feature_names)}"
                )
        if matrix.shape[1] != self.coefficients.shape[0]:
            raise ValueError(
                f"feature dimension {matrix.shape[1]} does not match predictor "
                f"dimension {self.coefficients.shape[0]}"
            )
        standardized = (matrix - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients

    def predict(self, features: Any) -> np.ndarray:
        return np.exp(self.predict_log(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": float(self.intercept),
            "coefficients": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "feature_names": list(self.feature_names) if self.feature_names else None,
            "ridge": float(self.ridge),
        }


def _fit_standardized_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    design = np.column_stack([np.ones(matrix.shape[0]), standardized])
    penalty = float(ridge) * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs, rcond=1e-12) @ rhs
    return coefficients, mean, scale


def _choose_ridge_loog(
    matrix: np.ndarray,
    target: np.ndarray,
    ridge_grid: Sequence[float],
) -> float:
    """Choose ridge using leave-one-fit-geometry-out CV only."""

    candidates = tuple(sorted({float(value) for value in ridge_grid}))
    if not candidates or any((not np.isfinite(value) or value < 0) for value in candidates):
        raise ValueError("ridge_grid must contain finite non-negative values")
    scores: list[tuple[float, float]] = []
    for ridge in candidates:
        losses: list[float] = []
        for held_out in range(matrix.shape[0]):
            train = np.delete(np.arange(matrix.shape[0]), held_out)
            coefficients, mean, scale = _fit_standardized_ridge(
                matrix[train], target[train], ridge
            )
            standardized = (matrix[held_out] - mean) / scale
            prediction = float(coefficients[0] + standardized @ coefficients[1:])
            losses.append((prediction - target[held_out]) ** 2)
        scores.append((float(np.mean(losses)), ridge))
    # Sorting by loss and then ridge makes ties deterministic and prefers the
    # simpler (smaller penalty) candidate.
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def fit_fixed_log_error_predictor(
    fit_features: Any,
    fit_errors: Any,
    *,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    feature_names: Sequence[str] | None = None,
    ridge: float | None = None,
) -> FixedLogErrorPredictor:
    """Fit a low-degree log-error predictor on fit geometries only.

    The frozen protocol permits at most the Euclidean covering radius and one
    GAP-kernel posterior-variance feature. The intercept plus feature count must
    not exceed the number of fit geometries. Ridge is selected by leave-one-fit
    geometry-out CV; a caller cannot inject a post-hoc ridge value. The
    ``ridge`` parameter is retained only to provide a clear error for stale
    callers and is never used to tune the model.
    """

    if ridge is not None:
        raise ValueError("ridge must be selected by training-set LOOG; pass ridge_grid instead")
    matrix, inferred_names = _feature_matrix(fit_features)
    errors = _error_vector(
        fit_errors,
        matrix.shape[0],
        names=tuple(fit_features.keys()) if isinstance(fit_features, Mapping) else None,
    )
    if feature_names is not None:
        names = tuple(feature_names)
        if len(names) != matrix.shape[1]:
            raise ValueError("feature_names length does not match feature dimension")
    else:
        names = inferred_names
    if names is None:
        raise ValueError(
            "feature_names are required for array inputs so Euclidean and GAP-kernel "
            "columns cannot be swapped or relabeled"
        )
    if matrix.shape[1] > len(PREDICTOR_FEATURE_NAMES):
        raise ValueError(
            "predictor may use at most Euclidean radius plus GAP-kernel posterior variance"
        )
    if names is not None and any(name not in PREDICTOR_FEATURE_NAMES for name in names):
        raise ValueError(f"unsupported predictor features: {names}")
    if matrix.shape[1] + 1 > matrix.shape[0]:
        raise ValueError(
            f"intercept plus {matrix.shape[1]} feature(s) exceeds {matrix.shape[0]} fit geometries"
        )
    target = np.log(errors)
    selected_ridge = _choose_ridge_loog(matrix, target, ridge_grid)
    coefficients, mean, scale = _fit_standardized_ridge(matrix, target, selected_ridge)
    return FixedLogErrorPredictor(
        intercept=float(coefficients[0]),
        coefficients=np.asarray(coefficients[1:], dtype=np.float64),
        feature_mean=np.asarray(mean, dtype=np.float64),
        feature_scale=np.asarray(scale, dtype=np.float64),
        feature_names=names,
        ridge=selected_ridge,
    )


fit_log_error_predictor = fit_fixed_log_error_predictor


@dataclass(frozen=True)
class KernelGateResult:
    """Machine-readable result of the pre-registered kernel gate."""

    passed: bool
    ranking_correct: bool
    within_50_percent_count: int
    required_within_50_percent: int
    kernel_log_mae: float
    euclidean_baseline_log_mae: float
    improvement_fraction: float
    required_improvement_fraction: float
    held_out_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_kernel_gate(
    predicted_log_errors: Any,
    observed_errors: Any,
    euclidean_baseline_errors: Any,
    *,
    min_within_50_percent: int = 2,
    min_improvement_fraction: float = 0.20,
) -> KernelGateResult:
    """Apply the fixed three-part held-out kernel gate.

    The gate requires all held-out rankings to agree, at least two raw
    predictions to lie within ``[0.5, 1.5]`` times the observed error, and a
    20% reduction in log-MAE relative to an Euclidean-only baseline. The
    predictor outputs are log errors; ``observed_errors`` and the explicitly
    named ``euclidean_baseline_errors`` are positive raw errors.
    """

    predicted = np.asarray(predicted_log_errors, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_errors, dtype=np.float64).reshape(-1)
    baseline = np.asarray(euclidean_baseline_errors, dtype=np.float64).reshape(-1)
    if predicted.size == 0 or predicted.shape != observed.shape or predicted.shape != baseline.shape:
        raise ValueError("predicted, observed, and baseline arrays must have equal non-zero length")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("predicted log errors must be finite")
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0):
        raise ValueError("observed errors must be finite and positive")
    if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
        raise ValueError("baseline errors must be finite and positive")
    predicted_raw = np.exp(predicted)
    ranking_correct = bool(np.array_equal(np.argsort(predicted_raw), np.argsort(observed)))
    ratio = predicted_raw / observed
    within = int(np.count_nonzero((ratio >= 0.5) & (ratio <= 1.5)))
    kernel_mae = float(np.mean(np.abs(predicted - np.log(observed))))
    baseline_mae = float(np.mean(np.abs(np.log(baseline) - np.log(observed))))
    if baseline_mae <= 1e-15:
        improvement = 0.0 if kernel_mae > 1e-15 else 1.0
    else:
        improvement = float((baseline_mae - kernel_mae) / baseline_mae)
    required_improvement_fraction = float(min_improvement_fraction)
    if not np.isfinite(required_improvement_fraction):
        raise ValueError("min_improvement_fraction must be finite")
    passed = bool(
        ranking_correct
        and within >= int(min_within_50_percent)
        and improvement >= required_improvement_fraction
    )
    reasons: list[str] = []
    if not ranking_correct:
        reasons.append("held-out ranking mismatch")
    if within < int(min_within_50_percent):
        reasons.append("too few predictions within 50 percent")
    if improvement < required_improvement_fraction:
        reasons.append("kernel log-MAE improvement below gate")
    return KernelGateResult(
        passed=passed,
        ranking_correct=ranking_correct,
        within_50_percent_count=within,
        required_within_50_percent=int(min_within_50_percent),
        kernel_log_mae=kernel_mae,
        euclidean_baseline_log_mae=baseline_mae,
        improvement_fraction=improvement,
        required_improvement_fraction=required_improvement_fraction,
        held_out_count=int(predicted.size),
        reason="passed" if passed else "; ".join(reasons),
    )


kernel_gate = evaluate_kernel_gate


__all__ = [
    "DEFAULT_RIDGE_GRID",
    "FIT_SUPPORT_NAMES",
    "HELD_OUT_SUPPORT_NAMES",
    "PREDICTOR_FEATURE_NAMES",
    "FixedLogErrorPredictor",
    "KernelGateResult",
    "build_geometry_feature_table",
    "empirical_kernel_covering_radius",
    "empirical_kernel_distance",
    "evaluate_kernel_gate",
    "euclidean_covering_radius",
    "fit_fixed_log_error_predictor",
    "fit_log_error_predictor",
    "gap_kernel_posterior_variance",
    "empirical_gap_kernel_distance",
    "geometry_features",
    "kernel_gate",
    "kernel_posterior_std",
    "kernel_posterior_variance",
    "nearest_euclidean_distances",
    "rbf_kernel",
    "validate_support_partition",
]
