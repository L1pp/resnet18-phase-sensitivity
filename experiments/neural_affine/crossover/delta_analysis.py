"""NumPy-only decomposition of frozen NTK/head/full coordinate fields.

This module is intentionally a saved-field analysis boundary.  It does not
import torch, scipy, the renderer, or any training code.  Inputs are three
strictly aligned 41x41 two-coordinate fields per seed.  The public entry point
``analyze_three_seed_fields`` returns a JSON-compatible report plus an array
bundle and can atomically write both artifacts.

The historical S1 fields used by this analysis can come from different
backends (for example AMD/ROCm NTK or head fields and an A10/CUDA full field).
That provenance is retained in the report as supplementary evidence; this
module never changes a formal S1 verdict.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GRID_SHAPE = (41, 41)
GRID_POINTS = GRID_SHAPE[0] * GRID_SHAPE[1]
DEFAULT_COORD_SCALE = 223.0
DEFAULT_SEEDS = (20260816, 20260817, 20260818)
FIELD_ROLES = ("ntk", "head", "full")
DELTA_ROLES = ("head_minus_ntk", "full_minus_head")

_ROLE_KEYS: dict[str, tuple[str, ...]] = {
    "ntk": (
        "ntk_pred_px",
        "ntk_query_px",
        "ntk_query_norm",
        "ntk_pred_norm",
        "pred_px",
        "pred",
        "prediction",
        "field_pred",
    ),
    "head": (
        "head_only_pred_px",
        "head_pred_px",
        "head_only_pred_norm",
        "head_pred_norm",
        "query_pred_px",
        "query_pred_norm",
        "pred_px",
        "pred",
        "prediction",
        "field_pred",
    ),
    "full": (
        "trained_pred_px",
        "full_pred_px",
        "raw_full_pred_px",
        "trained_pred_norm",
        "full_pred_norm",
        "pred_px",
        "pred",
        "prediction",
        "field_pred",
    ),
}

_TRUE_KEYS = (
    "true_px",
    "true",
    "target_px",
    "target",
    "field_true",
    "query_true_px",
    "query_true_px_posthoc",
)


@dataclass(frozen=True)
class FieldSource:
    """Description of a saved field source.

    ``key`` is optional for NPZ files with conventional role-specific names.
    ``normalized`` should be set when a source is ambiguous; if left ``None``,
    the loader infers normalization only from an explicit ``*_norm`` key or
    from a field whose values are all within the model-output range ``[-2, 2]``.
    """

    path: Path
    key: str | None = None
    true_key: str | None = None
    normalized: bool | None = None
    coord_scale: float = DEFAULT_COORD_SCALE

    @classmethod
    def from_value(cls, value: str | os.PathLike[str] | "FieldSource") -> "FieldSource":
        if isinstance(value, cls):
            return value
        return cls(Path(value))


@dataclass(frozen=True)
class LoadedField:
    """A canonical saved field and its source metadata."""

    field_px: np.ndarray
    true_px: np.ndarray | None
    source: FieldSource
    field_key: str
    true_key: str | None
    normalized: bool
    query_indices_checked: bool
    source_sha256: str


@dataclass(frozen=True)
class SeedFieldBundle:
    """The three aligned fields and canonical target grid for one seed."""

    seed: int
    ntk: LoadedField
    head: LoadedField
    full: LoadedField
    true_px: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays to deterministic JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Atomically write a UTF-8 JSON file in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_save_npz(path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write a compressed NPZ bundle.

    ``np.savez_compressed`` is directed at a temporary file with an explicit
    ``.npz`` suffix so NumPy does not silently append another extension.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **{str(k): np.asarray(v) for k, v in arrays.items()})
        # Windows rejects fsync on a read-only descriptor for this temporary
        # file; reopening read/write keeps the durability check explicit.
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _array_mapping(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return {"array": np.asarray(np.load(path, allow_pickle=False))}, {}
    if path.suffix.lower() != ".npz":
        raise ValueError(f"saved field must be .npz or .npy: {path}")
    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: np.asarray(blob[key]) for key in blob.files}
    return arrays, {}


def _select_key(arrays: Mapping[str, np.ndarray], role: str, explicit: str | None) -> str:
    if role not in FIELD_ROLES:
        raise ValueError(f"unknown field role: {role}")
    if explicit is not None:
        if explicit not in arrays:
            raise KeyError(f"field key {explicit!r} not found; available={sorted(arrays)}")
        return explicit
    for key in _ROLE_KEYS[role]:
        if key in arrays:
            return key
    raise KeyError(f"no {role} field key in source; available={sorted(arrays)}")


def _select_true_key(arrays: Mapping[str, np.ndarray], explicit: str | None) -> str | None:
    if explicit is not None:
        if explicit not in arrays:
            raise KeyError(f"true key {explicit!r} not found; available={sorted(arrays)}")
        return explicit
    for key in _TRUE_KEYS:
        if key in arrays:
            return key
    return None


def _canonical_field(array: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{label} batch dimension must be exactly 1, got {value.shape}")
        value = value[0]
    if value.shape == (*GRID_SHAPE, 2):
        canonical = value
    elif value.shape == (GRID_POINTS, 2):
        canonical = value.reshape(*GRID_SHAPE, 2)
    elif value.shape == (1, GRID_POINTS, 2):
        canonical = value[0].reshape(*GRID_SHAPE, 2)
    elif value.shape == (GRID_POINTS * 2,):
        canonical = value.reshape(*GRID_SHAPE, 2)
    else:
        raise ValueError(
            f"{label} must be exactly 41x41x2 or a row-major 1681x2 field, got {value.shape}"
        )
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(canonical, dtype=np.float64)


def _check_query_indices(arrays: Mapping[str, np.ndarray], *, path: Path) -> bool:
    if "query_indices" not in arrays:
        return False
    indices = np.asarray(arrays["query_indices"])
    if indices.shape != (GRID_POINTS,):
        raise ValueError(f"query_indices must have shape ({GRID_POINTS},), got {indices.shape}: {path}")
    expected = np.arange(GRID_POINTS, dtype=indices.dtype)
    if not np.array_equal(indices, expected):
        raise ValueError(f"query_indices are not strict row-major 0..1680 order: {path}")
    return True


def _key_implies_normalized(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_norm") or "_norm_" in lowered or "normalized" in lowered


def load_field(source: FieldSource | str | os.PathLike[str], role: str) -> LoadedField:
    """Load one field, convert to pixels, and enforce the canonical shape."""

    description = FieldSource.from_value(source)
    path = description.path
    arrays, _ = _array_mapping(path)
    query_checked = _check_query_indices(arrays, path=path)
    field_key = _select_key(arrays, role, description.key)
    true_key = _select_true_key(arrays, description.true_key)
    field = _canonical_field(arrays[field_key], label=f"{role} field {path}")
    true = (
        _canonical_field(arrays[true_key], label=f"true field {path}")
        if true_key is not None
        else None
    )
    normalized = description.normalized
    if normalized is None:
        normalized = _key_implies_normalized(field_key) or float(np.max(np.abs(field))) <= 2.0
    if normalized:
        field = field * float(description.coord_scale)
        if true is not None and (
            _key_implies_normalized(true_key or "") or float(np.max(np.abs(true))) <= 2.0
        ):
            true = true * float(description.coord_scale)
    if true is not None and not np.all(np.isfinite(true)):
        raise ValueError(f"true field contains non-finite values: {path}")
    return LoadedField(
        field_px=field,
        true_px=true,
        source=description,
        field_key=field_key,
        true_key=true_key,
        normalized=bool(normalized),
        query_indices_checked=query_checked,
        source_sha256=_sha256_file(path),
    )


def _canonical_true_override(value: str | os.PathLike[str] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        arrays, _ = _array_mapping(Path(value))
        key = _select_true_key(arrays, None)
        if key is None:
            key = "array" if "array" in arrays else None
        if key is None:
            raise KeyError(f"no true field in override source: {value}")
        return _canonical_field(arrays[key], label=f"true override {value}")
    return _canonical_field(np.asarray(value), label="true override")


def load_seed_bundle(
    seed: int,
    sources: Mapping[str, FieldSource | str | os.PathLike[str]],
    *,
    true_source: str | os.PathLike[str] | np.ndarray | None = None,
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> SeedFieldBundle:
    """Load and strictly align NTK/head/full fields for one seed."""

    missing = [role for role in FIELD_ROLES if role not in sources]
    if missing:
        raise ValueError(f"seed {seed} missing field roles: {missing}")
    loaded: dict[str, LoadedField] = {}
    for role in FIELD_ROLES:
        source = FieldSource.from_value(sources[role])
        if source.coord_scale == DEFAULT_COORD_SCALE and coord_scale != DEFAULT_COORD_SCALE:
            source = FieldSource(
                path=source.path,
                key=source.key,
                true_key=source.true_key,
                normalized=source.normalized,
                coord_scale=coord_scale,
            )
        loaded[role] = load_field(source, role)
    override_true = _canonical_true_override(true_source)
    supplied_true = [item.true_px for item in loaded.values() if item.true_px is not None]
    if override_true is not None:
        true = override_true
        for candidate in supplied_true:
            if not np.array_equal(candidate, true):
                raise ValueError(f"seed {seed} true field disagrees with explicit override")
    elif supplied_true:
        true = supplied_true[0]
        for candidate in supplied_true[1:]:
            if not np.array_equal(candidate, true):
                raise ValueError(f"seed {seed} true fields are not strictly aligned")
    else:
        raise ValueError(f"seed {seed} has no true field; provide true_source")
    for role, item in loaded.items():
        if item.field_px.shape != true.shape:
            raise ValueError(f"seed {seed} {role} field shape does not match true field")
    return SeedFieldBundle(
        seed=int(seed), ntk=loaded["ntk"], head=loaded["head"], full=loaded["full"], true_px=true
    )


def load_three_seed_fields(
    sources_by_seed: Mapping[int | str, Mapping[str, FieldSource | str | os.PathLike[str]]],
    *,
    true_sources: Mapping[int | str, str | os.PathLike[str] | np.ndarray] | None = None,
    expected_seeds: Sequence[int] | None = DEFAULT_SEEDS,
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> list[SeedFieldBundle]:
    """Load exactly three seed bundles, in deterministic seed order."""

    normalized_sources = {int(seed): sources for seed, sources in sources_by_seed.items()}
    if expected_seeds is None:
        expected = tuple(sorted(normalized_sources))
    else:
        expected = tuple(int(seed) for seed in expected_seeds)
        if set(normalized_sources) != set(expected):
            raise ValueError(
                f"expected seed set {sorted(set(expected))}, got {sorted(set(normalized_sources))}"
            )
    if len(expected) != 3 or len(set(expected)) != 3:
        raise ValueError(f"three distinct seeds are required, got {expected}")
    true_sources = true_sources or {}
    bundles = [
        load_seed_bundle(
            seed,
            normalized_sources[seed],
            true_source=true_sources.get(seed, true_sources.get(str(seed))),
            coord_scale=coord_scale,
        )
        for seed in sorted(expected)
    ]
    reference = bundles[0].true_px
    for bundle in bundles[1:]:
        if not np.array_equal(bundle.true_px, reference):
            raise ValueError(f"seed {bundle.seed} true grid is not strictly aligned with seed {bundles[0].seed}")
    return bundles


def _grid_axes(true_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = _canonical_field(true_px, label="true grid")
    x = np.asarray(true[:, 0, 0], dtype=np.float64)
    y = np.asarray(true[0, :, 1], dtype=np.float64)
    expected_x = np.broadcast_to(x[:, None], GRID_SHAPE)
    expected_y = np.broadcast_to(y[None, :], GRID_SHAPE)
    if not np.array_equal(true[..., 0], expected_x) or not np.array_equal(true[..., 1], expected_y):
        raise ValueError("true grid is not a strict tx-major/ty-minor Cartesian grid")
    if np.any(np.diff(x) == 0) or np.any(np.diff(y) == 0):
        raise ValueError("true grid axes must contain distinct coordinates")
    return x, y


def _polynomial_design(true_px: np.ndarray, degree: int) -> tuple[np.ndarray, tuple[str, ...]]:
    x_axis, y_axis = _grid_axes(true_px)
    x = 2.0 * (x_axis - x_axis.min()) / max(float(x_axis.max() - x_axis.min()), 1e-30) - 1.0
    y = 2.0 * (y_axis - y_axis.min()) / max(float(y_axis.max() - y_axis.min()), 1e-30) - 1.0
    xx, yy = np.meshgrid(x, y, indexing="ij")
    columns = [np.ones(GRID_POINTS)]
    names = ["1"]
    if degree >= 1:
        columns.extend((xx.reshape(-1), yy.reshape(-1)))
        names.extend(("x", "y"))
    if degree >= 2:
        columns.extend((xx.reshape(-1) ** 2, (xx * yy).reshape(-1), yy.reshape(-1) ** 2))
        names.extend(("x2", "xy", "y2"))
    if degree < 0 or degree > 2:
        raise ValueError(f"only polynomial degrees 0, 1, and 2 are supported, got {degree}")
    return np.column_stack(columns), tuple(names)


def _fit_polynomial(delta_px: np.ndarray, true_px: np.ndarray, degree: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    design, names = _polynomial_design(true_px, degree)
    target = np.asarray(delta_px, dtype=np.float64).reshape(GRID_POINTS, 2)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    fitted = (design @ coefficients).reshape(*GRID_SHAPE, 2)
    residual = np.asarray(delta_px, dtype=np.float64) - fitted
    raw_sse = float(np.sum(target * target))
    residual_sse = float(np.sum(residual * residual))
    centered_sse = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    explained = 1.0 if raw_sse <= 1e-30 and residual_sse <= 1e-30 else 1.0 - residual_sse / max(raw_sse, 1e-30)
    centered_r2 = 1.0 if centered_sse <= 1e-30 and residual_sse <= 1e-30 else 1.0 - residual_sse / max(centered_sse, 1e-30)
    summary: dict[str, Any] = {
        "degree": int(degree),
        "terms": list(names),
        "n_parameters_per_output": int(design.shape[1]),
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "coefficients": coefficients.tolist(),
        "raw_sse": raw_sse,
        "residual_sse": residual_sse,
        # Keep the shorter historical field-tools spelling as an alias while
        # making the uncentered denominator explicit for new consumers.
        "explained_fraction": float(explained),
        "explained_fraction_raw": float(explained),
        "r2_centered": float(centered_r2),
        "residual_mae_px": float(np.linalg.norm(residual.reshape(-1, 2), axis=1).mean()),
        "residual_rmse_px": float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1)))),
    }
    return summary, coefficients, residual


def _anova_summary(field_px: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    field = np.asarray(field_px, dtype=np.float64)
    interaction_field = np.empty_like(field)
    component_rows: list[dict[str, Any]] = []
    total_raw = 0.0
    total_centered = 0.0
    total_interaction = 0.0

    def fraction(numerator: float, denominator: float) -> float:
        # A constant component has zero centered energy.  Numerical roundoff
        # in the ANOVA reconstruction must not turn that exact no-interaction
        # case into a spurious fraction of one.
        if denominator <= 1e-24:
            return 0.0 if abs(numerator) <= 1e-24 else float("nan")
        return float(numerator / denominator)

    for component in range(2):
        values = field[..., component]
        grand = float(values.mean())
        x_main = values.mean(axis=1, keepdims=True) - grand
        y_main = values.mean(axis=0, keepdims=True) - grand
        interaction = values - grand - x_main - y_main
        interaction_field[..., component] = interaction
        raw_energy = float(np.sum(values * values))
        centered_energy = float(np.sum((values - grand) ** 2))
        interaction_energy = float(np.sum(interaction * interaction))
        x_energy = float(np.sum(x_main * x_main))
        y_energy = float(np.sum(y_main * y_main))
        component_rows.append(
            {
                "component": int(component),
                "raw_energy": raw_energy,
                "centered_energy": centered_energy,
                "x_main_energy": x_energy,
                "y_main_energy": y_energy,
                "interaction_energy": interaction_energy,
                "x_main_fraction": fraction(x_energy, centered_energy),
                "y_main_fraction": fraction(y_energy, centered_energy),
                "interaction_fraction": fraction(interaction_energy, centered_energy),
                "interaction_fraction_raw": fraction(interaction_energy, raw_energy),
                "anova_closure_max_abs": float(np.max(np.abs(values - (grand + x_main + y_main + interaction)))),
            }
        )
        total_raw += raw_energy
        total_centered += centered_energy
        total_interaction += interaction_energy
    return (
        {
            "components": component_rows,
            "raw_energy": total_raw,
            "centered_energy": total_centered,
            "interaction_energy": total_interaction,
            "interaction_fraction": fraction(total_interaction, total_centered),
            "interaction_fraction_raw": fraction(total_interaction, total_raw),
            "anova_closure_max_abs": float(max(row["anova_closure_max_abs"] for row in component_rows)),
        },
        interaction_field,
    )


def _summary_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "mean_abs": float(np.mean(np.abs(array))),
        "rms": float(np.sqrt(np.mean(array * array))),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _jacobian(field_px: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    field = np.asarray(field_px, dtype=np.float64)
    jacobian = np.empty((*GRID_SHAPE, 2, 2), dtype=np.float64)
    edge_order = 2 if min(GRID_SHAPE) >= 3 else 1
    for output_component in range(2):
        derivative_x, derivative_y = np.gradient(
            field[..., output_component], x_axis, y_axis, edge_order=edge_order
        )
        jacobian[..., output_component, 0] = derivative_x
        jacobian[..., output_component, 1] = derivative_y
    return jacobian


def _jacobian_summary(jacobian: np.ndarray) -> dict[str, Any]:
    value = np.asarray(jacobian, dtype=np.float64)
    determinant = value[..., 0, 0] * value[..., 1, 1] - value[..., 0, 1] * value[..., 1, 0]
    negative = determinant < 0.0
    nonpositive = determinant <= 0.0
    return {
        "derivative_output0_dx": _summary_stats(value[..., 0, 0]),
        "derivative_output0_dy": _summary_stats(value[..., 0, 1]),
        "derivative_output1_dx": _summary_stats(value[..., 1, 0]),
        "derivative_output1_dy": _summary_stats(value[..., 1, 1]),
        "determinant": _summary_stats(determinant),
        "negative_count": int(np.count_nonzero(negative)),
        "negative_fraction": float(np.mean(negative)),
        "nonpositive_count": int(np.count_nonzero(nonpositive)),
        "nonpositive_fraction": float(np.mean(nonpositive)),
        "fold_count": int(np.count_nonzero(nonpositive)),
        "fold_fraction": float(np.mean(nonpositive)),
    }


def _fft_summary(residual_px: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(residual_px, dtype=np.float64)
    transformed = np.fft.fftshift(np.fft.fft2(residual, axes=(0, 1)), axes=(0, 1))
    power = np.sum(np.abs(transformed) ** 2, axis=-1)
    h, w = power.shape
    frequency_x = np.fft.fftshift(np.fft.fftfreq(h) * h)
    frequency_y = np.fft.fftshift(np.fft.fftfreq(w) * w)
    fx, fy = np.meshgrid(frequency_x, frequency_y, indexing="ij")
    radius = np.sqrt(fx * fx + fy * fy)
    total = float(np.sum(power))
    low = float(np.sum(power[radius <= 2.0]) / max(total, 1e-30))
    nonzero = radius > 0.0
    masked = np.where(nonzero, power, -np.inf)
    main_index = np.unravel_index(int(np.argmax(masked)), masked.shape)
    main_power = float(power[main_index])
    main_fx = float(frequency_x[main_index[0]])
    main_fy = float(frequency_y[main_index[1]])
    main_angle = float(math.atan2(main_fy, main_fx))
    top_flat = np.argsort(np.where(nonzero, power, -np.inf).reshape(-1))[::-1][:8]
    top_peaks: list[dict[str, Any]] = []
    for flat_index in top_flat:
        index = np.unravel_index(int(flat_index), power.shape)
        top_peaks.append(
            {
                "frequency_index_xy": [float(frequency_x[index[0]]), float(frequency_y[index[1]])],
                "power": float(power[index]),
                "energy_fraction": float(power[index] / max(total, 1e-30)),
                "direction_radians": float(math.atan2(float(frequency_y[index[1]]), float(frequency_x[index[0]]))),
            }
        )
    angles = np.arctan2(fy, fx)
    directional: list[dict[str, Any]] = []
    for sector in range(8):
        lower = -math.pi + sector * (2.0 * math.pi / 8.0)
        upper = lower + 2.0 * math.pi / 8.0
        mask = nonzero & (angles >= lower) & (angles < upper)
        directional.append(
            {
                "sector": int(sector),
                "lower_radians": float(lower),
                "upper_radians": float(upper),
                "energy_fraction": float(np.sum(power[mask]) / max(total, 1e-30)),
            }
        )
    return {
        "total_energy": total,
        "low_frequency_radius_le_2_fraction": low,
        "main_nonzero_frequency_index_xy": [main_fx, main_fy],
        "main_nonzero_power": main_power,
        "main_nonzero_energy_fraction": float(main_power / max(total, 1e-30)),
        "main_nonzero_direction_radians": main_angle,
        "main_nonzero_direction_degrees": float(math.degrees(main_angle)),
        "top_nonzero_peaks": top_peaks,
        "directional_octants": directional,
    }


def _pairwise_consistency(vectors_by_seed: Mapping[int, np.ndarray]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for left_seed, right_seed in itertools.combinations(sorted(vectors_by_seed), 2):
        left = np.asarray(vectors_by_seed[left_seed], dtype=np.float64).reshape(-1)
        right = np.asarray(vectors_by_seed[right_seed], dtype=np.float64).reshape(-1)
        if left.shape != right.shape:
            raise ValueError("coefficient vectors are not aligned across seeds")
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        cosine = float(np.dot(left, right) / max(left_norm * right_norm, 1e-30))
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std <= 1e-30 or right_std <= 1e-30:
            correlation: float | None = None
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        signs = np.sign(left) == np.sign(right)
        both_nonzero = (left != 0.0) & (right != 0.0)
        rows.append(
            {
                "seed_left": int(left_seed),
                "seed_right": int(right_seed),
                "cosine": cosine,
                "correlation": correlation,
                "sign_agreement": float(np.mean(signs)),
                "nonzero_sign_agreement": (
                    float(np.mean(signs[both_nonzero])) if np.any(both_nonzero) else None
                ),
            }
        )
    numeric = {
        metric: [row[metric] for row in rows if row[metric] is not None]
        for metric in ("cosine", "correlation", "sign_agreement", "nonzero_sign_agreement")
    }
    aggregate = {
        metric: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for metric, values in numeric.items()
        if values
    }
    return {"pairs": rows, "aggregate": aggregate}


def _frequency_consistency(freq_by_seed: Mapping[int, Sequence[float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for left_seed, right_seed in itertools.combinations(sorted(freq_by_seed), 2):
        left = np.asarray(freq_by_seed[left_seed], dtype=np.float64)
        right = np.asarray(freq_by_seed[right_seed], dtype=np.float64)
        rows.append(
            {
                "seed_left": int(left_seed),
                "seed_right": int(right_seed),
                "frequency_left": left.tolist(),
                "frequency_right": right.tolist(),
                "exact_match": bool(np.array_equal(left, right)),
                "euclidean_index_distance": float(np.linalg.norm(left - right)),
            }
        )
    return {"pairs": rows, "exact_match_fraction": float(np.mean([row["exact_match"] for row in rows]))}


def _source_record(item: LoadedField) -> dict[str, Any]:
    return {
        "path": str(item.source.path),
        "sha256": item.source_sha256,
        "field_key": item.field_key,
        "true_key": item.true_key,
        "normalized_input": bool(item.normalized),
        "coord_scale": float(item.source.coord_scale),
        "query_indices_checked": bool(item.query_indices_checked),
    }


def _analyze_one_field(
    field_px: np.ndarray,
    true_px: np.ndarray,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    include_warp: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    polynomial: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    residual_degree2 = None
    for degree in (1, 2):
        summary, coefficients, residual = _fit_polynomial(field_px, true_px, degree)
        polynomial[f"degree_{degree}"] = summary
        arrays[f"poly{degree}_coefficients"] = coefficients
        arrays[f"poly{degree}_residual"] = residual
        if degree == 2:
            residual_degree2 = residual
    if residual_degree2 is None:  # pragma: no cover - degree loop is fixed above
        raise AssertionError("degree-2 residual missing")
    anova, interaction = _anova_summary(field_px)
    jacobian = _jacobian(field_px, x_axis, y_axis)
    result: dict[str, Any] = {
        "polynomial": polynomial,
        "cross_axis_anova": anova,
        "endpoint_jacobian": _jacobian_summary(jacobian),
        "degree2_residual_fft": _fft_summary(residual_degree2),
    }
    arrays["anova_interaction"] = interaction
    arrays["jacobian"] = jacobian
    if include_warp:
        identity = np.zeros_like(jacobian)
        identity[..., 0, 0] = 1.0
        identity[..., 1, 1] = 1.0
        warp = identity + jacobian
        result["I_plus_grad_delta"] = _jacobian_summary(warp)
        arrays["I_plus_grad_delta"] = warp
    return result, arrays


def _analyze_fields(
    bundles: Sequence[SeedFieldBundle],
    *,
    evidence_boundary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(bundles) != 3:
        raise ValueError(f"exactly three seed bundles are required, got {len(bundles)}")
    seeds = [int(bundle.seed) for bundle in bundles]
    if len(set(seeds)) != 3:
        raise ValueError(f"seed IDs must be unique, got {seeds}")
    true = _canonical_field(bundles[0].true_px, label="true grid")
    x_axis, y_axis = _grid_axes(true)
    for bundle in bundles:
        if not np.array_equal(bundle.true_px, true):
            raise ValueError(f"seed {bundle.seed} true grid is not strictly aligned")
        for role in FIELD_ROLES:
            field = getattr(bundle, role).field_px
            if field.shape != (*GRID_SHAPE, 2) or not np.all(np.isfinite(field)):
                raise ValueError(f"seed {bundle.seed} {role} field is not canonical finite 41x41x2")

    boundary = {
        "classification": "supplementary_saved_field_reanalysis",
        "formal_s1_verdict_unchanged": True,
        "training_reexecuted": False,
        "mixed_backend_allowed": True,
        "mixed_backend_note": (
            "Inputs may combine historical NTK/head/full artifacts from different backends; "
            "the decomposition is descriptive function-space evidence and is not a clean-room "
            "formal S1 result."
        ),
    }
    if evidence_boundary:
        boundary.update(_jsonable(dict(evidence_boundary)))
    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "saved_field_delta_decomposition",
        "grid": {
            "shape": list(GRID_SHAPE),
            "order": "tx_major_ty_minor",
            "x_values_px": x_axis.tolist(),
            "y_values_px": y_axis.tolist(),
        },
        "seeds": seeds,
        "source_boundary": boundary,
        "per_seed": {},
        "cross_seed": {},
    }
    arrays: dict[str, np.ndarray] = {"true_px": true}
    coefficients: dict[str, dict[int, np.ndarray]] = {}
    main_frequencies: dict[str, dict[int, Sequence[float]]] = {}
    for bundle in bundles:
        seed = int(bundle.seed)
        ntk = bundle.ntk.field_px
        head = bundle.head.field_px
        full = bundle.full.field_px
        deltas = {
            "head_minus_ntk": head - ntk,
            "full_minus_head": full - head,
        }
        seed_report: dict[str, Any] = {
            "sources": {
                "ntk": _source_record(bundle.ntk),
                "head": _source_record(bundle.head),
                "full": _source_record(bundle.full),
            },
            "endpoints": {},
            "deltas": {},
        }
        for role, field in (("ntk", ntk), ("head", head), ("full", full)):
            endpoint_result, endpoint_arrays = _analyze_one_field(
                field, true, x_axis=x_axis, y_axis=y_axis, include_warp=False
            )
            seed_report["endpoints"][role] = endpoint_result
            arrays[f"seed{seed}_{role}_px"] = field
            arrays[f"seed{seed}_{role}_jacobian"] = endpoint_arrays["jacobian"]
        for delta_role, delta in deltas.items():
            delta_result, delta_arrays = _analyze_one_field(
                delta, true, x_axis=x_axis, y_axis=y_axis, include_warp=True
            )
            seed_report["deltas"][delta_role] = delta_result
            arrays[f"seed{seed}_{delta_role}_px"] = delta
            for key, value in delta_arrays.items():
                arrays[f"seed{seed}_{delta_role}_{key}"] = value
            for degree in (1, 2):
                coefficient = delta_result["polynomial"][f"degree_{degree}"]["coefficients"]
                coefficients.setdefault(delta_role + f"_degree_{degree}", {})[seed] = np.asarray(coefficient)
            frequency = delta_result["degree2_residual_fft"]["main_nonzero_frequency_index_xy"]
            main_frequencies.setdefault(delta_role, {})[seed] = frequency
        report["per_seed"][str(seed)] = seed_report
    for consistency_key, vectors in coefficients.items():
        report["cross_seed"][f"{consistency_key}_coefficient_consistency"] = _pairwise_consistency(vectors)
    for delta_role, frequencies in main_frequencies.items():
        report["cross_seed"][f"{delta_role}_degree2_main_frequency_consistency"] = _frequency_consistency(frequencies)
    return _jsonable(report), arrays


def analyze_three_seed_fields(
    bundles: Sequence[SeedFieldBundle],
    *,
    output_json: str | os.PathLike[str] | None = None,
    output_npz: str | os.PathLike[str] | None = None,
    evidence_boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze three loaded seed bundles and optionally write atomic artifacts.

    The returned dictionary is JSON-compatible.  The NPZ array bundle is
    written when ``output_npz`` is supplied; the same arrays are intentionally
    not embedded in the JSON report.
    """

    report, arrays = _analyze_fields(bundles, evidence_boundary=evidence_boundary)
    if output_npz is not None:
        atomic_save_npz(output_npz, arrays)
        report["artifacts"] = {"npz": str(Path(output_npz))}
    if output_json is not None:
        if output_npz is not None:
            report.setdefault("artifacts", {})["json"] = str(Path(output_json))
        atomic_write_json(output_json, report)
    return report


def analyze_from_sources(
    sources_by_seed: Mapping[int | str, Mapping[str, FieldSource | str | os.PathLike[str]]],
    *,
    true_sources: Mapping[int | str, str | os.PathLike[str] | np.ndarray] | None = None,
    expected_seeds: Sequence[int] | None = DEFAULT_SEEDS,
    coord_scale: float = DEFAULT_COORD_SCALE,
    output_json: str | os.PathLike[str] | None = None,
    output_npz: str | os.PathLike[str] | None = None,
    evidence_boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: load strict fields, then analyze them."""

    bundles = load_three_seed_fields(
        sources_by_seed,
        true_sources=true_sources,
        expected_seeds=expected_seeds,
        coord_scale=coord_scale,
    )
    return analyze_three_seed_fields(
        bundles,
        output_json=output_json,
        output_npz=output_npz,
        evidence_boundary=evidence_boundary,
    )


__all__ = [
    "DEFAULT_COORD_SCALE",
    "DEFAULT_SEEDS",
    "FIELD_ROLES",
    "DELTA_ROLES",
    "FieldSource",
    "LoadedField",
    "SeedFieldBundle",
    "analyze_from_sources",
    "analyze_three_seed_fields",
    "atomic_save_npz",
    "atomic_write_json",
    "load_field",
    "load_seed_bundle",
    "load_three_seed_fields",
]
