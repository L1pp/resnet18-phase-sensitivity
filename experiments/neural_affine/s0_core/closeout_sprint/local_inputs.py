"""Clean-room adapters for the local S0 input boundary.

This module is deliberately small and conservative.  It reads only files that
the caller has already put in the protocol/asset manifest, checks the file
hash before opening it, and recomputes all error quantities from ``pred`` and
``true``.  The ``err`` and ``u`` arrays found in historical ``field.npz``
files are never used.

The field on disk is a rendered XY field even when it came from a six-output
model (the A10 6D protocol renders the two coordinate outputs separately).
That distinction is represented by ``model_head_dim`` in the returned
provenance and is checked when supplied; a six-dimensional model checkpoint
must not be silently treated as a two-dimensional checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .assets import sha256_file


class LocalInputError(ValueError):
    """A manifest-locked local input is missing or violates its schema."""


def _path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _manifest_path(value: Path | str, *, manifest_base: Path | str | None, label: str) -> Path:
    """Resolve a manifest path without consulting the process cwd."""

    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    if manifest_base is None:
        raise LocalInputError(f"relative {label} requires explicit manifest_base")
    base = Path(manifest_base).expanduser()
    if base.suffix and not base.is_dir():
        base = base.parent
    return (base / raw).resolve(strict=False)


def _finite_array(value: Any, *, name: str, last_dim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or array.ndim < 1:
        raise LocalInputError(f"{name} must be a non-empty array")
    if last_dim is not None and (array.ndim < 1 or array.shape[-1] != last_dim):
        raise LocalInputError(f"{name} must have final dimension {last_dim}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise LocalInputError(f"{name} contains non-finite values")
    return array


def _scalar_metadata(blob: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in blob.files:
        if key in {"pred", "true", "err", "u", "err_affine", "coords", "support_ids", "seed_ids"}:
            continue
        value = np.asarray(blob[key])
        if value.ndim == 0:
            item = value.item()
            metadata[key] = item.item() if isinstance(item, np.generic) else item
        elif value.dtype.kind in "biuf" and value.size <= 4096:
            metadata[key] = value.tolist()
    return metadata


def _broadcast_coords(coords: Any, shape: tuple[int, ...], true: np.ndarray) -> tuple[np.ndarray, str]:
    """Return one coordinate per vector, preferring explicit field coords."""

    target = shape
    if coords is not None:
        array = _finite_array(coords, name="coords", last_dim=2)
        try:
            array = np.broadcast_to(array, target)
        except ValueError as exc:
            raise LocalInputError(
                f"coords shape {array.shape} cannot broadcast to rendered field {target}"
            ) from exc
        return np.array(array, copy=True), "input_coords"
    # ``true`` is the target pixel coordinate at each grid location.  It is a
    # legitimate independent geometry source and avoids inventing row/column
    # coordinates when an old field omitted its coords array.
    return np.array(true, copy=True), "true_target"


def _broadcast_ids(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    target = shape[:-1]
    if value is None:
        return np.arange(int(np.prod(target)), dtype=np.int64).reshape(target)
    array = np.asarray(value)
    if array.size == 0:
        raise LocalInputError(f"{name} cannot be empty")
    try:
        array = np.broadcast_to(array, target)
    except ValueError as exc:
        raise LocalInputError(f"{name} shape {array.shape} cannot broadcast to {target}") from exc
    return np.array(array, copy=True)


def _fit_affine(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([pred.reshape(-1, 2), np.ones(pred.size // 2, dtype=np.float64)])
    coeff, _, _, _ = np.linalg.lstsq(x, true.reshape(-1, 2), rcond=None)
    return coeff[:2].T, coeff[2]


def _check_expected_sha(path: Path, expected_sha256: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = sha256_file(path).lower()
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise LocalInputError(f"invalid expected sha256 for {path}")
        if actual != expected:
            raise LocalInputError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def inspect_checkpoint_head(
    path: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Safely inspect a manifest-bound checkpoint head.

    This deliberately uses only ``torch.load(..., weights_only=True)`` and
    accepts no fallback that could execute pickle payloads.
    """

    resolved = _path(path)
    digest = _check_expected_sha(resolved, _require_sha(expected_sha256, label="checkpoint"))
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInputError("formal checkpoint provenance requires PyTorch") from exc
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LocalInputError(f"weights_only checkpoint load failed: {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LocalInputError("checkpoint payload must be a mapping")
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise LocalInputError("checkpoint state must be a mapping")
    weight = state.get("fc.weight")
    bias = state.get("fc.bias")
    if weight is None or bias is None or not torch.is_tensor(weight) or not torch.is_tensor(bias):
        raise LocalInputError("checkpoint must expose tensor fc.weight and fc.bias")
    if weight.ndim != 2 or bias.ndim != 1 or int(weight.shape[0]) != int(bias.shape[0]):
        raise LocalInputError(
            f"checkpoint fc head shapes are invalid: weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "head_dim": int(weight.shape[0]),
        "head_in": int(weight.shape[1]),
        "state_keys": sorted(str(key) for key in state.keys()),
    }


def _require_sha(value: Any, *, label: str) -> str:
    if value in (None, ""):
        raise LocalInputError(f"{label} requires an explicit SHA256")
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise LocalInputError(f"{label} has an invalid SHA256")
    return text


def _normalise_id_list(value: Any, *, label: str, expected_count: int | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise LocalInputError(f"{label} must be a non-empty list")
    if not raw:
        raise LocalInputError(f"{label} cannot be empty")
    text = [str(item) for item in raw]
    if len(set(text)) != len(text):
        raise LocalInputError(f"{label} must contain unique IDs")
    if expected_count is not None and len(text) != int(expected_count):
        raise LocalInputError(
            f"{label} has {len(text)} IDs but field has {expected_count} appearances"
        )
    # Unicode string IDs preserve seed/appearance provenance without forcing
    # an artificial integer namespace across independent machines.
    return np.asarray(text, dtype="U")


def _load_appearance_asset(
    path: Path | str,
    *,
    expected_sha256: str,
    manifest_base: Path | str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = _manifest_path(path, manifest_base=manifest_base, label="appearance asset")
    digest = _check_expected_sha(resolved, _require_sha(expected_sha256, label="appearance asset"))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LocalInputError(f"invalid appearance asset {resolved}: {exc}") from exc
    if isinstance(payload, Mapping):
        raw = payload.get("appearance_ids", payload.get("ids", payload.get("appearances")))
    else:
        raw = payload
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = []
        for item in raw:
            if isinstance(item, Mapping):
                item = item.get("id", item.get("appearance_id", item.get("name")))
            values.append(item)
    else:
        raise LocalInputError(f"appearance asset {resolved} must contain an IDs list")
    ids = _normalise_id_list(values, label="appearance_ids")
    return ids, {"path": str(resolved), "sha256": digest, "count": int(len(ids))}


@dataclass(frozen=True)
class RawField:
    """Recomputed rendered field and its immutable provenance."""

    path: Path
    sha256: str
    task: str
    seed_id: str | None
    appearance_ids: np.ndarray | None
    pred: np.ndarray
    true: np.ndarray
    raw_error: np.ndarray
    affine_pred: np.ndarray
    u: np.ndarray
    coords: np.ndarray
    support_ids: np.ndarray
    metadata: Mapping[str, Any]
    coordinate_source: str
    model_head_dim: int | None = None
    appearance_asset: Mapping[str, Any] | None = None
    support_provenance: Mapping[str, Any] | None = None
    checkpoint_provenance: Mapping[str, Any] | None = None
    axis_semantics: str = "legacy_seed"
    legacy_seed_ids: np.ndarray | None = None

    @property
    def n_seeds(self) -> int:
        return 1 if self.axis_semantics == "appearance" else int(self.pred.shape[0])

    @property
    def n_appearances(self) -> int:
        return int(self.pred.shape[0]) if self.axis_semantics == "appearance" else 1

    @property
    def seed_ids(self) -> np.ndarray:
        """Backward-compatible scalar API alias; formal fields use seed_id."""

        if self.axis_semantics == "appearance":
            return np.asarray([self.seed_id], dtype="U")
        if self.legacy_seed_ids is not None:
            return np.asarray(self.legacy_seed_ids, copy=True)
        return np.arange(self.pred.shape[0], dtype=np.int64)

    @property
    def grid_shape(self) -> tuple[int, int]:
        if self.pred.ndim < 4:
            raise LocalInputError(f"rendered field must have [n, ny, nx, 2], got {self.pred.shape}")
        return int(self.pred.shape[-3]), int(self.pred.shape[-2])

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "task": self.task,
            "seed_id": self.seed_id,
            "appearance_ids": None if self.appearance_ids is None else np.asarray(self.appearance_ids).tolist(),
            "appearance_asset": None if self.appearance_asset is None else dict(self.appearance_asset),
            "model_head_dim": self.model_head_dim,
            "checkpoint_provenance": None if self.checkpoint_provenance is None else dict(self.checkpoint_provenance),
            "shape": list(self.pred.shape),
            "n_seeds": self.n_seeds,
            "n_appearances": self.n_appearances,
            "grid_shape": list(self.grid_shape),
            "coordinate_source": self.coordinate_source,
            "seed_ids": np.asarray(self.seed_ids).tolist(),
            "support_provenance": None if self.support_provenance is None else dict(self.support_provenance),
            "axis_semantics": self.axis_semantics,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CombinedRawField:
    """One task with appearance axis first and independent seed axis added."""

    task: str
    seed_ids: np.ndarray
    appearance_ids: np.ndarray
    pred: np.ndarray
    true: np.ndarray
    raw_error: np.ndarray
    affine_pred: np.ndarray
    u: np.ndarray
    coords: np.ndarray
    support_ids: np.ndarray
    model_head_dim: int
    sources: tuple[Mapping[str, Any], ...]
    support_provenance: tuple[Mapping[str, Any] | None, ...]

    @property
    def n_seeds(self) -> int:
        return int(self.pred.shape[3])

    @property
    def n_appearances(self) -> int:
        return int(self.pred.shape[0])

    def provenance(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "seed_ids": np.asarray(self.seed_ids).tolist(),
            "appearance_ids": np.asarray(self.appearance_ids).tolist(),
            "shape": list(self.pred.shape),
            "model_head_dim": self.model_head_dim,
            "sources": [dict(item) for item in self.sources],
            "support_provenance": [None if item is None else dict(item) for item in self.support_provenance],
        }


def load_raw_field(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    task: str,
    seed_id: Any = None,
    appearance_ids: Any = None,
    appearance_asset_path: Path | str | None = None,
    appearance_asset_sha256: str | None = None,
    manifest_base: Path | str | None = None,
    support_provenance: Mapping[str, Any] | None = None,
    checkpoint_path: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    model_head_dim: int | None = None,
    expected_model_head_dim: int | None = None,
) -> RawField:
    """Load and independently recompute one manifest-locked XY field.

    ``task`` is mandatory by design.  It prevents a Neural Affine, AMD, and
    A10 field from being accidentally pooled merely because their arrays have
    the same shape.  ``model_head_dim`` is provenance only; the field itself
    must always end in two rendered coordinate components.
    """

    if not isinstance(task, str) or not task.strip():
        raise LocalInputError("task is required and cannot be empty")
    if model_head_dim is not None:
        try:
            model_head_dim = int(model_head_dim)
        except (TypeError, ValueError) as exc:
            raise LocalInputError("model_head_dim must be an integer") from exc
        if model_head_dim <= 0:
            raise LocalInputError("model_head_dim must be positive")
    if expected_model_head_dim is not None and model_head_dim != int(expected_model_head_dim):
        raise LocalInputError(
            f"model head dimension mismatch: expected {expected_model_head_dim}, got {model_head_dim}"
        )
    checkpoint_provenance: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint_resolved = (
            _path(checkpoint_path)
            if Path(checkpoint_path).expanduser().is_absolute()
            else _manifest_path(checkpoint_path, manifest_base=manifest_base, label="checkpoint")
        )
        checkpoint_provenance = inspect_checkpoint_head(
            checkpoint_resolved,
            expected_sha256=_require_sha(checkpoint_sha256, label="checkpoint"),
        )
        if model_head_dim is None:
            model_head_dim = int(checkpoint_provenance["head_dim"])
        if int(checkpoint_provenance["head_dim"]) != int(model_head_dim):
            raise LocalInputError(
                "checkpoint head dimension does not match field model_head_dim: "
                f"{checkpoint_provenance['head_dim']} != {model_head_dim}"
            )
    resolved = _path(path) if Path(path).expanduser().is_absolute() else _manifest_path(path, manifest_base=manifest_base, label="field")
    digest = _check_expected_sha(resolved, expected_sha256)
    appearance_asset: dict[str, Any] | None = None
    if appearance_asset_path is not None:
        asset_ids, appearance_asset = _load_appearance_asset(
            appearance_asset_path,
            expected_sha256=_require_sha(appearance_asset_sha256, label="appearance asset"),
            manifest_base=manifest_base,
        )
        if appearance_ids is not None:
            explicit_ids = _normalise_id_list(appearance_ids, label="appearance_ids")
            if not np.array_equal(explicit_ids, asset_ids):
                raise LocalInputError("explicit appearance_ids do not match the hash-pinned appearance asset")
        appearance_ids = asset_ids
    axis_semantics = "appearance" if appearance_ids is not None else "legacy_seed"
    if axis_semantics == "appearance":
        if seed_id in (None, ""):
            raise LocalInputError("appearance-resolved field requires an explicit seed_id")
        seed_text = str(seed_id)
    else:
        seed_text = None
    try:
        with np.load(resolved, allow_pickle=False) as blob:
            keys = set(blob.files)
            if not {"pred", "true"}.issubset(keys):
                raise LocalInputError(f"{resolved} must contain pred and true; found {sorted(keys)}")
            pred = _finite_array(blob["pred"], name="pred", last_dim=2)
            true = _finite_array(blob["true"], name="true", last_dim=2)
            if pred.shape != true.shape:
                raise LocalInputError(f"pred shape {pred.shape} != true shape {true.shape}")
            if pred.ndim != 4:
                raise LocalInputError(
                    "rendered_xy field must have [n_seed, n_y, n_x, 2] shape; "
                    f"got {pred.shape}"
                )
            coords, coordinate_source = _broadcast_coords(
                blob["coords"] if "coords" in keys else None,
                pred.shape,
                true,
            )
            support_ids = _broadcast_ids(
                blob["support_ids"] if "support_ids" in keys else None,
                pred.shape,
                name="support_ids",
            )
            stored_appearance = None
            if "appearance_ids" in keys:
                stored_appearance = _normalise_id_list(
                    blob["appearance_ids"], label="appearance_ids", expected_count=pred.shape[0]
                )
            legacy_seed_value = None
            if "seed_ids" in keys:
                candidate = np.asarray(blob["seed_ids"])
                if candidate.ndim == 1 and candidate.size == pred.shape[0]:
                    legacy_seed_value = np.array(candidate, copy=True)
            if axis_semantics == "appearance":
                if stored_appearance is not None and not np.array_equal(stored_appearance, np.asarray(appearance_ids)):
                    raise LocalInputError("field appearance_ids do not match manifest provenance")
                resolved_appearances = _normalise_id_list(
                    appearance_ids, label="appearance_ids", expected_count=pred.shape[0]
                )
            else:
                # Legacy direct callers may still use the old first-axis-as-
                # seeds contract.  Formal manifest callers never take this
                # branch because they must provide appearance provenance.
                resolved_appearances = None
            metadata = _scalar_metadata(blob)
    except LocalInputError:
        raise
    except Exception as exc:
        raise LocalInputError(f"cannot read rendered field {resolved}: {type(exc).__name__}: {exc}") from exc

    raw_error = pred - true
    matrix, bias = _fit_affine(pred, true)
    affine_pred = pred @ matrix.T + bias
    u = affine_pred - true
    return RawField(
        path=resolved,
        sha256=digest,
        task=task.strip(),
        seed_id=seed_text,
        appearance_ids=resolved_appearances,
        pred=pred,
        true=true,
        raw_error=raw_error,
        affine_pred=affine_pred,
        u=u,
        coords=coords,
        support_ids=support_ids,
        metadata=metadata,
        coordinate_source=coordinate_source,
        model_head_dim=model_head_dim,
        appearance_asset=appearance_asset,
        support_provenance=None if support_provenance is None else dict(support_provenance),
        checkpoint_provenance=checkpoint_provenance,
        axis_semantics=axis_semantics,
        legacy_seed_ids=legacy_seed_value if axis_semantics == "legacy_seed" else None,
    )


def load_fields_from_manifest(
    entries: Iterable[Mapping[str, Any]],
    *,
    manifest_base: Path | str | None = None,
    expected_role: Sequence[str] = ("field_rendered_xy", "field_2d", "rendered_xy", "field"),
) -> dict[str, RawField]:
    """Load each manifest entry independently, with no cross-task pooling."""

    allowed_roles = {str(role) for role in expected_role}
    result: dict[str, RawField] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise LocalInputError(f"manifest entry {index} is not an object")
        role = str(entry.get("role", ""))
        if role not in allowed_roles:
            raise LocalInputError(f"manifest field entry {index} has disallowed role {role!r}")
        task = entry.get("task", entry.get("group"))
        if not isinstance(task, str) or not task.strip():
            raise LocalInputError(f"manifest field entry {index} requires task/group")
        path = entry.get("path")
        if not path:
            raise LocalInputError(f"manifest field entry {index} requires path")
        expected_sha = _require_sha(
            entry.get("sha256", entry.get("expected_sha256")),
            label=f"manifest field {index}",
        )
        if "seed_id" not in entry or entry.get("seed_id") in (None, ""):
            raise LocalInputError(f"manifest field entry {index} requires explicit seed_id")
        has_appearance_ids = "appearance_ids" in entry and entry.get("appearance_ids") is not None
        asset_spec = entry.get("appearance_asset", entry.get("appearance_asset_path"))
        if isinstance(asset_spec, Mapping):
            asset_path = asset_spec.get("path")
            asset_sha = asset_spec.get("sha256", asset_spec.get("expected_sha256"))
        else:
            asset_path = asset_spec
            asset_sha = entry.get("appearance_asset_sha256", entry.get("appearance_sha256"))
        if not has_appearance_ids and asset_path is None:
            raise LocalInputError(
                f"manifest field entry {index} requires appearance_ids or a hash-pinned appearance_asset"
            )
        if asset_path is not None:
            _require_sha(asset_sha, label=f"manifest appearance asset {index}")
        support_spec = entry.get("support", entry.get("support_provenance"))
        if support_spec is None and "support_ids" not in entry and "support_asset" not in entry:
            raise LocalInputError(
                f"manifest field entry {index} requires locked support_ids/support_asset or explicit support spec"
            )
        if support_spec is None and "support_ids" in entry:
            support_spec = {"kind": "explicit_protocol", "support_ids": entry.get("support_ids")}
        if support_spec is None and "support_asset" in entry:
            support_spec = entry.get("support_asset")
        support_asset = entry.get("support_asset")
        if support_asset is not None:
            if not isinstance(support_asset, Mapping):
                raise LocalInputError(f"manifest support_asset {index} must be an object with path and sha256")
            support_path = support_asset.get("path")
            support_sha = _require_sha(
                support_asset.get("sha256", support_asset.get("expected_sha256")),
                label=f"manifest support asset {index}",
            )
            if not support_path:
                raise LocalInputError(f"manifest support asset {index} requires path")
            support_resolved = _manifest_path(support_path, manifest_base=manifest_base, label="support asset")
            support_actual = _check_expected_sha(support_resolved, support_sha)
            support_spec = {
                "kind": "hash_pinned_asset",
                "path": str(support_resolved),
                "sha256": support_actual,
            }
        model_head_dim = entry.get("model_head_dim", entry.get("head_dim"))
        if model_head_dim is None:
            raise LocalInputError(f"manifest field entry {index} requires explicit model_head_dim")
        checkpoint_path = entry.get("checkpoint_path", entry.get("checkpoint"))
        if isinstance(checkpoint_path, Mapping):
            checkpoint_sha = checkpoint_path.get("sha256", checkpoint_path.get("expected_sha256"))
            checkpoint_path = checkpoint_path.get("path")
        else:
            checkpoint_sha = entry.get("checkpoint_sha256", entry.get("checkpoint_expected_sha256"))
        if not checkpoint_path:
            raise LocalInputError(f"manifest field entry {index} requires checkpoint_path")
        checkpoint_sha = _require_sha(checkpoint_sha, label=f"manifest checkpoint {index}")
        asset_id = str(entry.get("id", entry.get("asset_id", task))).strip()
        if not asset_id or asset_id in result:
            raise LocalInputError(f"duplicate/empty manifest field id: {asset_id!r}")
        result[asset_id] = load_raw_field(
            path,
            expected_sha256=expected_sha,
            task=task,
            seed_id=entry.get("seed_id"),
            appearance_ids=entry.get("appearance_ids"),
            appearance_asset_path=asset_path,
            appearance_asset_sha256=asset_sha,
            manifest_base=manifest_base,
            support_provenance=support_spec if isinstance(support_spec, Mapping) else {"value": support_spec},
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            model_head_dim=model_head_dim,
            expected_model_head_dim=entry.get("expected_model_head_dim"),
        )
    if not result:
        raise LocalInputError("manifest contains no field entries")
    return result


def combine_task_fields(fields: Sequence[RawField]) -> CombinedRawField:
    """Validate and stack independent same-task runs along the seed axis."""

    values = tuple(fields)
    if not values:
        raise LocalInputError("cannot combine an empty task field list")
    if any(not isinstance(item, RawField) for item in values):
        raise TypeError("combine_task_fields requires RawField values")
    first = values[0]
    if first.axis_semantics != "appearance" or first.appearance_ids is None or first.seed_id is None:
        raise LocalInputError("formal same-task combination requires explicit appearance and seed provenance")
    if first.checkpoint_provenance is None:
        raise LocalInputError("formal same-task combination requires checkpoint provenance")
    seed_ids = [str(item.seed_id) for item in values]
    if len(set(seed_ids)) != len(seed_ids):
        raise LocalInputError(f"duplicate seed_id within task {first.task!r}")
    appearance_ids = np.asarray(first.appearance_ids)
    for item in values:
        if item.task != first.task:
            raise LocalInputError("different tasks cannot be combined")
        if item.axis_semantics != "appearance" or item.appearance_ids is None:
            raise LocalInputError("all same-task fields must carry explicit appearance provenance")
        if not np.array_equal(np.asarray(item.appearance_ids), appearance_ids):
            raise LocalInputError("same-task fields have different appearance IDs or order")
        for name in ("true", "coords", "support_ids"):
            if not np.array_equal(np.asarray(getattr(item, name)), np.asarray(getattr(first, name))):
                raise LocalInputError(f"same-task fields disagree on {name}")
        if item.support_provenance != first.support_provenance:
            raise LocalInputError("same-task fields disagree on support provenance")
        if item.checkpoint_provenance is None:
            raise LocalInputError("same-task field is missing checkpoint provenance")
        if int(item.checkpoint_provenance.get("head_dim", -1)) != int(first.checkpoint_provenance.get("head_dim", -2)):
            raise LocalInputError("same-task fields disagree on checkpoint head dimension")
        if item.pred.shape != first.pred.shape or item.model_head_dim != first.model_head_dim:
            raise LocalInputError("same-task fields disagree on shape or model_head_dim")
    return CombinedRawField(
        task=first.task,
        seed_ids=np.asarray(seed_ids, dtype="U"),
        appearance_ids=np.asarray(appearance_ids, dtype="U"),
        pred=np.stack([item.pred for item in values], axis=3),
        true=np.stack([item.true for item in values], axis=3),
        raw_error=np.stack([item.raw_error for item in values], axis=3),
        affine_pred=np.stack([item.affine_pred for item in values], axis=3),
        u=np.stack([item.u for item in values], axis=3),
        coords=np.asarray(first.coords, copy=True),
        support_ids=np.asarray(first.support_ids, copy=True),
        model_head_dim=int(first.model_head_dim),
        sources=tuple(item.provenance() for item in values),
        support_provenance=tuple(item.support_provenance for item in values),
    )


def combine_fields_by_task(fields: Mapping[str, RawField]) -> dict[str, CombinedRawField]:
    if not isinstance(fields, Mapping) or not fields:
        raise LocalInputError("fields must be a non-empty mapping")
    grouped: dict[str, list[RawField]] = {}
    for field in fields.values():
        if not isinstance(field, RawField):
            raise TypeError("combine_fields_by_task requires RawField values")
        grouped.setdefault(field.task, []).append(field)
    return {task: combine_task_fields(items) for task, items in grouped.items()}


def _field_signal(field: RawField | CombinedRawField, signal: str) -> np.ndarray:
    key = str(signal).strip().lower()
    vector = np.asarray(field.u)
    raw_vector = np.asarray(field.raw_error)
    if isinstance(field, CombinedRawField):
        # [appearance, y, x, seed, component] -> [appearance, y, x, seed]
        choices = {
            "raw_norm": np.linalg.norm(raw_vector, axis=-1),
            "error_norm": np.linalg.norm(raw_vector, axis=-1),
            "u_norm": np.linalg.norm(vector, axis=-1),
            "u_x": vector[..., 0],
            "u_y": vector[..., 1],
            "raw_x": raw_vector[..., 0],
            "raw_y": raw_vector[..., 1],
        }
    else:
        choices = {
            "raw_norm": np.linalg.norm(raw_vector, axis=-1),
            "error_norm": np.linalg.norm(raw_vector, axis=-1),
            "u_norm": np.linalg.norm(vector, axis=-1),
            "u_x": vector[..., 0],
            "u_y": vector[..., 1],
            "raw_x": raw_vector[..., 0],
            "raw_y": raw_vector[..., 1],
        }
    if key not in choices:
        raise LocalInputError(f"unknown partial signal {signal!r}; choose from {sorted(choices)}")
    values = np.asarray(choices[key], dtype=np.float64)
    if values.ndim not in (3, 4):
        raise LocalInputError(f"partial signal must have [n, ny, nx] or [appearance, ny, nx, seed], got {values.shape}")
    return values


def _partial_coords(field: RawField | CombinedRawField) -> tuple[np.ndarray, int]:
    coords = np.asarray(field.coords, dtype=np.float64)
    if coords.ndim != 4 or coords.shape[-1] != 2:
        raise LocalInputError(f"field coords must have [appearance, ny, nx, 2], got {coords.shape}")
    if not np.allclose(coords, coords[0][None, ...], rtol=0.0, atol=0.0):
        raise LocalInputError(f"task {field.task!r} has appearance-dependent coordinates")
    n_appearance = int(field.n_appearances)
    return coords[0].reshape(-1, 2), n_appearance


def _partial_auxiliary(
    field: RawField | CombinedRawField,
    *,
    anchors: Any = None,
    anchors_provenance: Mapping[str, Any] | None = None,
    domain: Any = None,
) -> dict[str, Any]:
    formal = getattr(field, "axis_semantics", "appearance") == "appearance"
    if formal and anchors is None and anchors_provenance is None:
        raise LocalInputError(
            f"formal task {field.task!r} requires explicit protocol/locked anchors; "
            "G4 and G9 cannot be inferred interchangeably"
        )
    if anchors is not None and anchors_provenance is None:
        anchors_provenance = {"kind": "explicit_protocol", "name": "provided"}
    if anchors is None and anchors_provenance is None:
        # Explicit protocol value "default_g4" is still represented; a
        # caller that wants the preregistered G9/G4 distinction must pass its
        # anchor array and label explicitly.
        anchors_provenance = {"kind": "explicit_protocol", "name": "default_task_anchors"}
    if anchors_provenance is not None and not isinstance(anchors_provenance, Mapping):
        raise LocalInputError("anchors_provenance must be an object")
    anchor_array = None if anchors is None else np.asarray(anchors, dtype=np.float64)
    if anchor_array is not None:
        if anchor_array.ndim != 2 or anchor_array.shape[1] < 2 or not np.all(np.isfinite(anchor_array)):
            raise LocalInputError("anchors must have finite shape (n, d>=2)")
    anchor_name = "" if anchors_provenance is None else str(
        anchors_provenance.get("name", anchors_provenance.get("label", ""))
    )
    if anchor_name.upper() == "G4" and anchor_array is not None and anchor_array.shape[0] != 4:
        raise LocalInputError("G4 anchor provenance requires exactly four anchors")
    if anchor_name.upper() == "G9" and anchor_array is not None and anchor_array.shape[0] != 9:
        raise LocalInputError("G9 anchor provenance requires exactly nine anchors")
    result: dict[str, Any] = {
        "anchors": anchor_array,
        "anchors_provenance": None if anchors_provenance is None else dict(anchors_provenance),
        "domain": None if domain is None else np.asarray(domain, dtype=np.float64),
    }
    return result


def field_to_partial_group(
    field: RawField | CombinedRawField,
    *,
    signal: str = "u_norm",
    anchors: Any = None,
    anchors_provenance: Mapping[str, Any] | None = None,
    domain: Any = None,
) -> dict[str, Any]:
    """Convert one task field into the ``partial_out`` group contract.

    The returned signals have shape ``(n_grid, n_seed)``.  No rows from any
    other task are accepted or appended here; use ``partial_groups_from_fields``
    to retain the one-result-per-task boundary.
    """

    if not isinstance(field, (RawField, CombinedRawField)):
        raise TypeError("field_to_partial_group requires a RawField or CombinedRawField")
    values = _field_signal(field, signal)
    coords_one, n_appearance = _partial_coords(field)
    if isinstance(field, CombinedRawField):
        # Repeated appearances are rows, not seeds.  The vector contract keeps
        # both components and is the scientific gate input; scalar u_norm is
        # retained solely as a supplementary/legacy signal.
        vector = np.asarray(field.u, dtype=np.float64)
        raw_vector = np.asarray(field.raw_error, dtype=np.float64)
        signals_vector = vector
        scalar = values.reshape(n_appearance * coords_one.shape[0], field.n_seeds)
        coords = np.tile(coords_one, (n_appearance, 1))
        seed_ids = np.asarray(field.seed_ids, copy=True)
        support_ids = np.asarray(field.support_ids[0], copy=True)
        provenance = field.provenance()
        provenance["scalar_signal_is_supplementary"] = True
    else:
        if field.axis_semantics == "appearance":
            # One formal field is one seed; preserve appearance rows without
            # mislabelling them as independent seeds.
            signals_vector = np.asarray(field.u, dtype=np.float64)[..., None, :]
            scalar = values.reshape(n_appearance * coords_one.shape[0], 1)
            coords = np.tile(coords_one, (n_appearance, 1))
            seed_ids = np.asarray([field.seed_id], dtype="U")
            support_ids = np.asarray(field.support_ids[0], copy=True)
            provenance = field.provenance()
            provenance["scalar_signal_is_supplementary"] = True
        else:
            # Compatibility with the original scalar API where the first axis
            # was explicitly interpreted as independent seeds.
            signals_vector = None
            coords = coords_one
            scalar = values.reshape(values.shape[0], -1).T
            seed_ids = np.asarray(field.seed_ids, copy=True)
            support_ids = np.asarray(field.support_ids[0], copy=True)
            provenance = field.provenance()
            provenance["legacy_axis_semantics"] = True
    aux = _partial_auxiliary(
        field,
        anchors=anchors,
        anchors_provenance=anchors_provenance,
        domain=domain,
    )
    if scalar.shape[0] != coords.shape[0]:
        raise LocalInputError("partial coords/signals row count mismatch")
    group = {
        "coords": coords,
        "signals": scalar,
        "task": field.task,
        "signal_kind": str(signal),
        "seed_ids": seed_ids,
        "support_ids": support_ids,
        "model_head_dim": field.model_head_dim,
        "field_provenance": provenance,
        **aux,
    }
    if signals_vector is not None:
        group["signals_vector"] = signals_vector
        group["vector_schema"] = "[appearance,ny,nx,seed,component]"
        group["vector_gate_signal"] = "pooled_vector_flatten"
    if isinstance(field, RawField):
        group["source_path"] = str(field.path)
        group["source_sha256"] = field.sha256
    else:
        group["source_path"] = [item["path"] for item in field.sources]
        group["source_sha256"] = [item["sha256"] for item in field.sources]
    return group


def partial_groups_from_fields(
    fields: Mapping[str, RawField], *, signal: str = "u_norm", anchors_by_task: Mapping[str, Any] | None = None,
    anchors_provenance_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    domain_by_task: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build independent partial groups while rejecting task-name ambiguity."""

    if not isinstance(fields, Mapping) or not fields:
        raise LocalInputError("fields must be a non-empty mapping")
    combined = combine_fields_by_task(fields)
    result: dict[str, dict[str, Any]] = {}
    for name, field in fields.items():
        if not isinstance(name, str) or not name.strip():
            raise LocalInputError("partial group names cannot be empty")
    for task, field in combined.items():
        result[task] = field_to_partial_group(
            field,
            signal=signal,
            anchors=None if anchors_by_task is None else anchors_by_task.get(task),
            anchors_provenance=None if anchors_provenance_by_task is None else anchors_provenance_by_task.get(task),
            domain=None if domain_by_task is None else domain_by_task.get(task),
        )
    return result


def save_partial_group(path: Path | str, group: Mapping[str, Any]) -> Path:
    """Write only a new clean-room partial input artifact."""

    required = {"coords", "signals", "task", "source_path", "source_sha256"}
    missing = sorted(required - set(group))
    if missing:
        raise LocalInputError(f"partial group missing required provenance: {missing}")
    out = _path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    coords = _finite_array(group["coords"], name="coords", last_dim=2)
    signals = _finite_array(group["signals"], name="signals")
    if signals.ndim == 1:
        signals = signals[:, None]
    if signals.ndim != 2 or signals.shape[0] != coords.shape[0]:
        raise LocalInputError("partial signals must be a 2-D array aligned to coords")
    metadata = {
        "task": str(group["task"]),
        "signal_kind": str(group.get("signal_kind", "unknown")),
        "source_path": str(group["source_path"]),
        "source_sha256": str(group["source_sha256"]),
        "model_head_dim": group.get("model_head_dim"),
        "field_provenance": group.get("field_provenance", {}),
        "anchors_provenance": group.get("anchors_provenance"),
        "vector_schema": group.get("vector_schema"),
        "vector_gate_signal": group.get("vector_gate_signal"),
    }
    arrays: dict[str, Any] = {
        "coords": np.asarray(coords, dtype=np.float64),
        "signals": np.asarray(signals, dtype=np.float64),
        "x": np.asarray(coords[:, 0], dtype=np.float64),
        "y": np.asarray(coords[:, 1], dtype=np.float64),
        "seed_ids": np.asarray(group.get("seed_ids", np.arange(signals.shape[1]))),
        "support_ids": np.asarray(group.get("support_ids", np.arange(coords.shape[0]))),
        "task": np.asarray(metadata["task"]),
        "provenance_json": np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)),
    }
    if "signals_vector" in group:
        vector = np.asarray(group["signals_vector"], dtype=np.float64)
        if vector.ndim != 5 or vector.shape[-1] != 2 or vector.shape[-2] != signals.shape[1]:
            raise LocalInputError(
                "signals_vector must have [appearance,ny,nx,seed,2] aligned to scalar signals"
            )
        if vector.shape[0] * vector.shape[1] * vector.shape[2] != coords.shape[0]:
            raise LocalInputError("signals_vector spatial rows do not match coords")
        if not np.all(np.isfinite(vector)):
            raise LocalInputError("signals_vector contains non-finite values")
        arrays["signals_vector"] = vector
    if group.get("anchors") is not None:
        anchors = _finite_array(group["anchors"], name="anchors")
        if anchors.ndim != 2 or anchors.shape[1] < 2:
            raise LocalInputError("anchors must have shape (n, d>=2)")
        arrays["anchors"] = anchors
    if group.get("domain") is not None:
        domain = _finite_array(group["domain"], name="domain")
        if domain.shape != (2, 2):
            raise LocalInputError("domain must have shape (2, 2)")
        arrays["domain"] = domain
    np.savez_compressed(out, **arrays)
    return out


def validate_saved_partial_group(
    path: Path | str,
    *,
    expected_task: str | None = None,
    expected_source_sha256: Sequence[str] | str | None = None,
    expected_anchor_label: str | None = None,
) -> dict[str, Any]:
    """Fail-fast validation for a newly written partial-group artifact."""

    resolved = _path(path)
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    with np.load(resolved, allow_pickle=False) as blob:
        required = {"coords", "signals", "seed_ids", "support_ids", "task", "provenance_json"}
        missing = sorted(required - set(blob.files))
        if missing:
            raise LocalInputError(f"partial group is missing required arrays: {missing}")
        coords = _finite_array(blob["coords"], name="coords", last_dim=2)
        signals = _finite_array(blob["signals"], name="signals")
        if signals.ndim != 2 or signals.shape[0] != coords.shape[0]:
            raise LocalInputError("saved partial coords/signals are not aligned")
        task = str(np.asarray(blob["task"]).item())
        if expected_task is not None and task != str(expected_task):
            raise LocalInputError(f"partial task mismatch: expected {expected_task!r}, got {task!r}")
        provenance = json.loads(str(np.asarray(blob["provenance_json"]).item()))
        if not isinstance(provenance, Mapping):
            raise LocalInputError("partial provenance_json must be an object")
        if expected_source_sha256 is not None:
            expected = [str(expected_source_sha256)] if isinstance(expected_source_sha256, str) else [str(item) for item in expected_source_sha256]
            actual = provenance.get("field_provenance", {})
            if isinstance(actual, Mapping) and "sources" in actual:
                actual_values = [str(item.get("sha256")) for item in actual["sources"]]
            else:
                actual_values = [str(provenance.get("source_sha256"))]
            if actual_values != expected:
                raise LocalInputError(f"partial source SHA mismatch: expected {expected}, got {actual_values}")
        if expected_anchor_label is not None:
            anchor_meta = provenance.get("anchors_provenance") or {}
            if not isinstance(anchor_meta, Mapping) or str(anchor_meta.get("name", anchor_meta.get("label", ""))) != str(expected_anchor_label):
                raise LocalInputError("partial anchor protocol label mismatch")
        anchor_meta = provenance.get("anchors_provenance")
        if anchor_meta is not None and not isinstance(anchor_meta, Mapping):
            raise LocalInputError("anchors_provenance must be an object")
        if anchor_meta is not None and anchor_meta.get("kind") != "explicit_protocol":
            if "anchors" not in blob.files:
                raise LocalInputError("locked anchor provenance exists but anchors array is missing")
        if "anchors" in blob.files:
            anchors = _finite_array(blob["anchors"], name="anchors")
            if anchors.ndim != 2 or anchors.shape[1] < 2:
                raise LocalInputError("saved anchors array is invalid")
            label = str((anchor_meta or {}).get("name", (anchor_meta or {}).get("label", "")))
            if label.upper() == "G4" and anchors.shape[0] != 4:
                raise LocalInputError("saved G4 anchors must have four rows")
            if label.upper() == "G9" and anchors.shape[0] != 9:
                raise LocalInputError("saved G9 anchors must have nine rows")
        result = {
            "task": task,
            "coords_shape": list(coords.shape),
            "signals_shape": list(signals.shape),
            "keys": list(blob.files),
            "provenance": dict(provenance),
        }
        if "signals_vector" in blob.files:
            vector = _finite_array(blob["signals_vector"], name="signals_vector")
            if vector.ndim != 5 or vector.shape[-1] != 2 or vector.shape[-2] != signals.shape[1]:
                raise LocalInputError("saved signals_vector schema is invalid")
            result["signals_vector_shape"] = list(vector.shape)
    return result


__all__ = [
    "CombinedRawField",
    "LocalInputError",
    "RawField",
    "combine_fields_by_task",
    "combine_task_fields",
    "field_to_partial_group",
    "load_fields_from_manifest",
    "load_raw_field",
    "partial_groups_from_fields",
    "save_partial_group",
    "validate_saved_partial_group",
]
