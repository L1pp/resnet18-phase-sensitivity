"""Clean-room Gaussian-blob data and BatchNorm exposure adapters.

The historical experiment packages are intentionally not imported here.  The
renderer is reimplemented from the frozen protocol: a 4x supersampled
Gaussian with ``sigma=6`` followed by Pillow Lanczos downsampling.  Images are
kept as uint8 until the explicit ``uint8_to_nchw`` boundary, matching the
training convention of three identical channels and no ImageNet
normalisation.

The three adapter functions at the bottom of this module are suitable for a
protocol JSON file consumed by :func:`closeout_sprint.bn_control.run_bn_only_control`:

``model_adapter(checkpoint_path=...)`` -> a safe-loaded 2D ResNet18;
``data_adapter(max_exposure=...)`` -> a lazy finite batch stream;
``evaluator_adapter(...)`` -> a callable ``(model, exposure)`` evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .assets import sha256_file
from .evaluator import fit_affine


IMAGE_SIZE = 224
COORD_SCALE = float(IMAGE_SIZE - 1)
T_LOW_PX = 59.0
T_HIGH_PX = 164.0
N_GRID = 41
N_TX = 8
N_TY = 8
SUPER_SAMPLE = 4
BLOB_SIGMA = 6.0
CORNERS4_TIDS: tuple[int, ...] = (0, 7, 56, 63)


class BlobAdapterError(ValueError):
    """Renderer/checkpoint/adapter input is invalid."""


def _manifest_path(value: Path | str, *, manifest_base: Path | str | None, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    if manifest_base is None:
        raise BlobAdapterError(f"relative {label} requires explicit manifest_base")
    base = Path(manifest_base).expanduser()
    if base.suffix and not base.is_dir():
        base = base.parent
    return (base / raw).resolve(strict=False)


def _require_sha(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise BlobAdapterError(f"{label} requires a valid SHA256")
    return text


@dataclass(frozen=True)
class AppearanceRecord:
    appearance_id: str
    sigma: float


@dataclass(frozen=True)
class AppearanceSet:
    role: str
    path: Path
    sha256: str
    records: tuple[AppearanceRecord, ...]

    @property
    def appearance_ids(self) -> tuple[str, ...]:
        return tuple(item.appearance_id for item in self.records)

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(float(item.sigma) for item in self.records)

    def metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sha256": self.sha256,
            "count": len(self.records),
            "appearance_ids": list(self.appearance_ids),
            "sigmas": list(self.sigmas),
            "fingerprint": self.sha256,
        }


def _parse_appearance_records(payload: Any, *, role: str) -> tuple[AppearanceRecord, ...]:
    if isinstance(payload, Mapping):
        raw = payload.get("appearances", payload.get("records"))
        if raw is None:
            ids = payload.get("appearance_ids", payload.get("ids"))
            sigmas = payload.get("sigmas", payload.get("sigma_values"))
            ids_is_sequence = isinstance(ids, Sequence) and not isinstance(ids, (str, bytes))
            sigmas_is_sequence = isinstance(sigmas, Sequence) and not isinstance(sigmas, (str, bytes))
            if not ids_is_sequence or not sigmas_is_sequence:
                raise BlobAdapterError(
                    f"{role} appearance_ids and sigmas must both be non-string sequences"
                )
            if len(ids) != len(sigmas):
                raise BlobAdapterError(
                    f"{role} appearance_ids/sigmas length mismatch: {len(ids)} != {len(sigmas)}"
                )
            raw = [{"id": item, "sigma": sigma} for item, sigma in zip(ids, sigmas)]
    else:
        raw = payload
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise BlobAdapterError(f"{role} appearance asset must contain a non-empty records list")
    records: list[AppearanceRecord] = []
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            appearance_id = item.get("id", item.get("appearance_id", item.get("name")))
            sigma = item.get("sigma", item.get("sigma_px"))
        elif isinstance(item, (int, float, np.number)):
            appearance_id = f"{role}_{index:03d}"
            sigma = item
        else:
            raise BlobAdapterError(f"{role} appearance entry {index} must provide id and sigma")
        if appearance_id in (None, ""):
            raise BlobAdapterError(f"{role} appearance entry {index} has no id")
        try:
            sigma_value = float(sigma)
        except (TypeError, ValueError) as exc:
            raise BlobAdapterError(f"{role} appearance entry {index} has invalid sigma") from exc
        if not np.isfinite(sigma_value) or sigma_value <= 0:
            raise BlobAdapterError(f"{role} appearance entry {index} sigma must be positive")
        records.append(AppearanceRecord(str(appearance_id), sigma_value))
    if len({item.appearance_id for item in records}) != len(records):
        raise BlobAdapterError(f"{role} appearance IDs must be unique")
    return tuple(records)


def load_appearance_set(
    path: Path | str,
    *,
    expected_sha256: str,
    role: str,
    manifest_base: Path | str | None = None,
    expected_count: int | None = None,
) -> AppearanceSet:
    """Read a hash-pinned train/eval appearance list."""

    resolved = _manifest_path(path, manifest_base=manifest_base, label=f"{role} appearance asset")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    expected = _require_sha(expected_sha256, label=f"{role} appearance asset")
    actual = sha256_file(resolved).lower()
    if actual != expected:
        raise BlobAdapterError(f"{role} appearance SHA mismatch: expected {expected}, got {actual}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BlobAdapterError(f"invalid {role} appearance asset: {exc}") from exc
    records = _parse_appearance_records(payload, role=role)
    if expected_count is not None and len(records) != int(expected_count):
        raise BlobAdapterError(f"{role} requires {expected_count} appearances, got {len(records)}")
    return AppearanceSet(role=str(role), path=resolved, sha256=actual, records=records)


def validate_train_eval_appearance_sets(
    train: AppearanceSet,
    evaluation: AppearanceSet,
    *,
    expected_train_count: int = 48,
    expected_eval_count: int = 8,
) -> dict[str, Any]:
    """Verify role, count, disjoint IDs/sigmas, and preserve fingerprints."""

    if str(train.role).lower() not in {"train", "training"}:
        raise BlobAdapterError(f"train appearance role is {train.role!r}")
    if str(evaluation.role).lower() not in {"eval", "evaluation", "test"}:
        raise BlobAdapterError(f"eval appearance role is {evaluation.role!r}")
    if len(train.records) != int(expected_train_count):
        raise BlobAdapterError(f"training appearance count must be {expected_train_count}")
    if len(evaluation.records) != int(expected_eval_count):
        raise BlobAdapterError(f"evaluation appearance count must be {expected_eval_count}")
    if set(train.appearance_ids) & set(evaluation.appearance_ids):
        raise BlobAdapterError("train/eval appearance IDs overlap")
    if any(np.isclose(left, right, rtol=0.0, atol=1e-12) for left in train.sigmas for right in evaluation.sigmas):
        raise BlobAdapterError("train/eval sigma values overlap")
    return {
        "train": train.metadata(),
        "eval": evaluation.metadata(),
        "disjoint_ids": True,
        "disjoint_sigmas": True,
    }


def _validate_image_size(image_size: int) -> int:
    try:
        value = int(image_size)
    except (TypeError, ValueError) as exc:
        raise BlobAdapterError("image_size must be an integer") from exc
    if value < 16 or value % 1:
        raise BlobAdapterError("image_size must be at least 16")
    return value


def _normalise_xy(points: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) / float(_validate_image_size(image_size) - 1)


def integer_grid_px(
    *,
    n_tx: int = N_TX,
    n_ty: int = N_TY,
    low_px: float = T_LOW_PX,
    high_px: float = T_HIGH_PX,
) -> np.ndarray:
    """The frozen 8x8 integer translation grid in ``(x, y)`` order."""

    n_tx, n_ty = int(n_tx), int(n_ty)
    if n_tx < 2 or n_ty < 2:
        raise BlobAdapterError("integer grid requires at least two points per axis")
    xs = np.linspace(float(low_px), float(high_px), n_tx, dtype=np.float64)
    ys = np.linspace(float(low_px), float(high_px), n_ty, dtype=np.float64)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


def dense_grid_px(
    *,
    n_grid: int = N_GRID,
    low_px: float = T_LOW_PX,
    high_px: float = T_HIGH_PX,
) -> np.ndarray:
    """The frozen 41x41 dense translation grid in ``(x, y)`` order."""

    n_grid = int(n_grid)
    if n_grid < 2:
        raise BlobAdapterError("dense grid requires at least two points per axis")
    axis = np.linspace(float(low_px), float(high_px), n_grid, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


def corners4_tids() -> tuple[int, ...]:
    return CORNERS4_TIDS


def corners4_coordinates() -> np.ndarray:
    grid = integer_grid_px()
    return grid[np.asarray(CORNERS4_TIDS, dtype=np.int64)]


def render_blob_at(
    centers_px: Any,
    *,
    sigma: float = BLOB_SIGMA,
    image_size: int = IMAGE_SIZE,
    supersample: int = SUPER_SAMPLE,
) -> dict[str, np.ndarray]:
    """Render one or more Gaussian blobs without importing legacy code."""

    centers = np.asarray(centers_px, dtype=np.float64).reshape(-1, 2)
    if centers.size == 0 or not np.all(np.isfinite(centers)):
        raise BlobAdapterError("centers_px must be a finite non-empty [n, 2] array")
    image_size = _validate_image_size(image_size)
    supersample = int(supersample)
    if supersample < 1:
        raise BlobAdapterError("supersample must be a positive integer")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise BlobAdapterError("sigma must be finite and positive")

    # Pillow is used only for deterministic Lanczos resampling.  It is not a
    # historical project import and keeps the exact uint8 renderer protocol.
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("blob renderer requires Pillow") from exc

    high = image_size * supersample
    yy, xx = np.mgrid[0:high, 0:high].astype(np.float64)
    images = np.empty((len(centers), image_size, image_size), dtype=np.uint8)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for index, (cx, cy) in enumerate(centers):
        dx = xx - float(cx) * supersample
        dy = yy - float(cy) * supersample
        values = np.exp(-(dx * dx + dy * dy) / (2.0 * (sigma * supersample) ** 2))
        high_image = Image.fromarray(np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        images[index] = np.asarray(
            high_image.resize((image_size, image_size), resample=resampling), dtype=np.uint8
        )
    return {
        "images": images,
        "xy_norm": _normalise_xy(centers, image_size),
        "t_px": np.array(centers, copy=True),
        "true_px": np.array(centers, copy=True),
    }


def render_blob_dataset(
    kind: str = "dense",
    *,
    sigma: float = BLOB_SIGMA,
    image_size: int = IMAGE_SIZE,
    n_grid: int = N_GRID,
    flatten: bool = True,
) -> dict[str, np.ndarray]:
    """Return a clean corners4 or dense blob dataset."""

    key = str(kind).strip().lower()
    if key in {"corners4", "g4", "support"}:
        centers = corners4_coordinates()
        pack = render_blob_at(centers, sigma=sigma, image_size=image_size)
        pack["tids"] = np.asarray(CORNERS4_TIDS, dtype=np.int64)
        pack["grid_shape"] = np.asarray((4,), dtype=np.int64)
        return pack
    if key not in {"dense", "g41", "grid"}:
        raise BlobAdapterError(f"unknown blob dataset kind {kind!r}")
    n_grid = int(n_grid)
    centers = dense_grid_px(n_grid=n_grid)
    pack = render_blob_at(centers, sigma=sigma, image_size=image_size)
    pack["tids"] = np.arange(len(centers), dtype=np.int64)
    if not flatten:
        pack["images"] = pack["images"].reshape(n_grid, n_grid, image_size, image_size)
        pack["xy_norm"] = pack["xy_norm"].reshape(n_grid, n_grid, 2)
        pack["t_px"] = pack["t_px"].reshape(n_grid, n_grid, 2)
        pack["true_px"] = pack["true_px"].reshape(n_grid, n_grid, 2)
        pack["grid_shape"] = np.asarray((n_grid, n_grid), dtype=np.int64)
    else:
        pack["grid_shape"] = np.asarray((n_grid, n_grid), dtype=np.int64)
    return pack


def corners4_pack(**kwargs: Any) -> dict[str, np.ndarray]:
    return render_blob_dataset("corners4", **kwargs)


def dense_pack(**kwargs: Any) -> dict[str, np.ndarray]:
    return render_blob_dataset("dense", **kwargs)


def render_training_appearances(
    appearance_set: AppearanceSet,
    *,
    image_size: int = IMAGE_SIZE,
) -> dict[str, Any]:
    """Render deterministic appearance-major, corners4-minor training order."""

    centers = corners4_coordinates()
    images: list[np.ndarray] = []
    true: list[np.ndarray] = []
    appearance_ids: list[str] = []
    for record in appearance_set.records:
        pack = render_blob_at(centers, sigma=record.sigma, image_size=image_size)
        images.append(pack["images"])
        true.append(pack["true_px"])
        appearance_ids.extend([record.appearance_id] * len(centers))
    return {
        "images": np.concatenate(images, axis=0),
        "true_px": np.concatenate(true, axis=0),
        "appearance_ids": np.asarray(appearance_ids, dtype="U"),
        "support_tids": np.tile(np.asarray(CORNERS4_TIDS, dtype=np.int64), len(appearance_set.records)),
        "appearance_set": appearance_set.metadata(),
        "role": "train",
        "order": "appearance_major_then_corners4_tid",
    }


def render_evaluation_appearances(
    appearance_set: AppearanceSet,
    *,
    n_grid: int = N_GRID,
    image_size: int = IMAGE_SIZE,
) -> dict[str, Any]:
    """Render deterministic appearance-major, dense-grid-minor eval order."""

    centers = dense_grid_px(n_grid=n_grid)
    images: list[np.ndarray] = []
    true: list[np.ndarray] = []
    appearance_ids: list[str] = []
    for record in appearance_set.records:
        pack = render_blob_at(centers, sigma=record.sigma, image_size=image_size)
        images.append(pack["images"])
        true.append(pack["true_px"])
        appearance_ids.extend([record.appearance_id] * len(centers))
    return {
        "images": np.concatenate(images, axis=0),
        "true_px": np.concatenate(true, axis=0),
        "appearance_ids": np.asarray(appearance_ids, dtype="U"),
        "grid_shape": np.asarray((len(appearance_set.records), n_grid, n_grid), dtype=np.int64),
        "appearance_set": appearance_set.metadata(),
        "role": "eval",
        "order": "appearance_major_then_dense_grid",
    }


@dataclass(frozen=True)
class AppearanceEvalStream:
    """Lazy evaluation stream; one appearance is rendered at a time."""

    appearance_set: AppearanceSet
    n_grid: int = N_GRID
    image_size: int = IMAGE_SIZE

    def __post_init__(self) -> None:
        if int(self.n_grid) < 2:
            raise BlobAdapterError("evaluation n_grid must be at least two")

    def __len__(self) -> int:
        return int(len(self.appearance_set.records) * int(self.n_grid) * int(self.n_grid))

    def metadata(self) -> dict[str, Any]:
        return {
            "role": "eval",
            "appearance_set": self.appearance_set.metadata(),
            "grid_shape": [len(self.appearance_set.records), int(self.n_grid), int(self.n_grid)],
            "order": "appearance_major_then_dense_grid",
            "lazy": True,
        }

    def iter_batches(self, batch_size: int = 256) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise BlobAdapterError("batch_size must be positive")
        centers = dense_grid_px(n_grid=int(self.n_grid))
        for record in self.appearance_set.records:
            pack = render_blob_at(centers, sigma=record.sigma, image_size=int(self.image_size))
            for start in range(0, len(centers), batch_size):
                stop = min(start + batch_size, len(centers))
                yield pack["images"][start:stop], pack["true_px"][start:stop]


def uint8_to_nchw(images: Any) -> Any:
    """Convert grayscale uint8 images to the historical 3-channel float tensor."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("uint8_to_nchw requires PyTorch") from exc
    if torch.is_tensor(images):
        tensor = images
        if tensor.dtype == torch.uint8:
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(images)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim == 3:
        if array.shape[-2:] != (array.shape[-2], array.shape[-1]):
            raise BlobAdapterError(f"images must have [n, h, w], got {array.shape}")
        array = np.ascontiguousarray(array)
        tensor = torch.from_numpy(array).unsqueeze(1).repeat(1, 3, 1, 1).contiguous()
    elif array.ndim == 4 and array.shape[1] in {1, 3}:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1).contiguous()
    else:
        raise BlobAdapterError(f"images must have [n, h, w] or [n, c, h, w], got {array.shape}")
    return tensor.float().div(255.0)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("checkpoint/model adapters require PyTorch") from exc
    return torch


def inspect_resnet18_2d_checkpoint(
    path: Path | str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Safe-inspect a standard torchvision ResNet18 checkpoint head."""

    torch = _torch()
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest.lower() != str(expected_sha256).strip().lower():
        raise BlobAdapterError(f"checkpoint sha256 mismatch for {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise BlobAdapterError(f"weights_only checkpoint load failed: {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BlobAdapterError("checkpoint payload must be a mapping")
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise BlobAdapterError("checkpoint state must be a mapping")
    if "fc.weight" not in state or "fc.bias" not in state:
        raise BlobAdapterError("checkpoint must expose standard fc.weight/fc.bias keys")
    weight = state["fc.weight"]
    bias = state["fc.bias"]
    if not torch.is_tensor(weight) or not torch.is_tensor(bias):
        raise BlobAdapterError("checkpoint fc tensors are not tensors")
    if tuple(weight.shape) != (2, 512) or tuple(bias.shape) != (2,):
        raise BlobAdapterError(
            "2D loader refuses non-2D head: "
            f"fc.weight={tuple(weight.shape)}, fc.bias={tuple(bias.shape)}"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "payload_keys": [str(key) for key in payload.keys()],
        "head_dim": 2,
        "head_in": 512,
        "arch": str(payload.get("arch", "resnet18")),
    }


def load_resnet18_2d_checkpoint(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    device: str | Any = "cpu",
    eval_mode: bool = True,
) -> Any:
    """Construct torchvision ResNet18 and load only a verified 2D head."""

    torch = _torch()
    info = inspect_resnet18_2d_checkpoint(path, expected_sha256=expected_sha256)
    try:
        import torchvision.models as models
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("2D checkpoint loader requires torchvision") from exc
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    resolved = Path(info["path"])
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload.get("state_dict", payload))
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise BlobAdapterError(f"standard ResNet18 state load failed: {exc}") from exc
    model.to(device)
    if eval_mode:
        model.eval()
    return model


def _predict_px(model: Any, images: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    torch = _torch()
    array = np.asarray(images)
    if array.ndim == 4:
        array = array.reshape(-1, array.shape[-2], array.shape[-1])
    if array.ndim != 3:
        raise BlobAdapterError(f"images must flatten to [n, h, w], got {array.shape}")
    if int(batch_size) < 1:
        raise BlobAdapterError("batch_size must be positive")
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    for start in range(0, len(array), int(batch_size)):
        batch = uint8_to_nchw(array[start : start + int(batch_size)]).to(device)
        output = model(batch)
        if not torch.is_tensor(output) or output.ndim != 2 or output.shape[-1] != 2:
            raise BlobAdapterError(f"2D model output must have [n, 2], got {getattr(output, 'shape', None)}")
        chunks.append(output.detach().float().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(chunks, axis=0) * COORD_SCALE


def evaluate_blob_model(
    model: Any,
    exposure: int = 0,
    *,
    dense: Mapping[str, Any] | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate a model on the clean dense blob grid at one exposure."""

    if isinstance(dense, AppearanceEvalStream):
        prediction_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        for images_chunk, true_chunk in dense.iter_batches(batch_size=batch_size):
            prediction_chunks.append(_predict_px(model, images_chunk, batch_size=batch_size))
            target_chunks.append(np.asarray(true_chunk, dtype=np.float64).reshape(-1, 2))
        if not prediction_chunks:
            raise BlobAdapterError("lazy eval stream produced no batches")
        pred = np.concatenate(prediction_chunks, axis=0)
        true = np.concatenate(target_chunks, axis=0)
        pack: Mapping[str, Any] = dense.metadata()
    else:
        pack = dict(dense) if dense is not None else dense_pack()
        images = np.asarray(pack["images"])
        true = np.asarray(pack["true_px"], dtype=np.float64).reshape(-1, 2)
        pred = _predict_px(model, images, batch_size=batch_size)
    if len(pred) != len(true):
        raise BlobAdapterError("dense prediction/target row count mismatch")
    matrix, bias = fit_affine(pred, true)
    affine_pred = pred @ matrix.T + bias
    raw_mae = float(np.mean(np.linalg.norm(pred - true, axis=1)))
    affine_mae = float(np.mean(np.linalg.norm(affine_pred - true, axis=1)))
    grid_shape = np.asarray(pack.get("grid_shape", (N_GRID, N_GRID))).reshape(-1).astype(int).tolist()
    return {
        "exposure": int(exposure),
        "raw_mae_px": raw_mae,
        "affine_removed_mae_px": affine_mae,
        "n_eval": int(len(true)),
        "grid_shape": grid_shape,
        "role": str(pack.get("role", "eval")),
        "appearance_set": pack.get("appearance_set"),
        "appearance_count": int(len(pack.get("appearance_set", {}).get("appearance_ids", [])))
        if isinstance(pack.get("appearance_set"), Mapping)
        else None,
    }


@dataclass(frozen=True)
class BlobBatchStream:
    """Lazy deterministic stream that can supply any requested exposure."""

    max_exposure: int = 3000
    batch_size: int = 64
    sigma: float = BLOB_SIGMA
    support_images: np.ndarray | None = None
    appearance_set: AppearanceSet | None = None
    seed: int = 20260821
    role: str = "legacy_train"
    order_fingerprint: str | None = None
    train_eval_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if int(self.max_exposure) < 1:
            raise BlobAdapterError("max_exposure must be positive")
        if int(self.batch_size) < 1:
            raise BlobAdapterError("batch_size must be positive")

    def __iter__(self) -> Iterator[Any]:
        images = (
            np.asarray(self.support_images)
            if self.support_images is not None
            else np.asarray(render_training_appearances(self.appearance_set)["images"])
            if self.appearance_set is not None
            else np.asarray(corners4_pack(sigma=self.sigma)["images"])
        )
        if images.ndim != 3 or images.shape[0] < 1:
            raise BlobAdapterError("support_images must have [n, h, w]")
        # The order is a pure function of the declared seed and image count;
        # no hidden global RNG state is consulted.
        order = np.random.default_rng(int(self.seed)).permutation(len(images))
        for batch_index in range(int(self.max_exposure)):
            positions = np.arange(int(self.batch_size), dtype=np.int64) + batch_index * int(self.batch_size)
            ids = order[positions % len(order)]
            yield uint8_to_nchw(images[ids])

    def metadata(self) -> dict[str, Any]:
        if self.support_images is not None:
            count = int(len(self.support_images))
            appearance = None
        elif self.appearance_set is not None:
            count = int(len(self.appearance_set.records) * len(CORNERS4_TIDS))
            appearance = self.appearance_set.metadata()
        else:
            count = len(CORNERS4_TIDS)
            appearance = None
        order = np.random.default_rng(int(self.seed)).permutation(count)
        order_fingerprint = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
        return {
            "role": self.role,
            "seed": int(self.seed),
            "max_exposure": int(self.max_exposure),
            "batch_size": int(self.batch_size),
            "n_images": count,
            "appearance_set": appearance,
            "train_eval_split": None if self.train_eval_metadata is None else dict(self.train_eval_metadata),
            "order": "seeded_permutation_with_cyclic_batches",
            "order_fingerprint": order_fingerprint,
        }


def make_exposure_batches(
    *,
    max_exposure: int = 3000,
    batch_size: int = 64,
    sigma: float = BLOB_SIGMA,
    appearance_set: AppearanceSet | None = None,
    seed: int = 20260821,
    train_eval_metadata: Mapping[str, Any] | None = None,
) -> BlobBatchStream:
    return BlobBatchStream(
        max_exposure=max_exposure,
        batch_size=batch_size,
        sigma=sigma,
        appearance_set=appearance_set,
        seed=seed,
        role="train" if appearance_set is not None else "legacy_train",
        train_eval_metadata=train_eval_metadata,
    )


# Explicit adapter names for protocol JSON.  They intentionally accept only
# JSON-friendly keyword arguments and do not read any unlisted project asset.
def model_adapter(*, checkpoint_path: Path | str, expected_sha256: str | None = None, device: str = "cpu") -> Any:
    return load_resnet18_2d_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        device=device,
        eval_mode=True,
    )


def data_adapter(
    *,
    train_appearance_asset_path: Path | str | None = None,
    train_appearance_sha256: str | None = None,
    eval_appearance_asset_path: Path | str | None = None,
    eval_appearance_sha256: str | None = None,
    manifest_base: Path | str | None = None,
    max_exposure: int = 3000,
    batch_size: int = 64,
    sigma: float = BLOB_SIGMA,
    seed: int = 20260821,
    expected_train_count: int = 48,
    expected_eval_count: int = 8,
) -> BlobBatchStream:
    """Protocol data adapter: 48 train sigmas x corners4, disjoint from eval."""

    if train_appearance_asset_path is None:
        raise BlobAdapterError("formal BN data adapter requires train_appearance_asset_path")
    train = load_appearance_set(
        train_appearance_asset_path,
        expected_sha256=_require_sha(train_appearance_sha256, label="train appearance asset"),
        role="train",
        manifest_base=manifest_base,
        expected_count=expected_train_count,
    )
    if eval_appearance_asset_path is None:
        raise BlobAdapterError("formal BN data adapter requires eval_appearance_asset_path")
    evaluation = load_appearance_set(
        eval_appearance_asset_path,
        expected_sha256=_require_sha(eval_appearance_sha256, label="eval appearance asset"),
        role="eval",
        manifest_base=manifest_base,
        expected_count=expected_eval_count,
    )
    validate_train_eval_appearance_sets(
        train,
        evaluation,
        expected_train_count=expected_train_count,
        expected_eval_count=expected_eval_count,
    )
    return make_exposure_batches(
        max_exposure=max_exposure,
        batch_size=batch_size,
        sigma=sigma,
        appearance_set=train,
        seed=seed,
        train_eval_metadata=validate_train_eval_appearance_sets(
            train,
            evaluation,
            expected_train_count=expected_train_count,
            expected_eval_count=expected_eval_count,
        ),
    )


def evaluator_adapter(
    *,
    train_appearance_asset_path: Path | str | None = None,
    train_appearance_sha256: str | None = None,
    eval_appearance_asset_path: Path | str | None = None,
    eval_appearance_sha256: str | None = None,
    manifest_base: Path | str | None = None,
    batch_size: int = 256,
    sigma: float = BLOB_SIGMA,
    expected_train_count: int = 48,
    expected_eval_count: int = 8,
    n_grid: int = N_GRID,
):
    if train_appearance_asset_path is None or eval_appearance_asset_path is None:
        raise BlobAdapterError("formal BN evaluator requires both train/eval appearance assets")
    train = load_appearance_set(
        train_appearance_asset_path,
        expected_sha256=_require_sha(train_appearance_sha256, label="train appearance asset"),
        role="train",
        manifest_base=manifest_base,
        expected_count=expected_train_count,
    )
    evaluation = load_appearance_set(
        eval_appearance_asset_path,
        expected_sha256=_require_sha(eval_appearance_sha256, label="eval appearance asset"),
        role="eval",
        manifest_base=manifest_base,
        expected_count=expected_eval_count,
    )
    split_metadata = validate_train_eval_appearance_sets(
        train,
        evaluation,
        expected_train_count=expected_train_count,
        expected_eval_count=expected_eval_count,
    )
    dense = AppearanceEvalStream(evaluation, n_grid=n_grid)

    def evaluate(model: Any, exposure: int) -> dict[str, Any]:
        result = evaluate_blob_model(model, exposure, dense=dense, batch_size=batch_size)
        result["train_eval_split"] = split_metadata
        return result

    return evaluate


__all__ = [
    "AppearanceRecord",
    "AppearanceSet",
    "AppearanceEvalStream",
    "BLOB_SIGMA",
    "BlobAdapterError",
    "BlobBatchStream",
    "COORD_SCALE",
    "CORNERS4_TIDS",
    "IMAGE_SIZE",
    "N_GRID",
    "T_HIGH_PX",
    "T_LOW_PX",
    "corners4_coordinates",
    "corners4_pack",
    "corners4_tids",
    "data_adapter",
    "dense_grid_px",
    "dense_pack",
    "evaluate_blob_model",
    "evaluator_adapter",
    "inspect_resnet18_2d_checkpoint",
    "integer_grid_px",
    "load_resnet18_2d_checkpoint",
    "make_exposure_batches",
    "model_adapter",
    "load_appearance_set",
    "render_blob_at",
    "render_blob_dataset",
    "render_evaluation_appearances",
    "render_training_appearances",
    "uint8_to_nchw",
    "validate_train_eval_appearance_sets",
]
