"""Canonical renderer, support bank and deterministic batch streams.

The training API in :mod:`crossover.training` only receives the support-bank
arrays.  Dense arrays live behind :func:`load_dense_cache` and are intentionally
loaded by the independent evaluator after model predictions have been frozen.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .protocol import SUPPORT_NAMES, canonical_json_bytes, load_protocol, protocol_hash


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: np.ndarray, name: str = "array") -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes({"shape": list(value.shape)}))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def grid_points_px(protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    cfg = load_protocol() if protocol is None else dict(protocol)
    low, high = float(cfg["safe_box_px"]["low"]), float(cfg["safe_box_px"]["high"])
    levels = int(cfg["grid"]["levels"])
    xs = np.linspace(low, high, levels, dtype=np.float64)
    ys = np.linspace(low, high, levels, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.ascontiguousarray(np.stack((xx, yy), axis=-1).reshape(-1, 2), dtype=np.float64)


def support_tids(name: str, protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    cfg = load_protocol() if protocol is None else dict(protocol)
    key = str(name)
    if key not in cfg["supports"]:
        raise KeyError(f"unknown support set {name!r}; expected {SUPPORT_NAMES}")
    return np.asarray(cfg["supports"][key], dtype=np.int64).copy()


def support_points_px(name: str, protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    points = grid_points_px(protocol)
    tids = support_tids(name, protocol)
    return np.ascontiguousarray(points[tids], dtype=np.float64)


def dense_points_px(protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    cfg = load_protocol() if protocol is None else dict(protocol)
    low, high = float(cfg["safe_box_px"]["low"]), float(cfg["safe_box_px"]["high"])
    n = int(cfg["dense_grid"]["n"])
    xs = np.linspace(low, high, n, dtype=np.float64)
    ys = np.linspace(low, high, n, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.ascontiguousarray(np.stack((xx, yy), axis=-1).reshape(-1, 2), dtype=np.float64)


@lru_cache(maxsize=8)
def _hi_grid(image_size: int, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    high = int(image_size) * int(supersample)
    yy, xx = np.mgrid[0:high, 0:high].astype(np.float64)
    return np.ascontiguousarray(xx), np.ascontiguousarray(yy)


def _render_one(point: Sequence[float], cfg: Mapping[str, Any]) -> np.ndarray:
    image_size = int(cfg["image_size"])
    supersample = int(cfg["renderer"]["supersample"])
    sigma = float(cfg["renderer"]["sigma"])
    x, y = float(point[0]), float(point[1])
    xx, yy = _hi_grid(image_size, supersample)
    scale = float(supersample)
    values = np.exp(-(((xx - x * scale) ** 2) + ((yy - y * scale) ** 2)) / (2.0 * (sigma * scale) ** 2))
    image = Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8), mode="L")
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((image_size, image_size), resample=resampling.LANCZOS)
    result = np.asarray(image, dtype=np.uint8)
    if result.shape != (image_size, image_size) or result.dtype != np.uint8:
        raise RuntimeError(f"renderer invariant failed: shape={result.shape}, dtype={result.dtype}")
    return np.ascontiguousarray(result)


def render_points(
    points_px: np.ndarray | Sequence[Sequence[float]],
    *,
    workers: int = 1,
    protocol: Mapping[str, Any] | None = None,
) -> np.ndarray:
    cfg = load_protocol() if protocol is None else dict(protocol)
    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"points_px must be finite [N,2], got {points.shape}")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    if len(points) == 0:
        return np.empty((0, int(cfg["image_size"]), int(cfg["image_size"])), dtype=np.uint8)
    if int(workers) == 1 or len(points) == 1:
        images = [_render_one(point, cfg) for point in points]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            images = list(pool.map(lambda point: _render_one(point, cfg), points, chunksize=16))
    return np.ascontiguousarray(np.stack(images, axis=0), dtype=np.uint8)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
    temporary.replace(path)


def materialize_cache(out_dir: str | Path, *, workers: int = 1, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create portable support and dense renderer assets with a manifest."""

    cfg = load_protocol() if protocol is None else dict(protocol)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    bank_points = grid_points_px(cfg)
    dense_points = dense_points_px(cfg)
    bank_images = render_points(bank_points, workers=workers, protocol=cfg)
    dense_images = render_points(dense_points, workers=workers, protocol=cfg)
    _atomic_npy(output / "support_bank_points.npy", bank_points)
    _atomic_npy(output / "support_bank_images.npy", bank_images)
    _atomic_npy(output / "dense_points.npy", dense_points)
    _atomic_npy(output / "dense_images.npy", dense_images)
    files = {}
    for path in sorted(output.glob("*.npy")):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        files[path.name] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    cfg_protocol_hash = protocol_hash(payload=cfg)
    support_bank_manifest = {}
    for name in SUPPORT_NAMES:
        tids = support_tids(name, cfg)
        image_hash = hash_array(bank_images[tids], "support_images")
        point_hash = hash_array(bank_points[tids], "support_points")
        support_bank_manifest[name] = {
            "tids": tids.tolist(),
            "count": int(len(tids)),
            "image_hash": image_hash,
            "point_hash": point_hash,
            "input_hash": image_hash + point_hash,
        }
    manifest = {
        "schema_version": 1,
        "kind": "neural_affine_crossover_cache",
        "protocol_id": cfg["protocol_id"],
        "protocol_hash": cfg_protocol_hash,
        "files": files,
        "support_bank": support_bank_manifest,
        "dense_count": int(len(dense_points)),
        "bank_hash": hash_array(bank_images, "support_bank_images") + hash_array(bank_points, "support_bank_points"),
    }
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest


def load_support_bank(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(cache_dir)
    images = np.load(root / "support_bank_images.npy", allow_pickle=False, mmap_mode="r")
    points = np.load(root / "support_bank_points.npy", allow_pickle=False, mmap_mode="r")
    if images.shape[0] != 64 or points.shape != (64, 2) or images.dtype != np.uint8:
        raise ValueError(f"support bank shape/dtype invalid: {images.shape}/{images.dtype}, {points.shape}")
    return images, points


def load_dense_cache(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(cache_dir)
    images = np.load(root / "dense_images.npy", allow_pickle=False, mmap_mode="r")
    points = np.load(root / "dense_points.npy", allow_pickle=False, mmap_mode="r")
    if images.shape[0] != 41 * 41 or points.shape != (41 * 41, 2) or images.dtype != np.uint8:
        raise ValueError("dense cache shape/dtype invalid")
    return images, points


def images_to_tensor(images_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert renderer bytes to NCHW float32 using the frozen CPU path."""

    array = images_uint8.detach().cpu().numpy() if torch.is_tensor(images_uint8) else np.asarray(images_uint8)
    if array.ndim == 2:
        array = array[None, None]
    elif array.ndim == 3:
        array = array[:, None]
    elif array.ndim == 4 and array.shape[-1] in (1, 3):
        array = np.transpose(array, (0, 3, 1, 2))
    if array.ndim != 4 or array.shape[1] not in (1, 3):
        raise ValueError(f"expected [N,H,W] or NCHW grayscale/RGB images, got {array.shape}")
    # ``mmap_mode='r'`` support/dense arrays are read-only; make an explicit
    # CPU copy before handing bytes to Torch so no undefined writable alias is
    # created.
    result = torch.from_numpy(np.array(array, dtype=np.uint8, copy=True, order="C")).to(dtype=torch.float32).div_(255.0)
    if result.shape[1] == 1:
        result = result.expand(-1, 3, -1, -1).contiguous()
    return result


class BalancedBatchStream:
    """Deterministic shuffled epochs with exact batch-size tiling.

    A fresh instance with the same ``seed`` and ``count`` produces the exact
    same index sequence, which is how head-only and full runs share batches.
    """

    def __init__(self, count: int, batch_size: int, seed: int) -> None:
        if int(count) <= 0 or int(batch_size) <= 0:
            raise ValueError("count and batch_size must be positive")
        self.count = int(count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self._order = np.empty(0, dtype=np.int64)

    def _start_epoch(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch * 1000003)
        self._order = rng.permutation(self.count).astype(np.int64)
        self.cursor = 0
        self.epoch += 1

    def next_indices(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        remaining = self.batch_size
        while remaining:
            if self._order.size == 0 or self.cursor >= self._order.size:
                self._start_epoch()
            take = min(remaining, int(self._order.size - self.cursor))
            parts.append(self._order[self.cursor : self.cursor + take])
            self.cursor += int(take)
            remaining -= int(take)
        # The tail of one permutation is followed by the head of the next
        # permutation.  We never tile a tail, so a complete stream of
        # ``batches * batch_size`` indices differs by at most one per tid.
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.int64)

    def state(self) -> dict[str, Any]:
        return {"count": self.count, "batch_size": self.batch_size, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor, "order": self._order.tolist()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        if (int(state["count"]), int(state["batch_size"]), int(state["seed"])) != (self.count, self.batch_size, self.seed):
            raise ValueError("batch stream metadata mismatch")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._order = np.asarray(state.get("order", []), dtype=np.int64)

    def digest(self, batches: int = 16) -> str:
        probe = BalancedBatchStream(self.count, self.batch_size, self.seed)
        digest = hashlib.sha256()
        for _ in range(int(batches)):
            digest.update(probe.next_indices().tobytes(order="C"))
        return digest.hexdigest()


def materialize_batch_stream(
    path: str | Path,
    *,
    count: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    """Materialize one paired stream and verify global balance.

    The formal matrix uses 3000x64 indices.  The resulting file is shared by
    head-only and full conditions for the same support/seed.
    """

    stream = BalancedBatchStream(count, batch_size, seed)
    schedule = np.stack([stream.next_indices() for _ in range(int(batches))], axis=0).astype(np.int64)
    counts = np.bincount(schedule.reshape(-1), minlength=int(count))
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError(f"balanced stream invariant failed: {counts.tolist()}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, schedule, allow_pickle=False)
    temporary.replace(target)
    digest = sha256_file(target)
    manifest = {
        "schema_version": 1,
        "kind": "paired_batch_stream",
        "path": target.name,
        "count": int(count),
        "batch_size": int(batch_size),
        "batches": int(batches),
        "seed": int(seed),
        "shape": list(schedule.shape),
        "sha256": digest,
        "counts": counts.tolist(),
        "balanced_max_minus_min": int(counts.max() - counts.min()),
    }
    manifest_path = target.with_suffix(".json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def load_batch_stream(path: str | Path, *, count: int, batch_size: int, batches: int | None = None) -> np.ndarray:
    schedule = np.load(Path(path), allow_pickle=False, mmap_mode="r")
    if schedule.ndim != 2 or schedule.shape[1] != int(batch_size) or schedule.dtype != np.int64 or (schedule.size and (int(np.max(schedule)) >= int(count) or int(np.min(schedule)) < 0)):
        raise ValueError(f"invalid batch stream {path}: shape={schedule.shape}, dtype={schedule.dtype}")
    if batches is not None and schedule.shape[0] < int(batches):
        raise ValueError(f"batch stream has {schedule.shape[0]} batches; {batches} required")
    return schedule


__all__ = [
    "BalancedBatchStream",
    "dense_points_px",
    "grid_points_px",
    "hash_array",
    "images_to_tensor",
    "load_batch_stream",
    "load_dense_cache",
    "load_support_bank",
    "materialize_cache",
    "materialize_batch_stream",
    "render_points",
    "sha256_file",
    "support_points_px",
    "support_tids",
]
