"""Self-contained canonical data generation for the S1 Neural Affine task.

The module intentionally has no dependency on the historical experiment
packages.  It is the single source of truth for the four-corner support set,
the official 41 x 41 dense grid, the sigma=6 blob renderer, and the cache
manifest used by the clean reproduction.

The public API is deliberately small:

``support_points_px``
    Return the four canonical corner points in pixel coordinates.
``dense_points_px``
    Return the row-major 41 x 41 evaluation grid.
``render_points``
    Render arbitrary points with optional thread parallelism.
``materialize_cache`` / ``load_cache``
    Write/read an auditable ``.npy`` + ``.npz`` cache.
``probe_indices`` / ``dataset_hash``
    Return the frozen 25-point probe and the full-cache SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = MODULE_ROOT / "protocol.json"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Serialize protocol/manifest fragments deterministically."""

    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash a file without loading the complete file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the non-negotiable S1 geometry/rendering invariants."""

    required = {
        "image_size",
        "coord_scale",
        "safe_box_px",
        "anchor_grid",
        "dense_grid",
        "renderer",
        "probes",
        "cache",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"protocol missing required keys: {missing}")

    image_size = int(payload["image_size"])
    coord_scale = float(payload["coord_scale"])
    if image_size != 224 or coord_scale != 223.0:
        raise ValueError(f"S1 image/coordinate scale drifted: {image_size=}, {coord_scale=}")

    box = payload["safe_box_px"]
    low, high = float(box["low"]), float(box["high"])
    if (low, high) != (59.0, 164.0):
        raise ValueError(f"S1 safe box drifted: {(low, high)}")

    anchors = payload["anchor_grid"]
    if (int(anchors["nx"]), int(anchors["ny"])) != (8, 8):
        raise ValueError("S1 anchor grid must be 8x8")
    if [int(value) for value in anchors["tids"]] != [0, 7, 56, 63]:
        raise ValueError("S1 support tids must be [0, 7, 56, 63]")

    dense = payload["dense_grid"]
    if int(dense["n"]) != 41 or dense.get("order") != "tx_major_ty_minor":
        raise ValueError("S1 dense grid must be row-major 41x41 in tx-major/ty-minor order")

    renderer = payload["renderer"]
    if float(renderer["sigma"]) != 6.0 or int(renderer["supersample"]) != 4:
        raise ValueError("S1 renderer must use sigma=6 and supersample=4")
    if renderer.get("downsample") != "PIL.Image.Resampling.LANCZOS":
        raise ValueError("S1 renderer must use PIL Lanczos downsampling")
    if renderer.get("dtype") != "uint8":
        raise ValueError("S1 rendered images must be uint8")

    probes = [int(value) for value in payload["probes"]["indices"]]
    if len(probes) != 25 or len(set(probes)) != 25 or any(value < 0 or value >= 41 * 41 for value in probes):
        raise ValueError("S1 probe list must contain 25 unique dense-grid indices")
    return dict(payload)


@lru_cache(maxsize=4)
def load_protocol(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and validate the frozen JSON protocol."""

    protocol_path = Path(path) if path is not None else PROTOCOL_PATH
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"protocol must be a JSON object: {protocol_path}")
    return _validate_protocol(payload)


def _protocol(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    return load_protocol(None if path is None else str(Path(path).resolve()))


def _grid_points(n: int, low: float, high: float) -> np.ndarray:
    """Build the historical ``indexing='ij'`` grid in a stable dtype/order."""

    xs = np.linspace(float(low), float(high), int(n), dtype=np.float64)
    ys = np.linspace(float(low), float(high), int(n), dtype=np.float64)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


def support_points_px(protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the four canonical 8x8-grid corner points in pixel units."""

    payload = _protocol() if protocol is None else _validate_protocol(protocol)
    box = payload["safe_box_px"]
    anchors = payload["anchor_grid"]
    grid = _grid_points(int(anchors["nx"]), float(box["low"]), float(box["high"]))
    return np.ascontiguousarray(grid[np.asarray(anchors["tids"], dtype=np.int64)], dtype=np.float64)


def dense_points_px(protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the canonical 41x41 dense grid in row-major order."""

    payload = _protocol() if protocol is None else _validate_protocol(protocol)
    box = payload["safe_box_px"]
    n = int(payload["dense_grid"]["n"])
    return np.ascontiguousarray(_grid_points(n, float(box["low"]), float(box["high"])), dtype=np.float64)


def probe_indices(protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the frozen 25 dense-grid probe indices as int64."""

    payload = _protocol() if protocol is None else _validate_protocol(protocol)
    return np.asarray(payload["probes"]["indices"], dtype=np.int64).copy()


@lru_cache(maxsize=8)
def _high_resolution_grid(image_size: int, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    high = int(image_size) * int(supersample)
    yy, xx = np.mgrid[0:high, 0:high].astype(np.float64)
    return np.ascontiguousarray(xx), np.ascontiguousarray(yy)


def _render_blob_one(point: Sequence[float], payload: Mapping[str, Any]) -> np.ndarray:
    image_size = int(payload["image_size"])
    renderer = payload["renderer"]
    supersample = int(renderer["supersample"])
    sigma = float(renderer["sigma"])
    x, y = float(point[0]), float(point[1])
    xx, yy = _high_resolution_grid(image_size, supersample)
    scale = float(supersample)
    dx = xx - x * scale
    dy = yy - y * scale
    values = np.exp(-(dx * dx + dy * dy) / (2.0 * (sigma * scale) ** 2))
    image = Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8), mode="L")
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((image_size, image_size), resample=resampling.LANCZOS)
    output = np.asarray(resized, dtype=np.uint8)
    if output.shape != (image_size, image_size) or output.dtype != np.uint8 or output.ndim != 2:
        raise RuntimeError(f"renderer invariant failed: {output.shape} {output.dtype}")
    return np.ascontiguousarray(output)


def _normalise_points(points_px: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_px must have shape [N,2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_px contains non-finite coordinates")
    return np.ascontiguousarray(points)


def render_points(
    points_px: np.ndarray | Sequence[Sequence[float]],
    workers: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Render points in stable input order as ``uint8[N,224,224]``.

    Thread parallelism is deliberately at the outer sample level.  Each
    worker uses the same cached high-resolution coordinate grid, and
    ``executor.map`` preserves the point order needed for deterministic
    hashes and checkpoint inputs.
    """

    payload = _protocol() if protocol is None else _validate_protocol(protocol)
    points = _normalise_points(points_px)
    count = len(points)
    image_size = int(payload["image_size"])
    images = np.empty((count, image_size, image_size), dtype=np.uint8)
    if count == 0:
        return images

    n_workers = int(payload["renderer"].get("workers", 1) if workers is None else workers)
    if n_workers < 1:
        raise ValueError(f"workers must be >= 1, got {n_workers}")

    if n_workers == 1 or count == 1:
        for index, point in enumerate(points):
            images[index] = _render_blob_one(point, payload)
        return images

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for index, image in enumerate(pool.map(lambda point: _render_blob_one(point, payload), points, chunksize=16)):
            images[index] = image
    return images


def _array_hash(digest: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json({"shape": list(arr.shape)}))
    digest.update(b"\0")
    digest.update(arr.tobytes(order="C"))


def dataset_hash(images: np.ndarray, points: np.ndarray) -> str:
    """Return the full deterministic SHA-256 for an image/point cache."""

    image_array = np.asarray(images)
    point_array = _normalise_points(points)
    if image_array.ndim != 3 or image_array.shape[0] != point_array.shape[0]:
        raise ValueError(f"images/points length mismatch: {image_array.shape}, {point_array.shape}")
    digest = hashlib.sha256()
    digest.update(b"neural-affine-s1-clean-dataset-v1\0")
    _array_hash(digest, "points", point_array)
    _array_hash(digest, "images", image_array)
    return digest.hexdigest()


def sample_sha256(image: np.ndarray, point: Sequence[float]) -> str:
    """Hash one rendered image together with its target point."""

    digest = hashlib.sha256()
    digest.update(b"neural-affine-s1-clean-sample-v1\0")
    _array_hash(digest, "point", _normalise_points(np.asarray(point, dtype=np.float64).reshape(1, 2))[0])
    _array_hash(digest, "image", np.asarray(image))
    return digest.hexdigest()


def sample_hashes(images: np.ndarray, points: np.ndarray) -> list[str]:
    image_array = np.asarray(images)
    point_array = _normalise_points(points)
    if image_array.ndim != 3 or image_array.shape[0] != len(point_array):
        raise ValueError(f"images/points length mismatch: {image_array.shape}, {point_array.shape}")
    return [sample_sha256(image, point) for image, point in zip(image_array, point_array)]


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(path)


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_entry(path: Path, array: np.ndarray | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if array is not None:
        entry.update({"shape": list(array.shape), "dtype": str(array.dtype)})
    return entry


def materialize_cache(
    out_dir: Path | str,
    workers: int | None = None,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render and atomically materialize the canonical dense/support cache.

    The primary portable files are ``images.npy`` and ``points.npy``.  The
    support arrays and a compressed ``cache.npz`` bundle are written as well,
    so a cloud run can choose either memory-mapped NPY or a single NPZ input.
    ``manifest.json`` records every file hash, every sample hash, the frozen
    25 probes, and the full dataset hash.
    """

    payload = _protocol() if protocol is None else _validate_protocol(protocol)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    dense_points = dense_points_px(payload)
    support_points = support_points_px(payload)
    dense_images = render_points(dense_points, workers=workers, protocol=payload)
    support_images = render_points(support_points, workers=workers, protocol=payload)

    expected_support = dense_points[[0, 40, 1640, 1680]]
    if not np.array_equal(support_points, expected_support):
        raise RuntimeError("support points are not the four dense-grid corners")
    dense_probe_indices = probe_indices(payload)
    probe_images = dense_images[dense_probe_indices]
    probe_points = dense_points[dense_probe_indices]

    images_path = output / "images.npy"
    points_path = output / "points.npy"
    support_images_path = output / "support_images.npy"
    support_points_path = output / "support_points.npy"
    npz_path = output / "cache.npz"
    manifest_path = output / "manifest.json"

    _atomic_save_npy(images_path, dense_images)
    _atomic_save_npy(points_path, dense_points)
    _atomic_save_npy(support_images_path, support_images)
    _atomic_save_npy(support_points_path, support_points)
    _atomic_save_npz(
        npz_path,
        {
            "images": dense_images,
            "points": dense_points,
            "support_images": support_images,
            "support_points": support_points,
            "support_tids": np.asarray(payload["anchor_grid"]["tids"], dtype=np.int64),
            "probe_indices": dense_probe_indices,
            "probe_images": probe_images,
            "probe_points": probe_points,
        },
    )

    sample_digests = sample_hashes(dense_images, dense_points)
    support_sample_digests = sample_hashes(support_images, support_points)
    full_digest = dataset_hash(dense_images, dense_points)
    # Match ``s1clean.config.protocol_hash``: protocol provenance is the hash
    # of canonical JSON, while the cache file entries below use raw file-byte
    # hashes.  Keeping the two domains separate prevents whitespace changes in
    # protocol.json from looking like a data change.
    protocol_digest = _sha256_bytes(_canonical_json(payload))
    files = {
        "images.npy": _file_entry(images_path, dense_images),
        "points.npy": _file_entry(points_path, dense_points),
        "support_images.npy": _file_entry(support_images_path, support_images),
        "support_points.npy": _file_entry(support_points_path, support_points),
        "cache.npz": _file_entry(npz_path),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": payload.get("protocol_id", "neural_affine_s1_clean_v1"),
        "protocol_sha256": protocol_digest,
        "image_size": int(payload["image_size"]),
        "coord_scale": float(payload["coord_scale"]),
        "safe_box_px": dict(payload["safe_box_px"]),
        "renderer": dict(payload["renderer"]),
        "support": {
            "tids": [int(value) for value in payload["anchor_grid"]["tids"]],
            "points_px": support_points.tolist(),
            "dense_indices": [0, 40, 1640, 1680],
            "count": int(len(support_points)),
            "sample_sha256": support_sample_digests,
        },
        "dense": {
            "grid_n": int(payload["dense_grid"]["n"]),
            "count": int(len(dense_points)),
            "sample_sha256": sample_digests,
        },
        "probes": {
            "indices": [int(value) for value in dense_probe_indices],
            "points_px": probe_points.tolist(),
            "sample_sha256": [sample_digests[int(value)] for value in dense_probe_indices],
        },
        "files": files,
        "sample_sha256": sample_digests,
        "dataset_sha256": full_digest,
        "cache_sha256": full_digest,
        "full_cache_sha256": full_digest,
        "full_cache_file_sha256": files["cache.npz"]["sha256"],
        "hash_algorithm": "sha256",
    }
    _atomic_write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def _load_array(path: Path, bundle: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if path.exists():
        return np.load(path, allow_pickle=False)
    if key in bundle:
        return np.asarray(bundle[key])
    raise FileNotFoundError(f"cache is missing {path.name} and NPZ key {key!r}")


def load_cache(out_dir: Path | str, *, verify: bool = True) -> dict[str, Any]:
    """Load the cache and, by default, verify file and dataset hashes."""

    output = Path(out_dir)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("cache manifest must be a JSON object")

    bundle: dict[str, np.ndarray] = {}
    npz_path = output / "cache.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as loaded:
            bundle = {key: np.asarray(loaded[key]) for key in loaded.files}

    images = _load_array(output / "images.npy", bundle, "images")
    points = _load_array(output / "points.npy", bundle, "points")
    support_images = _load_array(output / "support_images.npy", bundle, "support_images")
    support_points = _load_array(output / "support_points.npy", bundle, "support_points")
    images = np.ascontiguousarray(images)
    points = _normalise_points(points)
    support_images = np.ascontiguousarray(support_images)
    support_points = _normalise_points(support_points)
    if images.ndim != 3 or images.dtype != np.uint8:
        raise ValueError(f"invalid dense image array: {images.shape} {images.dtype}")
    if support_images.ndim != 3 or support_images.dtype != np.uint8:
        raise ValueError(f"invalid support image array: {support_images.shape} {support_images.dtype}")

    if verify:
        for name, entry in dict(manifest.get("files", {})).items():
            path = output / str(entry["path"])
            if path.exists() and sha256_file(path) != str(entry["sha256"]):
                raise ValueError(f"cache file hash mismatch: {path}")
        expected = manifest.get("dataset_sha256")
        if expected is not None and dataset_hash(images, points) != str(expected):
            raise ValueError("dense dataset hash mismatch")

    probes = np.asarray(manifest.get("probes", {}).get("indices", []), dtype=np.int64)
    if len(probes) and np.any((probes < 0) | (probes >= len(images))):
        raise ValueError("manifest probe index is outside the dense cache")
    support_tids = np.asarray(manifest.get("support", {}).get("tids", [0, 7, 56, 63]), dtype=np.int64)
    result = {
        "images": images,
        "points": points,
        "dense_images": images,
        "dense_points": points,
        "support_images": support_images,
        "support_points": support_points,
        "support_tids": support_tids,
        "probe_indices": probes,
        "probe_images": images[probes] if len(probes) else np.empty((0,) + images.shape[1:], dtype=np.uint8),
        "probe_points": points[probes] if len(probes) else np.empty((0, 2), dtype=np.float64),
        "manifest": dict(manifest),
    }
    return result


__all__ = [
    "PROTOCOL_PATH",
    "dataset_hash",
    "dense_points_px",
    "load_cache",
    "load_protocol",
    "materialize_cache",
    "probe_indices",
    "render_points",
    "sample_sha256",
    "sample_hashes",
    "sha256_file",
    "support_points_px",
]
