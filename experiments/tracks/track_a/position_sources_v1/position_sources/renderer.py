"""Deterministic compact-blob/triangle and strict-torus renderers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import load_protocol, write_json


def hash_array(array: np.ndarray, name: str = "array") -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _grid(low: float, high: float, n: int, *, endpoint: bool = True) -> np.ndarray:
    if int(n) < 2:
        raise ValueError("grid n must be at least 2")
    xs = np.linspace(float(low), float(high), int(n), endpoint=bool(endpoint), dtype=np.float64)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    return np.ascontiguousarray(np.stack((xx, yy), axis=-1).reshape(-1, 2), dtype=np.float64)


def dense_domain_points(protocol: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return every integer point in the frozen inclusive primary box."""

    cfg = load_protocol() if protocol is None else dict(protocol)
    box = cfg["renderer"]["primary_position_box_px"]
    low = int(box["low"])
    high = int(box["high"])
    axis = np.arange(low, high + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.ascontiguousarray(np.stack((xx, yy), axis=-1).reshape(-1, 2), dtype=np.float64)


def primary_points(protocol: Mapping[str, Any] | None = None, *, n: int | None = None) -> np.ndarray:
    # ``n`` is accepted only for backwards-compatible local smoke callers;
    # formal preparation always uses the complete integer domain.
    if n is not None:
        cfg = load_protocol() if protocol is None else dict(protocol)
        box = cfg["renderer"]["primary_position_box_px"]
        return _grid(box["low"], box["high"], int(n))
    return dense_domain_points(protocol)


def _hash_score(x: int, y: int, seed: int) -> int:
    payload = f"{int(seed)}:{int(x)}:{int(y)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _quotient_residue(points: np.ndarray, *, low: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(points, dtype=np.int64) - int(low)
    quotient = offsets // 32
    residue = offsets % 32
    return quotient, residue


def _coverage(points: np.ndarray, *, low: int) -> dict[str, Any]:
    quotient, residue = _quotient_residue(points, low=low)
    cells = sorted({(int(row[0]), int(row[1])) for row in quotient})
    return {
        "quotient_cells": [[x, y] for x, y in cells],
        "quotient_cell_count": len(cells),
        "x_residues": sorted({int(value) for value in residue[:, 0]}),
        "y_residues": sorted({int(value) for value in residue[:, 1]}),
    }


def split_dense_domain(protocol: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deterministic 80/20 quotient-cell split with coverage repair."""

    cfg = load_protocol() if protocol is None else dict(protocol)
    renderer = cfg["renderer"]
    low = int(renderer["primary_position_box_px"]["low"])
    domain = dense_domain_points(cfg).astype(np.int64)
    seed = int(renderer["dense_split"]["hash_seed"])
    train_fraction = float(renderer["dense_split"]["train_fraction"])
    quotient, residue = _quotient_residue(domain, low=low)
    train_indices: set[int] = set()
    test_indices: set[int] = set()
    for qx in range(4):
        for qy in range(4):
            cell = np.flatnonzero((quotient[:, 0] == qx) & (quotient[:, 1] == qy))
            ordered = sorted(cell.tolist(), key=lambda index: _hash_score(int(domain[index, 0]), int(domain[index, 1]), seed))
            train_count = int(round(len(ordered) * train_fraction))
            train_indices.update(ordered[:train_count])
            test_indices.update(ordered[train_count:])

    # The quotient-cell split already guarantees all 16 cells.  Deterministic
    # swaps below make the per-axis residue guarantee explicit rather than a
    # probability claim.
    def missing(indices: set[int]) -> tuple[set[int], set[int]]:
        rows = domain[sorted(indices)]
        q, r = _quotient_residue(rows, low=low)
        return set(range(32)).difference(set(map(int, r[:, 0]))), set(range(32)).difference(set(map(int, r[:, 1])))

    for target_set, other_set in ((train_indices, test_indices), (test_indices, train_indices)):
        for axis in (0, 1):
            missing_values = missing(target_set)[axis]
            for value in sorted(missing_values):
                candidates = [
                    index
                    for index in sorted(other_set)
                    if int(residue[index, axis]) == value
                ]
                if not candidates:
                    raise AssertionError(f"cannot repair residue coverage for axis={axis}, value={value}")
                chosen = candidates[0]
                # Pick the first target element whose removal leaves target
                # coverage intact; a full 1024-point cell makes this stable.
                replacement_candidates = [
                    index
                    for index in sorted(target_set)
                    if index not in {chosen} and int(residue[index, axis]) != value
                ]
                if not replacement_candidates:
                    raise AssertionError("coverage repair has no safe replacement")
                replacement = replacement_candidates[0]
                target_set.remove(replacement)
                other_set.add(replacement)
                other_set.remove(chosen)
                target_set.add(chosen)

    train = np.ascontiguousarray(domain[sorted(train_indices)], dtype=np.float64)
    test = np.ascontiguousarray(domain[sorted(test_indices)], dtype=np.float64)
    if len(train) + len(test) != len(domain) or set(map(tuple, train)).intersection(set(map(tuple, test))):
        raise AssertionError("dense split overlap/completeness failure")
    coverage = {
        "domain_count": int(len(domain)),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "train_fraction": float(len(train) / len(domain)),
        "train": _coverage(train, low=low),
        "test": _coverage(test, low=low),
        "method": renderer["dense_split"]["method"],
        "hash_seed": seed,
        "no_overlap": True,
        "complete_domain": True,
    }
    for split_name in ("train", "test"):
        split_coverage = coverage[split_name]
        if split_coverage["quotient_cell_count"] != 16 or split_coverage["x_residues"] != list(range(32)) or split_coverage["y_residues"] != list(range(32)):
            raise AssertionError(f"coverage requirement failed for {split_name}: {split_coverage}")
    return train, test, coverage


def torus_points(protocol: Mapping[str, Any] | None = None, *, n: int | None = None) -> np.ndarray:
    cfg = load_protocol() if protocol is None else dict(protocol)
    renderer = cfg["renderer"]
    size = float(renderer["torus_period_px"])
    if n is not None:
        return _grid(0.0, size, int(n), endpoint=False)
    axis = np.arange(int(size), dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.ascontiguousarray(np.stack((xx, yy), axis=-1).reshape(-1, 2), dtype=np.float64)


def _periodic_delta(values: np.ndarray, centers: float, period: float) -> np.ndarray:
    return (values - float(centers) + period / 2.0) % period - period / 2.0


def _pixel_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(int(size), dtype=np.float64) + 0.5
    return np.meshgrid(axis, axis, indexing="xy")


def _gaussian(xx: np.ndarray, yy: np.ndarray, x: float, y: float, sigma: float, *, period: float | None) -> np.ndarray:
    if period is None:
        dx = xx - float(x)
        dy = yy - float(y)
    else:
        dx = _periodic_delta(xx, float(x), float(period))
        dy = _periodic_delta(yy, float(y), float(period))
    return np.exp(-(dx * dx + dy * dy) / (2.0 * float(sigma) ** 2))


def _filled_triangle(xx: np.ndarray, yy: np.ndarray, x: float, y: float, *, period: float | None) -> np.ndarray:
    """Rasterize one fixed scalene triangle around the requested centroid."""

    vertices = np.asarray([[-12.0, -8.0], [12.0, -6.0], [0.0, 14.0]], dtype=np.float64)
    # The three offsets have centroid exactly (0,0).  For torus rendering,
    # use periodic deltas so the primitive can cross a seam without a crop.
    if period is None:
        px = xx - float(x)
        py = yy - float(y)
    else:
        px = _periodic_delta(xx, float(x), float(period))
        py = _periodic_delta(yy, float(y), float(period))
    ax, ay = vertices[0]
    bx, by = vertices[1]
    cx, cy = vertices[2]
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
    c = 1.0 - a - b
    return ((a >= 0.0) & (b >= 0.0) & (c >= 0.0)).astype(np.float64)


def render_one(
    point_px: Sequence[float],
    *,
    image_size: int,
    sigma_px: float,
    primitive: str = "blob",
    torus: bool = False,
) -> np.ndarray:
    point = np.asarray(point_px, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("point_px must be a finite pair")
    size = int(image_size)
    if size < 8:
        raise ValueError("image_size must be at least 8")
    if float(sigma_px) <= 0:
        raise ValueError("sigma_px must be positive")
    period = float(size) if torus else None
    xx, yy = _pixel_grid(size)
    base = _gaussian(xx, yy, float(point[0]), float(point[1]), float(sigma_px), period=period)
    if primitive == "blob":
        values = base
    elif primitive == "triangle":
        values = _filled_triangle(xx, yy, float(point[0]), float(point[1]), period=period)
    else:
        raise ValueError(f"unknown primitive: {primitive}")
    image = np.clip(np.rint(values * 255.0), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def render_points(
    points_px: np.ndarray | Sequence[Sequence[float]],
    *,
    image_size: int,
    sigma_px: float,
    primitive: str = "blob",
    torus: bool = False,
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"points_px must have shape [N,2], got {points.shape}")
    if torus:
        points = points % float(image_size)
    else:
        if np.any(points < 0) or np.any(points >= float(image_size)):
            raise ValueError("non-torus point is outside image")
    if len(points) == 0:
        return np.empty((0, int(image_size), int(image_size)), dtype=np.uint8)
    return np.ascontiguousarray(
        np.stack(
            [
                render_one(
                    point,
                    image_size=int(image_size),
                    sigma_px=float(sigma_px),
                    primitive=primitive,
                    torus=torus,
                )
                for point in points
            ],
            axis=0,
        )
    )


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
    temporary.replace(path)


def split_torus_domain(protocol: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = load_protocol() if protocol is None else dict(protocol)
    renderer = cfg["renderer"]
    period = int(renderer["torus_period_px"])
    seed = int(renderer["dense_split"]["hash_seed"])
    domain = torus_points(cfg).astype(np.int64)
    ordered = sorted(
        range(len(domain)),
        key=lambda index: _hash_score(int(domain[index, 0]), int(domain[index, 1]), seed),
    )
    train_count = int(round(len(ordered) * float(renderer["dense_split"]["train_fraction"])))
    train = np.ascontiguousarray(domain[sorted(ordered[:train_count])], dtype=np.float64)
    test = np.ascontiguousarray(domain[sorted(ordered[train_count:])], dtype=np.float64)
    return train, test, {
        "domain_count": int(len(domain)),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "train_fraction": float(len(train) / len(domain)),
        "period": period,
        "method": "sha256_point_split",
        "hash_seed": seed,
        "no_overlap": True,
        "complete_domain": True,
    }


def materialize_family_cache(
    out_dir: str | Path,
    *,
    family: str,
    protocol: Mapping[str, Any] | None = None,
    quick: bool = False,
    primitive: str = "blob",
    materialize_images: bool | None = None,
) -> dict[str, Any]:
    """Materialize a family cache with labels represented by point arrays.

    ``quick`` is explicitly recorded in the cache manifest and is intended
    only for local smoke/preflight; it is not a formal replacement for the
    frozen grid.
    """

    cfg = load_protocol() if protocol is None else dict(protocol)
    if family not in cfg["families"]:
        raise KeyError(f"unknown family: {family}")
    if primitive not in tuple(cfg["renderer"]["primitives"]):
        raise ValueError(f"unknown primitive: {primitive}")
    family_cfg = cfg["families"][family]
    image_size = int(family_cfg["input_size"])
    if family == "torus":
        train, evaluation, split_metadata = split_torus_domain(cfg)
        torus = True
    else:
        train, evaluation, split_metadata = split_dense_domain(cfg)
        torus = False
    if quick:
        train = np.ascontiguousarray(train[:64], dtype=np.float64)
        evaluation = np.ascontiguousarray(evaluation[:64], dtype=np.float64)
        split_metadata = dict(split_metadata)
        split_metadata["quick_subset"] = True
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_npy(root / "train_points.npy", train)
    _atomic_npy(root / "eval_points.npy", evaluation)
    domain_points = torus_points(cfg) if torus else dense_domain_points(cfg)
    _atomic_npy(root / "domain_points.npy", domain_points)
    should_materialize = bool(quick) if materialize_images is None else bool(materialize_images)
    image_names: list[str] = []
    if should_materialize:
        train_images = render_points(
            train,
            image_size=image_size,
            sigma_px=float(cfg["renderer"]["sigma_px"]),
            primitive=primitive,
            torus=torus,
        )
        eval_images = render_points(
            evaluation,
            image_size=image_size,
            sigma_px=float(cfg["renderer"]["sigma_px"]),
            primitive=primitive,
            torus=torus,
        )
        _atomic_npy(root / "train_images.npy", train_images)
        _atomic_npy(root / "eval_images.npy", eval_images)
        image_names = ["train_images.npy", "eval_images.npy"]
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.npy")):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        files[path.name] = {
            "bytes": int(path.stat().st_size),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    manifest = {
        "schema_version": 1,
        "kind": "position_sources_family_cache",
        "protocol_id": cfg["protocol_id"],
        "mode": "quick" if quick else "formal",
        "family": family,
        "primitive": primitive,
        "image_size": image_size,
        "torus": torus,
        "train_count": int(len(train)),
        "eval_count": int(len(evaluation)),
        "domain_count": int(len(domain_points)),
        "images_materialized": should_materialize,
        "files": files,
        "train_points_hash": hash_array(train, "train_points"),
        "eval_points_hash": hash_array(evaluation, "eval_points"),
        "domain_points_hash": hash_array(domain_points, "domain_points"),
        "split_metadata": split_metadata,
    }
    if should_materialize:
        manifest["train_images_hash"] = hash_array(train_images, "train_images")
        manifest["eval_images_hash"] = hash_array(eval_images, "eval_images")
        manifest["image_files"] = image_names
    write_json(root / "CACHE_MANIFEST.json", manifest)
    return manifest


def load_cache(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    path = Path(root)
    manifest = json.loads((path / "CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    train_points = np.load(path / "train_points.npy", allow_pickle=False, mmap_mode="r")
    eval_points = np.load(path / "eval_points.npy", allow_pickle=False, mmap_mode="r")
    train_images_path = path / "train_images.npy"
    eval_images_path = path / "eval_images.npy"
    train_images = np.load(train_images_path, allow_pickle=False, mmap_mode="r") if train_images_path.exists() else None
    eval_images = np.load(eval_images_path, allow_pickle=False, mmap_mode="r") if eval_images_path.exists() else None
    if train_points.ndim != 2 or train_points.shape[1] != 2 or eval_points.ndim != 2 or eval_points.shape[1] != 2:
        raise ValueError("point arrays must have shape [N,2]")
    if (train_images is None) != (eval_images is None):
        raise ValueError("train/eval images must be both materialized or both absent")
    if train_images is not None and eval_images is not None:
        if train_images.ndim != 3 or eval_images.ndim != 3 or train_images.dtype != np.uint8 or eval_images.dtype != np.uint8:
            raise ValueError("image arrays must be uint8 [N,H,W]")
        if train_images.shape[0] != train_points.shape[0] or eval_images.shape[0] != eval_points.shape[0]:
            raise ValueError("point/image count mismatch")
    return train_points, train_images, eval_points, eval_images, manifest


__all__ = [
    "hash_array",
    "load_cache",
    "materialize_family_cache",
    "dense_domain_points",
    "primary_points",
    "render_one",
    "render_points",
    "split_dense_domain",
    "split_torus_domain",
    "torus_points",
]
