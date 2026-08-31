from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .config import CONFIG


def _periodic_delta(values: np.ndarray, center: float, period: float) -> np.ndarray:
    return (values - float(center) + period / 2.0) % period - period / 2.0


def _pixel_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(size, dtype=np.float64) + 0.5
    return np.meshgrid(axis, axis, indexing="xy")


def triangle_vertices() -> np.ndarray:
    vertices = np.asarray(CONFIG["triangle_vertices_xy"], dtype=np.float64)
    if vertices.shape != (3, 2) or not np.isfinite(vertices).all():
        raise ValueError(f"triangle_vertices_xy must be finite [3,2], got {vertices.shape}")
    if not np.allclose(vertices.mean(axis=0), 0.0, atol=1e-8):
        raise ValueError("triangle vertices must be centered at the requested position")
    return vertices


def is_fully_visible(point_xy: Sequence[float], *, image_size: int | None = None) -> bool:
    size = int(CONFIG["image_size"] if image_size is None else image_size)
    point = np.asarray(point_xy, dtype=np.float64)
    vertices = triangle_vertices() + point[None, :]
    return bool(
        np.all(vertices[:, 0] >= 0.0)
        and np.all(vertices[:, 0] <= size)
        and np.all(vertices[:, 1] >= 0.0)
        and np.all(vertices[:, 1] <= size)
    )


def render_one(point_xy: Sequence[float], *, torus: bool, image_size: int | None = None) -> np.ndarray:
    size = int(CONFIG["image_size"] if image_size is None else image_size)
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("point_xy must be a finite pair")
    if not torus and (np.any(point < 0.0) or np.any(point >= float(size))):
        raise ValueError("finite-renderer point is outside the canvas")
    point = point % float(size) if torus else point
    xx, yy = _pixel_grid(size)
    if torus:
        px = _periodic_delta(xx, float(point[0]), float(size))
        py = _periodic_delta(yy, float(point[1]), float(size))
    else:
        px = xx - float(point[0])
        py = yy - float(point[1])

    vertices = triangle_vertices()
    ax, ay = vertices[0]
    bx, by = vertices[1]
    cx, cy = vertices[2]
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
    c = 1.0 - a - b
    image = ((a >= 0.0) & (b >= 0.0) & (c >= 0.0)).astype(np.uint8) * np.uint8(255)
    return np.ascontiguousarray(np.repeat(image[None, :, :], 3, axis=0))


def render_many(
    points_xy: Iterable[Sequence[float]],
    *,
    torus: bool,
    require_fully_visible: bool = False,
) -> np.ndarray:
    points = list(points_xy)
    if require_fully_visible and torus:
        raise ValueError("fully-visible filtering is only meaningful for finite rendering")
    if require_fully_visible and any(not is_fully_visible(point) for point in points):
        raise ValueError("finite renderer received a point whose actual triangle is clipped")
    images = [render_one(point, torus=torus) for point in points]
    if not images:
        size = int(CONFIG["image_size"])
        return np.empty((0, 3, size, size), dtype=np.uint8)
    return np.stack(images, axis=0)


def position_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(CONFIG["grid_values"], dtype=np.int64)
    points = np.asarray([(x, y) for y in values for x in values], dtype=np.float32)
    ix = np.tile(np.arange(len(values), dtype=np.int64), len(values))
    iy = np.repeat(np.arange(len(values), dtype=np.int64), len(values))
    return points, ix, iy


def finite_position_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only the configured grid points whose actual triangle is visible."""

    points, ix, iy = position_grid()
    keep = np.asarray([is_fully_visible(point) for point in points], dtype=bool)
    return points[keep], ix[keep], iy[keep]


def grid_points_with_visibility() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the full 16x16 grid and the finite fully-visible mask."""

    points, ix, iy = position_grid()
    visible = np.asarray([is_fully_visible(point) for point in points], dtype=bool)
    return points, ix, iy, visible
