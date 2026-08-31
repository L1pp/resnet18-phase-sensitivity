"""Unified dense evaluation labels: train / interpolation / hull."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .protocol import N_DENSE, dense_grid_px, integer_grid_px, load_protocol, regime_tids


def _hull_ok(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Axis-aligned bbox hull in translation plane (2D convex hull of a grid is the bbox of extremes).
    For 2D point sets we use scipy-free convex hull via gift wrapping if n small.
    """
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    q = np.asarray(query, dtype=np.float64)
    if len(pts) <= 2:
        # degenerate: inside if on the segment / points
        d = np.min(np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=-1), axis=1)
        return d <= 1e-6
    # 2D convex hull (monotone chain)
    hull = _convex_hull(pts)
    return np.array([_inside_hull(hull, xy) for xy in q], dtype=bool)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.array(sorted(map(tuple, points.tolist())))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _inside_hull(hull: np.ndarray, xy: np.ndarray, eps: float = 1e-7) -> bool:
    n = len(hull)
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        cross = (b[0] - a[0]) * (xy[1] - a[1]) - (b[1] - a[1]) * (xy[0] - a[0])
        if cross < -eps:
            return False
    return True


def eval_point_labels(train_tids: Sequence[int]) -> Dict[str, np.ndarray]:
    proto = load_protocol()
    dense = dense_grid_px()
    integer = integer_grid_px()
    train_t = integer[np.array(list(train_tids), dtype=int)]
    # nearest train translation (integer support)
    d = np.linalg.norm(dense[:, None, :] - train_t[None, :, :], axis=-1)
    nearest = d.min(axis=1)
    exact = nearest <= 1e-6
    in_hull = _hull_ok(train_t, dense)
    # distance to hull: 0 if inside else min distance to hull edges
    hull_d = np.zeros(len(dense))
    hull = _convex_hull(train_t)
    for i, xy in enumerate(dense):
        if in_hull[i]:
            hull_d[i] = 0.0
        else:
            hull_d[i] = _dist_to_polygon(hull, xy)
    interp = in_hull & (~exact)
    outside = ~in_hull
    return {
        "exact_train": exact,
        "interpolation": interp,
        "in_hull": in_hull,
        "outside_hull": outside,
        "nearest_train_px": nearest,
        "hull_distance_px": hull_d,
        "t_px": dense,
    }


def _dist_to_polygon(hull: np.ndarray, xy: np.ndarray) -> float:
    n = len(hull)
    best = np.inf
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(xy - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
        proj = a + t * ab
        best = min(best, float(np.linalg.norm(xy - proj)))
    return float(best)


def subset_metrics(pred_t: np.ndarray, true_t: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if not np.any(mask):
        return {"n": 0, "t_mae_px": float("nan")}
    err = np.linalg.norm(pred_t[mask] - true_t[mask], axis=-1)
    return {"n": int(mask.sum()), "t_mae_px": float(np.mean(err)), "t_max_px": float(np.max(err))}
