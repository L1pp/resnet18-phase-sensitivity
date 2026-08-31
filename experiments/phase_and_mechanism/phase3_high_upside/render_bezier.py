"""Quadratic Bézier renders for Track A. Reuses Phase 1 geometry, writes only Phase 3 cache."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from phase1_gap_rep.common import IMAGE_SIZE, px_to_norm, render_quadratic
from phase1_gap_rep.generate_data import _assert_in_canvas, build_shapes, profile_spec

_SHAPES = None


def shapes_px() -> np.ndarray:
    global _SHAPES
    if _SHAPES is None:
        _SHAPES, _ = build_shapes(profile_spec("factorial"))
    return _SHAPES


def render_at(shape_id: int, t_px: np.ndarray) -> np.ndarray:
    q = shapes_px()[int(shape_id)]
    t = np.asarray(t_px, dtype=np.float64).reshape(2)
    p_px = q + t[None, :]
    _assert_in_canvas(p_px, int(shape_id), 0)
    return render_quadratic(px_to_norm(p_px).reshape(-1))


def render_many(shape_ids: np.ndarray, t_px: np.ndarray) -> np.ndarray:
    images = np.zeros((len(shape_ids), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, (sid, t) in enumerate(zip(shape_ids, t_px)):
        images[i] = render_at(int(sid), t)
    return images
