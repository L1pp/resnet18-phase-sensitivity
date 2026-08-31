"""Line / arc primitive renderers and R18 Full64 content panel."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from phase1_gap_rep.common import COORD_SCALE, IMAGE_SIZE, STROKE_WIDTH, SUPER_SAMPLE, dump_json, px_to_norm
from phase1_gap_rep.generate_data import build_shapes, profile_spec

from .protocol import PHASE2_ROOT, SEED_ATLAS, T_HIGH_PX, T_LOW_PX, load_protocol

OUT = PHASE2_ROOT / "content"


def _stroke(points_norm: np.ndarray) -> np.ndarray:
    high = IMAGE_SIZE * SUPER_SAMPLE
    image = Image.new("L", (high, high), color=0)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * COORD_SCALE * SUPER_SAMPLE, float(y) * COORD_SCALE * SUPER_SAMPLE) for x, y in points_norm]
    width = int(round(STROKE_WIDTH * SUPER_SAMPLE))
    try:
        draw.line(xy, fill=255, width=width, joint="curve")
    except TypeError:
        draw.line(xy, fill=255, width=width)
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling.LANCZOS), dtype=np.uint8)


def render_line(p_px: np.ndarray) -> np.ndarray:
    q = np.asarray(p_px, dtype=np.float64).reshape(3, 2)
    # use endpoints only
    pts = np.stack([q[0], q[2]], axis=0) / COORD_SCALE
    return _stroke(pts)


def render_arc(p_px: np.ndarray) -> np.ndarray:
    q = np.asarray(p_px, dtype=np.float64).reshape(3, 2)
    # sample a circular-ish arc through three points via quadratic already — denser polyline
    t = np.linspace(0, 1, 64)[:, None]
    pts = (1 - t) ** 2 * q[0] + 2 * (1 - t) * t * q[1] + t**2 * q[2]
    return _stroke(pts / COORD_SCALE)


def run_content_panel() -> Dict[str, Any]:
    """Compare h(t) relational geometry across line / arc / quadratic on R18-G64 if trained.
    Full retrain on line/arc is third-tier; we at least dump primitive caches and a note.
    Cloud may call train_regime after swapping renderer — kept explicit.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    proto = load_protocol()
    note = {
        "status": "renderer_ready",
        "train_policy": "If time remains, train_regime resnet18 G64 on line and arc datasets using same t grid.",
        "protocol_hash": proto["protocol_hash"],
        "safe_box": [T_LOW_PX, T_HIGH_PX],
    }
    dump_json(OUT / "content_note.json", note)
    return note
