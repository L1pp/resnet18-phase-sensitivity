"""Load frozen factorial 8x8 images without re-rendering."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from phase1_gap_rep.common import IMAGE_SIZE, fingerprint, profile_spec
from phase1_gap_rep.generate_data import load_split

from .protocol import GEOMETRY_FP, canonical_regime, load_protocol, regime_tids

SPLIT_NAMES = ("train", "val", "test")


def load_factorial_grid() -> Dict[str, np.ndarray]:
    spec = profile_spec("factorial")
    n_s, n_t = spec.n_shapes, spec.n_translations
    images = np.zeros((n_s, n_t, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    p = np.zeros((n_s, n_t, 6), dtype=np.float32)
    q = np.zeros((n_s, n_t, 6), dtype=np.float32)
    t = np.zeros((n_s, n_t, 2), dtype=np.float32)
    filled = np.zeros((n_s, n_t), dtype=bool)
    for split in SPLIT_NAMES:
        data = load_split(spec, split)
        sid = data["shape_id"].astype(np.int64)
        tid = data["translation_id"].astype(np.int64)
        images[sid, tid] = data["images"]
        p[sid, tid] = data["P"]
        q[sid, tid] = data["Q"]
        t[sid, tid] = data["t"]
        filled[sid, tid] = True
    if not bool(np.all(filled)):
        raise RuntimeError("incomplete factorial grid; refuse to re-render")
    fp = fingerprint(spec)
    if fp != GEOMETRY_FP:
        raise RuntimeError(f"geometry fingerprint {fp} != {GEOMETRY_FP}")
    return {"images": images, "P": p, "Q": q, "t": t}


def pairs_for_regime(regime: str, shape_ids: Sequence[int] | None = None) -> List[Tuple[int, int]]:
    proto = load_protocol()
    name = canonical_regime(regime)
    tids = proto["regimes"][name]
    if shape_ids is None:
        shape_ids = list(range(64))
    return [(int(s), int(t)) for s in shape_ids for t in tids]


def gather_pairs(grid: Dict[str, np.ndarray], pairs: Sequence[Tuple[int, int]]) -> Dict[str, np.ndarray]:
    sids = np.array([p[0] for p in pairs], dtype=np.int64)
    tids = np.array([p[1] for p in pairs], dtype=np.int64)
    return {
        "images": grid["images"][sids, tids],
        "P": grid["P"][sids, tids],
        "Q": grid["Q"][sids, tids],
        "t": grid["t"][sids, tids],
        "shape_id": sids,
        "translation_id": tids,
    }
