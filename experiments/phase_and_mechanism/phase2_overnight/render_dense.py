"""Dense 41x41 evaluation cache. Does not touch factorial/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from phase1_gap_rep.common import COORD_SCALE, IMAGE_SIZE, dump_json, px_to_norm, render_quadratic
from phase1_gap_rep.generate_data import _assert_in_canvas, build_shapes, profile_spec

from .protocol import N_DENSE, PHASE2_ROOT, dense_grid_px, load_protocol

CACHE_DIR = PHASE2_ROOT / "eval_cache"


def cache_paths() -> Dict[str, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "dir": CACHE_DIR,
        "images": CACHE_DIR / "images_s32_g41.uint8",
        "P": CACHE_DIR / "P_s32_g41.npy",
        "t": CACHE_DIR / "t_s32_g41.npy",
        "meta": CACHE_DIR / "cache_meta.json",
    }


def _memmap_images(path: Path, create: bool) -> np.memmap:
    n_s, n_g = 32, N_DENSE * N_DENSE
    shape = (n_s, n_g, IMAGE_SIZE, IMAGE_SIZE)
    if create:
        return np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape)


def build_dense_cache(force: bool = False) -> Dict[str, Any]:
    proto = load_protocol()
    paths = cache_paths()
    n_s = len(proto["eval_shape_ids"])
    n_g = N_DENSE * N_DENSE
    if paths["images"].exists() and paths["P"].exists() and paths["meta"].exists() and not force:
        meta = __import__("json").loads(paths["meta"].read_text(encoding="utf-8"))
        if meta.get("protocol_hash") == proto["protocol_hash"]:
            print(f"[phase2] dense cache exists n={n_s * n_g}")
            return meta
    spec = profile_spec("factorial")
    shapes_px, _records = build_shapes(spec)
    t_px = dense_grid_px()
    eval_ids = [int(x) for x in proto["eval_shape_ids"]]
    P = np.zeros((n_s, n_g, 6), dtype=np.float64)
    T = np.zeros((n_s, n_g, 2), dtype=np.float64)
    images = _memmap_images(paths["images"], create=True)
    total = n_s * n_g
    done = 0
    for si, sid in enumerate(eval_ids):
        q = shapes_px[int(sid)]
        for gi, t in enumerate(t_px):
            p_px = q + t[None, :]
            _assert_in_canvas(p_px, int(sid), gi)
            p_norm = px_to_norm(p_px).reshape(-1)
            images[si, gi] = render_quadratic(p_norm)
            P[si, gi] = p_norm
            T[si, gi] = t
            done += 1
            if done % 2000 == 0 or done == total:
                print(f"[phase2] render dense {done}/{total}")
    images.flush()
    np.save(paths["P"], P)
    np.save(paths["t"], T)
    meta = {
        "protocol_hash": proto["protocol_hash"],
        "n_shapes": n_s,
        "n_grid": n_g,
        "eval_shape_ids": eval_ids,
        "image_size": IMAGE_SIZE,
        "coord_scale": COORD_SCALE,
        "bytes_images": int(paths["images"].stat().st_size),
    }
    dump_json(paths["meta"], meta)
    print(f"[phase2] wrote dense cache {paths['images']}")
    return meta


def load_dense_cache() -> Dict[str, Any]:
    proto = load_protocol()
    paths = cache_paths()
    if not paths["images"].exists():
        build_dense_cache()
    images = _memmap_images(paths["images"], create=False)
    P = np.load(paths["P"])
    t = np.load(paths["t"])
    return {
        "images": images,
        "P": P,
        "t": t,
        "eval_shape_ids": [int(x) for x in proto["eval_shape_ids"]],
        "t_flat_px": dense_grid_px(),
        "protocol_hash": proto["protocol_hash"],
    }


def flatten_images(cache: Dict[str, Any]) -> np.ndarray:
    img = cache["images"]
    return np.asarray(img.reshape(img.shape[0] * img.shape[1], IMAGE_SIZE, IMAGE_SIZE))
