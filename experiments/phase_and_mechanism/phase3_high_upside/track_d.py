"""Track D: cross-resolution coordinate semantics. No fine-tune."""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from phase1_gap_rep.common import setup_matplotlib_chinese

from .backbones_xy import build_xy_backbone
from .io_util import dump_json
from .primitives import make_batch
from .protocol import PHASE3_ROOT, PRIMITIVE_FAMILIES, SEED_TRACK_C, TRACK_D_SIZES, record_failure
from .trainer import load_best, predict_xy

OUT = PHASE3_ROOT / "track_d"


def _best_c_dir() -> Optional[str]:
    all6 = PHASE3_ROOT / "track_c" / "models" / f"all6__{SEED_TRACK_C}" / "checkpoints" / "best_slim.pt"
    if all6.exists():
        return str(all6.parent.parent)
    panel = PHASE3_ROOT / "track_c" / "panel.json"
    if not panel.exists():
        return None
    blob = __import__("json").loads(panel.read_text(encoding="utf-8"))
    rows = [r for r in blob.get("lofo", []) if r.get("ok")]
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r.get("seen_mae_px", 1e9)))
    return str(PHASE3_ROOT / "track_c" / "models" / best["name"])


def _eval_size(model, size: int, size_mode: str, seed: int) -> Dict[str, Any]:
    blob = make_batch(PRIMITIVE_FAMILIES, 600, seed, image_size=size, size_mode=size_mode)
    pred = predict_xy(model, blob["images"])
    target = blob["xy_norm"]
    scale = float(size - 1)
    err_norm = np.linalg.norm(pred - target, axis=-1)
    err_px = err_norm * scale
    # calibration: pred vs target slope per axis
    slopes = []
    for axis in (0, 1):
        x = target[:, axis]
        y = pred[:, axis]
        if np.std(x) < 1e-8:
            slopes.append(None)
            continue
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        slopes.append(slope)
    center = np.abs(target - 0.5).max(axis=1) < 0.2
    border = np.abs(target - 0.5).max(axis=1) > 0.35
    return {
        "size": size,
        "size_mode": size_mode,
        "norm_mae": float(np.mean(err_norm)),
        "px_mae": float(np.mean(err_px)),
        "slope_x": slopes[0],
        "slope_y": slopes[1],
        "center_px_mae": float(np.mean(err_px[center])) if np.any(center) else None,
        "border_px_mae": float(np.mean(err_px[border])) if np.any(border) else None,
        "n": int(len(target)),
    }


def _plot(rows: List[Dict[str, Any]], path) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for mode in ("relative", "absolute"):
        sub = [r for r in rows if r.get("size_mode") == mode]
        if not sub:
            continue
        ax.plot([r["size"] for r in sub], [r["norm_mae"] for r in sub], marker="o", label=mode)
    ax.set_xlabel("canvas size")
    ax.set_ylabel("normalized MAE")
    ax.set_title("Track D 跨分辨率归一化坐标误差")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_track_d() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = _best_c_dir()
    if model_dir is None:
        payload = {"ok": False, "error": "no Track C model"}
        dump_json(OUT / "panel.json", payload)
        return payload
    model = load_best(build_xy_backbone("resnet18")[0], model_dir)
    rows: List[Dict[str, Any]] = []
    for size in TRACK_D_SIZES:
        for mode in ("relative", "absolute"):
            try:
                rows.append(_eval_size(model, size, mode, 20260841 + size))
            except Exception as exc:
                record_failure(f"track_d_{size}_{mode}", str(exc), {"tb": traceback.format_exc()[-2000:]})
                rows.append({"ok": False, "size": size, "size_mode": mode, "error": str(exc)})
    _plot([r for r in rows if "norm_mae" in r], OUT / "figures" / "norm_mae_vs_size.png")
    rel224 = next((r for r in rows if r.get("size") == 224 and r.get("size_mode") == "relative"), None)
    rel_other = [r for r in rows if r.get("size") != 224 and r.get("size_mode") == "relative" and "norm_mae" in r]
    flag = None
    if rel224 and rel_other and all(r["norm_mae"] < max(0.04, 2.5 * rel224["norm_mae"]) for r in rel_other):
        flag = "HIGH_UPSIDE: emergent resolution-transferable coordinate system"
    payload = {"model_dir": model_dir, "rows": rows, "high_upside_flag": flag}
    dump_json(OUT / "panel.json", payload)
    return payload
