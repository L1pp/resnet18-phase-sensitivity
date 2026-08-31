"""Track E: random architecture prior vs learned features."""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase1_gap_rep.common import COORD_SCALE, setup_matplotlib_chinese

from .backbones_xy import build_xy_backbone
from .io_util import dump_json
from .protocol import (
    N_STEPS_REF,
    PHASE3_ROOT,
    SEED_TRACK_E,
    TRACK_E_ARCH,
    TRACK_E_MODES,
    TRACK_E_REGIMES,
    record_failure,
)
from .trainer import load_best, mae_px, predict_xy, train_xy

OUT = PHASE3_ROOT / "track_e"


def _load_regime(regime: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        from phase2_overnight.data import gather_pairs, load_factorial_grid, pairs_for_regime
        from phase2_overnight.protocol import pair_split
    except Exception as exc:
        record_failure("track_e_import", str(exc))
        return None
    try:
        grid = load_factorial_grid()
    except Exception as exc:
        record_failure("track_e_factorial_missing", str(exc))
        return None
    pairs = pairs_for_regime(regime)
    split = pair_split(pairs, SEED_TRACK_E)
    train = gather_pairs(grid, [tuple(p) for p in split["train"]])
    val = gather_pairs(grid, [tuple(p) for p in split["val"]])
    return {
        "train_images": train["images"],
        "train_xy": train["t"].astype(np.float64),
        "val_images": val["images"],
        "val_xy": val["t"].astype(np.float64),
    }


def _dense_test_xy(n: int = 512) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from phase2_overnight.render_dense import load_dense_cache
        cache = load_dense_cache()
        images = np.asarray(cache["images"]).reshape(-1, 224, 224)
        t = cache["t"].reshape(-1, 2) / COORD_SCALE
        rng = np.random.default_rng(20260842)
        idx = rng.choice(len(images), size=min(n, len(images)), replace=False)
        return images[idx], t[idx]
    except Exception:
        return None


def _jacobian_quality(model, images: np.ndarray, xy: np.ndarray) -> Dict[str, float]:
    """Finite-difference behavioral Jacobian of predicted (x,y) vs GT (x,y)."""
    pred = predict_xy(model, images)
    # local slope via pairing nearest neighbors in GT
    if len(xy) < 8:
        return {"det_mean": None}
    dpred = pred[1:] - pred[:-1]
    dxy = xy[1:] - xy[:-1]
    valid = np.linalg.norm(dxy, axis=-1) > 1e-4
    if not np.any(valid):
        return {"det_mean": None}
    ratios = np.linalg.norm(dpred[valid], axis=-1) / np.linalg.norm(dxy[valid], axis=-1)
    return {"step_ratio_mean": float(np.mean(ratios)), "step_ratio_std": float(np.std(ratios))}


def run_one(arch: str, mode: str, regime: str, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    name = f"{arch}__{mode}__{regime}__{SEED_TRACK_E}"
    out = OUT / "models" / name
    hidden = 128 if mode == "R1" else None
    model, _ = build_xy_backbone(arch, mlp_hidden=hidden)
    amp = arch in {"resnet18", "densenet121"} and mode == "R3"
    clip = 1.0 if arch == "efficientnet_b0" else None
    steps = 2000 if mode in {"R0", "R1"} else N_STEPS_REF
    summary = train_xy(
        model,
        data["train_images"],
        data["train_xy"],
        data["val_images"],
        data["val_xy"],
        out,
        seed=SEED_TRACK_E,
        arch=arch,
        image_size=224,
        steps=steps,
        mode=mode,
        amp=amp,
        clip_grad_norm=clip,
        extra={"regime": regime},
    )
    model = load_best(build_xy_backbone(arch, mlp_hidden=hidden)[0], out)
    val_mae = mae_px(predict_xy(model, data["val_images"]), data["val_xy"], 224)
    test = _dense_test_xy()
    test_mae = None
    jac = {}
    if test is not None:
        test_mae = mae_px(predict_xy(model, test[0]), test[1], 224)
        jac = _jacobian_quality(model, test[0][:128], test[1][:128])
    return {
        "ok": True,
        "name": name,
        "arch": arch,
        "mode": mode,
        "regime": regime,
        "val_mae_px": val_mae,
        "test_mae_px": test_mae,
        "init_hash": summary["init_hash"],
        "jacobian": jac,
        "amp": summary["amp"],
        "clip_grad_norm": clip,
    }


def _plot(rows: List[Dict[str, Any]], path) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for arch in TRACK_E_ARCH:
        xs, ys = [], []
        for mode in TRACK_E_MODES:
            hit = next((r for r in rows if r.get("ok") and r.get("arch") == arch and r.get("mode") == mode and r.get("regime") == "G64"), None)
            if hit and hit.get("val_mae_px") is not None:
                xs.append(mode)
                ys.append(hit["val_mae_px"])
        if xs:
            ax.plot(xs, ys, marker="o", label=arch)
    ax.set_ylabel("val MAE (px)")
    ax.set_title("Track E random frozen → full feature learning（G64）")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_track_e() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for regime in TRACK_E_REGIMES:
        data = _load_regime(regime)
        if data is None:
            rows.append({"ok": False, "regime": regime, "error": "factorial_unavailable"})
            continue
        for arch in TRACK_E_ARCH:
            for mode in TRACK_E_MODES:
                try:
                    rows.append(run_one(arch, mode, regime, data))
                except Exception as exc:
                    record_failure(f"track_e_{arch}_{mode}_{regime}", str(exc), {"tb": traceback.format_exc()[-3000:]})
                    rows.append({"ok": False, "arch": arch, "mode": mode, "regime": regime, "error": str(exc)})
    _plot(rows, OUT / "figures" / "g64_modes.png")
    payload = {"rows": rows}
    dump_json(OUT / "panel.json", payload)
    return payload
