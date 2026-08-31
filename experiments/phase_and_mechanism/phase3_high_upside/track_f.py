"""Track F optional: translation vs rotation/scale inversion. One seed only."""

from __future__ import annotations

import traceback
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .backbones_xy import build_xy_backbone
from .io_util import dump_json
from .primitives import render_polygon
from .protocol import PHASE3_ROOT, SEED_TRACK_F, TIME_BUDGET_SEC, record_failure
from .trainer import TrainingExploded, mae_px, predict_xy, train_xy

OUT = PHASE3_ROOT / "track_f"


def _sample(n: int, seed: int, hold_combo: bool) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 224, 224), dtype=np.uint8)
    y = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        cx = float(rng.uniform(70.0, 154.0))
        cy = float(rng.uniform(70.0, 154.0))
        theta = float(rng.uniform(0.0, 180.0))
        scale = float(rng.uniform(0.7, 1.4))
        if hold_combo and (cx > 120 and theta > 90 and scale > 1.05):
            cx = float(rng.uniform(70.0, 118.0))
        images[i] = render_polygon(cx, cy, {"n_sides": 3, "radius": 16.0 * scale, "theta_deg": theta}, 224)
        y[i] = np.array([cx / 223.0, cy / 223.0, theta / 180.0, np.log(scale)], dtype=np.float64)
    return images, y


def run_track_f(elapsed_sec: float = 0.0) -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    if elapsed_sec > 0.8 * TIME_BUDGET_SEC:
        payload = {"ok": False, "skipped": True, "reason": "time_budget"}
        dump_json(OUT / "panel.json", payload)
        return payload
    try:
        tr_img, tr_y = _sample(3000, SEED_TRACK_F, hold_combo=True)
        va_img, va_y = _sample(400, SEED_TRACK_F + 1, hold_combo=False)
        te_img, te_y = _sample(600, SEED_TRACK_F + 2, hold_combo=False)
        model, _ = build_xy_backbone("resnet18")
        # replace 2-d head with 4-d
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 4)
        out = OUT / "models" / f"affine_r18__{SEED_TRACK_F}"
        summary = train_xy(
            model,
            tr_img,
            tr_y,
            va_img,
            va_y,
            out,
            seed=SEED_TRACK_F,
            arch="resnet18",
            image_size=224,
            steps=3000,
            mode="R3",
            amp=True,
            extra={"target": "tx_ty_theta_logs"},
        )
        payload_ckpt = torch.load(out / "checkpoints" / "best_slim.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload_ckpt["model_state"])
        model.eval()
        pred = predict_xy(model, te_img)
        t_mae = mae_px(pred[:, :2], te_y[:, :2], 224)
        th_mae = float(np.mean(np.abs(pred[:, 2] - te_y[:, 2])) * 180.0)
        s_mae = float(np.mean(np.abs(np.exp(pred[:, 3]) - np.exp(te_y[:, 3]))))
        payload = {
            "ok": True,
            "val_mae_px_like": summary["best_val_mae_px"],
            "test_t_mae_px": t_mae,
            "test_theta_mae_deg": th_mae,
            "test_scale_mae": s_mae,
            "note": "translation vs rotation/scale; one seed only",
        }
    except Exception as exc:
        record_failure("track_f", str(exc), {"tb": traceback.format_exc()[-3000:]})
        payload = {"ok": False, "error": str(exc)}
    dump_json(OUT / "panel.json", payload)
    return payload
