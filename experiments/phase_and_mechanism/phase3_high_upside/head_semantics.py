"""Frozen-head semantics: M_Δ z should shift t̂ by Δ and leave q̂ almost unchanged."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from phase1_gap_rep.common import COORD_SCALE, decompose_controls

from .operators import apply_affine
from .protocol import delta_key


def predict_from_gap(head: nn.Module, z: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run a frozen linear (or MLP) head on GAP features. Output is model-native coords."""
    head.eval()
    device = next(head.parameters()).device
    z_arr = np.asarray(z, dtype=np.float32)
    outs = []
    with torch.no_grad():
        for start in range(0, len(z_arr), batch_size):
            tensor = torch.from_numpy(z_arr[start : start + batch_size]).to(device)
            outs.append(head(tensor).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def extract_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    clf = getattr(model, "classifier", None)
    if isinstance(clf, nn.Linear):
        return clf
    if isinstance(clf, nn.Sequential):
        for module in reversed(list(clf)):
            if isinstance(module, nn.Linear):
                return module
    raise RuntimeError(f"cannot extract head from {type(model)}")


def _as_controls(pred: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
    if pred.shape[-1] == 6:
        parts = decompose_controls(pred)
        return parts["t"], parts["q"]
    if pred.shape[-1] == 2:
        return pred, None
    raise ValueError(f"unsupported head output dim {pred.shape[-1]}")


def _to_px(t_norm: np.ndarray, already_px: bool) -> np.ndarray:
    if already_px:
        return np.asarray(t_norm, dtype=np.float64)
    return np.asarray(t_norm, dtype=np.float64) * COORD_SCALE


def head_shift_metrics(
    head: nn.Module,
    z: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    delta_px: Sequence[float],
    already_px: bool = False,
) -> Dict[str, float]:
    """hat t(M z) ≈ hat t(z) + Δ, and relative geometry stays put when 6-d."""
    pred0 = predict_from_gap(head, z)
    pred1 = predict_from_gap(head, apply_affine(z, matrix, bias).astype(np.float32))
    t0, q0 = _as_controls(pred0)
    t1, q1 = _as_controls(pred1)
    t0_px = _to_px(t0, already_px)
    t1_px = _to_px(t1, already_px)
    delta = np.asarray(delta_px, dtype=np.float64).reshape(1, 2)
    shift = t1_px - t0_px
    err = shift - delta
    out = {
        "n": int(len(z)),
        "delta_px": [float(delta_px[0]), float(delta_px[1])],
        "mean_shift_px": [float(shift[:, 0].mean()), float(shift[:, 1].mean())],
        "shift_mae_px": float(np.mean(np.linalg.norm(err, axis=-1))),
        "shift_rmse_px": float(np.sqrt(np.mean(np.sum(err * err, axis=-1)))),
        "abs_shift_mae_px": float(np.mean(np.linalg.norm(shift, axis=-1))),
    }
    if q0 is not None and q1 is not None:
        q_err = np.mean(np.abs(q1 - q0), axis=(-1, -2)) * (1.0 if already_px else COORD_SCALE)
        out["q_mae_px"] = float(np.mean(q_err))
    return out


def evaluate_head_battery(
    head: nn.Module,
    z: np.ndarray,
    ops: Mapping[str, Mapping[str, Any]],
    deltas: Sequence[Sequence[float]],
    already_px: bool = False,
) -> Dict[str, Any]:
    rows = []
    for delta in deltas:
        key = delta_key(delta)
        if key not in ops:
            continue
        blob = ops[key]
        rows.append(
            head_shift_metrics(
                head,
                z,
                np.asarray(blob["matrix"]),
                np.asarray(blob["bias"]),
                delta,
                already_px=already_px,
            )
        )
    if not rows:
        return {"rows": [], "mean_shift_mae_px": None, "mean_q_mae_px": None}
    q_vals = [r["q_mae_px"] for r in rows if "q_mae_px" in r]
    return {
        "rows": rows,
        "mean_shift_mae_px": float(np.mean([r["shift_mae_px"] for r in rows])),
        "mean_q_mae_px": float(np.mean(q_vals)) if q_vals else None,
    }
