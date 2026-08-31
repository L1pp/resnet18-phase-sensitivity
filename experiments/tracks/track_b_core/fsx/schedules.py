from __future__ import annotations

import math
from typing import Any

import numpy as np


def _check_update_step(update_step: int, total_steps: int) -> None:
    if not 1 <= int(update_step) <= int(total_steps):
        raise ValueError(f"update_step must be in [1,{total_steps}]")


def common_lr_for_update(
    update_step: int,
    *,
    total_steps: int = 3000,
    base_lr: float = 1e-3,
    eta_min: float = 1e-5,
) -> float:
    """LR used by an update when CosineAnnealingLR advances after optimizer.step()."""
    _check_update_step(update_step, total_steps)
    completed_before = int(update_step) - 1
    progress = completed_before / int(total_steps)
    return float(eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def common_lr_after_update(
    completed_steps: int,
    *,
    total_steps: int = 3000,
    base_lr: float = 1e-3,
    eta_min: float = 1e-5,
) -> float:
    if not 0 <= int(completed_steps) <= int(total_steps):
        raise ValueError(f"completed_steps must be in [0,{total_steps}]")
    progress = int(completed_steps) / int(total_steps)
    return float(eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def b4_causal_lr_for_update(
    update_step: int,
    *,
    total_steps: int = 20000,
    warmup_steps: int = 500,
    base_lr: float = 1e-3,
    eta_min: float = 1e-5,
) -> float:
    """Exact B4 update schedule: 1..500 warmup, 501..20000 cosine."""
    _check_update_step(update_step, total_steps)
    step = int(update_step)
    if not 0 < int(warmup_steps) < int(total_steps):
        raise ValueError("warmup_steps must be inside the run")
    if step <= warmup_steps:
        return float(base_lr * step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return float(eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def mean_mse(prediction: Any, truth: Any) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if pred.shape != target.shape or pred.size == 0 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("prediction/truth must be same-shape nonempty finite arrays")
    return float(np.mean((pred - target) ** 2))


def common_loss(prediction: Any, truth: Any) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    mse = mean_mse(pred, target)
    return float(mse + 0.25 * np.mean(np.abs(pred - target)))


def b7_dense_absolute_loss(endpoint_prediction: Any, endpoint_truth: Any) -> float:
    pred = np.asarray(endpoint_prediction)
    truth = np.asarray(endpoint_truth)
    if pred.shape != (128, 2) or truth.shape != (128, 2):
        raise ValueError("B7 matched absolute loss requires [128,2] endpoint occurrences")
    return mean_mse(pred, truth)


def b7_relative_loss(
    endpoint_prediction: Any,
    endpoint_truth: Any,
    anchor_prediction: Any,
    anchor_truth: Any,
) -> dict[str, float]:
    pred = np.asarray(endpoint_prediction, dtype=np.float64)
    truth = np.asarray(endpoint_truth, dtype=np.float64)
    if pred.shape != (128, 2) or truth.shape != (128, 2):
        raise ValueError("B7 relative loss requires left64 then right64 endpoint arrays")
    pair_loss = mean_mse(pred[64:] - pred[:64], truth[64:] - truth[:64])
    anchor_loss = mean_mse(anchor_prediction, anchor_truth)
    return {"pair_mean_mse": pair_loss, "anchor_mean_mse": anchor_loss, "total": pair_loss + anchor_loss}
