from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _fit_affine(pred_px: np.ndarray, true_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_px, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_px, dtype=np.float64).reshape(-1, 2)
    design = np.concatenate([pred, np.ones((len(pred), 1), dtype=np.float64)], axis=1)
    coeff, *_ = np.linalg.lstsq(design, true, rcond=None)
    corrected = design @ coeff
    return coeff, corrected


def _mae_px(pred: np.ndarray, true: np.ndarray) -> float:
    delta = np.asarray(pred, dtype=np.float64).reshape(-1, 2) - np.asarray(true, dtype=np.float64).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(delta, axis=-1)))


def closed_form_identity(support_px: np.ndarray, dense_px: np.ndarray) -> dict[str, Any]:
    coeff, _ = _fit_affine(support_px, support_px)
    design = np.concatenate(
        [np.asarray(dense_px, dtype=np.float64), np.ones((len(dense_px), 1), dtype=np.float64)], axis=1
    )
    pred = design @ coeff
    _, corrected = _fit_affine(pred, dense_px)
    return {
        "kind": "closed_form_affine_identity",
        "raw_mae_px": _mae_px(pred, dense_px),
        "affine_residual_mae_px": _mae_px(corrected, dense_px),
        "coeff": coeff.tolist(),
    }


class ExplicitXYMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def run_explicit_xy_mlp(
    support_px: np.ndarray,
    dense_px: np.ndarray,
    out_dir: str | Path,
    *,
    seed: int = 20260816,
    coord_scale: float = 223.0,
    steps: int = 3000,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    from .models import seed_all

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    chosen = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(seed)
    model = ExplicitXYMLP().to(chosen)
    support = np.asarray(support_px, dtype=np.float32) / float(coord_scale)
    reps = (batch_size + len(support) - 1) // len(support)
    tiled = np.tile(support, (reps, 1))[:batch_size]
    x = torch.from_numpy(tiled).to(chosen)
    anchor = torch.from_numpy(support).to(chosen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    best_anchor = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.mean((pred - x) ** 2) + 0.25 * torch.mean(torch.abs(pred - x))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite MLP loss at step {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        row: dict[str, float | int] = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                anchor_pred = model(anchor)
            anchor_mae = float(torch.linalg.vector_norm(anchor_pred - anchor, dim=-1).mean().cpu()) * coord_scale
            row["anchor_mae_px"] = anchor_mae
            if anchor_mae < best_anchor:
                best_anchor = anchor_mae
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
    if best_state is None:
        raise RuntimeError("MLP did not produce a best checkpoint")
    torch.save({"model_state": best_state, "seed": seed, "kind": "explicit_xy_mlp"}, target_dir / "best.pt")
    model.load_state_dict(best_state)
    model.to(chosen).eval()
    dense_norm = torch.from_numpy(np.asarray(dense_px, dtype=np.float32) / coord_scale).to(chosen)
    with torch.no_grad():
        pred_px = model(dense_norm).cpu().numpy().astype(np.float64) * coord_scale
        anchor_pred_px = model(anchor).cpu().numpy().astype(np.float64) * coord_scale
    coeff, corrected = _fit_affine(pred_px, dense_px)
    result = {
        "kind": "explicit_xy_mlp",
        "seed": seed,
        "steps": steps,
        "anchor_mae_px": _mae_px(anchor_pred_px, support_px),
        "raw_mae_px": _mae_px(pred_px, dense_px),
        "affine_residual_mae_px": _mae_px(corrected, dense_px),
        "affine_coeff": coeff.tolist(),
    }
    np.savez_compressed(
        target_dir / "predictions.npz",
        pred_px=pred_px,
        true_px=np.asarray(dense_px, dtype=np.float64),
        anchor_pred_px=anchor_pred_px,
        anchor_true_px=np.asarray(support_px, dtype=np.float64),
    )
    (target_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (target_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
