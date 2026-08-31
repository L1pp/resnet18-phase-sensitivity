"""L3: coordinate MLP width × support × last/full. Dense pretrain then sparse same-task FT."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from phase1_gap_rep.common import seed_everything
from phase2_overnight.protocol import dense_grid_px, integer_grid_px
from phase3_high_upside.io_util import state_dict_hash
from phase3_high_upside.trainer import xy_loss
from today_shortcycle.calib import apply_affine, fit_affine, mae_px
from today_shortcycle.fields import save_field

from .blob_io import device, rms, write_json
from .const import (
    CODE_REV,
    COORD_SCALE,
    LR_MIN_USED,
    LR_USED,
    MLP_DUMP,
    MLP_PRETRAIN_BOX,
    MLP_PRETRAIN_STEPS,
    MLP_SPARSE_STEPS,
    RESULTS_ROOT,
    SEED,
    SUPPORT_REGIMES,
    WD,
)


class LinearXY(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPXY(nn.Module):
    def __init__(self, hidden: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [2] + list(hidden) + [2]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ARCH = {
    "linear": tuple(),
    "mlp_s": (64, 64),
    "mlp_m": (128, 128),
    "mlp_w": (256, 256, 256),
}


def _build(arch: str) -> nn.Module:
    if arch == "linear":
        return LinearXY()
    hidden = ARCH[arch]
    return MLPXY(hidden)


def _set_stage(model: nn.Module, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if stage == "full":
        for p in model.parameters():
            p.requires_grad = True
        return
    if stage != "last":
        raise ValueError(stage)
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("no Linear")
    for p in last.parameters():
        p.requires_grad = True


def _xy_from_tids(tids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    grid = integer_grid_px()
    t_px = grid[np.asarray(list(tids), dtype=int)]
    xy = (t_px / COORD_SCALE).astype(np.float32)
    return xy, xy.copy()


def _eval_field(model: nn.Module, dev: torch.device) -> Dict[str, Any]:
    t_dense = dense_grid_px()
    xy = (t_dense / COORD_SCALE).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xy).to(dev)).cpu().numpy().reshape(1, 41, 41, 2) * COORD_SCALE
    true = t_dense.reshape(1, 41, 41, 2).astype(np.float64)
    a, b = fit_affine(pred, true)
    u = apply_affine(pred, a, b) - true
    return {
        "pred": pred,
        "true": true,
        "u": u,
        "mean_box_mae_px": mae_px(pred, true),
        "affine_removed_mae_px": mae_px(apply_affine(pred, a, b), true),
    }


def _anchor(model: nn.Module, xy: np.ndarray, dev: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(xy).to(dev)).cpu().numpy() * COORD_SCALE
    return mae_px(p, xy * COORD_SCALE)


def _tangent_r(model: nn.Module, xy_s: np.ndarray, dev: torch.device) -> Dict[str, float]:
    """Cheap R(v) at sparse step0: unit-g / GD / AdamW one-step."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {}
    snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    x = torch.from_numpy(xy_s).to(dev)
    y = x
    model.train()
    for p in params:
        if p.grad is not None:
            p.grad = None
    pred = model(x)
    loss, _, _ = xy_loss(pred, y)
    loss.backward()
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    def _eval() -> np.ndarray:
        return np.asarray(_eval_field(model, dev)["pred"], dtype=np.float64)

    def _sup() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(x).detach().cpu().numpy() * COORD_SCALE

    f0 = _eval()
    s0 = _sup()
    gnorm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item())
    if gnorm < 1e-12:
        return {"amp_unit": float("nan")}
    eps = 1e-3
    model.load_state_dict(snap)
    model.to(dev)
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(eps * g / gnorm)
    amp_unit = rms(_eval() - f0) / max(rms(_sup() - s0), 1e-12)
    model.load_state_dict(snap)
    model.to(dev)
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(-float(LR_USED) * g)
    d_gd = _eval()
    amp_gd = rms(d_gd - f0) / max(rms(_sup() - s0), 1e-12)
    box_gd = mae_px(d_gd, np.asarray(_eval_field(model, dev)["true"]))
    model.load_state_dict(snap)
    model.to(dev)
    opt = torch.optim.AdamW(params, lr=LR_USED, weight_decay=WD)
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()
    d_ad = _eval()
    s_ad = _sup()
    amp_adam = rms(d_ad - f0) / max(rms(s_ad - s0), 1e-12)
    model.load_state_dict(snap)
    model.to(dev)
    true = dense_grid_px().reshape(1, 41, 41, 2)
    return {
        "amp_unit": float(amp_unit),
        "amp_gd": float(amp_gd),
        "amp_adam": float(amp_adam),
        "d_box_gd": float(mae_px(d_gd, true) - mae_px(f0, true)),
        "d_box_adam": float(mae_px(d_ad, true) - mae_px(f0, true)),
    }


def _pretrain(arch: str, dev: torch.device, out: Path) -> nn.Module:
    slim = out / "pretrained_slim.pt"
    seed_everything(SEED)
    model = _build(arch).to(dev)
    _set_stage(model, "full")
    if slim.exists():
        payload = torch.load(slim, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model.to(dev)
    xy, y = _xy_from_tids(SUPPORT_REGIMES["G64"])
    x = torch.from_numpy(xy).to(dev)
    yt = torch.from_numpy(y).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_USED, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_PRETRAIN_STEPS, eta_min=LR_MIN_USED)
    for step in range(1, MLP_PRETRAIN_STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss, _, _ = xy_loss(pred, yt)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == MLP_PRETRAIN_STEPS:
            train_mae = _anchor(model, xy, dev)
            if train_mae < MLP_PRETRAIN_BOX:
                box = _eval_field(model, dev)["mean_box_mae_px"]
                if box < MLP_PRETRAIN_BOX or arch == "linear":
                    break
    pre = _eval_field(model, dev)
    torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()}}, slim)
    save_field(out / "field_pretrain.npz", pre["pred"], pre["true"], u=pre["u"])
    write_json(out / "pretrain.json", {"box": pre["mean_box_mae_px"], "steps_cap": MLP_PRETRAIN_STEPS})
    return model


def run_one(arch: str, stage: str, support: str, dev: torch.device) -> Dict[str, Any]:
    run_id = f"{arch}__{stage}__{support}"
    out = RESULTS_ROOT / "l3_mlp" / run_id
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "train_summary.json"
    if summary_path.exists() and (out / "field_sparse_final.npz").exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    pre_root = RESULTS_ROOT / "l3_mlp" / f"_pretrain_{arch}"
    pre_root.mkdir(parents=True, exist_ok=True)
    model = _pretrain(arch, dev, pre_root)
    seed_everything(SEED)
    _set_stage(model, stage)
    pre = _eval_field(model, dev)
    xy_s, y_s = _xy_from_tids(SUPPORT_REGIMES[support])
    tangent = _tangent_r(model, xy_s, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_USED, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_SPARSE_STEPS, eta_min=LR_MIN_USED)
    xc = torch.from_numpy(xy_s).to(dev)
    yc = torch.from_numpy(y_s).to(dev)
    series: List[Dict[str, Any]] = []
    t0 = time.time()

    def dump(step: int) -> None:
        fld = _eval_field(model, dev)
        sub = out / "traj" / f"step_{int(step):04d}"
        sub.mkdir(parents=True, exist_ok=True)
        save_field(sub / "field.npz", fld["pred"], fld["true"], u=fld["u"])
        series.append(
            {
                "step": int(step),
                "mean_box": float(fld["mean_box_mae_px"]),
                "u": float(fld["affine_removed_mae_px"]),
                "train_anchor": float(_anchor(model, xy_s, dev)),
            }
        )

    dump(0)
    for step in range(1, MLP_SPARSE_STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(xc)
        loss, _, _ = xy_loss(pred, yc)
        loss.backward()
        opt.step()
        sched.step()
        if step in MLP_DUMP:
            dump(step)
    final = _eval_field(model, dev)
    save_field(out / "field_sparse_final.npz", final["pred"], final["true"], u=final["u"])
    write_json(out / "traj" / "series.json", {"rows": series})
    summary = {
        "code_rev": CODE_REV,
        "arch": arch,
        "stage": stage,
        "support": support,
        "n_support": len(SUPPORT_REGIMES[support]),
        "pretrain_box": float(pre["mean_box_mae_px"]),
        "sparse_box": float(final["mean_box_mae_px"]),
        "box_ratio": float(final["mean_box_mae_px"] / max(pre["mean_box_mae_px"], 1e-6)),
        "sparse_train_anchor": float(_anchor(model, xy_s, dev)),
        "elapsed_sec": float(time.time() - t0),
        "tangent_step0": tangent,
        "init_hash": state_dict_hash(model.state_dict()),
        "series": series,
    }
    write_json(summary_path, summary)
    print(
        f"[L3] {run_id} pre={summary['pretrain_box']:.4f} "
        f"sparse={summary['sparse_box']:.4f} ratio={summary['box_ratio']:.2f}",
        flush=True,
    )
    return summary


def run() -> Dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    dev = device()
    rows = []
    supports = ("corners4", "G9", "G16", "G64")
    for arch in ("linear", "mlp_s", "mlp_m", "mlp_w"):
        stages = ("full",) if arch == "linear" else ("last", "full")
        for stage in stages:
            for support in supports:
                rows.append(run_one(arch, stage, support, dev))
    table = {"code_rev": CODE_REV, "rows": rows}
    write_json(RESULTS_ROOT / "l3_mlp" / "summary.json", table)
    from .plots import plot_mlp_phase

    plot_mlp_phase(table)
    print("[L3] done", flush=True)
    return table
