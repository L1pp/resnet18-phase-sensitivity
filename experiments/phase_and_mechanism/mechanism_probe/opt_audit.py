"""S2: 10-step 2D optimizer audit from original blob_G64. AdamW is always reset in current FT."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from error_extension.freeze import enter_train
from functional_drift.blob2d import corners4_pack, render_blob_at, train_anchor_mae_px
from phase1_gap_rep.common import images_to_tensor, seed_everything
from phase2_overnight.protocol import integer_grid_px
from phase3_high_upside.trainer import xy_loss
from today_shortcycle.fields import save_field

from .blob_io import (
    add_flat,
    device,
    load_blob,
    official_eval,
    pack_grads,
    pack_params,
    prepare_stage,
    restore_state,
    snapshot_state,
    support_loss,
    trainable,
    write_json,
)
from .const import (
    ADAMW_BETAS,
    CODE_REV,
    LR_USED,
    OPT_STEPS,
    RESULTS_ROOT,
    SEED,
    STAGES,
    SUPPORT_REGIMES,
    TRAIN_BATCH,
    WARMUP_DENSE_STEPS,
    WD,
)

FT_BATCH = 64


def tile_to_batch(images: np.ndarray, xy_norm: np.ndarray, batch: int):
    n = int(len(images))
    reps = int(np.ceil(float(batch) / float(n)))
    return np.tile(images, (reps, 1, 1))[:batch], np.tile(xy_norm, (reps, 1))[:batch]


def _g64_pack() -> Dict[str, np.ndarray]:
    grid = integer_grid_px()
    tids = np.asarray(list(SUPPORT_REGIMES["G64"]), dtype=int)
    return render_blob_at(grid[tids])


def _step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    images: np.ndarray,
    xy_norm: np.ndarray,
    batch: int,
) -> float:
    enter_train(model)
    dev = next(model.parameters()).device
    n = len(images)
    last = 0.0
    for start in range(0, n, batch):
        enter_train(model)
        xb = images_to_tensor(images[start : start + batch]).to(dev)
        yb = torch.from_numpy(np.ascontiguousarray(xy_norm[start : start + batch]).astype(np.float32)).to(dev)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss, _, _ = xy_loss(pred, yb)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
    return last


def _make_opt(model: nn.Module, kind: str) -> torch.optim.Optimizer:
    params = trainable(model)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=LR_USED, momentum=0.0, weight_decay=WD)
    if kind == "adamw_small_lr":
        return torch.optim.AdamW(params, lr=LR_USED * 0.1, weight_decay=WD, betas=ADAMW_BETAS)
    return torch.optim.AdamW(params, lr=LR_USED, weight_decay=WD, betas=ADAMW_BETAS)


def _dump(model: nn.Module, out: Path, step: int, support_images: np.ndarray, support_true: np.ndarray) -> Dict[str, Any]:
    off = official_eval(model, name=f"{out.name}_step{step}")
    rec = {
        "step": int(step),
        "mean_box_mae_px": float(off["mean_box_mae_px"]),
        "affine_removed_mae_px": float(off["affine_removed_mae_px"]),
        "train_anchor_mae_px": float(train_anchor_mae_px(model, support_images, support_true)),
    }
    step_dir = out / "traj" / f"step_{int(step):04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    save_field(step_dir / "field.npz", off["pred"], off["true"], u=off["u"])
    write_json(step_dir / "metrics.json", rec)
    enter_train(model)
    print(
        f"  dump step={step} box={rec['mean_box_mae_px']:.3f} "
        f"anchor={rec['train_anchor_mae_px']:.3f}",
        flush=True,
    )
    return rec


def _run_simple(
    stage: str,
    kind: str,
    corners: Dict[str, np.ndarray],
    out: Path,
) -> Dict[str, Any]:
    seed_everything(SEED)
    model = load_blob()
    prepare_stage(model, stage)
    opt = _make_opt(model, kind)
    series = []
    series.append(_dump(model, out, 0, corners["images"], corners["true_px"]))
    train_img, train_xy = tile_to_batch(corners["images"], corners["xy_norm"], FT_BATCH)
    t0 = time.time()
    for step in range(1, OPT_STEPS + 1):
        _step(model, opt, train_img, train_xy, FT_BATCH)
        if step in (1, OPT_STEPS):
            series.append(_dump(model, out, step, corners["images"], corners["true_px"]))
    summary = {
        "code_rev": CODE_REV,
        "stage": stage,
        "opt": kind,
        "keep_state_proxy": False,
        "elapsed_sec": float(time.time() - t0),
        "series": series,
        "box_step0": series[0]["mean_box_mae_px"],
        "box_step1": next(r["mean_box_mae_px"] for r in series if r["step"] == 1),
        "box_step10": next(r["mean_box_mae_px"] for r in series if r["step"] == OPT_STEPS),
        "anchor_step1": next(r["train_anchor_mae_px"] for r in series if r["step"] == 1),
        "note": "fresh optimizer; current functional_drift FT is this AdamW-reset case; train in enter_train + tile-to-64",
    }
    write_json(out / "summary.json", summary)
    return summary


def _run_keep_state(
    stage: str,
    corners: Dict[str, np.ndarray],
    dense: Dict[str, np.ndarray],
    out: Path,
) -> Dict[str, Any]:
    seed_everything(SEED)
    model = load_blob()
    prepare_stage(model, stage)
    opt = _make_opt(model, "adamw_reset")
    series = []
    series.append(_dump(model, out, 0, corners["images"], corners["true_px"]))
    t0 = time.time()
    dense_img, dense_xy = tile_to_batch(dense["images"], dense["xy_norm"], FT_BATCH)
    train_img, train_xy = tile_to_batch(corners["images"], corners["xy_norm"], FT_BATCH)
    for _ in range(WARMUP_DENSE_STEPS):
        _step(model, opt, dense_img, dense_xy, FT_BATCH)
    enter_train(model)
    warm = official_eval(model, name=f"{stage}_after_warmup")
    warm_rec = {
        "step": "after_dense_warmup",
        "warmup_steps": WARMUP_DENSE_STEPS,
        "mean_box_mae_px": float(warm["mean_box_mae_px"]),
        "train_anchor_mae_px": float(train_anchor_mae_px(model, corners["images"], corners["true_px"])),
        "dense_anchor_mae_px": float(train_anchor_mae_px(model, dense["images"], dense["true_px"])),
    }
    save_field(out / "field_after_warmup.npz", warm["pred"], warm["true"], u=warm["u"])
    write_json(out / "warmup.json", warm_rec)
    print(
        f"  warmup box={warm_rec['mean_box_mae_px']:.3f} "
        f"corner_anchor={warm_rec['train_anchor_mae_px']:.3f}",
        flush=True,
    )
    for step in range(1, OPT_STEPS + 1):
        _step(model, opt, train_img, train_xy, FT_BATCH)
        if step in (1, OPT_STEPS):
            rec = _dump(model, out, step, corners["images"], corners["true_px"])
            series.append(rec)
    summary = {
        "code_rev": CODE_REV,
        "stage": stage,
        "opt": "adamw_keep_state_proxy",
        "keep_state_proxy": True,
        "warmup_task": "G64_blob_same_freeze",
        "warmup_steps": WARMUP_DENSE_STEPS,
        "elapsed_sec": float(time.time() - t0),
        "series": series,
        "warmup": warm_rec,
        "box_step0": series[0]["mean_box_mae_px"],
        "box_after_warmup": warm_rec["mean_box_mae_px"],
        "box_step1": next(r["mean_box_mae_px"] for r in series if r["step"] == 1),
        "box_step10": next(r["mean_box_mae_px"] for r in series if r["step"] == OPT_STEPS),
        "anchor_step1": next(r["train_anchor_mae_px"] for r in series if r["step"] == 1),
        "ambiguity": (
            "Cannot restore original blob pretrain Adam moments from slim.pt. "
            "Proxy: 30 dense G64 steps with the SAME freeze stage, then sparse 4-corner "
            "without creating a new optimizer."
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def run() -> Dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    corners = corners4_pack()
    dense = _g64_pack()
    kinds = ("adamw_reset", "sgd", "adamw_small_lr")
    rows: List[Dict[str, Any]] = []
    root = RESULTS_ROOT / "s2_opt_audit"
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "adamw_reset_fact.json",
        {
            "fact": "sparse FT in functional_drift always constructs a new AdamW; slim.pt has no optimizer state",
            "code": "torch.optim.AdamW(params, lr=1e-3, ...) at the start of each unfreeze run",
            "existing_step1_shock_is": "AdamW reset",
        },
    )
    for stage in STAGES:
        for kind in kinds:
            out = root / f"{stage}__{kind}"
            summary_path = out / "summary.json"
            if summary_path.exists():
                rec = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
                print(f"[S2] skip existing {out.name}", flush=True)
            else:
                out.mkdir(parents=True, exist_ok=True)
                print(f"[S2] {stage} {kind}", flush=True)
                rec = _run_simple(stage, kind, corners, out)
            rows.append(rec)
        out = root / f"{stage}__adamw_keep_state_proxy"
        if (out / "summary.json").exists():
            rec = __import__("json").loads((out / "summary.json").read_text(encoding="utf-8"))
            print(f"[S2] skip existing {out.name}", flush=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            print(f"[S2] {stage} adamw_keep_state_proxy", flush=True)
            rec = _run_keep_state(stage, corners, dense, out)
        rows.append(rec)
    table = {"code_rev": CODE_REV, "rows": rows}
    write_json(root / "summary.json", table)
    from .plots import plot_opt_audit

    plot_opt_audit(table)
    print("[S2] done", flush=True)
    return table
