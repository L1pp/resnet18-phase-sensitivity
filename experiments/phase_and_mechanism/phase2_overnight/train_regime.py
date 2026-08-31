"""Equal-step trainer for Phase 2 regimes. Does not write into phase1 directories."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase1_gap_rep.common import (
    BATCH_SIZE,
    COORD_SCALE,
    WEIGHT_DECAY,
    decompose_controls,
    dump_json,
    fingerprint,
    profile_spec,
    seed_everything,
)
from phase1_gap_rep.train import FactorialDataset, TrainingExploded, _amp_context, combined_loss, coordinate_errors

from .backbones import build_backbone
from .ckpt import save_slim
from .data import gather_pairs, load_factorial_grid, pairs_for_regime
from .eval_dense import eval_point_labels, subset_metrics
from .features import extract_stage_gaps
from .geometry import control_metrics
from .probe import position_readout
from .protocol import (
    GEOMETRY_FP,
    PHASE2_ROOT,
    SEED_ATLAS,
    TRAJECTORY_EPOCHS,
    canonical_regime,
    load_protocol,
    pair_split,
)
from .render_dense import load_dense_cache

MODELS_ROOT = PHASE2_ROOT / "models"


def _workers() -> int:
    if os.name == "nt":
        return 0
    return 4


def n_steps_ref() -> int:
    proto = load_protocol()
    pairs = pairs_for_regime("G64")
    split = pair_split(pairs, SEED_ATLAS)
    n_train = len(split["train"])
    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    return steps_per_epoch * 100


def model_dir(arch: str, regime: str, seed: int, variant: str = "standard", run_name: Optional[str] = None) -> Path:
    if run_name:
        return MODELS_ROOT / run_name
    tag = arch if variant == "standard" else f"{arch}_{variant}"
    return MODELS_ROOT / f"{tag}__{canonical_regime(regime)}__{seed}"


def _eval_mae(model, images, targets) -> float:
    from phase1_gap_rep.train import _predict

    pred = _predict(model, images)
    return float(coordinate_errors(pred, targets)["mae_px"])


def train_regime(
    arch: str,
    regime: str,
    seed: int,
    variant: str = "standard",
    shape_ids: Optional[Sequence[int]] = None,
    steps: Optional[int] = None,
    trajectory: bool = False,
    skip_if_done: bool = True,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    proto = load_protocol()
    out = model_dir(arch, regime, seed, variant, run_name=run_name)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "tables" / "train_summary.json"
    if skip_if_done and summary_path.exists():
        print(f"[phase2 train] skip done {out.name}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    seed_everything(seed)
    grid = load_factorial_grid()
    pairs = pairs_for_regime(regime, shape_ids)
    split = pair_split(pairs, seed)
    train = gather_pairs(grid, [tuple(p) for p in split["train"]])
    val = gather_pairs(grid, [tuple(p) for p in split["val"]])
    steps = int(steps or n_steps_ref())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    model, tag = build_backbone(arch, variant)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    # cosine over `steps` optimizer updates
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    nw = _workers()
    loader = DataLoader(
        FactorialDataset(train["images"], train["P"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=nw,
        pin_memory=bool(nw),
        persistent_workers=bool(nw),
        generator=torch.Generator().manual_seed(seed),
    )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    save_slim(model, ckpt_dir / "init.pt", {"epoch": 0, "arch": arch, "variant": variant, "seed": seed})

    # init probe (GAP)
    try:
        z_tr = extract_stage_gaps(model, train["images"][: min(len(train["images"]), 2048)], stages=["gap"])["gap"]
        z_va = extract_stage_gaps(model, val["images"][: min(len(val["images"]), 512)], stages=["gap"])["gap"]
        t_tr = train["t"][: len(z_tr)].astype(np.float64)
        t_va = val["t"][: len(z_va)].astype(np.float64)
        init_probe = position_readout(z_tr, t_tr, z_va, t_va)
    except Exception as exc:
        init_probe = {"error": str(exc)}
    dump_json(out / "tables" / "init_probe.json", init_probe)

    # 1-epoch smoke
    model.train()
    it = iter(loader)
    smoke_loss = 0.0
    n_smoke = 0
    try:
        xb, yb = next(it)
    except StopIteration:
        raise RuntimeError("empty loader")
    xb, yb = xb.to(dev), yb.to(dev)
    with _amp_context(dev, amp):
        pred = model(xb)
        loss, _, _ = combined_loss(pred, yb, 0.25)
    if not math.isfinite(float(loss.detach().cpu())):
        raise TrainingExploded("smoke loss not finite")
    print(f"[phase2 train] {out.name} smoke_loss={float(loss):.4f} steps={steps}")

    last_path = ckpt_dir / "last.pt"
    start_step = 0
    best_val = math.inf
    history: List[Dict[str, Any]] = []
    if last_path.exists():
        payload = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            opt.load_state_dict(payload["optimizer_state"])
        start_step = int(payload.get("step") or 0)
        best_val = float(payload.get("best_val_mae_px", math.inf))
        print(f"[phase2 train] resume {out.name} step={start_step}")

    step = start_step
    epoch = 0
    t0 = time.time()
    traj_saved = set()
    iterator = iter(loader)
    model.train()
    while step < steps:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = iter(loader)
            xb, yb = next(iterator)
            val_mae = _eval_mae(model, val["images"], val["P"])
            train_mae = _eval_mae(model, train["images"][:512], train["P"][:512])
            history.append({"epoch": epoch, "step": step, "train_mae_px": train_mae, "val_mae_px": val_mae})
            if val_mae < best_val:
                best_val = val_mae
                save_slim(
                    model,
                    ckpt_dir / "best_slim.pt",
                    {"epoch": epoch, "step": step, "val_mae_px": val_mae, "arch": arch, "seed": seed, "fingerprint": GEOMETRY_FP},
                )
            if trajectory and epoch in TRAJECTORY_EPOCHS and epoch not in traj_saved:
                save_slim(model, ckpt_dir / f"epoch_{epoch:03d}.pt", {"epoch": epoch, "step": step})
                traj_saved.add(epoch)
            print(f"[phase2 train] {out.name} epoch={epoch} step={step}/{steps} val={val_mae:.3f}px")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "best_val_mae_px": best_val,
                    "arch": arch,
                    "variant": variant,
                    "seed": seed,
                },
                last_path,
            )
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(set_to_none=True)
        with _amp_context(dev, amp):
            pred = model(xb)
            loss, _, _ = combined_loss(pred, yb, 0.25)
        if not math.isfinite(float(loss.detach().cpu())):
            raise TrainingExploded(f"loss exploded at step {step}")
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

    val_mae = _eval_mae(model, val["images"], val["P"])
    if val_mae < best_val or not (ckpt_dir / "best_slim.pt").exists():
        best_val = val_mae
        save_slim(model, ckpt_dir / "best_slim.pt", {"epoch": epoch, "step": step, "val_mae_px": val_mae, "arch": arch, "seed": seed})
    if trajectory:
        save_slim(model, ckpt_dir / "epoch_100.pt", {"epoch": 100, "step": step})
    save_slim(model, ckpt_dir / "last_slim.pt", {"epoch": epoch, "step": step, "val_mae_px": val_mae})

    summary = {
        "arch": arch,
        "variant": variant,
        "regime": canonical_regime(regime),
        "seed": seed,
        "steps": step,
        "n_train": int(len(train["images"])),
        "n_val": int(len(val["images"])),
        "unique_images": int(len(train["images"])),
        "presentations_per_image": float(step * BATCH_SIZE / max(1, len(train["images"]))),
        "best_val_mae_px": best_val,
        "final_val_mae_px": val_mae,
        "init_probe": init_probe,
        "elapsed_sec": time.time() - t0,
        "protocol_hash": proto["protocol_hash"],
        "fingerprint": fingerprint(profile_spec("factorial")),
        "tag": tag,
    }
    dump_json(summary_path, summary)
    dump_json(out / "tables" / "history.json", history)
    return summary


def dense_eval_model(
    arch: str,
    regime: str,
    seed: int,
    variant: str = "standard",
    run_name: Optional[str] = None,
    eval_shape_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    from phase1_gap_rep.train import _predict
    from .ckpt import load_arch_checkpoint

    out = model_dir(arch, regime, seed, variant, run_name=run_name)
    path = out / "checkpoints" / "best_slim.pt"
    model = load_arch_checkpoint(path, arch, variant)
    cache = load_dense_cache()
    keep = list(range(cache["images"].shape[0]))
    if eval_shape_ids is not None:
        eval_set = set(int(x) for x in eval_shape_ids)
        keep = [i for i, sid in enumerate(cache["eval_shape_ids"]) if int(sid) in eval_set]
        if not keep:
            raise RuntimeError("dense eval shape filter is empty")
    n_s = len(keep)
    n_g = cache["images"].shape[1]
    flat = np.asarray(cache["images"])[keep].reshape(n_s * n_g, 224, 224)
    P = cache["P"][keep].reshape(n_s * n_g, 6)
    pred = _predict(model, flat)
    metrics, _vec, _pt, _tt = control_metrics(pred, P)
    true_t = cache["t"][keep].reshape(n_s * n_g, 2)
    pred_t = decompose_controls(pred)["t"] * COORD_SCALE
    tids = load_protocol()["regimes"][canonical_regime(regime)]
    labels = eval_point_labels(tids)

    def mask_rep(m):
        return np.repeat(m[None, :], n_s, axis=0).reshape(-1)

    err = np.linalg.norm(pred_t - true_t, axis=-1)
    nearest = mask_rep(labels["nearest_train_px"])
    hull_d = mask_rep(labels["hull_distance_px"])

    def _bins(dist, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (dist >= lo) & (dist < hi)
            if np.any(m):
                rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "t_mae_px": float(np.mean(err[m]))})
        return rows

    out_m = {
        "overall": metrics,
        "exact_train": subset_metrics(pred_t, true_t, mask_rep(labels["exact_train"])),
        "interpolation": subset_metrics(pred_t, true_t, mask_rep(labels["interpolation"])),
        "in_hull": subset_metrics(pred_t, true_t, mask_rep(labels["in_hull"])),
        "outside_hull": subset_metrics(pred_t, true_t, mask_rep(labels["outside_hull"])),
        "n_eval_shapes": n_s,
        "eval_shape_ids": [int(cache["eval_shape_ids"][i]) for i in keep],
        "nearest_train_px_mean": float(np.mean(labels["nearest_train_px"])),
        "hull_distance_px_mean": float(np.mean(labels["hull_distance_px"])),
        "mae_vs_nearest_train": _bins(nearest, np.linspace(0.0, float(nearest.max() + 1e-6), 9)),
        "mae_vs_hull_distance": _bins(hull_d, np.linspace(0.0, float(hull_d.max() + 1e-6), 9)),
    }
    dump_json(out / "tables" / "dense_eval.json", out_m)
    return out_m
