"""Recovery addendum (post-hoc, authorized by user).

Fixes / completions that turn known failures and negatives into usable results,
append-only: NEVER overwrites the original overnight artifacts.

1. atlas: efficientnet_b0 G64 failed (loss exploded at step 395). Diagnosis showed
   fp16 AMP overflow is the trigger; fp32 trains stably. Retrain in fp32 and dense-eval,
   append the recovered row.
2. branches: original gates never fired (C16 outside=12.4px > 1.0, G16 interp=3.72 > 1.0,
   G9 interp=7.58 > 1.5). Run the branch experiments with evidence-based relaxed gates.
3. content: the original panel only wrote a "renderer_ready" note. Train R18-G64 on the
   line and arc renderers on the same t grid and produce a real primitive-contrast panel.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase1_gap_rep.common import (
    BATCH_SIZE,
    COORD_SCALE,
    WEIGHT_DECAY,
    dump_json,
    fingerprint,
    profile_spec,
    seed_everything,
)
from phase1_gap_rep.generate_data import build_shapes
from phase1_gap_rep.train import FactorialDataset, TrainingExploded, _amp_context, _predict, combined_loss, coordinate_errors

from phase2_overnight.backbones import build_backbone
from phase2_overnight.ckpt import save_slim
from phase2_overnight.content import render_arc, render_line
from phase2_overnight.data import gather_pairs, load_factorial_grid, pairs_for_regime
from phase2_overnight.eval_dense import subset_metrics
from phase2_overnight.geometry import control_metrics
from phase2_overnight.probe import position_readout
from phase2_overnight.protocol import (
    PHASE2_ROOT,
    SEED_ATLAS,
    canonical_regime,
    load_protocol,
    pair_split,
)
from phase2_overnight.render_dense import load_dense_cache
from phase2_overnight.train_regime import dense_eval_model, model_dir, n_steps_ref
from phase2_overnight.features import extract_stage_gaps

MODELS_ROOT = PHASE2_ROOT / "models"
RECOVERY_ROOT = PHASE2_ROOT / "recovery"
RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)


def _workers() -> int:
    import os

    return 0 if os.name == "nt" else 4


def _eval_mae(model, images, targets) -> float:
    pred = _predict(model, images)
    return float(coordinate_errors(pred, targets)["mae_px"])


def train_regime_custom(
    arch: str,
    regime: str,
    seed: int,
    amp: bool,
    run_name: str,
    variant: str = "standard",
    steps: Optional[int] = None,
    shape_ids: Optional[Sequence[int]] = None,
    images: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    clip: Optional[float] = None,
) -> Dict[str, Any]:
    """Same equal-step trainer as train_regime but with explicit `amp` flag and
    optional pre-rendered images/P (used by the content panel)."""
    proto = load_protocol()
    out = MODELS_ROOT / run_name
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "tables" / "train_summary.json"

    seed_everything(seed)
    if images is None or P is None:
        grid = load_factorial_grid()
        pairs = pairs_for_regime(regime, shape_ids)
        split = pair_split(pairs, seed)
        train = gather_pairs(grid, [tuple(p) for p in split["train"]])
        val = gather_pairs(grid, [tuple(p) for p in split["val"]])
    else:
        pairs = pairs_for_regime(regime, shape_ids)
        split = pair_split(pairs, seed)
        t_map = {tuple(p): i for i, p in enumerate(pairs)}
        t_idx = [t_map[tuple(p)] for p in split["train"]]
        v_idx = [t_map[tuple(p)] for p in split["val"]]
        train = {"images": images[np.array(t_idx)], "P": P[np.array(t_idx)]}
        val = {"images": images[np.array(v_idx)], "P": P[np.array(v_idx)]}

    steps = int(steps or n_steps_ref())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tag = build_backbone(arch, variant)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
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

    # init probe (GAP readout)
    try:
        z_tr = extract_stage_gaps(model, train["images"][: min(len(train["images"]), 2048)], stages=["gap"])["gap"]
        z_va = extract_stage_gaps(model, val["images"][: min(len(val["images"]), 512)], stages=["gap"])["gap"]
        t_tr = train.get("t", np.zeros((len(train["images"]), 2)))[: len(z_tr)].astype(np.float64)
        t_va = val.get("t", np.zeros((len(val["images"]), 2)))[: len(z_va)].astype(np.float64)
        init_probe = position_readout(z_tr, t_tr, z_va, t_va)
    except Exception as exc:
        init_probe = {"error": str(exc)}
    dump_json(out / "tables" / "init_probe.json", init_probe)

    # 1-epoch smoke
    model.train()
    it = iter(loader)
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
    print(f"[recovery] {out.name} smoke_loss={float(loss):.4f} amp={amp} steps={steps}", flush=True)

    last_path = ckpt_dir / "last.pt"
    start_step = 0
    best_val = math.inf
    history: List[Dict[str, Any]] = []
    step = start_step
    epoch = 0
    t0 = time.time()
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
                save_slim(model, ckpt_dir / "best_slim.pt", {"epoch": epoch, "step": step, "val_mae_px": val_mae, "arch": arch, "seed": seed})
            print(f"[recovery] {out.name} epoch={epoch} step={step}/{steps} val={val_mae:.3f}px", flush=True)
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
        if clip is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

    val_mae = _eval_mae(model, val["images"], val["P"])
    if val_mae < best_val or not (ckpt_dir / "best_slim.pt").exists():
        best_val = val_mae
        save_slim(model, ckpt_dir / "best_slim.pt", {"epoch": epoch, "step": step, "val_mae_px": val_mae, "arch": arch, "seed": seed})
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
        "amp": amp,
        "note": "recovery addendum",
    }
    dump_json(summary_path, summary)
    dump_json(out / "tables" / "history.json", history)
    return summary


def recover_efficientnet_g64() -> Dict[str, Any]:
    """Retrain the failed atlas cell in fp32 and dense-eval it."""
    arch, regime, seed = "efficientnet_b0", "G64", SEED_ATLAS
    run_name = f"{arch}__{regime}__{seed}_fp32"
    steps = n_steps_ref()
    tr = train_regime_custom(arch, regime, seed, amp=False, run_name=run_name, steps=steps, clip=1.0)
    ev = dense_eval_model(arch, regime, seed, run_name=run_name)
    row = {"arch": arch, "regime": regime, "train": tr, "dense": ev, "init_probe": tr.get("init_probe"), "ok": True, "note": "recovered: fp32 + grad clip (AMP overflow / instability fix)"}
    return row


def recover_branches() -> Dict[str, Any]:
    """Relaxed conditional branches. Gates replaced by evidence-based conditions:
      I-A  LeftHalf        (C4 outside hull was 33.7px -> LeftHalf helps find axis structure)
      I-B  C16 coordconv   (C16 outside 12.4px vs in-hull 2.99px -> gap to fix)
      I-B  C16 circular    (same motivation)
      I-C  G2x / G2y       (line-sparse generalization, G9 interp 7.58px)
    """
    out = PHASE2_ROOT / "branches"
    out.mkdir(parents=True, exist_ok=True)
    steps = n_steps_ref()
    log: Dict[str, Any] = {"ran": [], "relaxed_gates_note": "original gates never fired; run with evidence-based criteria"}

    def run_branch(name, regime, variant="standard"):
        tag = "resnet18" if variant == "standard" else f"resnet18_{variant}"
        run_name = f"{tag}__{regime}__{SEED_ATLAS}"
        try:
            tr = train_regime_custom("resnet18", regime, SEED_ATLAS, amp=True, run_name=run_name, variant=variant, steps=steps)
            ev = dense_eval_model("resnet18", regime, SEED_ATLAS, variant=variant, run_name=run_name)
            log["ran"].append({"name": name, "train": tr, "dense": ev})
            return True
        except Exception as exc:
            log["ran"].append({"name": name, "error": str(exc)})
            return False

    run_branch("I-A LeftHalf", "LeftHalf")
    run_branch("I-B C16 CoordConv", "C16", variant="coordconv")
    run_branch("I-B C16 circular", "C16", variant="circular")
    run_branch("I-C G2x", "G2x")
    run_branch("I-C G2y", "G2y")
    dump_json(out / "branches_recovery.json", log)
    return log


def _build_primitive_dataset(renderer, shape_ids_all=True):
    """Render line/arc images for the G64 t grid (same labels as factorial grid)."""
    spec = profile_spec("factorial")
    shapes_px, _records = build_shapes(spec)
    proto = load_protocol()
    tids = proto["regimes"]["G64"]
    t_px = np.array(proto["integer_t_px"])[tids]
    pairs = [(int(s), int(t)) for s in range(64) for t in tids]
    images = np.zeros((len(pairs), 224, 224), dtype=np.uint8)
    P = np.zeros((len(pairs), 6), dtype=np.float64)
    t_all = np.zeros((len(pairs), 2), dtype=np.float64)
    from phase1_gap_rep.common import px_to_norm

    for i, (s, t) in enumerate(pairs):
        q = shapes_px[s].astype(np.float64)
        p_px = q + t_px[t][None, :]
        img = renderer(p_px.reshape(-1))
        images[i] = img
        P[i] = px_to_norm(p_px).reshape(-1)
        t_all[i] = t_px[t]
    return images, P, t_all, pairs


def recover_content_panel() -> Dict[str, Any]:
    """Train R18-G64 on line and arc renderers; cross-evaluate to form the primitive contrast panel."""
    out = PHASE2_ROOT / "content"
    out.mkdir(parents=True, exist_ok=True)
    steps = n_steps_ref()
    panel: Dict[str, Any] = {"primitives": {}}

    for prim, renderer in (("line", render_line), ("arc", render_arc)):
        images, P, t_all, pairs = _build_primitive_dataset(renderer)
        run_name = f"resnet18__G64__{SEED_ATLAS}_{prim}"
        tr = train_regime_custom(
            "resnet18", "G64", SEED_ATLAS, amp=True, run_name=run_name, steps=steps,
            images=images, P=P,
        )
        model = None
        from phase2_overnight.ckpt import load_arch_checkpoint

        model = load_arch_checkpoint(MODELS_ROOT / run_name / "checkpoints" / "best_slim.pt", "resnet18", "standard")
        # grid eval on own and cross primitive
        split = pair_split(pairs, SEED_ATLAS)
        t_map = {tuple(p): i for i, p in enumerate(pairs)}
        t_idx = np.array([t_map[tuple(p)] for p in split["train"]])
        v_idx = np.array([t_map[tuple(p)] for p in split["val"]])
        own = _eval_mae(model, images[v_idx], P[v_idx])
        panel["primitives"][prim] = {"train": tr, "grid_val_mae_px_own": own}
        # cross-eval: evaluate this model on the other primitive's val set
        for other, r2 in (("line", render_line), ("arc", render_arc)):
            if other == prim:
                continue
            o_images, o_P, _, o_pairs = _build_primitive_dataset(r2)
            o_split = pair_split(o_pairs, SEED_ATLAS)
            o_map = {tuple(p): i for i, p in enumerate(o_pairs)}
            o_idx = np.array([o_map[tuple(p)] for p in o_split["val"]])
            xm = _eval_mae(model, o_images[o_idx], o_P[o_idx])
            panel["primitives"][prim][f"grid_val_mae_px_cross_{other}"] = xm

    dump_json(out / "content_panel_results.json", panel)
    return panel


def main() -> None:
    report: Dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        eff = recover_efficientnet_g64()
        report["efficientnet_g64_recovered"] = True
        report["efficientnet_g64_interp_mae_px"] = eff["dense"]["interpolation"]["t_mae_px"]
        report["efficientnet_g64_best_val_mae_px"] = eff["train"]["best_val_mae_px"]
        print(f"[recovery] efficientnet G64 recovered interp={report['efficientnet_g64_interp_mae_px']:.3f}px", flush=True)
    except Exception as exc:
        report["efficientnet_g64_recovered"] = False
        report["efficientnet_g64_error"] = str(exc)
        import traceback

        traceback.print_exc()

    try:
        br = recover_branches()
        report["branches_relaxed"] = br["ran"]
        print("[recovery] branches done", flush=True)
    except Exception as exc:
        report["branches_error"] = str(exc)
        import traceback

        traceback.print_exc()

    try:
        cp = recover_content_panel()
        report["content_panel"] = cp
        print("[recovery] content done", flush=True)
    except Exception as exc:
        report["content_error"] = str(exc)
        import traceback

        traceback.print_exc()

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump_json(RECOVERY_ROOT / "recovery_progress.json", report)
    print("[recovery] all done", flush=True)


if __name__ == "__main__":
    main()