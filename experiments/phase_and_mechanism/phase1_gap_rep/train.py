"""Train ResNet18+GAP absolute-coordinate regression.

New L1 runs write under results/phase1_gap_rep/{run}/ and never touch
the frozen factorial wreckage (data, checkpoints, history, metrics).
"""

from __future__ import annotations

import csv
import math
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import (
    ANALYSIS_GATE_MAE_PX,
    BATCH_SIZE,
    COORD_SCALE,
    EARLY_STOP_MAE_PX,
    RESULTS_ROOT,
    SEED,
    TRAIN_RUN_ORDER,
    WEIGHT_DECAY,
    TrainRun,
    build_model,
    device,
    dirs,
    dump_json,
    ensure_run_dirs,
    fingerprint,
    images_to_tensor,
    profile_spec,
    run_dirs,
    seed_everything,
    setup_matplotlib_chinese,
    train_run_config,
    train_run_spec,
    write_run_metadata,
)
from .generate_data import load_split


class FactorialDataset(Dataset):
    def __init__(self, images: np.ndarray, targets: np.ndarray) -> None:
        self.images = np.ascontiguousarray(images)
        self.targets = np.ascontiguousarray(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        target = torch.from_numpy(self.targets[index].copy())
        return image, target


class TrainingExploded(RuntimeError):
    pass


def _amp_context(dev: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16)


def _write_csv(path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_history(path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _predict(model: nn.Module, images: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    dev = next(model.parameters()).device
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images_to_tensor(images[start : start + batch_size]).to(dev)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def coordinate_errors(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    endpoint = float(np.mean(np.abs(diff[:, [0, 1, 4, 5]])))
    interior = float(np.mean(np.abs(diff[:, [2, 3]])))
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_px": mae * COORD_SCALE,
        "rmse_px": rmse * COORD_SCALE,
        "endpoint_mae_px": endpoint * COORD_SCALE,
        "interior_mae_px": interior * COORD_SCALE,
    }


def combined_loss(pred: torch.Tensor, target: torch.Tensor, l1_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    return mse + l1_weight * l1, mse, l1


def load_model(profile: str, checkpoint: str = "best") -> nn.Module:
    spec = profile_spec(profile)
    name = "resnet18_best.pt" if checkpoint == "best" else "resnet18_last.pt"
    path = dirs(spec)["checkpoints"] / name
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; train after the dataset is reviewed")
    payload = torch.load(path, map_location=device(), weights_only=False)
    if payload.get("fingerprint") != fingerprint(spec):
        raise RuntimeError("checkpoint fingerprint mismatch")
    model = build_model().to(device())
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_run_model(run_name: str, checkpoint: str = "best") -> nn.Module:
    name = "resnet18_best.pt" if checkpoint == "best" else "resnet18_last.pt"
    path = run_dirs(run_name)["checkpoints"] / name
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    payload = torch.load(path, map_location=device(), weights_only=False)
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    if payload.get("fingerprint") != fingerprint(spec):
        raise RuntimeError("checkpoint fingerprint mismatch")
    model = build_model().to(device())
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def evaluate(profile: str = "factorial") -> Dict[str, Any]:
    spec = profile_spec(profile)
    model = load_model(spec.name, "best")
    metrics: Dict[str, Any] = {"profile": spec.name, "fingerprint": fingerprint(spec)}
    for split in ("train", "val", "test"):
        data = load_split(spec, split)
        pred = _predict(model, data["images"])
        errors = coordinate_errors(pred, data["P"])
        metrics[split] = errors
        print(
            f"[eval:{spec.name}] {split} MAE={errors['mae_px']:.3f}px "
            f"RMSE={errors['rmse_px']:.3f}px endpoint={errors['endpoint_mae_px']:.3f}px "
            f"P1={errors['interior_mae_px']:.3f}px"
        )
    test_mae = float(metrics["test"]["mae_px"])
    metrics["analysis_gate_mae_px"] = ANALYSIS_GATE_MAE_PX
    metrics["ready_for_representation_analysis"] = bool(test_mae <= ANALYSIS_GATE_MAE_PX)
    dump_json(dirs(spec)["tables"] / "metrics.json", metrics)
    flat = {"profile": spec.name}
    for split, errors in metrics.items():
        if isinstance(errors, dict):
            for key, value in errors.items():
                flat[f"{split}_{key}"] = value
    _write_csv(dirs(spec)["tables"] / "metrics.csv", [flat])
    if not metrics["ready_for_representation_analysis"]:
        print(
            f"[eval:{spec.name}] test MAE={test_mae:.3f}px 未到分析门槛 {ANALYSIS_GATE_MAE_PX}px；"
            "先看训练曲线，不要强行做表征分解。"
        )
    return metrics


def evaluate_run(run_name: str) -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    model = load_run_model(run.name, "best")
    d = run_dirs(run.name)
    metrics: Dict[str, Any] = {
        "run": run.name,
        "data_profile": spec.name,
        "fingerprint": fingerprint(spec),
        "loss": f"MSE + {run.l1_weight} * L1",
    }
    for split in ("train", "val", "test"):
        data = load_split(spec, split)
        pred = _predict(model, data["images"])
        errors = coordinate_errors(pred, data["P"])
        metrics[split] = errors
        print(
            f"[eval:{run.name}] {split} MAE={errors['mae_px']:.3f}px "
            f"RMSE={errors['rmse_px']:.3f}px endpoint={errors['endpoint_mae_px']:.3f}px "
            f"P1={errors['interior_mae_px']:.3f}px"
        )
    test_mae = float(metrics["test"]["mae_px"])
    metrics["analysis_gate_mae_px"] = ANALYSIS_GATE_MAE_PX
    metrics["ready_for_representation_analysis"] = bool(test_mae <= ANALYSIS_GATE_MAE_PX)
    dump_json(d["tables"] / "metrics.json", metrics)
    flat = {"run": run.name, "data_profile": spec.name}
    for split, errors in metrics.items():
        if isinstance(errors, dict):
            for key, value in errors.items():
                flat[f"{split}_{key}"] = value
    _write_csv(d["tables"] / "metrics.csv", [flat])
    write_run_metadata(run, "eval", {"metrics": metrics})
    if not metrics["ready_for_representation_analysis"]:
        print(
            f"[eval:{run.name}] test MAE={test_mae:.3f}px 未到分析门槛 {ANALYSIS_GATE_MAE_PX}px；"
            "先看训练曲线，不要强行做表征分解。"
        )
    return metrics


def _build_optimizer(run: TrainRun, model: nn.Module):
    if run.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=run.lr, weight_decay=WEIGHT_DECAY)
    if run.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=run.lr,
            momentum=run.momentum,
            weight_decay=WEIGHT_DECAY,
        )
    raise ValueError(f"unknown optimizer {run.optimizer}")


def _build_scheduler(run: TrainRun, optimizer: torch.optim.Optimizer):
    if run.schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.epochs, eta_min=run.lr_min)
    if run.schedule == "linear":
        end_factor = run.lr_min / run.lr if run.lr else 1.0
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=max(1, run.epochs),
        )
    if run.schedule == "constant":
        return None
    raise ValueError(f"unknown schedule {run.schedule}")


def _exploded(train_loss: float, val_mae: float) -> bool:
    if not math.isfinite(train_loss) or not math.isfinite(val_mae):
        return True
    return val_mae > 800.0


def _plot_history(history: Sequence[Mapping[str, Any]], path, title: str) -> None:
    if not history:
        return
    plt = setup_matplotlib_chinese()
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(epochs, [float(row["train_mae_px"]) for row in history], label="train MAE")
    ax.plot(epochs, [float(row["val_mae_px"]) for row in history], label="val MAE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (px)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _train_loop(run: TrainRun, force: bool = False, resume: bool = False, note: str = "") -> Dict[str, Any]:
    spec = profile_spec(run.data_profile)
    d = ensure_run_dirs(run.name)
    best_path = d["checkpoints"] / "resnet18_best.pt"
    last_path = d["checkpoints"] / "resnet18_last.pt"
    history_path = d["tables"] / "history.csv"
    if force and resume:
        raise RuntimeError("use either --force (scratch) or --resume, not both")
    if best_path.exists() and not force and not resume:
        raise RuntimeError(f"checkpoint exists: {best_path}; use --resume to continue or --force to restart")
    if resume and not last_path.exists() and not best_path.exists():
        raise FileNotFoundError(f"cannot resume: missing {last_path} and {best_path}")

    train_data = load_split(spec, "train")
    val_data = load_split(spec, "val")
    seed_everything(SEED)
    dev = device()
    amp = bool(dev.type == "cuda") if run.amp is None else bool(run.amp and dev.type == "cuda")
    model = build_model().to(dev)
    history: List[Dict[str, Any]] = _read_history(history_path) if resume else []
    start_epoch = 0
    best_val = math.inf
    source_val = None

    if resume:
        ckpt_path = last_path if last_path.exists() else best_path
        payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if payload.get("fingerprint") != fingerprint(spec):
            raise RuntimeError("checkpoint fingerprint mismatch")
        model.load_state_dict(payload["model_state"])
        start_epoch = int(payload.get("epoch") or 0)
        if history:
            start_epoch = max(start_epoch, int(history[-1]["epoch"]))
        best_val = float(payload.get("best_val_mae_px", payload.get("val_mae_px", math.inf)))
        if best_path.exists():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            best_val = min(best_val, float(best_payload.get("val_mae_px", best_val)))
        print(
            f"[train:{run.name}] resume weights@epoch {start_epoch} best_val={best_val:.3f}px "
            f"lr={run.lr} -> {run.lr_min} remaining={run.epochs - start_epoch}"
        )
    elif run.resume_from:
        src = dirs(run.resume_from)["checkpoints"] / "resnet18_best.pt"
        if not src.exists():
            raise FileNotFoundError(f"cannot init from {src}")
        payload = torch.load(src, map_location=dev, weights_only=False)
        if payload.get("fingerprint") != fingerprint(spec):
            raise RuntimeError("source checkpoint fingerprint mismatch")
        model.load_state_dict(payload["model_state"])
        source_val = float(payload.get("val_mae_px", math.inf))
        torch.save(
            {
                "model_state": payload["model_state"],
                "epoch": 0,
                "fingerprint": fingerprint(spec),
                "run": run.name,
                "source": str(src),
                "source_val_mae_px": source_val,
            },
            d["checkpoints"] / "resnet18_init.pt",
        )
        print(
            f"[train:{run.name}] init from {src} source_val={source_val:.3f}px "
            f"lr={run.lr} -> {run.lr_min} epochs={run.epochs} L1={run.l1_weight}"
        )
    else:
        print(
            f"[train:{run.name}] scratch {run.optimizer} lr={run.lr} -> {run.lr_min} "
            f"schedule={run.schedule} epochs={run.epochs} L1={run.l1_weight}"
        )

    optimizer = _build_optimizer(run, model)
    scheduler = _build_scheduler(run, optimizer)
    if resume:
        payload = torch.load(last_path if last_path.exists() else best_path, map_location=dev, weights_only=False)
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(
        FactorialDataset(train_data["images"], train_data["P"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED + start_epoch),
    )
    epochs_no_improve = 0
    stage_best = math.inf
    stopped_reason = "max_epochs"
    dump_json(
        d["config"] / "train_config.json",
        {
            **train_run_config(run),
            "fingerprint": fingerprint(spec),
            "resume": resume,
            "resume_from_epoch": start_epoch,
            "source_val_mae_px": source_val,
            "note": note,
        },
    )
    epoch = start_epoch
    end_epoch = run.epochs
    if start_epoch >= end_epoch:
        print(f"[train:{run.name}] already finished at epoch {start_epoch}")
        return evaluate_run(run.name)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        running = 0.0
        running_mse = 0.0
        running_l1 = 0.0
        seen = 0
        for images, targets in loader:
            images, targets = images.to(dev), targets.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(dev, amp):
                pred = model(images)
                loss, mse, l1 = combined_loss(pred, targets, run.l1_weight)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if run.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip_grad_norm)
                optimizer.step()
            running += float(loss.detach()) * len(images)
            running_mse += float(mse.detach()) * len(images)
            running_l1 += float(l1.detach()) * len(images)
            seen += len(images)
        train_loss = running / max(1, seen)
        train_mse = running_mse / max(1, seen)
        train_l1 = running_l1 / max(1, seen)
        model.eval()
        val_pred = _predict(model, val_data["images"])
        val_errors = coordinate_errors(val_pred, val_data["P"])
        train_pred = _predict(model, train_data["images"])
        train_errors = coordinate_errors(train_pred, train_data["P"])
        lr_used = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        if _exploded(train_loss, val_errors["mae_px"]):
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_l1": train_l1,
                "train_mae_px": train_errors["mae_px"],
                "val_mae_px": val_errors["mae_px"],
                "val_rmse_px": val_errors["rmse_px"],
                "lr_used": lr_used,
                "lr_next": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            _write_csv(history_path, history)
            raise TrainingExploded(
                f"epoch {epoch} train_loss={train_loss} val_mae={val_errors['mae_px']:.3f}px"
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_l1": train_l1,
            "train_mae_px": train_errors["mae_px"],
            "val_mae_px": val_errors["mae_px"],
            "val_rmse_px": val_errors["rmse_px"],
            "lr_used": lr_used,
            "lr_next": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        _write_csv(history_path, history)
        _plot_history(history, d["figures"] / "train_mae.png", f"{run.name} MAE")
        print(
            f"[train:{run.name}] epoch {epoch}/{end_epoch} "
            f"train_mae={train_errors['mae_px']:.3f}px val_mae={val_errors['mae_px']:.3f}px "
            f"loss={train_loss:.5g} lr={lr_used:.4g}"
        )
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "fingerprint": fingerprint(spec),
            "run": run.name,
            "val_mae_px": val_errors["mae_px"],
            "best_val_mae_px": best_val,
        }
        if val_errors["mae_px"] < stage_best - 1e-12:
            stage_best = val_errors["mae_px"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if val_errors["mae_px"] < best_val - 1e-12:
            best_val = val_errors["mae_px"]
            payload["best_val_mae_px"] = best_val
            torch.save(payload, best_path)
        torch.save(payload, last_path)
        if val_errors["mae_px"] <= EARLY_STOP_MAE_PX:
            stopped_reason = "val_mae_le_0.2px"
            break
        if epochs_no_improve >= run.patience:
            stopped_reason = f"patience_{run.patience}"
            break

    _write_csv(history_path, history)
    _plot_history(history, d["figures"] / "train_mae.png", f"{run.name} MAE")
    summary = {
        "run": run.name,
        "best_val_mae_px": best_val,
        "final_epoch": epoch,
        "stopped_reason": stopped_reason,
        "early_stop_mae_px": EARLY_STOP_MAE_PX,
        "patience": run.patience,
        "max_epochs": end_epoch,
        "resume": resume,
        "resume_from_epoch": start_epoch,
        "source_val_mae_px": source_val,
        "lr": run.lr,
        "lr_min": run.lr_min,
        "schedule": run.schedule,
        "optimizer": run.optimizer,
        "l1_weight": run.l1_weight,
        "note": note,
    }
    dump_json(d["tables"] / "train_summary.json", summary)
    write_run_metadata(run, "train", summary)
    print(f"[train:{run.name}] stopped_reason={stopped_reason} best_val_mae={best_val:.3f}px")
    return evaluate_run(run.name)


def train_run(run_name: str, force: bool = False, resume: bool = False) -> Dict[str, Any]:
    run = train_run_spec(run_name)
    try:
        return _train_loop(run, force=force, resume=resume)
    except TrainingExploded as exc:
        if run.sgd_fallback_lr is None:
            raise
        print(f"[train:{run.name}] exploded: {exc}; fallback constant lr={run.sgd_fallback_lr}")
        d = run_dirs(run.name)
        hist = d["tables"] / "history.csv"
        if hist.exists():
            exploded = d["tables"] / "history_exploded.csv"
            if exploded.exists():
                hist.replace(d["tables"] / "history_exploded_fallback.csv")
            else:
                hist.replace(exploded)
        dump_json(
            d["tables"] / "train_summary.json",
            {
                "run": run.name,
                "stopped_reason": "exploded_nan",
                "note": "SGD lr=1.0 NaN at epoch 1; same-process fallback lr=0.1 also NaN. "
                "Use a fresh process: python phase1_gap_rep/run.py train --run sgd_l1_fixed",
                "sgd_fallback_lr": run.sgd_fallback_lr,
            },
        )
        print(
            f"[train:{run.name}] CUDA NaN 污染同进程，改在新进程跑 sgd_l1_fixed（固定 lr=0.1）"
        )
        return {"run": run.name, "exploded": True, "fallback": "sgd_l1_fixed"}


def _plot_run_comparison() -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    drawn = False
    for name in TRAIN_RUN_ORDER:
        history = _read_history(run_dirs(name)["tables"] / "history.csv")
        if not history:
            continue
        epochs = [int(row["epoch"]) for row in history]
        ax.plot(epochs, [float(row["val_mae_px"]) for row in history], label=f"{name} val")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MAE (px)")
    ax.set_title("L1 三组训练 val MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = RESULTS_ROOT / "figures" / "l1_runs_val_mae.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[all_l1] comparison plot -> {out}")


def train_all_l1_runs(force: bool = False, resume: bool = False) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name in TRAIN_RUN_ORDER:
        d = run_dirs(name)
        summary_path = d["tables"] / "train_summary.json"
        metrics_path = d["tables"] / "metrics.json"
        last_path = d["checkpoints"] / "resnet18_last.pt"
        if summary_path.exists() and metrics_path.exists() and not force:
            print(f"[all_l1] skip finished {name}")
            results[name] = {"skipped": True, "summary": str(summary_path)}
            continue
        run_resume = bool((resume or last_path.exists()) and last_path.exists() and not force)
        print(f"[all_l1] starting {name} force={force} resume={run_resume}")
        results[name] = train_run(name, force=force, resume=run_resume)
    _plot_run_comparison()
    print("[all_l1] done")
    return results


def train(profile: str = "factorial", force: bool = False, resume: bool = False) -> Dict[str, Any]:
    raise RuntimeError(
        "legacy factorial trainer is disabled so wreckage is not overwritten; "
        "use --run adamw_l1_resume|adamw_l1_scratch|sgd_l1|all_l1"
    )
