"""Generic (x,y) trainer. fp32 default. No silent resume. Smoke then full train."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from phase1_gap_rep.common import images_to_tensor, seed_everything

from .io_util import dump_json, state_dict_hash
from .protocol import (
    AMP_ALLOWED_ARCH,
    BATCH_SIZE,
    L1_WEIGHT,
    LR,
    LR_MIN,
    N_STEPS_REF,
    WEIGHT_DECAY,
)


class TrainingExploded(RuntimeError):
    pass


class XYDataset(Dataset):
    def __init__(self, images: np.ndarray, xy: np.ndarray):
        self.images = np.asarray(images)
        self.xy = np.asarray(xy, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.images))

    def __getitem__(self, idx: int):
        return self.images[idx], self.xy[idx]


def _amp_context(dev: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=dev.type, dtype=torch.float16)


def xy_loss(pred: torch.Tensor, target: torch.Tensor, l1_weight: float = L1_WEIGHT):
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    return mse + l1_weight * l1, mse, l1


def _workers() -> int:
    return 0 if os.name == "nt" else 4


@torch.no_grad()
def predict_xy(model: nn.Module, images: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for start in range(0, len(images), batch_size):
        batch = images_to_tensor(images[start : start + batch_size]).to(device)
        outs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def mae_px(pred: np.ndarray, target: np.ndarray, image_size: int) -> float:
    scale = float(image_size - 1)
    err = np.linalg.norm((pred - target) * scale, axis=-1)
    return float(np.mean(err))


def replace_xy_head(model: nn.Module, hidden: Optional[int] = None) -> nn.Module:
    """Replace classifier with Linear(D,2) or MLP. Returns the new head module."""

    def _make(in_f: int) -> nn.Module:
        if hidden is None:
            return nn.Linear(in_f, 2)
        return nn.Sequential(nn.Linear(in_f, int(hidden)), nn.GELU(), nn.Linear(int(hidden), 2))

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = _make(model.head.in_features)
        return model.head
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = _make(model.fc.in_features)
        return model.fc
    clf = getattr(model, "classifier", None)
    if isinstance(clf, nn.Linear):
        model.classifier = _make(clf.in_features)
        return model.classifier
    if isinstance(clf, nn.Sequential):
        for i in range(len(clf) - 1, -1, -1):
            if isinstance(clf[i], nn.Linear):
                in_f = clf[i].in_features
                clf[i] = _make(in_f)
                return clf[i]
    raise RuntimeError(f"cannot replace xy head on {type(model)}")


def apply_train_mode(model: nn.Module, mode: str) -> None:
    """R0/R1 freeze backbone; R2 last stage; R3 full."""
    mode = mode.upper()
    for param in model.parameters():
        param.requires_grad = True
    if mode in {"R3", "FULL", "S"}:
        return
    if mode in {"R0", "R1", "LINEAR", "MLP"}:
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name for key in ("fc", "classifier", "head"))
        return
    if mode == "R2":
        last_keys = ("layer4", "denseblock4", "features.7", "features.8")
        for name, param in model.named_parameters():
            last = any(key in name for key in last_keys)
            head = any(key in name for key in ("fc", "classifier", "head"))
            param.requires_grad = last or head
        return
    raise ValueError(mode)


def _should_amp(arch: str, requested: bool) -> bool:
    if not requested:
        return False
    return arch in AMP_ALLOWED_ARCH


def train_xy(
    model: nn.Module,
    train_images: np.ndarray,
    train_xy: np.ndarray,
    val_images: np.ndarray,
    val_xy: np.ndarray,
    out_dir: Path,
    *,
    seed: int,
    arch: str,
    image_size: int,
    steps: int = N_STEPS_REF,
    mode: str = "R3",
    amp: bool = False,
    clip_grad_norm: Optional[float] = None,
    resume: bool = False,
    skip_if_done: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    ckpt_dir = out_dir / "checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables / "train_summary.json"
    if skip_if_done and summary_path.exists():
        return __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    if resume:
        raise RuntimeError("resume=true is forbidden unless this is an explicit recorded resume experiment")

    seed_everything(seed)
    apply_train_mode(model, mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = bool(amp) and device.type == "cuda" and _should_amp(arch, True)
    if amp and not use_amp and arch not in AMP_ALLOWED_ARCH:
        use_amp = False

    init_hash = state_dict_hash(model.state_dict())
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "init_hash": init_hash, "seed": seed}, ckpt_dir / "init.pt")
    dump_json(out_dir / "config" / "init_info.json", {"init_hash": init_hash, "seed": seed, "resume": False})

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = DataLoader(
        XYDataset(train_images, train_xy),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=_workers(),
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    xb, yb = next(iter(loader))
    xb, yb = images_to_tensor(xb.numpy() if torch.is_tensor(xb) and xb.dtype == torch.uint8 else np.asarray(xb)).to(device), yb.to(device)
    if xb.dtype != torch.float32 or xb.max() > 2:
        xb = images_to_tensor(np.asarray(train_images[: BATCH_SIZE])).to(device)
        yb = torch.from_numpy(np.asarray(train_xy[: BATCH_SIZE], dtype=np.float32)).to(device)
    with _amp_context(device, use_amp):
        pred = model(xb)
        loss, _, _ = xy_loss(pred, yb)
    if not math.isfinite(float(loss.detach().cpu())):
        raise TrainingExploded("smoke loss not finite")
    smoke_loss = float(loss.detach().cpu())

    iterator = iter(loader)
    best_val = math.inf
    history: List[Dict[str, Any]] = []
    grad_norms: List[float] = []
    t0 = time.time()
    step = 0
    epoch = 0
    model.train()
    while step < steps:
        try:
            images_b, xy_b = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = iter(loader)
            images_b, xy_b = next(iterator)
            val_mae = mae_px(predict_xy(model, val_images), val_xy, image_size)
            history.append({"epoch": epoch, "step": step, "val_mae_px": val_mae, "grad_norm_mean": float(np.mean(grad_norms[-50:]) if grad_norms else 0.0)})
            if val_mae < best_val:
                best_val = val_mae
                torch.save(
                    {
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch,
                        "step": step,
                        "val_mae_px": val_mae,
                        "init_hash": init_hash,
                        "resume": False,
                    },
                    ckpt_dir / "best_slim.pt",
                )
        if torch.is_tensor(images_b) and images_b.dtype == torch.uint8:
            xb = images_to_tensor(images_b.numpy()).to(device)
        else:
            xb = images_to_tensor(np.asarray(images_b)).to(device)
        yb = xy_b.to(device) if torch.is_tensor(xy_b) else torch.from_numpy(np.asarray(xy_b)).to(device)
        opt.zero_grad(set_to_none=True)
        with _amp_context(device, use_amp):
            pred = model(xb)
            loss, _, _ = xy_loss(pred, yb)
        if not math.isfinite(float(loss.detach().cpu())):
            raise TrainingExploded(f"loss exploded at step {step}")
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(clip_grad_norm)))
        else:
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum().cpu())
            gn = math.sqrt(total)
        grad_norms.append(gn)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

    val_mae = mae_px(predict_xy(model, val_images), val_xy, image_size)
    if val_mae < best_val or not (ckpt_dir / "best_slim.pt").exists():
        best_val = val_mae
        torch.save(
            {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "step": step,
                "val_mae_px": val_mae,
                "init_hash": init_hash,
                "resume": False,
            },
            ckpt_dir / "best_slim.pt",
        )
    summary = {
        "arch": arch,
        "mode": mode,
        "seed": seed,
        "steps": step,
        "n_train": int(len(train_images)),
        "n_val": int(len(val_images)),
        "image_size": image_size,
        "best_val_mae_px": best_val,
        "final_val_mae_px": val_mae,
        "smoke_loss": smoke_loss,
        "amp": use_amp,
        "clip_grad_norm": clip_grad_norm,
        "resume": False,
        "init_hash": init_hash,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
        "elapsed_sec": time.time() - t0,
        "trainable": int(sum(p.numel() for p in params)),
    }
    if extra:
        summary.update(extra)
    dump_json(summary_path, summary)
    dump_json(tables / "history.json", history)
    return summary


def load_best(model: nn.Module, out_dir: Path) -> nn.Module:
    path = Path(out_dir) / "checkpoints" / "best_slim.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model
