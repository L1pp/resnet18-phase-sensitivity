"""Shared IO and 2D blob helpers for the local mechanism probe."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from error_extension.field_metrics import dump_json, jsonable
from error_extension.freeze import apply_unfreeze_stage, enter_train, trainable_param_count
from functional_drift.blob2d import corners4_pack, dense_pack
from phase1_gap_rep.common import images_to_tensor, seed_everything
from phase3_high_upside.trainer import predict_xy, xy_loss
from spatial_field.heads import build_head, load_slim
from today_shortcycle.calib import apply_affine, fit_affine, mae_px
from today_shortcycle.fields import save_field

from .const import (
    BLOB_CKPT,
    BLOB_SHA256_PREFIX,
    EVAL_BATCH,
    FORBIDDEN_INIT_SUBSTR,
    HEAD_2,
    IMAGE_SIZE,
    N_DENSE,
    PACK_ROOT,
    RESULTS_ROOT,
    SEED,
)


def file_sha256_prefix(path: Path, n: int = 16) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:n]


def assert_blob_gate() -> str:
    path = Path(BLOB_CKPT)
    if not path.exists():
        raise FileNotFoundError(path)
    text = str(path).replace("\\", "/")
    for bad in FORBIDDEN_INIT_SUBSTR:
        if bad in text and "spatial_field_dynamics" not in text:
            raise RuntimeError(f"refusing init {path}")
    prefix = file_sha256_prefix(path)
    if prefix != BLOB_SHA256_PREFIX:
        raise RuntimeError(f"blob sha {prefix} != {BLOB_SHA256_PREFIX}")
    return prefix


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_blob(dev: Optional[torch.device] = None) -> nn.Module:
    assert_blob_gate()
    model = build_head(HEAD_2)
    load_slim(model, BLOB_CKPT, HEAD_2)
    return model.to(dev or device())


def snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state(model: nn.Module, snap: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(snap)
    model.to(next(model.parameters()).device)


def prepare_stage(model: nn.Module, stage: str) -> Dict[str, Any]:
    info = apply_unfreeze_stage(model, stage)
    enter_train(model)
    info["n_trainable"] = int(trainable_param_count(model))
    return info


def predict_px(model: nn.Module, images: np.ndarray, batch_size: int = EVAL_BATCH) -> np.ndarray:
    model.eval()
    flat = np.asarray(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    bs = int(batch_size)
    while True:
        try:
            pred_norm = predict_xy(model, flat, batch_size=bs)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 4:
                raise
            bs = max(4, bs // 2)
            print(f"[oom] eval batch -> {bs}", flush=True)
    return np.asarray(pred_norm, dtype=np.float64) * float(IMAGE_SIZE - 1)


_DENSE_CACHE: Optional[Dict[str, np.ndarray]] = None


def get_dense() -> Dict[str, np.ndarray]:
    global _DENSE_CACHE
    if _DENSE_CACHE is None:
        print("[cache] rendering official 41x41 blob field (RAM only, no uint8 disk)", flush=True)
        t0 = time.time()
        _DENSE_CACHE = dense_pack()
        print(f"[cache] dense render {time.time() - t0:.1f}s", flush=True)
    return _DENSE_CACHE


def official_eval(model: nn.Module, name: str = "probe") -> Dict[str, Any]:
    pack = get_dense()
    images = pack["images"].reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    pred = predict_px(model, images).reshape(1, N_DENSE, N_DENSE, 2)
    true = np.asarray(pack["true_px"], dtype=np.float64).reshape(1, N_DENSE, N_DENSE, 2)
    a, b = fit_affine(pred, true)
    pred_aff = apply_affine(pred, a, b)
    u = pred_aff - true
    return {
        "pred": pred,
        "true": true,
        "err": pred - true,
        "u": u,
        "A": a,
        "b": b,
        "mean_box_mae_px": mae_px(pred, true),
        "affine_removed_mae_px": mae_px(pred_aff, true),
        "n_eval": 1,
        "n_grid": N_DENSE,
        "name": name,
    }


def support_loss(model: nn.Module, images: np.ndarray, xy_norm: np.ndarray) -> torch.Tensor:
    from error_extension.freeze import enter_train

    enter_train(model)
    dev = next(model.parameters()).device
    xb = images_to_tensor(images).to(dev)
    yb = torch.from_numpy(np.ascontiguousarray(xy_norm).astype(np.float32)).to(dev)
    pred = model(xb)
    loss, _, _ = xy_loss(pred, yb)
    return loss


def trainable(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def pack_params(params: Sequence[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float() for p in params])


def pack_grads(params: Sequence[nn.Parameter]) -> torch.Tensor:
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        else:
            chunks.append(p.grad.detach().reshape(-1).float())
    return torch.cat(chunks)


def add_flat(params: Sequence[nn.Parameter], flat: torch.Tensor, scale: float) -> None:
    offset = 0
    vec = flat.detach()
    for param in params:
        n = param.numel()
        param.data.add_(
            scale * vec[offset : offset + n].reshape_as(param).to(device=param.device, dtype=param.dtype)
        )
        offset += n


def rms(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(a))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def write_json(path: Path, payload: Any) -> None:
    dump_json(path, jsonable(payload))


def ensure_results() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT
