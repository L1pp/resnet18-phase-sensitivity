"""Self-contained fp32 trainer for the S1 Neural Affine reproduction."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import mae_px
from .models import canonical_variant, load_model_state, state_dict_hash


FORMAL_BATCH_SIZE = 64
FORMAL_STEPS = 3000
FORMAL_LR = 1e-3
FORMAL_WEIGHT_DECAY = 1e-4
FORMAL_LR_MIN = 1e-5
FORMAL_L1_WEIGHT = 0.25
ANCHOR_EVERY = 100
RESUME_EVERY = 200


def _cfg(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _json_value(value: Any) -> Any:
    """Convert common NumPy/Torch scalar values for metadata JSON."""

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_value)
        handle.write("\n")
    temp.replace(path)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _images_to_tensor(images: Any) -> torch.Tensor:
    """Convert uint8 grayscale/RGB arrays to NCHW float32 in [0, 1]."""

    if torch.is_tensor(images):
        array = images.detach().cpu().numpy()
    else:
        array = np.asarray(images)
    if array.ndim == 2:
        array = array[None, None, :, :]
    elif array.ndim == 3:
        # S1 renderers use [N,H,W] grayscale. A single [C,H,W] image is also
        # accepted when its first dimension is a channel count.
        if array.shape[0] in (1, 3, 5) and array.shape[1] == array.shape[2] and array.shape[0] != len(array):
            array = array[None, ...]
        else:
            array = array[:, None, :, :]
    elif array.ndim == 4:
        if array.shape[-1] in (1, 3, 5) and array.shape[1] not in (1, 3, 5):
            array = np.transpose(array, (0, 3, 1, 2))
    else:
        raise ValueError(f"images must be [N,H,W], NCHW or NHWC, got {array.shape}")

    if array.ndim != 4:
        raise ValueError(f"failed to convert images to NCHW: {array.shape}")
    if array.shape[1] == 1:
        array = np.repeat(array, 3, axis=1)
    if array.shape[1] not in (3, 5):
        raise ValueError(f"expected one, three or five image channels, got {array.shape}")
    result = torch.from_numpy(np.ascontiguousarray(array))
    if result.dtype == torch.uint8:
        result = result.float().div_(255.0)
    else:
        result = result.float()
        if result.numel() and float(result.detach().abs().max()) > 1.0:
            result = result.div(255.0)
    return result


@torch.no_grad()
def predict_images(
    model: nn.Module,
    images_uint8: Any,
    device: str | torch.device | None = None,
    batch_size: int = FORMAL_BATCH_SIZE,
) -> np.ndarray:
    """Forward images in bounded batches and return normalized [N,2] outputs."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if device is None:
        try:
            device_obj = next(model.parameters()).device
        except StopIteration:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)
    model.to(device_obj)
    was_training = bool(model.training)
    model.eval()
    array = np.asarray(images_uint8) if not torch.is_tensor(images_uint8) else images_uint8.detach().cpu().numpy()
    n = int(array.shape[0]) if array.ndim >= 3 else 1
    if n == 0:
        if was_training:
            model.train()
        return np.empty((0, 2), dtype=np.float32)
    outputs: list[np.ndarray] = []
    for start in range(0, n, int(batch_size)):
        batch = _images_to_tensor(array[start : start + int(batch_size)]).to(device_obj)
        outputs.append(model(batch).float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _source_payload(source: Any) -> Mapping[str, Any]:
    if isinstance(source, (str, Path)):
        value = _torch_load(source, map_location="cpu")
    else:
        value = source
    if not isinstance(value, Mapping):
        raise TypeError("init_source must be a checkpoint path or mapping")
    if "model_state" not in value:
        if value and all(torch.is_tensor(item) for item in value.values()):
            return {"model_state": value}
        raise KeyError("init_source lacks model_state")
    return value


def _metadata(
    *,
    model: nn.Module,
    variant: str,
    seed: int,
    protocol_hash: str,
    code_hash: str,
    init_hash: str,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "variant": variant,
        "seed": int(seed),
        "protocol_hash": str(protocol_hash),
        "code_hash": str(code_hash),
        "init_hash": str(init_hash),
        **extra,
    }
    payload["model_state"] = _clone_state(model)
    return payload


def _check_runtime_error(exc: RuntimeError, batch_size: int) -> RuntimeError:
    text = str(exc).lower()
    if "out of memory" in text or "cuda error" in text and "memory" in text:
        return RuntimeError(
            f"training ran out of memory with fixed batch_size={batch_size}; "
            "batch size was not changed automatically"
        )
    return exc


def xy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = FORMAL_L1_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, MSE and L1 terms for the frozen S1 objective."""

    mse = F.mse_loss(prediction, target)
    l1 = F.l1_loss(prediction, target)
    return mse + float(l1_weight) * l1, mse, l1


def train_run(
    model: nn.Module,
    train_images_uint8: Any,
    train_targets_norm: Any,
    anchor_images_uint8: Any,
    anchor_targets_norm: Any,
    out_dir: str | Path,
    config: Any,
    seed: int,
    init_source: Any = None,
    device: str | torch.device | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train one independent S1 run and write resumable evidence.

    The formal defaults are frozen at batch 64, 3000 steps, AdamW 1e-3/1e-4,
    ``MSE + 0.25 L1``, and cosine learning rate to 1e-5.  Small smoke tests
    may explicitly override ``steps`` and ``batch_size`` in ``config``; no
    error path ever changes the requested batch size.
    """

    out = Path(out_dir)
    ckpt = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    steps = int(_cfg(config, "steps", FORMAL_STEPS))
    batch_size = int(_cfg(config, "batch_size", FORMAL_BATCH_SIZE))
    anchor_every = int(_cfg(config, "anchor_every", ANCHOR_EVERY))
    resume_every = int(_cfg(config, "resume_every", RESUME_EVERY))
    lr = float(_cfg(config, "lr", FORMAL_LR))
    weight_decay = float(_cfg(config, "weight_decay", FORMAL_WEIGHT_DECAY))
    lr_min = float(_cfg(config, "lr_min", FORMAL_LR_MIN))
    l1_weight = float(_cfg(config, "l1_weight", FORMAL_L1_WEIGHT))
    if steps <= 0 or batch_size <= 0 or anchor_every <= 0 or resume_every <= 0:
        raise ValueError("steps, batch_size, anchor_every and resume_every must be positive")

    train_images = np.asarray(train_images_uint8)
    anchor_images = np.asarray(anchor_images_uint8)
    train_targets = np.asarray(train_targets_norm, dtype=np.float32).reshape(-1, 2)
    anchor_targets = np.asarray(anchor_targets_norm, dtype=np.float32).reshape(-1, 2)
    if len(train_images) != len(train_targets) or len(anchor_images) != len(anchor_targets):
        raise ValueError("image and target counts do not match")
    if len(train_images) == 0 or len(anchor_images) == 0:
        raise ValueError("S1 training and anchor sets must be non-empty")

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    model = model.float().to(device_obj)
    variant = canonical_variant(_cfg(config, "variant", getattr(model, "variant", "vanilla")))
    protocol_hash = str(_cfg(config, "protocol_hash", "unspecified"))
    code_hash = str(_cfg(config, "code_hash", "s1clean-train-v1"))

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps), eta_min=lr_min
    )

    resume_path = ckpt / "resume.pt"
    resumed = False
    step = 0
    best_anchor = math.inf
    best_step = 0
    history: list[dict[str, Any]] = []
    init_state: dict[str, torch.Tensor]
    init_hash: str
    order: torch.Tensor = torch.empty(0, dtype=torch.long)
    cursor = 0

    if resume and resume_path.exists():
        payload = _torch_load(resume_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"invalid resume checkpoint at {resume_path}")
        for key, expected in (("variant", variant), ("protocol_hash", protocol_hash), ("code_hash", code_hash)):
            if str(payload.get(key)) != str(expected):
                raise RuntimeError(f"resume metadata mismatch for {key}: {payload.get(key)!r} != {expected!r}")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        _move_optimizer_state(optimizer, device_obj)
        step = int(payload.get("step", 0))
        best_anchor = float(payload.get("best_anchor_mae_px", math.inf))
        best_step = int(payload.get("best_step", 0))
        history = list(payload.get("history", []))
        init_hash = str(payload.get("init_hash", ""))
        order = torch.as_tensor(payload.get("batch_order", []), dtype=torch.long)
        cursor = int(payload.get("batch_cursor", 0))
        if payload.get("rng_state") is not None:
            _restore_rng_state(payload["rng_state"])
        resumed = True
        init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    else:
        if init_source is not None:
            source = _source_payload(init_source)
            source_variant = source.get("variant")
            if source_variant is not None and canonical_variant(str(source_variant)) != variant:
                raise RuntimeError(f"init_source variant mismatch: {source_variant!r} != {variant!r}")
            model.load_state_dict(source["model_state"], strict=True)
        init_state = _clone_state(model)
        init_hash = state_dict_hash(init_state)
        _atomic_torch_save(
            _metadata(
                model=model,
                variant=variant,
                seed=seed,
                protocol_hash=protocol_hash,
                code_hash=code_hash,
                init_hash=init_hash,
                init_source=str(init_source) if isinstance(init_source, (str, Path)) else None,
            ),
            ckpt / "init.pt",
        )

    # A resumed run's init checkpoint must remain available and must describe
    # the same initial state.  Recreate it only when a prior checkpoint is not
    # present, never overwriting a historical exact-init artifact.
    init_path = ckpt / "init.pt"
    if not init_path.exists():
        _atomic_torch_save(
            {
                "model_state": init_state,
                "variant": variant,
                "seed": int(seed),
                "protocol_hash": protocol_hash,
                "code_hash": code_hash,
                "init_hash": init_hash,
            },
            init_path,
        )

    _atomic_json(
        {
            "variant": variant,
            "seed": int(seed),
            "protocol_hash": protocol_hash,
            "code_hash": code_hash,
            "batch_size": batch_size,
            "steps": steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_min": lr_min,
            "l1_weight": l1_weight,
            "anchor_every": anchor_every,
            "resume_every": resume_every,
            "device": str(device_obj),
            "resumed": resumed,
        },
        out / "config.json",
    )

    def save_resume(reason: str) -> None:
        payload = _metadata(
            model=model,
            variant=variant,
            seed=seed,
            protocol_hash=protocol_hash,
            code_hash=code_hash,
            init_hash=init_hash,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            step=int(step),
            best_anchor_mae_px=float(best_anchor) if math.isfinite(best_anchor) else None,
            best_step=int(best_step),
            history=history,
            batch_order=order,
            batch_cursor=int(cursor),
            rng_state=_rng_state(),
            reason=str(reason),
        )
        _atomic_torch_save(payload, resume_path)
        _atomic_json(history, out / "history.json")

    def save_best(score: float, at_step: int) -> None:
        payload = _metadata(
            model=model,
            variant=variant,
            seed=seed,
            protocol_hash=protocol_hash,
            code_hash=code_hash,
            init_hash=init_hash,
            step=int(at_step),
            best_anchor_mae_px=float(score),
        )
        _atomic_torch_save(payload, ckpt / "best.pt")

    def evaluate_anchor() -> float:
        prediction = predict_images(model, anchor_images, device_obj, batch_size=batch_size)
        return mae_px(prediction, anchor_targets)

    started = time.time()
    last_loss = math.nan
    try:
        model.train()
        while step < steps:
            if order.numel() == 0 or cursor >= int(order.numel()):
                order = torch.randperm(len(train_images), dtype=torch.long)
                cursor = 0
            stop = min(cursor + batch_size, int(order.numel()))
            batch_indices = order[cursor:stop]
            cursor = stop
            # Tile the final short batch to keep the formal batch size and its
            # optimizer shape fixed; four support images therefore naturally
            # repeat into a 64-sample training batch.
            if len(batch_indices) < batch_size:
                repeats = (batch_size + len(batch_indices) - 1) // len(batch_indices)
                batch_indices = batch_indices.repeat(repeats)[:batch_size]
            x = _images_to_tensor(train_images[batch_indices.numpy()]).to(device_obj)
            y = torch.from_numpy(train_targets[batch_indices.numpy()]).to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            try:
                prediction = model(x)
                loss, _, _ = xy_loss(prediction, y, l1_weight=l1_weight)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"non-finite loss at step {step + 1}")
                loss.backward()
                for parameter in params:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                        raise FloatingPointError(f"non-finite gradient at step {step + 1}")
                optimizer.step()
                scheduler.step()
            except RuntimeError as exc:
                raise _check_runtime_error(exc, batch_size) from exc
            last_loss = float(loss.detach().cpu())
            step += 1

            should_eval = step % anchor_every == 0 or step == steps
            row: dict[str, Any] = {
                "step": int(step),
                "train_loss": float(last_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if should_eval:
                model.eval()
                score = evaluate_anchor()
                model.train()
                row["anchor_mae_px"] = float(score)
                if not math.isfinite(score):
                    raise FloatingPointError(f"non-finite anchor metric at step {step}")
                if score < best_anchor:
                    best_anchor = float(score)
                    best_step = int(step)
                    save_best(best_anchor, best_step)
            history.append(row)

            if step % resume_every == 0 or step == steps:
                save_resume("periodic")

        model.eval()
        final_prediction = predict_images(model, anchor_images, device_obj, batch_size=batch_size)
        final_anchor = mae_px(final_prediction, anchor_targets)
        final_payload = _metadata(
            model=model,
            variant=variant,
            seed=seed,
            protocol_hash=protocol_hash,
            code_hash=code_hash,
            init_hash=init_hash,
            step=int(step),
            final_anchor_mae_px=float(final_anchor),
        )
        _atomic_torch_save(final_payload, ckpt / "final.pt")
        if not (ckpt / "best.pt").exists():
            best_anchor = float(final_anchor)
            best_step = int(step)
            save_best(best_anchor, best_step)

        best_payload = _torch_load(ckpt / "best.pt", map_location="cpu")
        model.load_state_dict(best_payload["model_state"], strict=True)
        model.to(device_obj).eval()
        best_prediction = predict_images(model, anchor_images, device_obj, batch_size=batch_size)
        np.savez_compressed(
            out / "predictions_anchor.npz",
            prediction_norm=best_prediction.astype(np.float32),
            target_norm=anchor_targets.astype(np.float32),
            step=np.asarray(best_step, dtype=np.int64),
        )
        summary = {
            "variant": variant,
            "seed": int(seed),
            "protocol_hash": protocol_hash,
            "code_hash": code_hash,
            "init_hash": init_hash,
            "steps": int(step),
            "batch_size": int(batch_size),
            "best_step": int(best_step),
            "best_anchor_mae_px": float(best_anchor),
            "final_anchor_mae_px": float(final_anchor),
            "elapsed_sec": float(time.time() - started),
            "resumed": bool(resumed),
            "device": str(device_obj),
            "n_train": int(len(train_images)),
            "n_anchor": int(len(anchor_images)),
            "checkpoints": {
                "init": str(ckpt / "init.pt"),
                "best": str(ckpt / "best.pt"),
                "final": str(ckpt / "final.pt"),
                "resume": str(resume_path),
            },
        }
        _atomic_json(summary, out / "summary.json")
        _atomic_json(history, out / "history.json")
        return summary
    except Exception:
        # Preserve a complete recovery point for technical failures.  The
        # original exception is re-raised; callers can classify OOM/nonfinite
        # without a silent hyperparameter change.
        if step > 0:
            try:
                save_resume("failure")
            except Exception:
                pass
        raise


__all__ = ["predict_images", "train_run", "xy_loss"]
