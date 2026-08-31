"""Support-only training regimes with deterministic resume artifacts."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import BalancedBatchStream, hash_array, images_to_tensor, sha256_file
from .models import (
    backbone_hash,
    backbone_parameter_hash,
    clone_state_dict,
    load_exact_init,
    state_dict_hash,
)
from .protocol import REGIME_NAMES, load_protocol, protocol_hash, write_json


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    temporary.replace(path)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return optimizer.state_dict()


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def mae_px(prediction_norm: np.ndarray, target_norm: np.ndarray, coord_scale: float = 223.0) -> float:
    pred = np.asarray(prediction_norm, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_norm, dtype=np.float64).reshape(-1, 2)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {pred.shape}/{target.shape}")
    # Primary metric is per-point Euclidean vector MAE in pixels.  Coordinate
    # L1 is not interchangeable with this value and is not used for gates.
    return float(np.mean(np.linalg.norm((pred - target) * float(coord_scale), axis=-1)))


def xy_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.25) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(prediction, target)
    l1 = F.l1_loss(prediction, target)
    return mse + float(l1_weight) * l1, mse, l1


@torch.no_grad()
def predict_images(model: nn.Module, images_uint8: np.ndarray, device: str | torch.device = "cpu", batch_size: int = 64) -> np.ndarray:
    device_obj = torch.device(device)
    model = model.to(device_obj)
    was_training = bool(model.training)
    model.eval()
    array = np.asarray(images_uint8)
    if array.ndim < 3:
        raise ValueError(f"images must have batch dimension: {array.shape}")
    outputs: list[np.ndarray] = []
    for start in range(0, int(array.shape[0]), int(batch_size)):
        batch = images_to_tensor(array[start : start + int(batch_size)]).to(device_obj)
        outputs.append(model(batch).float().cpu().numpy())
    if was_training:
        model.train()
    if not outputs:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _stream_seed(seed: int, support_name: str) -> int:
    digest = hashlib.sha256(str(support_name).encode("utf-8")).digest()
    return int(seed) * 1009 + int.from_bytes(digest[:4], "little")


def _set_regime_trainability(model: nn.Module, regime: str) -> None:
    if regime not in REGIME_NAMES:
        raise ValueError(f"unknown regime: {regime}")
    if regime == "head_only":
        backbone = getattr(model, "backbone", None)
        for parameter in model.parameters():
            parameter.requires_grad = False
        if not hasattr(model, "fc"):
            raise ValueError("head_only requires model.fc")
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        # Keep the complete module in eval mode for the entire head-only run.
        # Linear layers still compute gradients and update in eval mode, while
        # BN buffers cannot move.  This also survives predict_images() calls at
        # anchor checkpoints without recursively re-enabling backbone BN.
        model.eval()
    elif regime == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False


def _init_payload(model: nn.Module, *, seed: int, support_name: str, regime: str, p_hash: str, init_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "condition_init",
        "seed": int(seed),
        "support": str(support_name),
        "regime": str(regime),
        "protocol_hash": str(p_hash),
        "init_hash": str(init_hash),
        "backbone_init_hash": backbone_hash(model),
        "backbone_parameter_init_hash": backbone_parameter_hash(model),
        "model_state": clone_state_dict(model),
    }


def _write_receipt(
    run_dir: Path,
    *,
    protocol_sha: str,
    support_hash: str,
    dense_labels_read: bool,
    stream_digest: str | None = None,
    asset_bundle_sha256: str | None = None,
    package_id: str | None = None,
    package_manifest_sha256: str | None = None,
    init_source_protocol_hash: str | None = None,
    init_source_file_sha256: str | None = None,
    init_state_hash: str | None = None,
) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "receipt.json":
            files[str(path.relative_to(run_dir)).replace("\\", "/")] = {
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    receipt = {
        "schema_version": 1,
        "kind": "condition_receipt",
        "protocol_sha256": str(protocol_sha),
        "support_input_hash": str(support_hash),
        "dense_labels_read_by_training": bool(dense_labels_read),
        "paired_batch_stream_sha256": stream_digest,
        "asset_bundle_sha256": asset_bundle_sha256,
        "package_id": package_id,
        "package_manifest_sha256": package_manifest_sha256,
        "init_source_protocol_hash": init_source_protocol_hash,
        "init_source_file_sha256": init_source_file_sha256,
        "init_state_hash": init_state_hash,
        "files": files,
    }
    write_json(run_dir / "receipt.json", receipt)
    return receipt


def _fit_frozen_feature(model: nn.Module, images_uint8: np.ndarray, targets_norm: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(model, "forward_features") or not hasattr(model, "fc"):
        raise ValueError("frozen_feature requires model.forward_features and model.fc")
    model.eval().to(device)
    with torch.no_grad():
        features = model.forward_features(images_to_tensor(images_uint8).to(device)).float().cpu().numpy().astype(np.float64)
    design = np.concatenate((features, np.ones((len(features), 1), dtype=np.float64)), axis=1)
    coefficients = np.linalg.pinv(design, rcond=1e-10) @ np.asarray(targets_norm, dtype=np.float64)
    weight = coefficients[:-1].T.astype(np.float32)
    bias = coefficients[-1].astype(np.float32)
    with torch.no_grad():
        model.fc.weight.copy_(torch.from_numpy(weight).to(model.fc.weight.device))
        model.fc.bias.copy_(torch.from_numpy(bias).to(model.fc.bias.device))
    return features, weight, bias


def run_condition(
    model: nn.Module,
    support_images_uint8: np.ndarray,
    support_points_px: np.ndarray,
    out_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    seed: int,
    support_name: str,
    regime: str,
    init_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    steps: int | None = None,
    resume: bool = True,
    batch_stream: np.ndarray | None = None,
    batch_stream_sha256: str | None = None,
    asset_bundle_sha256: str | None = None,
    package_id: str | None = None,
    package_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one support/regime/seed condition using support arrays only."""

    cfg = load_protocol() if protocol is None else dict(protocol)
    if regime not in REGIME_NAMES:
        raise ValueError(f"unknown regime {regime!r}")
    images = np.asarray(support_images_uint8)
    points_px = np.asarray(support_points_px, dtype=np.float64).reshape(-1, 2)
    if images.ndim != 3 or images.dtype != np.uint8 or len(images) != len(points_px) or len(images) == 0:
        raise ValueError(f"invalid support arrays: images={images.shape}/{images.dtype}, points={points_px.shape}")
    scale = float(cfg["coord_scale"])
    targets = (points_px / scale).astype(np.float32)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    p_hash = protocol_hash(payload=cfg)
    effective_package_id = str(package_id) if package_id is not None else "neural_affine_crossover_clean"
    support_hash = hash_array(images, "support_images") + hash_array(points_px, "support_points")
    requested_steps = int(cfg["training"]["steps"] if steps is None else steps)
    if requested_steps <= 0:
        raise ValueError("steps must be positive")
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    lr_min = float(cfg["training"]["scheduler"]["eta_min"])
    l1_weight = float(cfg["training"]["loss"]["l1_weight"])
    resume_path = output / "checkpoints" / "resume.pt"
    init_path = output / "checkpoints" / "init.pt"
    best_path = output / "checkpoints" / "best.pt"
    final_path = output / "checkpoints" / "final.pt"
    output.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    # The model is always populated from the exact seed asset before a fresh
    # condition starts; this makes all three regimes share an identical init.
    init_source_protocol_hash: str | None = None
    init_source_file_sha256: str | None = None
    if init_checkpoint is not None:
        # Always load the immutable exact init first, including when a resume
        # file exists.  This gives resume validation the original init hash
        # rather than the fresh model RNG state.
        # Formal asset validation has already authenticated the legacy S1
        # source hash.  State this compatibility exception explicitly rather
        # than relying on load_exact_init's default strictness.
        init_payload = load_exact_init(
            init_checkpoint,
            model,
            expected_seed=seed,
            expected_protocol_hash=p_hash,
            allow_source_protocol_mismatch=True,
        )
        init_hash = str(init_payload["init_hash"])
        init_source_protocol_hash = str(init_payload.get("protocol_hash"))
        init_source_file_sha256 = sha256_file(init_checkpoint)
    else:
        init_hash = state_dict_hash(model.state_dict())
        init_source_protocol_hash = p_hash
    stream_seed = _stream_seed(seed, support_name)
    stream = BalancedBatchStream(len(images), batch_size, stream_seed)
    paired = None if batch_stream is None else np.asarray(batch_stream)
    if paired is not None:
        if paired.ndim != 2 or paired.shape[1] != batch_size or paired.shape[0] < requested_steps or paired.dtype != np.int64:
            raise ValueError(f"paired batch stream must be [>=steps,{batch_size}] int64, got {paired.shape}/{paired.dtype}")
        if paired.size and (int(np.min(paired)) < 0 or int(np.max(paired)) >= len(images)):
            raise ValueError("paired batch stream contains out-of-range indices")
        if batch_stream_sha256 is None:
            batch_stream_sha256 = hash_array(paired, "paired_batch_stream")
    paired_cursor = 0
    stream_digest = str(batch_stream_sha256) if batch_stream_sha256 is not None else stream.digest(batches=max(16, requested_steps))
    _set_regime_trainability(model, regime)
    backbone_init = backbone_hash(model)
    backbone_parameter_init = backbone_parameter_hash(model)
    started = time.perf_counter()

    if regime == "frozen_feature":
        # Closed-form fitting is computed from the initial feature map.  No
        # dense array is reachable from this function.
        initial_state = clone_state_dict(model)
        initial_backbone_hash = backbone_hash(model)
        features, fitted_weight, fitted_bias = _fit_frozen_feature(model, images, targets, device_obj)
        prediction = predict_images(model, images, device_obj, batch_size=batch_size)
        init_record = _init_payload(model, seed=seed, support_name=support_name, regime=regime, p_hash=p_hash, init_hash=init_hash)
        init_record["model_state"] = initial_state
        init_record["backbone_init_hash"] = initial_backbone_hash
        _atomic_torch(output / "checkpoints" / "init.pt", init_record)
        _atomic_torch(
            best_path,
            {
                "schema_version": 1,
                "kind": "condition_checkpoint",
                "seed": int(seed),
                "support": support_name,
                "regime": regime,
                "protocol_hash": p_hash,
                "init_hash": init_hash,
                "step": 0,
                "model_state": clone_state_dict(model),
                "feature_head_weight": fitted_weight,
                "feature_head_bias": fitted_bias,
            },
        )
        _atomic_torch(output / "checkpoints" / "final.pt", _torch_load(best_path))
        _atomic_npz(output / "predictions_support.npz", prediction_norm=prediction, target_norm=targets, points_px=points_px)
        summary = {
            "schema_version": 1,
            "seed": int(seed),
            "support": support_name,
            "support_count": int(len(images)),
            "regime": regime,
            "protocol_id": cfg["protocol_id"],
            "package_id": effective_package_id,
            "package_manifest_sha256": package_manifest_sha256,
            "protocol_hash": p_hash,
            "asset_bundle_sha256": asset_bundle_sha256,
            "init_hash": init_hash,
            "init_state_hash": init_hash,
            "init_source_protocol_hash": init_source_protocol_hash,
            "init_source_file_sha256": init_source_file_sha256,
            "stream_seed": stream_seed,
            "stream_digest": stream_digest,
            "steps_requested": requested_steps,
            "steps_completed": 0,
            "batch_size": batch_size,
            "support_mae_px": mae_px(prediction, targets, scale),
            "backbone_init_hash": initial_backbone_hash,
            "backbone_final_hash": backbone_hash(model),
            "backbone_parameter_init_hash": backbone_parameter_hash(model),
            "backbone_parameter_final_hash": backbone_parameter_hash(model),
            "elapsed_sec": float(time.perf_counter() - started),
            "dense_labels_read": False,
            "status": "completed",
        }
        write_json(output / "summary.json", summary)
        _write_receipt(output, protocol_sha=p_hash, support_hash=support_hash, dense_labels_read=False, stream_digest=stream_digest, asset_bundle_sha256=asset_bundle_sha256, package_id=effective_package_id, package_manifest_sha256=package_manifest_sha256, init_source_protocol_hash=init_source_protocol_hash, init_source_file_sha256=init_source_file_sha256, init_state_hash=init_hash)
        return summary

    # Training regimes use the exact same stream schedule.  A resumed run
    # restores the stream cursor from the checkpoint before consuming a batch.
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError(f"{regime} has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(requested_steps, 1), eta_min=lr_min)
    step = 0
    best_score = math.inf
    best_step = 0
    history: list[dict[str, Any]] = []
    resumed = False
    if resume and resume_path.exists():
        payload = _torch_load(resume_path)
        if not isinstance(payload, Mapping):
            raise ValueError("resume checkpoint is not a mapping")
        resume_identity = {
            "protocol_hash": p_hash,
            "asset_bundle_sha256": asset_bundle_sha256,
            "package_id": effective_package_id,
            "package_manifest_sha256": package_manifest_sha256,
            "support": support_name,
            "regime": regime,
            "seed": int(seed),
            "init_hash": init_hash,
            "init_state_hash": init_hash,
            "init_source_protocol_hash": init_source_protocol_hash,
            "init_source_file_sha256": init_source_file_sha256,
            "requested_steps": requested_steps,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_min": lr_min,
            "l1_weight": l1_weight,
            "stream_sha256": batch_stream_sha256,
        }
        for key, expected in resume_identity.items():
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"resume metadata mismatch for {key}")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        _move_optimizer_state(optimizer, device_obj)
        stream_state = payload["stream_state"]
        if paired is not None:
            if stream_state.get("mode") != "paired_file" or str(stream_state.get("sha256")) != str(batch_stream_sha256):
                raise ValueError("paired batch stream source mismatch on resume")
            paired_cursor = int(stream_state.get("cursor", 0))
        else:
            stream.load_state(stream_state)
        step = int(payload.get("step", 0))
        # save_resume serializes an uninitialized best score as JSON null;
        # restore that state as +inf so a short run can resume safely.
        raw_best_score = payload.get("best_score", math.inf)
        best_score = math.inf if raw_best_score is None else float(raw_best_score)
        best_step = int(payload.get("best_step", 0))
        history = list(payload.get("history", []))
        resumed = True
    else:
        initial = _init_payload(model, seed=seed, support_name=support_name, regime=regime, p_hash=p_hash, init_hash=init_hash)
        _atomic_torch(init_path, initial)

    model = model.to(device_obj)
    started = time.perf_counter()

    def save_resume(reason: str) -> None:
        _atomic_torch(
            resume_path,
            {
                "schema_version": 1,
                "kind": "condition_resume",
                "seed": int(seed),
                "support": support_name,
                "regime": regime,
                "protocol_hash": p_hash,
                "init_hash": init_hash,
                "init_state_hash": init_hash,
                "init_source_protocol_hash": init_source_protocol_hash,
                "init_source_file_sha256": init_source_file_sha256,
                "package_id": effective_package_id,
                "package_manifest_sha256": package_manifest_sha256,
                "requested_steps": requested_steps,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "lr_min": lr_min,
                "l1_weight": l1_weight,
                "stream_sha256": batch_stream_sha256,
                "asset_bundle_sha256": asset_bundle_sha256,
                "step": int(step),
                "best_score": None if not math.isfinite(best_score) else float(best_score),
                "best_step": int(best_step),
                "model_state": clone_state_dict(model),
                "optimizer_state": _clone_optimizer_state(optimizer),
                "scheduler_state": scheduler.state_dict(),
                "stream_state": ({"mode": "paired_file", "sha256": batch_stream_sha256, "cursor": int(paired_cursor)} if paired is not None else stream.state()),
                "history": history,
                "reason": reason,
            },
        )

    def save_checkpoint(path: Path, at_step: int, score: float | None = None) -> None:
        _atomic_torch(
            path,
            {
                "schema_version": 1,
                "kind": "condition_checkpoint",
                "seed": int(seed),
                "support": support_name,
                "regime": regime,
                "protocol_hash": p_hash,
                "init_hash": init_hash,
                "step": int(at_step),
                "best_score": None if score is None else float(score),
                "model_state": clone_state_dict(model),
            },
        )

    try:
        while step < requested_steps:
            if paired is not None:
                indices = np.asarray(paired[paired_cursor], dtype=np.int64)
                paired_cursor += 1
            else:
                indices = stream.next_indices()
            x = images_to_tensor(images[indices]).to(device_obj)
            y = torch.from_numpy(targets[indices]).to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            if regime == "head_only":
                with torch.no_grad():
                    features = model.forward_features(x)
                prediction_tensor = model.fc(features)
            else:
                prediction_tensor = model(x)
            loss, mse, l1 = xy_loss(prediction_tensor, y, l1_weight=l1_weight)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            for parameter in params:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                    raise FloatingPointError(f"non-finite gradient at step {step + 1}")
            optimizer.step()
            scheduler.step()
            step += 1
            row = {"step": int(step), "loss": float(loss.detach().cpu()), "mse": float(mse.detach().cpu()), "l1": float(l1.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"])}
            if step % int(cfg["training"]["anchor_eval_every"]) == 0 or step == requested_steps:
                eval_pred = predict_images(model, images, device_obj, batch_size=batch_size)
                score = mae_px(eval_pred, targets, scale)
                row["support_mae_px"] = float(score)
                if score < best_score:
                    best_score = float(score)
                    best_step = int(step)
                    save_checkpoint(best_path, best_step, best_score)
            history.append(row)
            if step % int(cfg["training"]["resume_every"]) == 0 or step == requested_steps:
                save_resume("periodic")
        model.eval()
        final_prediction = predict_images(model, images, device_obj, batch_size=batch_size)
        final_score = mae_px(final_prediction, targets, scale)
        save_checkpoint(final_path, step, final_score)
        if not best_path.exists():
            best_score, best_step = final_score, step
            save_checkpoint(best_path, step, final_score)
        _atomic_npz(output / "predictions_support.npz", prediction_norm=final_prediction, target_norm=targets, points_px=points_px)
        summary = {
            "schema_version": 1,
            "seed": int(seed),
            "support": support_name,
            "support_count": int(len(images)),
            "regime": regime,
            "protocol_id": cfg["protocol_id"],
            "package_id": effective_package_id,
            "package_manifest_sha256": package_manifest_sha256,
            "protocol_hash": p_hash,
            "asset_bundle_sha256": asset_bundle_sha256,
            "init_hash": init_hash,
            "init_state_hash": init_hash,
            "init_source_protocol_hash": init_source_protocol_hash,
            "init_source_file_sha256": init_source_file_sha256,
            "stream_seed": stream_seed,
            "stream_digest": stream_digest,
            "steps_requested": requested_steps,
            "steps_completed": int(step),
            "batch_size": batch_size,
            "best_step": int(best_step),
            "best_support_mae_px": float(best_score),
            "support_mae_px": float(final_score),
            "backbone_init_hash": backbone_init,
            "backbone_final_hash": backbone_hash(model),
            "backbone_parameter_init_hash": backbone_parameter_init,
            "backbone_parameter_final_hash": backbone_parameter_hash(model),
            "elapsed_sec": float(time.perf_counter() - started),
            "resumed": resumed,
            "dense_labels_read": False,
            "status": "completed",
        }
        write_json(output / "history.json", history)
        write_json(output / "summary.json", summary)
        _write_receipt(output, protocol_sha=p_hash, support_hash=support_hash, dense_labels_read=False, stream_digest=stream_digest, asset_bundle_sha256=asset_bundle_sha256, package_id=effective_package_id, package_manifest_sha256=package_manifest_sha256, init_source_protocol_hash=init_source_protocol_hash, init_source_file_sha256=init_source_file_sha256, init_state_hash=init_hash)
        return summary
    except Exception:
        if step > 0:
            try:
                save_resume("failure")
            except Exception:
                pass
        raise


__all__ = ["mae_px", "predict_images", "run_condition", "xy_loss"]
