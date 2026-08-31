"""Independent dense evaluator.

This module is deliberately separate from training.  It obtains predictions
from the frozen checkpoint first and only then opens ``dense_points.npy``;
there is no path for dense labels to enter support-set optimization.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data import dense_points_px, images_to_tensor, load_dense_cache, sha256_file
from .models import backbone_hash, backbone_parameter_hash, build_model, state_dict_hash
from .protocol import load_protocol, protocol_hash, write_json
from .training import mae_px, predict_images


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_prediction_archive(path: Path, *, prediction_norm: np.ndarray, prediction_px: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction_norm=np.asarray(prediction_norm, dtype=np.float32), prediction_px=np.asarray(prediction_px, dtype=np.float64))
    temporary.replace(path)
    return sha256_file(path)


def fit_affine(prediction_px: np.ndarray, target_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction_px, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_px, dtype=np.float64).reshape(-1, 2)
    design = np.concatenate((pred, np.ones((len(pred), 1), dtype=np.float64)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients[:2].T, coefficients[2]


def affine_removed_mae_px(prediction_px: np.ndarray, target_px: np.ndarray) -> float:
    matrix, bias = fit_affine(prediction_px, target_px)
    transformed = np.asarray(prediction_px, dtype=np.float64) @ matrix.T + bias
    return float(np.mean(np.linalg.norm(transformed - np.asarray(target_px, dtype=np.float64), axis=-1)))


def _load_frozen_model(run_dir: Path, *, model_factory: Any, device: torch.device) -> tuple[torch.nn.Module, Mapping[str, Any], Path]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # The primary crossover statistic is the pre-registered 3000-step
    # endpoint.  Selecting ``best.pt`` by support error would use the support
    # set to choose an off-support field and can change the scientific result.
    # ``best.pt`` remains an optional diagnostic artifact only.
    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing endpoint checkpoint: {checkpoint_path}")
    payload = _torch_load(checkpoint_path)
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise ValueError(f"invalid checkpoint: {checkpoint_path}")
    model = model_factory()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, summary, checkpoint_path


def evaluate_run(
    run_dir: str | Path,
    cache_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    model_factory: Any | None = None,
    allow_small_cache: bool = False,
    asset_bundle_sha256: str | None = None,
    package_id: str | None = None,
    package_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently re-forward one run on all 41x41 dense images."""

    cfg = load_protocol() if protocol is None else dict(protocol)
    p_hash = protocol_hash(payload=cfg)
    run = Path(run_dir)
    started = time.perf_counter()
    factory = build_model if model_factory is None else model_factory
    device_obj = torch.device(device)
    model, summary, checkpoint_path = _load_frozen_model(run, model_factory=factory, device=device_obj)
    if str(summary.get("protocol_hash")) != p_hash:
        raise ValueError("run protocol hash does not match evaluator protocol")
    expected_bundle = asset_bundle_sha256 if asset_bundle_sha256 is not None else summary.get("asset_bundle_sha256")
    if asset_bundle_sha256 is not None and str(summary.get("asset_bundle_sha256")) != str(asset_bundle_sha256):
        raise ValueError("run asset_bundle_sha256 does not match evaluator input")
    expected_package_id = package_id if package_id is not None else summary.get("package_id")
    expected_package_manifest_sha = package_manifest_sha256 if package_manifest_sha256 is not None else summary.get("package_manifest_sha256")
    if package_id is not None and str(summary.get("package_id")) != str(package_id):
        raise ValueError("run package_id does not match evaluator input")
    if package_manifest_sha256 is not None and str(summary.get("package_manifest_sha256")) != str(package_manifest_sha256):
        raise ValueError("run package_manifest_sha256 does not match evaluator input")

    # Only dense images are read before the prediction becomes immutable.
    dense_image_path = Path(cache_dir) / "dense_images.npy"
    dense_images = np.load(dense_image_path, allow_pickle=False, mmap_mode="r")
    if dense_images.ndim != 3 or dense_images.dtype != np.uint8:
        raise ValueError("dense image cache is not uint8 [N,H,W]")
    if not allow_small_cache and dense_images.shape != (1681, int(cfg["image_size"]), int(cfg["image_size"])):
        raise ValueError(f"formal dense image shape must be (1681,224,224), got {dense_images.shape}")
    prediction_norm = predict_images(model, dense_images, device_obj, batch_size=int(cfg["evaluation"]["batch_size"]))
    prediction_px = prediction_norm.astype(np.float64) * float(cfg["coord_scale"])
    if not allow_small_cache and (prediction_px.shape != (1681, 2) or not np.all(np.isfinite(prediction_px))):
        raise ValueError(f"formal prediction field must be finite (1681,2), got {prediction_px.shape}")
    prediction_frozen_sha = hashlib.sha256(np.ascontiguousarray(prediction_px).tobytes(order="C")).hexdigest()
    prediction_archive_path = run / "predictions_dense.npz"
    prediction_archive_sha = _atomic_prediction_archive(prediction_archive_path, prediction_norm=prediction_norm, prediction_px=prediction_px)

    # Dense labels are intentionally opened after the forward above.
    dense_points_path = Path(cache_dir) / "dense_points.npy"
    dense_points = np.load(dense_points_path, allow_pickle=False, mmap_mode="r")
    true_px = np.asarray(dense_points, dtype=np.float64)
    if not allow_small_cache:
        expected_points = dense_points_px(cfg)
        if true_px.shape != (1681, 2) or not np.array_equal(true_px, expected_points):
            raise ValueError("formal dense labels do not match 41x41 tx-major/ty-minor Cartesian grid")
        if not np.all(np.isfinite(true_px)):
            raise ValueError("formal dense labels are non-finite")
    raw = mae_px(prediction_norm, true_px / float(cfg["coord_scale"]), float(cfg["coord_scale"]))
    residual = affine_removed_mae_px(prediction_px, true_px)
    checkpoint_payload = _torch_load(checkpoint_path)
    backbone_final = backbone_hash(model)
    backbone_parameter_final = backbone_parameter_hash(model)
    backbone_init = str(summary.get("backbone_init_hash", ""))
    backbone_parameter_init = str(summary.get("backbone_parameter_init_hash", ""))
    head_backbone_unchanged = None
    full_backbone_changed = None
    full_backbone_parameter_changed = None
    if str(summary.get("regime")) == "head_only":
        head_backbone_unchanged = bool(backbone_final == backbone_init)
    elif str(summary.get("regime")) == "full":
        full_backbone_changed = bool(backbone_final != backbone_init)
        full_backbone_parameter_changed = bool(backbone_parameter_final != backbone_parameter_init)
    result = {
        "schema_version": 1,
        "kind": "independent_dense_evaluation",
        "status": "passed" if (head_backbone_unchanged is not False and full_backbone_changed is not False and full_backbone_parameter_changed is not False and np.isfinite(raw) and np.isfinite(residual)) else "failed",
        "run_dir": str(run.resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "allow_small_cache": bool(allow_small_cache),
        "seed": int(summary["seed"]),
        "support": str(summary["support"]),
        "regime": str(summary["regime"]),
        "protocol_hash": p_hash,
        "asset_bundle_sha256": expected_bundle,
        "package_id": expected_package_id,
        "package_manifest_sha256": expected_package_manifest_sha,
        "checkpoint": {
            "path": str(checkpoint_path),
            "selection": "endpoint_final",
            "sha256": sha256_file(checkpoint_path),
            "state_hash": state_dict_hash(checkpoint_payload["model_state"]),
        },
        "dense_images": {"path": str(dense_image_path), "sha256": sha256_file(dense_image_path), "count": int(len(dense_images))},
        "dense_labels": {"path": str(dense_points_path), "sha256": sha256_file(dense_points_path), "count": int(len(dense_points)), "read_after_prediction": True},
        "prediction_frozen_sha256": prediction_frozen_sha,
        "prediction_archive": {
            "path": str(prediction_archive_path),
            "sha256": prediction_archive_sha,
            "shape": list(prediction_px.shape),
            "read_before_dense_labels": True,
        },
        "metrics": {"raw_mae_px": raw, "affine_removed_mae_px": residual},
        "backbone": {"init_hash": backbone_init, "final_hash": backbone_final, "parameter_init_hash": backbone_parameter_init, "parameter_final_hash": backbone_parameter_final, "head_only_bitwise_unchanged": head_backbone_unchanged, "full_backbone_changed": full_backbone_changed, "full_backbone_parameter_changed": full_backbone_parameter_changed},
        "elapsed_sec": float(time.perf_counter() - started),
    }
    write_json(run / "independent_evaluator.json", result)
    write_json(
        run / "evaluator_receipt.json",
        {
            "schema_version": 1,
            "kind": "independent_evaluator_receipt",
            "prediction_archive_sha256": prediction_archive_sha,
            "asset_bundle_sha256": expected_bundle,
            "package_id": expected_package_id,
            "package_manifest_sha256": expected_package_manifest_sha,
            "prediction_archive_path": str(prediction_archive_path),
            "dense_labels_read_after_atomic_archive": True,
            "dense_label_sha256": sha256_file(dense_points_path),
        },
    )
    return result


__all__ = ["affine_removed_mae_px", "evaluate_run", "fit_affine"]
