"""Supplemental 60-epoch baseline experiment.

This module is deliberately separate from ``resnet_phase_experiment.py``.  It
reuses that module's synthetic data, model construction, seed and training
configuration, but trains only the original ``baseline`` ResNet18 for 60
epochs.  The default command runs the experiment; ``--check-only`` performs
read-only architecture/data/output checks and never starts optimization.

The output directory is intentionally outside the main experiment's
checkpoints and analysis folders::

    outputs/supplemental/baseline_60ep/

An existing non-empty output directory is never silently overwritten.  Pass
``--force`` to overwrite individual files in that directory (no recursive
deletion is performed).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import resnet_phase_experiment as rpe


# Keep the extension explicit while taking every base setting from the main
# experiment.  The three milestone epochs are part of the supplemental design,
# not a change to the original 20-epoch experiment.
LONG_EPOCHS = 60
MILESTONES = (10, 20, 40, 60)
VARIANT = "baseline"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "supplemental" / "baseline_60ep"


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write a deterministic CSV, preserving the first-seen column order."""

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _torch_load(path: Path, device: torch.device) -> Mapping[str, object]:
    """Load a checkpoint across torch versions with/without ``weights_only``."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"checkpoint {path} is not a mapping")
    return checkpoint


def _make_scaler(enabled: bool):
    """Construct the CUDA scaler using the current or legacy AMP API."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (TypeError, RuntimeError):
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(
    images: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = rpe.OfflineLineDataset(images, targets)
    return DataLoader(
        dataset,
        batch_size=rpe.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=rpe.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    """Evaluate a completed epoch in ``eval()`` mode, including pixel MAE."""

    model.eval()
    mse_sum = 0.0
    abs_sum = 0.0
    count = 0
    # Match the primary script's mean-per-batch MSE, then weight each batch by
    # its sample count.  This keeps the final metric exact even for a short
    # final batch and makes it the value used by ReduceLROnPlateau/best.
    criterion = nn.MSELoss()
    scale_px = float(rpe.IMAGE_SIZE - 1)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with rpe._autocast_context(device, amp_enabled):
                predictions = model(images)
                mse = criterion(predictions, targets)
            mse_sum += float(mse.item()) * len(images)
            abs_sum += float((predictions - targets).abs().sum().item())
            count += int(len(images))
    denom = float(max(1, count))
    mae = abs_sum / denom
    return {
        "mse": mse_sum / denom,
        "mae": mae,
        "mae_px": mae * scale_px,
        "count": float(count),
    }


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler,
    epoch: int,
    best_epoch: int,
    best_val_mse: float,
    best_val_mae_px: float,
    history_row: Mapping[str, object],
    checkpoint_kind: str,
) -> Dict[str, object]:
    """Build a complete, restartable checkpoint payload."""

    return {
        "variant": VARIANT,
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val_mse),
        "best_val_mae_px": float(best_val_mae_px),
        "data_fingerprint": rpe.config_fingerprint(),
        "config": rpe._jsonable_config(),
        "training_config": {
            "batch_size": int(rpe.BATCH_SIZE),
            "epochs": LONG_EPOCHS,
            "learning_rate": float(rpe.LEARNING_RATE),
            "weight_decay": float(rpe.WEIGHT_DECAY),
            "num_workers": int(rpe.NUM_WORKERS),
            "amp": bool(rpe.AMP),
            "grad_clip_norm": float(rpe.GRAD_CLIP_NORM),
            "scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=4)",
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history_row": dict(history_row),
        "saved_at_utc": _utc_now(),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def _assert_base_configuration() -> None:
    """Guard against accidentally drifting from the requested main config."""

    if VARIANT not in rpe.VARIANTS:
        raise RuntimeError(f"main script no longer exposes {VARIANT!r}")
    if int(rpe.BATCH_SIZE) != 64:
        raise RuntimeError(f"expected main BATCH_SIZE=64, got {rpe.BATCH_SIZE!r}")
    if int(rpe.NUM_WORKERS) != 0:
        raise RuntimeError(f"expected Windows-safe NUM_WORKERS=0, got {rpe.NUM_WORKERS!r}")


def _check_data_cache() -> Tuple[bool, Dict[str, object]]:
    """Read and validate all three pre-generated splits without creating them."""

    expected_counts = {
        "train": len(rpe._centres_for_offsets(rpe.TRAIN_OFFSETS)),
        "val": len(rpe._centres_for_offsets(rpe.VAL_OFFSETS)),
        "dense_test": len(rpe._dense_centres()),
    }
    status: Dict[str, object] = {
        "data_dir": str(rpe.DATA_DIR),
        "fingerprint": rpe.config_fingerprint(),
        "expected_counts_from_config": expected_counts,
        "splits": {},
    }
    ready = True
    manifest_path = rpe.DATA_DIR / "manifest.json"
    manifest_counts: Mapping[str, object] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != rpe.config_fingerprint():
            raise RuntimeError(
                f"manifest fingerprint mismatch ({manifest.get('fingerprint')!r} != {rpe.config_fingerprint()!r})"
            )
        raw_counts = manifest.get("counts", {})
        if not isinstance(raw_counts, Mapping):
            raise RuntimeError("manifest counts is not a mapping")
        manifest_counts = raw_counts
        status["manifest"] = {"present": True, "counts": dict(raw_counts)}
    except (FileNotFoundError, OSError, json.JSONDecodeError, RuntimeError, TypeError, AttributeError) as exc:
        ready = False
        status["manifest"] = {"present": False, "error": str(exc)}

    expected_dense = rpe._dense_centres()
    for split in ("train", "val", "dense_test"):
        try:
            images, targets, x_px = rpe._load_npz_split(split)
            actual_count = int(len(targets))
            config_count_ok = actual_count == int(expected_counts[split]) and len(images) == actual_count and len(x_px) == actual_count
            manifest_count = manifest_counts.get(split)
            manifest_count_ok = (
                isinstance(manifest_count, (int, float))
                and int(manifest_count) == actual_count
                and float(manifest_count) == float(actual_count)
            )
            target_definition_ok = bool(
                np.allclose(
                    targets,
                    x_px / float(rpe.IMAGE_SIZE - 1),
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            dense_sequence_ok = True
            if split == "dense_test":
                dense_sequence_ok = bool(
                    len(x_px) == len(expected_dense)
                    and np.array_equal(x_px, expected_dense)
                    and np.all(np.diff(x_px) == 1.0)
                    and np.all(x_px == np.floor(x_px))
                )
            split_info = {
                "present": True,
                "count": actual_count,
                "expected_count_from_config": int(expected_counts[split]),
                "manifest_count": manifest_count,
                "config_count_ok": config_count_ok,
                "manifest_count_ok": manifest_count_ok,
                "target_definition_ok": target_definition_ok,
                "dense_sequence_ok": dense_sequence_ok,
                "image_shape": list(images.shape),
                "target_shape": list(targets.shape),
                "x_px_shape": list(x_px.shape),
            }
            if not (config_count_ok and manifest_count_ok and target_definition_ok and dense_sequence_ok):
                ready = False
                split_info["error"] = "count/target-definition/dense-sequence validation failed"
        except (FileNotFoundError, RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            ready = False
            split_info = {"present": False, "error": str(exc)}
        status["splits"][split] = split_info
    return ready, status


def _output_status() -> Dict[str, object]:
    files = sorted(str(path.relative_to(OUTPUT_DIR)) for path in OUTPUT_DIR.rglob("*") if path.is_file()) if OUTPUT_DIR.exists() else []
    return {"path": str(OUTPUT_DIR), "exists": OUTPUT_DIR.exists(), "files": files, "nonempty": bool(files)}


def check_only() -> int:
    """Perform read-only checks; a missing prepare cache is reported, not made."""

    _assert_base_configuration()
    print(f"[check-only] module: {Path(rpe.__file__).resolve()}")
    print(f"[check-only] fingerprint: {rpe.config_fingerprint()}")
    print(f"[check-only] device: {rpe.choose_device()}")

    # Architecture smoke test uses a single deterministic sample and no grad.
    rpe.seed_everything(rpe.SEED)
    device = rpe.choose_device()
    model = rpe.build_model(VARIANT).to(device).eval()
    sample = torch.from_numpy(rpe.render_line(112.3)).float().div_(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample)
    if tuple(output.shape) != (1, 1):
        raise RuntimeError(f"unexpected baseline output shape: {tuple(output.shape)}")
    print(f"[check-only] architecture: output={tuple(output.shape)}, parameters={sum(p.numel() for p in model.parameters())}")

    data_ready, data_status = _check_data_cache()
    if data_ready:
        print("[check-only] data cache: ready (train/val/dense_test validated)")
    else:
        print("[check-only] data cache: not ready; run the main script's `prepare` stage before training")
        for split, info in data_status["splits"].items():
            if not info.get("present", False):
                print(f"  - {split}: {info.get('error', 'missing')}")

    output_status = _output_status()
    if output_status["nonempty"]:
        print("[check-only] output directory is non-empty; a normal run will refuse to overwrite it")
    else:
        print(f"[check-only] output directory is available: {OUTPUT_DIR}")
    print("[check-only] no training was started")
    # This command is useful before prepare, so missing data is a visible
    # warning rather than a mutation or an optimization failure.
    return 0


def train_long(force: bool = False) -> Dict[str, object]:
    """Train baseline for 60 epochs, then run dense analysis and plots."""

    _assert_base_configuration()
    data_ready, data_status = _check_data_cache()
    if not data_ready:
        raise RuntimeError("data cache is not ready; run `python resnet_phase_experiment.py prepare` first")

    output_status = _output_status()
    if output_status["nonempty"] and not force:
        raise FileExistsError(
            f"refusing to overwrite existing supplemental output files in {OUTPUT_DIR}; pass --force to overwrite files"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rpe.seed_everything(rpe.SEED)
    train_images, train_targets, _ = rpe._load_npz_split("train")
    val_images, val_targets, _ = rpe._load_npz_split("val")
    device = rpe.choose_device()
    amp_enabled = bool(rpe.AMP and device.type == "cuda")

    train_generator = torch.Generator().manual_seed(rpe.SEED)
    train_loader = _loader(train_images, train_targets, device, shuffle=True, generator=train_generator)
    train_eval_loader = _loader(train_images, train_targets, device, shuffle=False)
    val_loader = _loader(val_images, val_targets, device, shuffle=False)

    # build_model('baseline') calls torchvision.resnet18(weights=None) and
    # changes only fc, exactly matching the primary baseline definition.
    model = rpe.build_model(VARIANT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rpe.LEARNING_RATE, weight_decay=rpe.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.MSELoss()
    scaler = _make_scaler(amp_enabled)

    history_path = OUTPUT_DIR / "history.csv"
    best_path = OUTPUT_DIR / "baseline_best.pt"
    best_val_mse = float("inf")
    best_val_mae_px = float("inf")
    best_epoch = 0
    history: List[Dict[str, object]] = []

    print(
        f"[baseline_60ep] device={device}, AMP={amp_enabled}, batch={rpe.BATCH_SIZE}, "
        f"train={len(train_targets)}, val={len(val_targets)}"
    )
    for epoch in range(1, LONG_EPOCHS + 1):
        model.train()
        online_mse_sum = 0.0
        online_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with rpe._autocast_context(device, amp_enabled):
                predictions = model(images)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            if rpe.GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), rpe.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            online_mse_sum += float(loss.detach().item()) * len(images)
            online_count += int(len(images))

        # These are all measured with the same completed weights and eval()
        # mode, so BN statistics and dropout cannot mix training/eval notions.
        train_eval = _evaluate(model, train_eval_loader, device, amp_enabled)
        val_eval = _evaluate(model, val_loader, device, amp_enabled)
        lr_used = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_eval["mse"])
        lr_next = float(optimizer.param_groups[0]["lr"])
        row: Dict[str, object] = {
            "epoch": int(epoch),
            "train_mse": online_mse_sum / float(max(1, online_count)),
            "val_mse": val_eval["mse"],
            "val_mae": val_eval["mae"],
            # Keep the historical ``lr`` column as the post-scheduler value,
            # while making the optimizer value used in this epoch explicit.
            "lr": lr_next,
            "lr_used": lr_used,
            "lr_next": lr_next,
            "train_eval_mse": train_eval["mse"],
            "train_eval_mae": train_eval["mae"],
            "train_eval_mae_px": train_eval["mae_px"],
            "val_eval_mse": val_eval["mse"],
            "val_eval_mae": val_eval["mae"],
            "val_eval_mae_px": val_eval["mae_px"],
        }
        history.append(row)
        # Persist after every epoch so a long run remains inspectable after an
        # interruption.  This intentionally overwrites only this CSV.
        _write_rows(history_path, history)
        print(
            f"[baseline_60ep] epoch {epoch:03d}/{LONG_EPOCHS}: "
            f"val_eval_mae_px={val_eval['mae_px']:.6g} lr_used={lr_used:.6g} lr_next={lr_next:.6g}"
        )

        if val_eval["mse"] < best_val_mse:
            best_val_mse = float(val_eval["mse"])
            best_val_mae_px = float(val_eval["mae_px"])
            best_epoch = epoch
            _save_checkpoint(
                best_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_mse=best_val_mse,
                    best_val_mae_px=best_val_mae_px,
                    history_row=row,
                    checkpoint_kind="best",
                ),
            )

        if epoch in MILESTONES:
            _save_checkpoint(
                OUTPUT_DIR / f"baseline_ep{epoch}.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_mse=best_val_mse,
                    best_val_mae_px=best_val_mae_px,
                    history_row=row,
                    checkpoint_kind=f"milestone_ep{epoch}",
                ),
            )

    final_path = OUTPUT_DIR / "baseline_final.pt"
    _save_checkpoint(
        final_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=LONG_EPOCHS,
            best_epoch=best_epoch,
            best_val_mse=best_val_mse,
            best_val_mae_px=best_val_mae_px,
            history_row=history[-1],
            checkpoint_kind="final",
        ),
    )

    training_summary: Dict[str, object] = {
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val_mse),
        "best_val_mae_px": float(best_val_mae_px),
        "final_epoch": LONG_EPOCHS,
        "history_rows": len(history),
        "checkpoint_paths": [
            str(OUTPUT_DIR / f"baseline_ep{epoch}.pt") for epoch in MILESTONES
        ]
        + [str(best_path), str(final_path)],
        "data": data_status,
    }
    _write_training_plot(history)
    dense_summary = dense_analysis(device=device)
    _write_dense_plot(dense_summary)
    metadata = _write_metadata(training_summary, dense_summary, device)
    print(f"[baseline_60ep] best epoch={best_epoch}, val_eval_mae_px={best_val_mae_px:.6g}")
    print(f"[baseline_60ep] outputs: {OUTPUT_DIR}")
    return metadata


def _load_model_checkpoint(path: Path, device: torch.device) -> nn.Module:
    checkpoint = _torch_load(path, device)
    if checkpoint.get("data_fingerprint") != rpe.config_fingerprint():
        raise RuntimeError(f"checkpoint {path} has a mismatched data fingerprint")
    if checkpoint.get("variant") != VARIANT:
        raise RuntimeError(f"checkpoint {path} is not a baseline checkpoint")
    model = rpe.build_model(VARIANT).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _predict(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    loader = _loader(images, np.zeros(len(images), dtype=np.float32), device, shuffle=False)
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for images_batch, _ in loader:
            with rpe._autocast_context(device, bool(rpe.AMP and device.type == "cuda")):
                output = model(images_batch.to(device, non_blocking=True))
            outputs.append(output.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(outputs).astype(np.float64, copy=False) if outputs else np.empty(0, dtype=np.float64)


def _residue_peak_to_peak(x_px: np.ndarray, errors_px: np.ndarray, period: int) -> float:
    detrended = rpe._linear_detrend(errors_px, x_px)
    means: List[float] = []
    residues = np.mod(np.asarray(x_px, dtype=np.int64), period)
    for residue in range(period):
        values = detrended[residues == residue]
        if len(values):
            means.append(float(values.mean()))
    return float(max(means) - min(means)) if means else float("nan")


def _exact_fft_metrics(errors_px: np.ndarray, x_px: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Return FFT rows and exact target-bin metrics (no local ±2-bin peak)."""

    detrended = rpe._linear_detrend(errors_px, x_px)
    window = np.hanning(len(detrended))
    raw_spectrum = np.abs(np.fft.rfft(detrended * window))
    # For a real sinusoid, Hann-windowed magnitude is scaled by approximately
    # window.sum()/2.  Report this normalized amplitude while retaining the
    # raw FFT magnitude for auditability.  All target bins are non-DC, so the
    # DC convention does not affect the requested period metrics.
    amplitude_scale = max(float(window.sum() / 2.0), 1e-12)
    spectrum = raw_spectrum / amplitude_scale
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0)
    fft_rows = [
        {
            "bin": int(index),
            "frequency_cycles_per_pixel": float(freq),
            "raw_fft_magnitude_px": float(raw_spectrum[index]),
            "amplitude_px": float(amplitude),
            "preprocessing": "linear_detrend_then_hann",
        }
        for index, (freq, amplitude) in enumerate(zip(frequencies, spectrum))
    ]
    target_rows: List[Dict[str, object]] = []
    non_dc_bins = np.arange(1, len(spectrum), dtype=np.int64)
    for period in rpe.PERIODS:
        target_frequency = 1.0 / float(period)
        target_bin = int(np.argmin(np.abs(frequencies - target_frequency)))
        noise_bins = non_dc_bins[non_dc_bins != target_bin]
        noise_floor = float(np.median(spectrum[noise_bins])) if len(noise_bins) else float("nan")
        amplitude = float(spectrum[target_bin])
        target_rows.append(
            {
                "period": int(period),
                "target_frequency_cycles_per_pixel": target_frequency,
                "target_bin": target_bin,
                "target_bin_frequency_cycles_per_pixel": float(frequencies[target_bin]),
                "raw_target_bin_fft_magnitude_px": float(raw_spectrum[target_bin]),
                "target_bin_amplitude_px": amplitude,
                "noise_floor_median_amplitude_px_excluding_target_bin": noise_floor,
                "target_bin_noise_ratio": amplitude / max(noise_floor, 1e-12)
                if np.isfinite(noise_floor)
                else float("nan"),
                "preprocessing": "linear_detrend_then_hann; exact target bin; DC excluded",
            }
        )
    return np.asarray(spectrum), fft_rows + target_rows


def dense_analysis(*, device: torch.device) -> List[Dict[str, object]]:
    images, targets, x_px = rpe._load_npz_split("dense_test")
    checkpoint_labels = [(f"ep{epoch}", OUTPUT_DIR / f"baseline_ep{epoch}.pt") for epoch in MILESTONES]
    checkpoint_labels.append(("best", OUTPUT_DIR / "baseline_best.pt"))
    summaries: List[Dict[str, object]] = []
    scale_px = float(rpe.IMAGE_SIZE - 1)
    for label, checkpoint_path in checkpoint_labels:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing dense-analysis checkpoint: {checkpoint_path}")
        model = _load_model_checkpoint(checkpoint_path, device)
        pred = _predict(model, images, device)
        errors = pred - targets.astype(np.float64)
        errors_px = errors * scale_px
        pred_px = pred * scale_px
        true_px = targets.astype(np.float64) * scale_px
        bias_px = float(errors_px.mean()) if len(errors_px) else float("nan")
        debiased_mae_px = float(np.abs(errors_px - bias_px).mean()) if len(errors_px) else float("nan")
        adjacent = np.diff(pred_px)
        steps = {
            "adjacent_step_mean_px": float(adjacent.mean()) if len(adjacent) else float("nan"),
            "adjacent_step_mae_vs1_px": float(np.abs(adjacent - 1.0).mean()) if len(adjacent) else float("nan"),
            "nonmonotonic_fraction": float(np.mean(adjacent <= 0.0)) if len(adjacent) else float("nan"),
        }
        prediction_rows = [
            {
                "index": int(index),
                "x_px": float(x_px[index]),
                "true_x": float(targets[index]),
                "pred_x": float(pred[index]),
                "error": float(errors[index]),
                "true_x_px": float(true_px[index]),
                "pred_x_px": float(pred_px[index]),
                "error_px": float(errors_px[index]),
            }
            for index in range(len(x_px))
        ]
        _write_rows(OUTPUT_DIR / f"predictions_{label}.csv", prediction_rows)

        residue_p2p = {
            f"detrended_residue_mean_peak_to_peak_p{period}_px": _residue_peak_to_peak(x_px, errors_px, period)
            for period in rpe.PERIODS
        }
        spectrum, fft_rows_all = _exact_fft_metrics(errors_px, x_px)
        fft_rows = [row for row in fft_rows_all if "bin" in row]
        target_rows = [row for row in fft_rows_all if "target_bin" in row]
        _write_rows(OUTPUT_DIR / f"fft_{label}.csv", fft_rows)
        _write_rows(OUTPUT_DIR / f"fft_targets_{label}.csv", target_rows)

        summary: Dict[str, object] = {
            "checkpoint": label,
            "checkpoint_path": str(checkpoint_path),
            "dense_count": int(len(x_px)),
            "mae_px": float(np.abs(errors_px).mean()) if len(errors_px) else float("nan"),
            "rmse_px": float(np.sqrt(np.mean(errors_px**2))) if len(errors_px) else float("nan"),
            "bias_px": bias_px,
            "debiased_mae_px": debiased_mae_px,
            "max_abs_error_px": float(np.abs(errors_px).max()) if len(errors_px) else float("nan"),
            **steps,
            **residue_p2p,
        }
        for target in target_rows:
            period = int(target["period"])
            summary[f"fft_target_bin_amplitude_p{period}_px"] = target["target_bin_amplitude_px"]
            summary[f"fft_target_bin_noise_ratio_p{period}"] = target["target_bin_noise_ratio"]
        summaries.append(summary)
    _write_rows(OUTPUT_DIR / "dense_summary.csv", summaries)
    return summaries


def _write_training_plot(history: Sequence[Mapping[str, object]]) -> None:
    plt = rpe._import_matplotlib()
    epochs = [int(row["epoch"]) for row in history]
    val_mae_px = [float(row["val_eval_mae_px"]) for row in history]
    # Plot the optimizer value used during each epoch.  ``lr`` remains the
    # post-scheduler compatibility column in history; ``lr_used`` avoids a
    # one-epoch visual shift when the scheduler changes the learning rate.
    lrs_used = [float(row["lr_used"]) for row in history]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    axes[0].plot(epochs, val_mae_px, color="#1f77b4", marker=".", linewidth=1.4)
    axes[0].set_ylabel("validation MAE (px)")
    axes[0].set_title("Baseline 60-epoch validation MAE (full run)")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(alpha=0.25)

    later = [(epoch, mae) for epoch, mae in zip(epochs, val_mae_px) if epoch >= 8]
    zoom_epochs = [item[0] for item in later] or epochs
    zoom_mae = [item[1] for item in later] or val_mae_px
    axes[1].plot(zoom_epochs, zoom_mae, color="#2ca02c", marker=".", linewidth=1.4)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation MAE (px)")
    axes[1].set_title("Validation MAE (epoch 8+ zoom; linear axis)")
    if zoom_mae:
        low = min(zoom_mae)
        high = max(zoom_mae)
        padding = max((high - low) * 0.08, 1e-6)
        axes[1].set_ylim(max(0.0, low - padding), high + padding)
    axes[1].set_xlim(min(zoom_epochs), max(zoom_epochs))
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, lrs_used, color="#d62728", marker=".", linewidth=1.4)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate used")
    axes[2].set_title("Learning rate used in each epoch")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_val_mae_lr.png", dpi=160)
    plt.close(fig)


def _write_dense_plot(summaries: Sequence[Mapping[str, object]]) -> None:
    plt = rpe._import_matplotlib()
    milestone_labels = {f"ep{epoch}" for epoch in MILESTONES}
    labels = [str(row["checkpoint"]) for row in summaries if str(row["checkpoint"]) in milestone_labels]
    selected = [row for row in summaries if str(row["checkpoint"]) in labels]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, (name, color) in enumerate(
        (("mae_px", "#1f77b4"), ("bias_px", "#ff7f0e"), ("debiased_mae_px", "#2ca02c"))
    ):
        values = [float(row[name]) for row in selected]
        ax.bar(x + (offset - 1) * width, values, width=width, label=name, color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("dense error (px)")
    ax.set_title("Baseline dense-test milestone errors")
    ax.set_ylim(bottom=min(0.0, min((float(row["bias_px"]) for row in selected), default=0.0)))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dense_milestone_metrics.png", dpi=160)
    plt.close(fig)


def _write_metadata(
    training_summary: Mapping[str, object],
    dense_summary: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "status": "complete",
        "created_at_utc": _utc_now(),
        "script": str(Path(__file__).resolve()),
        "main_module": str(Path(rpe.__file__).resolve()),
        "variant": VARIANT,
        "device": str(device),
        "seed": int(rpe.SEED),
        "data_fingerprint": rpe.config_fingerprint(),
        "config": rpe._jsonable_config(),
        "training_config": {
            "epochs": LONG_EPOCHS,
            "batch_size": int(rpe.BATCH_SIZE),
            "learning_rate": float(rpe.LEARNING_RATE),
            "weight_decay": float(rpe.WEIGHT_DECAY),
            "amp_requested": bool(rpe.AMP),
            "amp_enabled": bool(rpe.AMP and device.type == "cuda"),
            "grad_clip_norm": float(rpe.GRAD_CLIP_NORM),
            "scheduler": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 4},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": rpe._safe_package_version("torch"),
            "torchvision": rpe._safe_package_version("torchvision"),
            "numpy": rpe._safe_package_version("numpy"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_summary": dict(training_summary),
        "dense_summary": [dict(row) for row in dense_summary],
        "files": sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="check architecture/data/output without training")
    parser.add_argument("--force", action="store_true", help="allow overwriting individual files in the output directory")
    args = parser.parse_args(argv)
    if args.check_only:
        return check_only()
    train_long(force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
