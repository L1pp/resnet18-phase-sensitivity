"""Phase 1.7 multi-seed confirmation of the frozen Phase 1.6 B_t protocol.

Does not change the B_t estimator, rank, nulls, projection, or success
thresholds after seeing new-seed results. Writes only under
results/phase1_gap_rep/phase1_7_multiseed/.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .analyze import _labels, _load_split_codes
from .audit import (
    SPLIT_NAMES,
    _erasure_z,
    _frobenius_alignment,
    _functional_alignment,
    _histogram,
    _pc_spectrum,
    _predict_p,
    _principal_angles_deg,
    _qr_basis,
    _readouts,
    _subspace_energy,
    _to_py,
)
from .bt_robustness import (
    ENERGY_TOL_OK,
    ENERGY_TOL_STRICT,
    N_FOLD,
    N_NULL,
    _eval_cut,
    _fit_factor,
    _mae_px,
    _overlap_norm,
    _pack_head,
    _probe_fit,
    _probe_mae,
    _sample_matched,
    _summarize_null,
    _write_cells,
)
from .common import (
    ANALYSIS_GATE_MAE_PX,
    BATCH_SIZE,
    COORD_SCALE,
    EARLY_STOP_MAE_PX,
    RESULTS_ROOT,
    SEED,
    SCRIPT_DIR,
    TrainRun,
    build_model,
    decompose_controls,
    device,
    dirs,
    dump_json,
    environment_versions,
    fingerprint,
    gap_features,
    geometry_config,
    images_to_tensor,
    profile_spec,
    seed_everything,
    setup_matplotlib_chinese,
    train_run_config,
)
from .generate_data import load_split
from .train import (
    FactorialDataset,
    TrainingExploded,
    _amp_context,
    _build_optimizer,
    _build_scheduler,
    _exploded,
    _plot_history,
    _predict,
    _write_csv,
    combined_loss,
    coordinate_errors,
)

OUT_NAME = "phase1_7_multiseed"
SEEDS = (20260810, 20260811, 20260812)
EXISTING_RUN = {20260810: "adamw_l1_scratch"}
ANALYSIS_SEED = 20260810
PROTOCOL_FP = "195f37870e2cdb8685a7"
TRAINING_LEARNED_VAL_MAE = 1.0
R_T_MIN = 0.80
R_Q_MAX = 1.25
P_EXTREME = 0.05
FOLD_MEAN_MAX = 20.0
FOLD_MAX_MAX = 35.0
REVERSE_T_MAX = 0.50
REVERSE_Q_MIN = 2.0
CODE_FILES = (
    "common.py",
    "generate_data.py",
    "train.py",
    "extract.py",
    "analyze.py",
    "audit.py",
    "cuts.py",
    "bt_robustness.py",
    "bt_explore.py",
    "run.py",
    "phase17.py",
)


def _root() -> Path:
    return RESULTS_ROOT / OUT_NAME


def _seed_dir(seed: int) -> Path:
    path = _root() / f"seed_{seed}"
    for name in ("config", "checkpoints", "tables", "figures", "features", "nulls"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _cross_dir() -> Path:
    path = _root() / "cross_seed_summary"
    for name in ("tables", "figures"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _safe_histogram(samples: Sequence[float], bins: int = 40) -> Dict[str, Any]:
    array = np.asarray(samples, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"counts": [], "edges": []}
    lo, hi = float(finite.min()), float(finite.max())
    if (hi - lo) <= max(1e-12, abs(lo) * 1e-9):
        return {"counts": [int(finite.size)], "edges": [lo, lo + max(1e-12, abs(lo) * 1e-9)]}
    try:
        counts, edges = np.histogram(finite, bins=bins)
        return {"counts": counts.tolist(), "edges": edges.tolist()}
    except ValueError:
        return {"counts": [int(finite.size)], "edges": [lo, hi if hi > lo else lo + 1e-12]}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        array = state[key].detach().float().cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_state() -> Dict[str, Any]:
    repo = SCRIPT_DIR.parent

    def run(args: Sequence[str]) -> str:
        result = subprocess.run(args, cwd=repo, capture_output=True, text=True, check=False)
        return (result.stdout or "").strip()

    porcelain = run(["git", "status", "--porcelain"])
    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(porcelain),
        "status_porcelain": porcelain,
        "diff_stat": run(["git", "diff", "--stat"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _protocol_run() -> TrainRun:
    return TrainRun(
        name="phase17_protocol",
        optimizer="adamw",
        lr=1e-3,
        lr_min=1e-5,
        epochs=100,
        patience=50,
        schedule="cosine",
        l1_weight=0.25,
        data_profile="factorial",
    )


def freeze_protocol(force: bool = False) -> Dict[str, Any]:
    out = _cross_dir()
    dest = out / "frozen_protocol.json"
    if dest.exists() and not force:
        print("[phase17] freeze already exists; not overwriting")
        return json.loads(dest.read_text(encoding="utf-8"))
    spec = profile_spec("factorial")
    data_dir = dirs(spec)["data"]
    data_hashes = {}
    for path in sorted(data_dir.glob("*.npz")):
        data_hashes[path.name] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    code_hashes = {}
    for name in CODE_FILES:
        path = SCRIPT_DIR / name
        code_hashes[name] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    env = environment_versions()
    env["cuda"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
        env["cuda_runtime"] = torch.version.cuda
    payload = {
        "experiment": "phase1.7_multiseed",
        "question": "Do independent ResNet18 inits repeat a rank-2 translation-associated B_t with the same linear-erasure function?",
        "protocol_fingerprint_expected": PROTOCOL_FP,
        "protocol_fingerprint_actual": fingerprint(spec),
        "analysis_rng_seed": ANALYSIS_SEED,
        "training_seeds": list(SEEDS),
        "existing_seed_source": EXISTING_RUN,
        "bt_definition": {
            "Z_c": "Z - mu_train",
            "q": "geometry_comp 4D from decompose_controls",
            "X": "[tx, ty, q1, q2, q3, q4] mean/std on train only",
            "fit": "OLS np.linalg.lstsq, no ridge",
            "B_t": "QR span of two translation coefficient rows, rank fixed at 2",
            "erasure": "z' = mu + (I-P)(z-mu), centered",
            "reprobe": "OLS with intercept, fit erased train only",
            "identity_perm": "permute translation_id -> (tx,ty) globally; q and Z fixed; N=1000",
            "energy_matched": "PC-window rank-2, try 2% then 5%, N=1000, MAX_DRAWS=40000",
            "shape_fold": "seed 20260810 permutation of 64 shapes, 8 folds x 8, jackknife estimator stability",
            "constants": {
                "N_NULL": N_NULL,
                "N_FOLD": N_FOLD,
                "ENERGY_TOL_STRICT": ENERGY_TOL_STRICT,
                "ENERGY_TOL_OK": ENERGY_TOL_OK,
            },
        },
        "training": train_run_config(_protocol_run()),
        "geometry": geometry_config(spec),
        "replication_thresholds_frozen_before_new_seeds": {
            "note": "Copied from Phase1.6 decision rules for re-probe/fold; not A/B/C letters.",
            "R_t_min": R_T_MIN,
            "R_q_max": R_Q_MAX,
            "null_p_max": P_EXTREME,
            "fold_mean_angle_max_deg": FOLD_MEAN_MAX,
            "fold_max_angle_max_deg": FOLD_MAX_MAX,
            "training_learned_val_mae_px": TRAINING_LEARNED_VAL_MAE,
            "reverse_R_t_max": REVERSE_T_MAX,
            "reverse_R_q_min": REVERSE_Q_MIN,
            "primary": ["R_t", "R_q", "reprobe_null_p", "shape_fold_angles"],
            "secondary": ["original_head_collapse", "t_swap", "init_Bt_angles"],
        },
        "data_files": data_hashes,
        "code_hashes": code_hashes,
        "environment": env,
        "git": _git_state(),
        "do_not": [
            "change B_t estimator/rank/null/projection after seeing new seeds",
            "use test MAE gate to skip extract",
            "drop a poorly trained seed",
            "compare raw B_t principal angles across seeds",
            "M1/M2, new architecture, new data, commit/push",
        ],
    }
    if payload["protocol_fingerprint_actual"] != PROTOCOL_FP:
        raise RuntimeError(
            f"fingerprint {payload['protocol_fingerprint_actual']} != {PROTOCOL_FP}"
        )
    dump_json(dest, payload)
    dump_json(out / "code_hashes.json", code_hashes)
    dump_json(out / "environment.json", env)
    dump_json(out / "seed_manifest.json", {"seeds": list(SEEDS), "existing": EXISTING_RUN})
    print(f"[phase17] froze protocol fp={PROTOCOL_FP} git={payload['git']['head'][:12]}")
    return payload


def _extract_gap(model, spec) -> np.ndarray:
    n_s, n_t = spec.n_shapes, spec.n_translations
    z = np.zeros((n_s, n_t, 512), dtype=np.float32)
    filled = np.zeros((n_s, n_t), dtype=bool)
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for split in ("train", "val", "test"):
            data = load_split(spec, split)
            images = data["images"]
            sid = data["shape_id"].astype(np.int64)
            tid = data["translation_id"].astype(np.int64)
            feats = []
            for start in range(0, len(images), BATCH_SIZE):
                batch = images_to_tensor(images[start : start + BATCH_SIZE]).to(dev)
                feats.append(gap_features(model, batch).float().cpu().numpy())
            array = np.concatenate(feats, axis=0)
            z[sid, tid] = array
            filled[sid, tid] = True
            print(f"[phase17] extract {split} n={len(images)}")
    if not bool(np.all(filled)):
        raise RuntimeError("incomplete GAP features")
    return z


def _save_init(model, spec, seed_dir: Path, seed: int, reconstructed: bool) -> Dict[str, Any]:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_path = seed_dir / "checkpoints" / "resnet18_init.pt"
    torch.save({"model_state": state, "seed": seed, "reconstructed": reconstructed}, init_path)
    print("[phase17] forward Z_init (before optimizer)")
    z_init = _extract_gap(model, spec)
    np.savez_compressed(
        seed_dir / "features" / "Z_init.npz",
        Z=z_init,
        fingerprint=np.asarray(fingerprint(spec)),
        seed=np.asarray(seed),
    )
    info = {
        "seed": seed,
        "reconstructed_from_seed": reconstructed,
        "state_sha256": _state_sha256(state),
        "init_ckpt_sha256": _file_sha256(init_path),
        "Z_init_sha256": _file_sha256(seed_dir / "features" / "Z_init.npz"),
        "fingerprint": fingerprint(spec),
    }
    dump_json(seed_dir / "config" / "init_info.json", info)
    return info


def _task_metrics(pred: np.ndarray, labels: Mapping[str, np.ndarray], g: np.ndarray, masks) -> Dict[str, Any]:
    return {name: _pack_head(pred, labels, g, mask) for name, mask in masks.items()}


def _evaluate_model(model, spec, labels, g, masks, z_grid=None) -> Dict[str, Any]:
    split_pred = {}
    for split in ("train", "val", "test"):
        data = load_split(spec, split)
        pred = _predict(model, data["images"])
        split_pred[split] = {
            "coord": coordinate_errors(pred, data["P"]),
            "sid": data["shape_id"].astype(np.int64),
            "tid": data["translation_id"].astype(np.int64),
            "pred": pred,
        }
    pred_grid = np.zeros((spec.n_shapes, spec.n_translations, 6), dtype=np.float64)
    for split, item in split_pred.items():
        pred_grid[item["sid"], item["tid"]] = item["pred"]
    pred_flat = pred_grid.reshape(-1, 6)
    packed = _task_metrics(pred_flat, labels, g, masks)
    metrics = {
        split: {
            **split_pred[split]["coord"],
            "E_t": packed[split]["E_t"],
            "E_q": packed[split]["E_q"],
            "E_P": packed[split]["E_P"],
            "E_g": packed[split]["E_g"],
        }
        for split in ("train", "val", "test")
    }
    return metrics


def _copy_existing_seed(seed: int, seed_dir: Path, spec) -> None:
    src = RESULTS_ROOT / EXISTING_RUN[seed]
    for rel in (
        Path("tables") / "history.csv",
        Path("tables") / "metrics.json",
        Path("tables") / "metrics.csv",
        Path("tables") / "train_summary.json",
        Path("config") / "train_config.json",
        Path("features") / "Z.npz",
        Path("features") / "head.npz",
        Path("figures") / "train_mae.png",
    ):
        source = src / rel
        dest = seed_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, dest)
    hashes = {}
    for name in ("resnet18_best.pt", "resnet18_last.pt"):
        path = src / "checkpoints" / name
        if path.exists():
            hashes[name] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size, "source": str(path)}
    dump_json(
        seed_dir / "config" / "existing_source.json",
        {"source_run": EXISTING_RUN[seed], "checkpoint_hashes": hashes, "copied_trained_features": True},
    )


def train_seed(seed: int) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    seed_dir = _seed_dir(seed)
    run = TrainRun(
        name=f"phase17_seed_{seed}",
        optimizer="adamw",
        lr=1e-3,
        lr_min=1e-5,
        epochs=100,
        patience=50,
        schedule="cosine",
        l1_weight=0.25,
        data_profile="factorial",
    )
    labels_grid = _labels(spec.name)
    split_grid = _load_split_codes(spec.name)
    labels = {
        "P": labels_grid["P"].reshape(-1, 6),
        "Q": labels_grid["Q"].reshape(-1, 6),
        "t": labels_grid["t"].reshape(-1, 2),
    }
    g = decompose_controls(labels["P"])["geometry_comp"]
    masks = {name: split_grid.reshape(-1) == i for i, name in enumerate(SPLIT_NAMES)}

    if seed in EXISTING_RUN:
        print(f"[phase17] seed {seed}: reuse {EXISTING_RUN[seed]}, do not retrain")
        _copy_existing_seed(seed, seed_dir, spec)
        if not (seed_dir / "features" / "Z_init.npz").exists():
            seed_everything(seed)
            model = build_model().to(device())
            init_info = _save_init(model, spec, seed_dir, seed, reconstructed=True)
        else:
            init_info = json.loads((seed_dir / "config" / "init_info.json").read_text(encoding="utf-8"))
            print(f"[phase17] seed {seed}: Z_init exists")
        summary = json.loads((seed_dir / "tables" / "train_summary.json").read_text(encoding="utf-8"))
        metrics = json.loads((seed_dir / "tables" / "metrics.json").read_text(encoding="utf-8"))
        dump_json(
            seed_dir / "config" / "train_config.json",
            {**train_run_config(run), "seed": seed, "source": EXISTING_RUN[seed], "fingerprint": fingerprint(spec)},
        )
        return {"seed": seed, "reused": True, "init": init_info, "summary": summary, "metrics": metrics}

    best_path = seed_dir / "checkpoints" / "resnet18_best.pt"
    last_path = seed_dir / "checkpoints" / "resnet18_last.pt"
    history_path = seed_dir / "tables" / "history.csv"
    summary_path = seed_dir / "tables" / "train_summary.json"
    if summary_path.exists() and best_path.exists() and last_path.exists():
        print(f"[phase17] seed {seed}: training already finished")
        if not (seed_dir / "features" / "Z_init.npz").exists():
            seed_everything(seed)
            model = build_model().to(device())
            _save_init(model, spec, seed_dir, seed, reconstructed=True)
        return {
            "seed": seed,
            "reused": False,
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "skipped_train": True,
        }

    train_data = load_split(spec, "train")
    val_data = load_split(spec, "val")
    seed_everything(seed)
    dev = device()
    amp = bool(dev.type == "cuda")
    model = build_model().to(dev)
    resume = last_path.exists() and not summary_path.exists()
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    best_val = math.inf
    init_info: Dict[str, Any]
    if resume:
        payload = torch.load(last_path, map_location=dev, weights_only=False)
        if payload.get("fingerprint") != fingerprint(spec):
            raise RuntimeError("checkpoint fingerprint mismatch")
        model.load_state_dict(payload["model_state"])
        start_epoch = int(payload.get("epoch") or 0)
        best_val = float(payload.get("best_val_mae_px", payload.get("val_mae_px", math.inf)))
        if (seed_dir / "tables" / "history.csv").exists():
            import csv as _csv
            with (seed_dir / "tables" / "history.csv").open(encoding="utf-8", newline="") as handle:
                history = list(_csv.DictReader(handle))
        init_path = seed_dir / "config" / "init_info.json"
        init_info = json.loads(init_path.read_text(encoding="utf-8")) if init_path.exists() else {"resumed": True}
        print(f"[phase17] seed {seed}: resume epoch {start_epoch} best_val={best_val:.3f}")
    else:
        init_info = _save_init(model, spec, seed_dir, seed, reconstructed=False)
    optimizer = _build_optimizer(run, model)
    scheduler = _build_scheduler(run, optimizer)
    if resume:
        payload = torch.load(last_path, map_location=dev, weights_only=False)
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
        generator=torch.Generator().manual_seed(seed + start_epoch),
    )
    dump_json(
        seed_dir / "config" / "train_config.json",
        {**train_run_config(run), "seed": seed, "fingerprint": fingerprint(spec), "init": init_info, "resume": resume},
    )
    epochs_no_improve = 0
    stage_best = best_val if math.isfinite(best_val) else math.inf
    stopped_reason = "max_epochs"
    epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, run.epochs + 1):
            model.train()
            running = running_mse = running_l1 = 0.0
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
                    optimizer.step()
                running += float(loss.detach()) * len(images)
                running_mse += float(mse.detach()) * len(images)
                running_l1 += float(l1.detach()) * len(images)
                seen += len(images)
            train_loss = running / max(1, seen)
            val_pred = _predict(model, val_data["images"])
            val_errors = coordinate_errors(val_pred, val_data["P"])
            train_pred = _predict(model, train_data["images"])
            train_errors = coordinate_errors(train_pred, train_data["P"])
            lr_used = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            if _exploded(train_loss, val_errors["mae_px"]):
                raise TrainingExploded(f"epoch {epoch} val_mae={val_errors['mae_px']}")
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse": running_mse / max(1, seen),
                "train_l1": running_l1 / max(1, seen),
                "train_mae_px": train_errors["mae_px"],
                "val_mae_px": val_errors["mae_px"],
                "val_rmse_px": val_errors["rmse_px"],
                "lr_used": lr_used,
                "lr_next": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            _write_csv(history_path, history)
            _plot_history(history, seed_dir / "figures" / "train_mae.png", f"seed {seed} MAE")
            print(
                f"[phase17] seed {seed} epoch {epoch}/{run.epochs} "
                f"train={train_errors['mae_px']:.3f} val={val_errors['mae_px']:.3f} px"
            )
            payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch,
                "fingerprint": fingerprint(spec),
                "seed": seed,
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
    except TrainingExploded as exc:
        stopped_reason = f"exploded: {exc}"
        print(f"[phase17] seed {seed} TRAINING EXPLODED: {exc}")
        _write_csv(history_path, history)
    summary = {
        "seed": seed,
        "best_val_mae_px": best_val if math.isfinite(best_val) else None,
        "final_epoch": epoch,
        "stopped_reason": stopped_reason,
        "early_stop_mae_px": EARLY_STOP_MAE_PX,
        "patience": run.patience,
        "max_epochs": run.epochs,
        "lr": run.lr,
        "lr_min": run.lr_min,
        "schedule": run.schedule,
        "optimizer": run.optimizer,
        "l1_weight": run.l1_weight,
        "selection": "validation MAE only; test never used to pick epoch",
        "init": init_info,
    }
    dump_json(summary_path, summary)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device(), weights_only=False)
        eval_model = build_model().to(device())
        eval_model.load_state_dict(ckpt["model_state"])
        eval_model.eval()
        metrics = _evaluate_model(eval_model, spec, labels, g, masks)
        dump_json(
            seed_dir / "tables" / "metrics.json",
            {
                "seed": seed,
                "fingerprint": fingerprint(spec),
                "selected_by": "best_validation_mae",
                "test_gate_not_applied": True,
                **metrics,
            },
        )
    print(f"[phase17] seed {seed} stopped={stopped_reason} best_val={best_val}")
    return {"seed": seed, "reused": False, "init": init_info, "summary": summary}


def extract_trained(seed: int) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    seed_dir = _seed_dir(seed)
    z_path = seed_dir / "features" / "Z.npz"
    head_path = seed_dir / "features" / "head.npz"
    if z_path.exists() and head_path.exists():
        print(f"[phase17] seed {seed}: trained features exist")
        return {"seed": seed, "skipped": True}
    if seed in EXISTING_RUN:
        _copy_existing_seed(seed, seed_dir, spec)
        return {"seed": seed, "copied": True}
    best_path = seed_dir / "checkpoints" / "resnet18_best.pt"
    if not best_path.exists():
        last_path = seed_dir / "checkpoints" / "resnet18_last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"no checkpoint for seed {seed}")
        best_path = last_path
        print(f"[phase17] seed {seed}: no best ckpt, extracting last")
    payload = torch.load(best_path, map_location=device(), weights_only=False)
    if payload.get("fingerprint") != fingerprint(spec):
        raise RuntimeError("checkpoint fingerprint mismatch")
    model = build_model().to(device())
    model.load_state_dict(payload["model_state"])
    model.eval()
    z = _extract_gap(model, spec)
    weight = model.fc.weight.detach().float().cpu().numpy()
    bias = model.fc.bias.detach().float().cpu().numpy()
    np.savez_compressed(z_path, Z=z, fingerprint=np.asarray(fingerprint(spec)), seed=np.asarray(seed))
    np.savez_compressed(head_path, W=weight, b=bias, fingerprint=np.asarray(fingerprint(spec)), seed=np.asarray(seed))
    info = {
        "seed": seed,
        "from_checkpoint": best_path.name,
        "ckpt_epoch": int(payload.get("epoch") or -1),
        "ckpt_val_mae_px": payload.get("val_mae_px"),
        "test_gate_not_applied": True,
        "Z_sha256": _file_sha256(z_path),
        "head_sha256": _file_sha256(head_path),
    }
    dump_json(seed_dir / "features" / "extract_info.json", info)
    print(f"[phase17] seed {seed}: saved trained Z (no test gate)")
    return info


def _load_z_head(seed_dir: Path, spec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(seed_dir / "features" / "Z.npz", allow_pickle=False) as payload:
        if str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) != fingerprint(spec):
            raise RuntimeError("Z fingerprint mismatch")
        z = np.asarray(payload["Z"], dtype=np.float64)
    with np.load(seed_dir / "features" / "head.npz", allow_pickle=False) as payload:
        if str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) != fingerprint(spec):
            raise RuntimeError("head fingerprint mismatch")
        weight = np.asarray(payload["W"], dtype=np.float64)
        bias = np.asarray(payload["b"], dtype=np.float64)
    return z, weight, bias


def _t_swap(z_flat, mu, u_bt, weight, bias, labels, sid, test_mask) -> Dict[str, Any]:
    pred_donor_t, pred_receiver_t, pred_receiver_q = [], [], []
    n_pairs = 0
    test_idx = np.where(test_mask)[0]
    by_shape: Dict[int, List[int]] = {}
    for i in test_idx:
        by_shape.setdefault(int(sid[i]), []).append(int(i))
    for idxs in by_shape.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                zi = z_flat[i] - ((z_flat[i] - mu) @ u_bt) @ u_bt.T + ((z_flat[j] - mu) @ u_bt) @ u_bt.T
                zj = z_flat[j] - ((z_flat[j] - mu) @ u_bt) @ u_bt.T + ((z_flat[i] - mu) @ u_bt) @ u_bt.T
                pi = _predict_p(zi[None], weight, bias)[0]
                pj = _predict_p(zj[None], weight, bias)[0]
                ti = decompose_controls(pi[None])["t"][0]
                tj = decompose_controls(pj[None])["t"][0]
                qi = decompose_controls(pi[None])["q"].reshape(6)
                qj = decompose_controls(pj[None])["q"].reshape(6)
                pred_donor_t.append(float(np.mean(np.abs(ti - labels["t"][j]))) * COORD_SCALE)
                pred_donor_t.append(float(np.mean(np.abs(tj - labels["t"][i]))) * COORD_SCALE)
                pred_receiver_t.append(float(np.mean(np.abs(ti - labels["t"][i]))) * COORD_SCALE)
                pred_receiver_t.append(float(np.mean(np.abs(tj - labels["t"][j]))) * COORD_SCALE)
                pred_receiver_q.append(float(np.mean(np.abs(qi - labels["Q"][i]))) * COORD_SCALE)
                pred_receiver_q.append(float(np.mean(np.abs(qj - labels["Q"][j]))) * COORD_SCALE)
                n_pairs += 1
    return {
        "label": "latent off-manifold intervention, not image-level counterfactual",
        "n_pairs": n_pairs,
        "pred_t_vs_donor_mae_px": float(np.mean(pred_donor_t)) if pred_donor_t else None,
        "pred_t_vs_receiver_mae_px": float(np.mean(pred_receiver_t)) if pred_receiver_t else None,
        "pred_q_vs_receiver_mae_px": float(np.mean(pred_receiver_q)) if pred_receiver_q else None,
        "role": "secondary supportive; not used in replication verdict",
    }


def _save_null_draws(path: Path, kind: str, perms, bases, stats: Mapping[str, List[float]], extra=None) -> None:
    payload = {
        "kind": np.asarray(kind),
        "energy": np.asarray(stats["energy"], dtype=np.float64),
        "head_E_t": np.asarray(stats["head_E_t"], dtype=np.float64),
        "head_E_q": np.asarray(stats["head_E_q"], dtype=np.float64),
        "probe_E_t": np.asarray(stats["probe_E_t"], dtype=np.float64),
        "probe_E_g": np.asarray(stats["probe_E_g"], dtype=np.float64),
        "F_t": np.asarray(stats["F_t"], dtype=np.float64),
        "F_q": np.asarray(stats["F_q"], dtype=np.float64),
        "bases": np.stack(bases).astype(np.float32),
    }
    if perms is not None:
        payload["translation_id_perm"] = np.stack(perms).astype(np.int16)
    if extra:
        for key, value in extra.items():
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def analyze_trained(seed: int) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    seed_dir = _seed_dir(seed)
    z, weight, bias = _load_z_head(seed_dir, spec)
    split_grid = _load_split_codes(spec.name)
    labels_grid = _labels(spec.name)
    n_s, n_t, dim = z.shape
    z_flat = z.reshape(-1, dim)
    split_flat = split_grid.reshape(-1)
    sid = np.repeat(np.arange(n_s), n_t)
    tid = np.tile(np.arange(n_t), n_s)
    labels = {"P": labels_grid["P"].reshape(-1, 6), "Q": labels_grid["Q"].reshape(-1, 6), "t": labels_grid["t"].reshape(-1, 2)}
    g = decompose_controls(labels["P"])["geometry_comp"]
    masks = {name: split_flat == i for i, name in enumerate(SPLIT_NAMES)}
    train_mask = masks["train"]
    test_mask = masks["test"]
    mu = z_flat[train_mask].mean(0)
    train_centered = z_flat[train_mask] - mu
    t_train, g_train = labels["t"][train_mask], g[train_mask]
    t_mu, t_sig = t_train.mean(0), t_train.std(0) + 1e-12
    q_mu, q_sig = g_train.mean(0), g_train.std(0) + 1e-12
    _, u_bt = _fit_factor(train_centered, t_train, g_train, t_mu, t_sig, q_mu, q_sig)
    energy = _subspace_energy(train_centered, u_bt)
    total_energy = float(np.mean(np.sum(train_centered**2, axis=1)))
    w_t, w_q = _readouts(weight)
    angles_wt = _principal_angles_deg(u_bt, w_t.T)
    pred0 = _predict_p(z_flat, weight, bias)
    baseline = {name: _pack_head(pred0, labels, g, mask) for name, mask in masks.items()}
    t0 = _probe_fit(z_flat, labels["t"], train_mask)
    g0 = _probe_fit(z_flat, g, train_mask)
    baseline_probe = {f"{name}_E_t": _probe_mae(t0, labels["t"], mask) for name, mask in masks.items()}
    baseline_probe.update({f"{name}_E_g": _probe_mae(g0, g, mask) for name, mask in masks.items()})
    true_cut = _eval_cut(z_flat, u_bt, mu, weight, bias, labels, g, masks)
    const_t = {name: _mae_px(labels["t"][mask] - t_mu) for name, mask in masks.items()}
    print(
        f"[phase17] seed {seed} B_t energy={energy:.4f} "
        f"head_t={true_cut['head']['test']['E_t']:.3f} probe_t={true_cut['probe']['test_E_t']:.3f}"
    )

    perm_path = seed_dir / "nulls" / "identity_perm.npz"
    match_path = seed_dir / "nulls" / "energy_matched.npz"
    keys = ("energy", "head_E_t", "head_E_q", "probe_E_t", "probe_E_g", "F_t", "F_q")
    if perm_path.exists() and match_path.exists():
        print(f"[phase17] seed {seed}: loading saved null draws")
        with np.load(perm_path) as payload:
            perm_stats = {k: payload[k].astype(np.float64).tolist() for k in keys}
        with np.load(match_path) as payload:
            match_stats = {k: payload[k].astype(np.float64).tolist() for k in keys}
            mean_e = float(np.mean(payload["energy"]))
            rel = abs(mean_e - energy) / max(energy, 1e-12)
            window = int(np.asarray(payload["pc_window"]).reshape(-1)[0]) if "pc_window" in payload.files else -1
            match_info = {
                "pc_window": window,
                "n_collected": int(len(payload["energy"])),
                "mean_deleted_energy": mean_e,
                "rel_error": rel,
                "tol_requested": ENERGY_TOL_OK if rel > ENERGY_TOL_STRICT else ENERGY_TOL_STRICT,
                "matched_ok": bool(rel <= ENERGY_TOL_OK),
                "matched_strict": bool(rel <= ENERGY_TOL_STRICT),
                "loaded_from_disk": True,
            }
    else:
        rng = np.random.default_rng(ANALYSIS_SEED)
        unique_t = np.zeros((n_t, 2), dtype=np.float64)
        for t_id in range(n_t):
            unique_t[t_id] = labels["t"][tid == t_id][0]
        train_tid = tid[train_mask]
        perm_stats = {k: [] for k in keys}
        perm_ids, perm_bases = [], []
        print(f"[phase17] seed {seed} permutation N={N_NULL}")
        for i in range(N_NULL):
            if i % 100 == 0:
                print(f"[phase17] seed {seed} perm {i}/{N_NULL}")
            order = rng.permutation(n_t)
            fake_t = unique_t[order[train_tid]]
            _, basis_p = _fit_factor(train_centered, fake_t, g_train, t_mu, t_sig, q_mu, q_sig)
            ev = _eval_cut(z_flat, basis_p, mu, weight, bias, labels, g, masks)
            perm_ids.append(order)
            perm_bases.append(basis_p)
            perm_stats["energy"].append(_subspace_energy(train_centered, basis_p))
            perm_stats["head_E_t"].append(ev["head"]["test"]["E_t"])
            perm_stats["head_E_q"].append(ev["head"]["test"]["E_q"])
            perm_stats["probe_E_t"].append(ev["probe"]["test_E_t"])
            perm_stats["probe_E_g"].append(ev["probe"]["test_E_g"])
            perm_stats["F_t"].append(_functional_alignment(z_flat[test_mask] - mu, w_t, basis_p))
            perm_stats["F_q"].append(_functional_alignment(z_flat[test_mask] - mu, w_q, basis_p))
        _save_null_draws(perm_path, "identity_perm", perm_ids, perm_bases, perm_stats)

        pcs, pc_energy = _pc_spectrum(train_centered)
        print(f"[phase17] seed {seed} energy-matched rank-2")
        matched, match_e, match_info = _sample_matched(
            train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, ENERGY_TOL_STRICT
        )
        if not match_info["matched_strict"]:
            print(f"[phase17] 2% missed rel={match_info['rel_error']:.4f}, retry 5%")
            matched, match_e, match_info = _sample_matched(
                train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, ENERGY_TOL_OK
            )
        match_stats = {k: [] for k in keys}
        match_rel = []
        for i, basis_m in enumerate(matched):
            if i % 100 == 0:
                print(f"[phase17] seed {seed} matched {i}/{len(matched)}")
            ev = _eval_cut(z_flat, basis_m, mu, weight, bias, labels, g, masks)
            e = _subspace_energy(train_centered, basis_m)
            match_stats["energy"].append(e)
            match_rel.append(abs(e - energy) / max(energy, 1e-12))
            match_stats["head_E_t"].append(ev["head"]["test"]["E_t"])
            match_stats["head_E_q"].append(ev["head"]["test"]["E_q"])
            match_stats["probe_E_t"].append(ev["probe"]["test_E_t"])
            match_stats["probe_E_g"].append(ev["probe"]["test_E_g"])
            match_stats["F_t"].append(_functional_alignment(z_flat[test_mask] - mu, w_t, basis_m))
            match_stats["F_q"].append(_functional_alignment(z_flat[test_mask] - mu, w_q, basis_m))
        _save_null_draws(
            match_path,
            "energy_matched",
            None,
            matched,
            match_stats,
            extra={"rel_error": match_rel, "pc_window": [match_info["pc_window"]] * len(matched)},
        )

    F_t = _functional_alignment(z_flat[test_mask] - mu, w_t, u_bt)
    F_q = _functional_alignment(z_flat[test_mask] - mu, w_q, u_bt)
    print(f"[phase17] seed {seed} shape 8-fold")
    shuffled = np.random.default_rng(ANALYSIS_SEED).permutation(np.arange(n_s))
    folds = []
    for fold in range(N_FOLD):
        held = shuffled[fold * 8 : (fold + 1) * 8]
        keep = np.ones(n_s, dtype=bool)
        keep[held] = False
        fold_mask = train_mask & keep[sid]
        _, u_f = _fit_factor(z_flat[fold_mask] - mu, labels["t"][fold_mask], g[fold_mask], t_mu, t_sig, q_mu, q_sig)
        ev = _eval_cut(z_flat, u_f, mu, weight, bias, labels, g, masks)
        angles = _principal_angles_deg(u_bt, u_f)
        folds.append(
            {
                "fold": fold,
                "held_shapes": held.tolist(),
                "n_train_cells": int(fold_mask.sum()),
                "angles_deg": angles,
                "mean_angle_deg": float(np.mean(angles)),
                "max_angle_deg": float(np.max(angles)),
                "overlap_norm": _overlap_norm(u_bt, u_f),
                "energy": _subspace_energy(train_centered, u_f),
                "head": ev["head"],
                "probe": ev["probe"],
            }
        )

    z_no_wt = _erasure_z(z_flat, _qr_basis(w_t.T), mu, True)
    probe_t_nowt = _probe_fit(z_no_wt, labels["t"], train_mask)
    wt_redundancy = {
        "angles_Bt_Wt_deg": angles_wt,
        "overlap_Bt_Wt": _overlap_norm(u_bt, w_t.T),
        "erasure_Wt_head_test": _pack_head(_predict_p(z_no_wt, weight, bias), labels, g, test_mask),
        "erasure_Wt_probe_train_E_t": _probe_mae(probe_t_nowt, labels["t"], train_mask),
        "erasure_Wt_probe_val_E_t": _probe_mae(probe_t_nowt, labels["t"], masks["val"]),
        "erasure_Wt_probe_test_E_t": _probe_mae(probe_t_nowt, labels["t"], test_mask),
        "note": "Deleting W_t kills the current decoder; leftover z may still linearly decode t.",
    }
    swap = _t_swap(z_flat, mu, u_bt, weight, bias, labels, sid, test_mask)

    perm_block = {
        "n": N_NULL,
        "energy": _summarize_null(energy, perm_stats["energy"], True),
        "head_E_t": _summarize_null(true_cut["head"]["test"]["E_t"], perm_stats["head_E_t"], True),
        "head_E_q": _summarize_null(true_cut["head"]["test"]["E_q"], perm_stats["head_E_q"], True),
        "head_E_q_preserve": _summarize_null(true_cut["head"]["test"]["E_q"], perm_stats["head_E_q"], False),
        "probe_E_t": _summarize_null(true_cut["probe"]["test_E_t"], perm_stats["probe_E_t"], True),
        "probe_E_g": _summarize_null(true_cut["probe"]["test_E_g"], perm_stats["probe_E_g"], True),
        "probe_E_g_preserve": _summarize_null(true_cut["probe"]["test_E_g"], perm_stats["probe_E_g"], False),
        "F_t": _summarize_null(F_t, perm_stats["F_t"], True),
        "F_q": _summarize_null(F_q, perm_stats["F_q"], True),
        "histogram": {k: _safe_histogram(v) for k, v in perm_stats.items()},
    }
    match_block = {
        "n": len(match_stats["energy"]),
        "info": match_info,
        "energy": _summarize_null(energy, match_stats["energy"], True),
        "head_E_t": _summarize_null(true_cut["head"]["test"]["E_t"], match_stats["head_E_t"], True),
        "head_E_q": _summarize_null(true_cut["head"]["test"]["E_q"], match_stats["head_E_q"], True),
        "head_E_q_preserve": _summarize_null(true_cut["head"]["test"]["E_q"], match_stats["head_E_q"], False),
        "probe_E_t": _summarize_null(true_cut["probe"]["test_E_t"], match_stats["probe_E_t"], True),
        "probe_E_g": _summarize_null(true_cut["probe"]["test_E_g"], match_stats["probe_E_g"], True),
        "probe_E_g_preserve": _summarize_null(true_cut["probe"]["test_E_g"], match_stats["probe_E_g"], False),
        "F_t": _summarize_null(F_t, match_stats["F_t"], True),
        "F_q": _summarize_null(F_q, match_stats["F_q"], True),
        "histogram": {k: _safe_histogram(v) for k, v in match_stats.items()},
    }
    fold_angles = [fold["angles_deg"] for fold in folds]
    mean_ang = float(np.mean([np.mean(item) for item in fold_angles]))
    max_ang = float(max(max(item) for item in fold_angles))
    r_t = true_cut["probe"]["test_E_t"] / max(const_t["test"], 1e-12)
    r_q = true_cut["probe"]["test_E_g"] / max(baseline_probe["test_E_g"], 1e-12)
    endpoints = {
        "P1_R_t": r_t,
        "P2_R_q": r_q,
        "P3_perm_probe_t_p": perm_block["probe_E_t"]["p"],
        "P3_perm_probe_t_percentile": perm_block["probe_E_t"]["percentile"],
        "P3_matched_probe_t_p": match_block["probe_E_t"]["p"],
        "P3_matched_probe_t_percentile": match_block["probe_E_t"]["percentile"],
        "P3_perm_probe_g_preserve_p": perm_block["probe_E_g_preserve"]["p"],
        "P3_matched_probe_g_preserve_p": match_block["probe_E_g_preserve"]["p"],
        "P4_fold_mean_angle_deg": mean_ang,
        "P4_fold_max_angle_deg": max_ang,
        "P4_fold_mean_overlap": float(np.mean([fold["overlap_norm"] for fold in folds])),
    }
    pattern = {
        "t_collapse": bool(r_t >= R_T_MIN),
        "q_preserve": bool(r_q <= R_Q_MAX),
        "null_extreme": bool(
            perm_block["probe_E_t"]["p"] <= P_EXTREME and match_block["probe_E_t"]["p"] <= P_EXTREME
        ),
        "fold_stable": bool(mean_ang <= FOLD_MEAN_MAX and max_ang <= FOLD_MAX_MAX),
        "reverse_t_still_readable": bool(r_t < REVERSE_T_MAX),
        "reverse_both_destroyed": bool(r_t >= R_T_MIN and r_q > REVERSE_Q_MIN),
        "null_indistinguishable": bool(
            perm_block["probe_E_t"]["p"] > 0.10 and match_block["probe_E_t"]["p"] > 0.10
        ),
    }
    pattern["matches_protocol"] = bool(
        pattern["t_collapse"]
        and pattern["q_preserve"]
        and pattern["null_extreme"]
        and pattern["fold_stable"]
        and not pattern["reverse_t_still_readable"]
        and not pattern["reverse_both_destroyed"]
        and not pattern["null_indistinguishable"]
    )
    payload = {
        "seed": seed,
        "analysis_rng_seed": ANALYSIS_SEED,
        "definition": {
            "rank": int(u_bt.shape[1]),
            "deleted_energy": energy,
            "deleted_energy_fraction": energy / max(total_energy, 1e-12),
            "total_train_centered_energy": total_energy,
            "angles_Bt_Wt_deg": angles_wt,
            "frobenius_Wt_on_Bt": _frobenius_alignment(w_t, u_bt),
            "frobenius_Wq_on_Bt": _frobenius_alignment(w_q, u_bt),
        },
        "baseline": baseline,
        "baseline_probe": baseline_probe,
        "constant_baseline": {"train_mean_t": t_mu.tolist(), "all": const_t, "test_E_t": const_t["test"]},
        "true_cut": {
            "energy": energy,
            "head": true_cut["head"],
            "probe": true_cut["probe"],
            "F_t_from_Bt": F_t,
            "F_q_from_Bt": F_q,
            "angles_Bt_Wt_deg": angles_wt,
        },
        "nulls": {"permutation": perm_block, "energy_matched": match_block},
        "shape_folds": folds,
        "wt_redundancy": wt_redundancy,
        "t_swap": swap,
        "primary_endpoints": endpoints,
        "pattern": pattern,
    }
    dump_json(seed_dir / "tables" / "bt_summary.json", _to_py(payload))
    _write_cells(seed_dir / "tables" / "cells.csv", sid, tid, split_flat, labels, g, pred0, true_cut["pred"], energy)
    fig_dir = seed_dir / "figures"
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(perm_stats["probe_E_t"], bins=40, alpha=0.85)
    ax.axvline(true_cut["probe"]["test_E_t"], color="C3", ls="--", label="真实 B_t")
    ax.set_title(f"seed {seed} identity-perm：translation re-probe")
    ax.set_xlabel("px")
    ax.set_ylabel("次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "perm_probe_t.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(match_stats["probe_E_t"], bins=40, alpha=0.85)
    ax.axvline(true_cut["probe"]["test_E_t"], color="C3", ls="--", label="真实 B_t")
    ax.set_title(f"seed {seed} 能量匹配：translation re-probe")
    ax.set_xlabel("px")
    ax.set_ylabel("次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "matched_probe_t.png", dpi=140)
    plt.close(fig)
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, [f["angles_deg"][0] for f in folds], marker="o", label="主角度 1")
    ax.plot(x, [f["angles_deg"][1] for f in folds], marker="s", label="主角度 2")
    ax.set_xlabel("shape fold")
    ax.set_ylabel("度")
    ax.set_title(f"seed {seed} estimator stability（非 unseen-shape）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fold_angles.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(seed_dir / "features" / "B_t.npz", U=u_bt.astype(np.float32), mu=mu.astype(np.float32))
    dump_json(seed_dir / "tables" / "fold_table.json", _to_py(folds))
    dump_json(seed_dir / "tables" / "wt_redundancy.json", _to_py(wt_redundancy))
    dump_json(seed_dir / "tables" / "t_swap.json", _to_py(swap))
    dump_json(seed_dir / "tables" / "primary_endpoints.json", _to_py(endpoints))
    print(f"[phase17] seed {seed} pattern={pattern['matches_protocol']} R_t={r_t:.3f} R_q={r_q:.3f}")
    return payload


def analyze_init(seed: int, trained: Mapping[str, Any]) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    seed_dir = _seed_dir(seed)
    path = seed_dir / "features" / "Z_init.npz"
    if not path.exists():
        return {"seed": seed, "missing_Z_init": True}
    with np.load(path, allow_pickle=False) as payload:
        z = np.asarray(payload["Z"], dtype=np.float64)
    split_grid = _load_split_codes(spec.name)
    labels_grid = _labels(spec.name)
    n_s, n_t, dim = z.shape
    z_flat = z.reshape(-1, dim)
    split_flat = split_grid.reshape(-1)
    tid = np.tile(np.arange(n_t), n_s)
    labels = {"P": labels_grid["P"].reshape(-1, 6), "Q": labels_grid["Q"].reshape(-1, 6), "t": labels_grid["t"].reshape(-1, 2)}
    g = decompose_controls(labels["P"])["geometry_comp"]
    masks = {name: split_flat == i for i, name in enumerate(SPLIT_NAMES)}
    train_mask = masks["train"]
    test_mask = masks["test"]
    mu = z_flat[train_mask].mean(0)
    train_centered = z_flat[train_mask] - mu
    t_train, g_train = labels["t"][train_mask], g[train_mask]
    t_mu, t_sig = t_train.mean(0), t_train.std(0) + 1e-12
    q_mu, q_sig = g_train.mean(0), g_train.std(0) + 1e-12
    t0 = _probe_fit(z_flat, labels["t"], train_mask)
    g0 = _probe_fit(z_flat, g, train_mask)
    _, u_init = _fit_factor(train_centered, t_train, g_train, t_mu, t_sig, q_mu, q_sig)
    energy = _subspace_energy(train_centered, u_init)
    z_e = _erasure_z(z_flat, u_init, mu, True)
    t_e = _probe_fit(z_e, labels["t"], train_mask)
    g_e = _probe_fit(z_e, g, train_mask)
    u_trained = None
    angles = None
    bt_path = seed_dir / "features" / "B_t.npz"
    if bt_path.exists():
        with np.load(bt_path) as payload:
            u_trained = np.asarray(payload["U"], dtype=np.float64)
        angles = _principal_angles_deg(u_init, u_trained)
    info = {
        "seed": seed,
        "role": "secondary / exploratory; not used in Phase1.7 replication verdict",
        "init_probe_test_E_t": _probe_mae(t0, labels["t"], test_mask),
        "init_probe_test_E_g": _probe_mae(g0, g, test_mask),
        "trained_probe_test_E_t": trained.get("baseline_probe", {}).get("test_E_t"),
        "init_Bt_energy": energy,
        "init_Bt_erasure_probe_test_E_t": _probe_mae(t_e, labels["t"], test_mask),
        "init_Bt_erasure_probe_test_E_g": _probe_mae(g_e, g, test_mask),
        "init_vs_trained_Bt_angles_deg_exploratory": angles,
        "note": "Do not compare raw B_t angles across seeds; feature bases are unaligned.",
    }
    dump_json(seed_dir / "tables" / "init_vs_trained.json", _to_py(info))
    print(
        f"[phase17] seed {seed} init probe t={info['init_probe_test_E_t']:.3f} "
        f"trained probe t={info['trained_probe_test_E_t']}"
    )
    return info


def _agg(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(v) for v in array],
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "sd_n3": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(array.size),
    }


def _seed_pattern_verdict(trained: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    best_val = summary.get("best_val_mae_px")
    learned = best_val is not None and math.isfinite(float(best_val)) and float(best_val) <= TRAINING_LEARNED_VAL_MAE
    if not learned:
        return "training_failure / uninterpretable mechanism run"
    if trained["pattern"]["matches_protocol"]:
        return "pattern"
    if trained["pattern"]["reverse_t_still_readable"] or trained["pattern"]["reverse_both_destroyed"] or trained["pattern"]["null_indistinguishable"]:
        return "opposite"
    return "weak"


def summarize_cross(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out = _cross_dir()
    statuses = [row["status"] for row in rows]
    n_pattern = sum(s == "pattern" for s in statuses)
    n_fail = sum(s.startswith("training_failure") for s in statuses)
    n_opp = sum(s == "opposite" for s in statuses)
    n_weak = sum(s == "weak" for s in statuses)
    if n_opp > 0 or n_pattern <= 1:
        letter = "Not replicated"
        if n_opp > 0:
            note = "存在功能方向相反的 seed，或真 B_t 与 null 不可分辨。"
        elif n_fail and n_pattern == 0:
            note = "没有可解释的 seed 重复该模式。"
        else:
            note = "只有一个或零个 seed 出现该模式。"
    elif n_pattern == 3 and n_fail == 0:
        letter = "Replicated"
        note = "三个独立训练 seed 均重复 translation re-probe 抽干、几何保住、null 极端、fold 稳定。"
    else:
        letter = "Mixed"
        note = "三个 seed 中只有两个表现出完整模式，或第三个较弱/训练失败但未出现相反功能。"
    master = []
    for row in rows:
        ep = row["trained"]["primary_endpoints"]
        base = row["trained"]["baseline"]["test"]
        probe0 = row["trained"]["baseline_probe"]
        cut = row["trained"]["true_cut"]
        init = row.get("init") or {}
        master.append(
            {
                "seed": row["seed"],
                "status": row["status"],
                "best_val_mae_px": row["summary"].get("best_val_mae_px"),
                "test_P_mae_px": base["E_P"],
                "test_t_mae_px": base["E_t"],
                "test_q_mae_px": base["E_q"],
                "Bt_energy": row["trained"]["definition"]["deleted_energy"],
                "Bt_energy_frac": row["trained"]["definition"]["deleted_energy_fraction"],
                "R_t": ep["P1_R_t"],
                "R_q": ep["P2_R_q"],
                "erased_probe_t": cut["probe"]["test_E_t"],
                "constant_t": row["trained"]["constant_baseline"]["test_E_t"],
                "erased_probe_g": cut["probe"]["test_E_g"],
                "baseline_probe_t": probe0["test_E_t"],
                "baseline_probe_g": probe0["test_E_g"],
                "perm_p_probe_t": ep["P3_perm_probe_t_p"],
                "matched_p_probe_t": ep["P3_matched_probe_t_p"],
                "fold_mean_deg": ep["P4_fold_mean_angle_deg"],
                "fold_max_deg": ep["P4_fold_max_angle_deg"],
                "fold_overlap": ep["P4_fold_mean_overlap"],
                "Wt_reprobe_t": row["trained"]["wt_redundancy"]["erasure_Wt_probe_test_E_t"],
                "init_probe_t": init.get("init_probe_test_E_t"),
                "t_swap_donor": row["trained"]["t_swap"]["pred_t_vs_donor_mae_px"],
                "t_swap_receiver": row["trained"]["t_swap"]["pred_t_vs_receiver_mae_px"],
                "t_swap_q": row["trained"]["t_swap"]["pred_q_vs_receiver_mae_px"],
            }
        )
    keys = ("R_t", "R_q", "erased_probe_t", "fold_mean_deg", "test_P_mae_px", "init_probe_t", "baseline_probe_t")
    aggregates = {key: _agg([row[key] for row in master if row[key] is not None]) for key in keys}
    dump_json(out / "tables" / "master.json", _to_py({"rows": master, "aggregates": aggregates, "verdict": letter}))
    _write_csv(out / "tables" / "master.csv", master)
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(len(master))
    ax.bar(xs - 0.15, [row["R_t"] for row in master], width=0.3, label="R_t 平移抽干")
    ax.bar(xs + 0.15, [row["R_q"] for row in master], width=0.3, label="R_q 几何保住")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(row["seed"]) for row in master])
    ax.set_ylabel("比值")
    ax.set_title("Phase1.7 主终点（n=3，不是总体推断）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "primary_R.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(xs - 0.15, [row["init_probe_t"] or 0 for row in master], width=0.3, label="初始 GAP 线性 t")
    ax.bar(xs + 0.15, [row["baseline_probe_t"] for row in master], width=0.3, label="训后 GAP 线性 t")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(row["seed"]) for row in master])
    ax.set_ylabel("test MAE (px)")
    ax.set_title("初始 vs 训后 translation linear probe（次要）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "init_vs_trained_probe_t.png", dpi=140)
    plt.close(fig)
    payload = {
        "verdict": letter,
        "note": note,
        "n_pattern": n_pattern,
        "n_training_failure": n_fail,
        "n_opposite": n_opp,
        "n_weak": n_weak,
        "master": master,
        "aggregates": aggregates,
        "statuses": statuses,
    }
    dump_json(out / "summary.json", _to_py(payload))
    return payload


def write_handoff(cross: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    master = cross["master"]
    lines = [
        "# Phase 1.7 多 seed 确认（交接）",
        "",
        f"**Verdict: {cross['verdict']}**",
        "",
        cross["note"],
        "",
        "本轮不寻找新 cut，不改模型，不改 `B_t` 定义。三个训练 seed：`20260810`（复用冻结 `adamw_l1_scratch`）、`20260811`、`20260812`。数据 fingerprint `195f37870e2cdb8685a7`。分析 RNG 固定为 `20260810`。",
        "",
        "不要把下面的 mean±SD 当成强统计推断（n=3）。1000 次 null 的 p 只描述模型内极端性，不能替代跨训练 seed 不确定性。",
        "",
        "## 1. 三个 seed 是否都完成主任务？",
        "",
    ]
    for row in master:
        lines.append(
            f"- seed {row['seed']}: status `{row['status']}`；best val {row['best_val_mae_px']}；"
            f"test P/t/q = {row['test_P_mae_px']:.3f} / {row['test_t_mae_px']:.3f} / {row['test_q_mae_px']:.3f} px"
        )
    lines += [
        "",
        "## 2–7. `B_t` 选择性线性擦除是否跨 seed 重复？",
        "",
        "| seed | R_t | R_q | erased probe t | const t | perm p | matched p | fold mean/max ° | W_t 后再探 t |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in master:
        lines.append(
            f"| {row['seed']} | {row['R_t']:.3f} | {row['R_q']:.3f} | {row['erased_probe_t']:.3f} | "
            f"{row['constant_t']:.3f} | {row['perm_p_probe_t']:.4g} | {row['matched_p_probe_t']:.4g} | "
            f"{row['fold_mean_deg']:.2f}/{row['fold_max_deg']:.2f} | {row['Wt_reprobe_t']:.3f} |"
        )
    agg = cross["aggregates"]
    lines += [
        "",
        f"R_t mean/median/range = {agg['R_t']['mean']:.3f} / {agg['R_t']['median']:.3f} / {agg['R_t']['range']:.3f}（n=3）",
        f"R_q mean/median/range = {agg['R_q']['mean']:.3f} / {agg['R_q']['median']:.3f} / {agg['R_q']['range']:.3f}",
        "",
        "## 8–9. 随机初始 GAP 的位置可读性",
        "",
        "| seed | 初始 probe t | 训后 probe t | 改善（px） |",
        "|---|---:|---:|---:|",
    ]
    for row in master:
        init_t = row["init_probe_t"]
        trained_t = row["baseline_probe_t"]
        delta = None if init_t is None else init_t - trained_t
        lines.append(
            f"| {row['seed']} | {init_t if init_t is None else f'{init_t:.3f}'} | {trained_t:.3f} | "
            f"{'' if delta is None else f'{delta:.3f}'} |"
        )
    lines += [
        "",
        "初始 vs 训后 `B_t` 主角度只作 seed 内 exploratory，见各 `tables/init_vs_trained.json`。禁止跨 seed 比 raw `B_t` 角度。",
        "",
        "## 10. Verdict",
        "",
        f"**{cross['verdict']}** — {cross['note']}",
        "",
        "## 11. 是否值得进入 unseen-translation / unseen-shape？",
        "",
    ]
    if cross["verdict"] == "Replicated":
        lines.append("机制在独立训练 seed 上重复，值得进入规划阶段讨论 unseen-t / unseen-shape。本轮不自动开那些实验。")
    elif cross["verdict"] == "Mixed":
        lines.append("尚未稳定到可以默认开 unseen split。先由规划决定是否加到 5 个 seed，而不是改 cut 或改网络。")
    else:
        lines.append("不要进入 unseen-t / unseen-shape。先解释为什么该现象不能跨训练重复。")
    lines += [
        "",
        "产物：`results/phase1_gap_rep/phase1_7_multiseed/`。未 commit / push。停止，等待规划。",
        "",
    ]
    text = "\n".join(lines)
    (_cross_dir() / "HANDOFF.md").write_text(text, encoding="utf-8")
    (_root() / "HANDOFF.md").write_text(text, encoding="utf-8")
    print(text)


def run_seed(seed: int) -> Dict[str, Any]:
    print(f"[phase17] ===== seed {seed} =====")
    train_info = train_seed(seed)
    extract_trained(seed)
    summary_bt = _seed_dir(seed) / "tables" / "bt_summary.json"
    if summary_bt.exists():
        print(f"[phase17] seed {seed}: analysis exists, skip recompute")
        trained = json.loads(summary_bt.read_text(encoding="utf-8"))
    else:
        trained = analyze_trained(seed)
    init_path = _seed_dir(seed) / "tables" / "init_vs_trained.json"
    if init_path.exists() and summary_bt.exists():
        init = json.loads(init_path.read_text(encoding="utf-8"))
    else:
        init = analyze_init(seed, trained)
    summary = train_info.get("summary") or json.loads((_seed_dir(seed) / "tables" / "train_summary.json").read_text(encoding="utf-8"))
    status = _seed_pattern_verdict(trained, summary)
    dump_json(
        _seed_dir(seed) / "tables" / "seed_status.json",
        {"seed": seed, "status": status, "pattern": trained["pattern"], "best_val_mae_px": summary.get("best_val_mae_px")},
    )
    return {"seed": seed, "status": status, "summary": summary, "trained": trained, "init": init, "train_info": train_info}


def run_phase17() -> Dict[str, Any]:
    freeze_protocol(force=False)
    rows = [run_seed(seed) for seed in SEEDS]
    cross = summarize_cross(rows)
    write_handoff(cross, rows)
    print(f"[phase17] verdict={cross['verdict']} wrote {_root()}")
    return {"verdict": cross["verdict"], "statuses": [row["status"] for row in rows]}


if __name__ == "__main__":
    run_phase17()
