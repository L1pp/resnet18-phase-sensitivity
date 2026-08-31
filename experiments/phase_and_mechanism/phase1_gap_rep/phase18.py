"""Phase 1.8 unseen 2D translation-identity interpolation.

Retrains three ResNet18+GAP seeds on a frozen translation holdout of the
same 4096 factorial images. Does not fine-tune Phase 1.7 weights.
Does not write into factorial/ wreckage.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audit import SPLIT_NAMES, _pc_spectrum, _subspace_energy, _to_py
from .bt_robustness import N_NULL, _fit_factor, _pack_head
from .common import (
    BATCH_SIZE,
    COORD_SCALE,
    EARLY_STOP_MAE_PX,
    IMAGE_SIZE,
    RESULTS_ROOT,
    TrainRun,
    build_model,
    decompose_controls,
    device,
    dump_json,
    environment_versions,
    fingerprint,
    gap_features,
    images_to_tensor,
    profile_spec,
    seed_everything,
    setup_matplotlib_chinese,
    train_run_config,
)
from .cross_probe import (
    axis_mae_px,
    constant_t_mae,
    cross_probe_block,
    energy_flags,
    eval_cross_cut,
    per_axis_report,
    probe_predict,
    ratios,
    sample_energy_matched_strict,
    save_null_draws,
    summarize_null,
    tx_ty_from_tid,
)
from .generate_data import load_split
from .phase17 import PROTOCOL_FP, SEEDS, _file_sha256, _state_sha256
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

OUT_NAME = "phase1_8_unseen_t"
ANALYSIS_SEED = 20260810
N_TX = 8
N_TY = 8
VAL_TX_TY: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (1, 4),
    (2, 6),
    (3, 2),
    (4, 5),
    (5, 3),
    (6, 1),
    (6, 6),
)
TEST_TX_TY: Tuple[Tuple[int, int], ...] = (
    (1, 3),
    (2, 1),
    (2, 5),
    (3, 6),
    (4, 2),
    (5, 4),
    (6, 3),
    (5, 1),
)
R_T_OOD_MIN = 0.80
R_Q_OOD_MAX = 1.10


def _root() -> Path:
    path = RESULTS_ROOT / OUT_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _split_dir() -> Path:
    path = _root() / "split"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def tid_from_xy(tx: int, ty: int, n_ty: int = N_TY) -> int:
    return int(tx) * int(n_ty) + int(ty)


def xy_from_tid(tid: int, n_ty: int = N_TY) -> Tuple[int, int]:
    return int(tid) // int(n_ty), int(tid) % int(n_ty)


def _protocol_run(seed: int) -> TrainRun:
    return TrainRun(
        name=f"phase18_seed_{seed}",
        optimizer="adamw",
        lr=1e-3,
        lr_min=1e-5,
        epochs=100,
        patience=50,
        schedule="cosine",
        l1_weight=0.25,
        data_profile="factorial",
    )


def load_factorial_grid(spec) -> Dict[str, np.ndarray]:
    n_s, n_t = spec.n_shapes, spec.n_translations
    images = np.zeros((n_s, n_t, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    p = np.zeros((n_s, n_t, 6), dtype=np.float32)
    q = np.zeros((n_s, n_t, 6), dtype=np.float32)
    t = np.zeros((n_s, n_t, 2), dtype=np.float32)
    filled = np.zeros((n_s, n_t), dtype=bool)
    for split in SPLIT_NAMES:
        data = load_split(spec, split)
        sid = data["shape_id"].astype(np.int64)
        tid = data["translation_id"].astype(np.int64)
        images[sid, tid] = data["images"]
        p[sid, tid] = data["P"]
        q[sid, tid] = data["Q"]
        t[sid, tid] = data["t"]
        filled[sid, tid] = True
    if not bool(np.all(filled)):
        raise RuntimeError("incomplete factorial grid; refuse to re-render")
    fp = fingerprint(spec)
    if fp != PROTOCOL_FP:
        raise RuntimeError(f"geometry fingerprint {fp} != {PROTOCOL_FP}")
    return {"images": images, "P": p, "Q": q, "t": t}


def build_translation_split(spec) -> Dict[str, Any]:
    n_s, n_t = spec.n_shapes, spec.n_translations
    if spec.n_tx != N_TX or spec.n_ty != N_TY:
        raise RuntimeError(f"expected 8x8 translation grid, got {spec.n_tx}x{spec.n_ty}")
    val_tids = sorted(tid_from_xy(tx, ty) for tx, ty in VAL_TX_TY)
    test_tids = sorted(tid_from_xy(tx, ty) for tx, ty in TEST_TX_TY)
    if set(val_tids) & set(test_tids):
        raise RuntimeError("val/test translation identities overlap")
    for tx, ty in (*VAL_TX_TY, *TEST_TX_TY):
        if not (1 <= tx <= 6 and 1 <= ty <= 6):
            raise RuntimeError(f"holdout ({tx},{ty}) is not an interior interpolation cell")
    split_codes = np.zeros((n_s, n_t), dtype=np.int8)
    for tid in val_tids:
        split_codes[:, tid] = 1
    for tid in test_tids:
        split_codes[:, tid] = 2
    train_tids = [i for i in range(n_t) if i not in val_tids and i not in test_tids]
    counts = {
        "train": int((split_codes == 0).sum()),
        "val": int((split_codes == 1).sum()),
        "test": int((split_codes == 2).sum()),
    }
    expected = {"train": 48 * n_s, "val": 8 * n_s, "test": 8 * n_s}
    if counts != expected:
        raise RuntimeError(f"split counts {counts} != {expected}")
    payload = {
        "kind": "unseen_2d_translation_identity_interpolation",
        "not": ["unseen_tx_scalar", "unseen_ty_scalar", "extrapolation"],
        "val_tx_ty": [list(item) for item in VAL_TX_TY],
        "test_tx_ty": [list(item) for item in TEST_TX_TY],
        "val_translation_ids": val_tids,
        "test_translation_ids": test_tids,
        "train_translation_ids": train_tids,
        "split_codes": split_codes.astype(int).tolist(),
        "geometry_fingerprint": fingerprint(spec),
        "counts": counts,
    }
    digest = hashlib.sha256(
        json.dumps(
            {k: payload[k] for k in ("kind", "val_tx_ty", "test_tx_ty", "split_codes", "geometry_fingerprint")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    payload["split_hash"] = digest[:20]
    payload["split_sha256"] = digest
    return payload


def freeze_split(spec, force: bool = False) -> Dict[str, Any]:
    dest = _split_dir() / "split_manifest.json"
    npz_path = _split_dir() / "split.npz"
    if dest.exists() and npz_path.exists() and not force:
        payload = json.loads(dest.read_text(encoding="utf-8"))
        print(f"[phase18] split already frozen hash={payload['split_hash']}")
        return payload
    payload = build_translation_split(spec)
    split_codes = np.asarray(payload["split_codes"], dtype=np.int8)
    np.savez_compressed(
        npz_path,
        split_codes=split_codes,
        val_translation_ids=np.asarray(payload["val_translation_ids"], dtype=np.int32),
        test_translation_ids=np.asarray(payload["test_translation_ids"], dtype=np.int32),
        train_translation_ids=np.asarray(payload["train_translation_ids"], dtype=np.int32),
        geometry_fingerprint=np.asarray(payload["geometry_fingerprint"]),
        split_hash=np.asarray(payload["split_hash"]),
    )
    dump_json(dest, payload)
    print(f"[phase18] froze split hash={payload['split_hash']} counts={payload['counts']}")
    return payload


def load_split_codes() -> Tuple[np.ndarray, Dict[str, Any]]:
    manifest = json.loads((_split_dir() / "split_manifest.json").read_text(encoding="utf-8"))
    with np.load(_split_dir() / "split.npz", allow_pickle=False) as payload:
        codes = np.asarray(payload["split_codes"], dtype=np.int8)
        stored = str(np.asarray(payload["split_hash"]).reshape(-1)[0])
    if stored != manifest["split_hash"]:
        raise RuntimeError("split hash mismatch between npz and manifest")
    return codes, manifest


def gather(grid: Mapping[str, np.ndarray], split_codes: np.ndarray, code: int) -> Dict[str, np.ndarray]:
    sid, tid = np.where(split_codes == code)
    return {
        "images": grid["images"][sid, tid],
        "P": grid["P"][sid, tid],
        "Q": grid["Q"][sid, tid],
        "t": grid["t"][sid, tid],
        "shape_id": sid.astype(np.int32),
        "translation_id": tid.astype(np.int32),
    }


def extract_gap_grid(model, images: np.ndarray) -> np.ndarray:
    n_s, n_t = images.shape[:2]
    z = np.zeros((n_s, n_t, 512), dtype=np.float32)
    model.eval()
    dev = next(model.parameters()).device
    flat = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(flat), BATCH_SIZE):
            batch = images_to_tensor(flat[start : start + BATCH_SIZE]).to(dev)
            feats.append(gap_features(model, batch).float().cpu().numpy())
    z = np.concatenate(feats, axis=0).reshape(n_s, n_t, 512)
    return z


def export_slim_best(src: Path, dest: Path) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    slim = {key: value for key, value in payload.items() if key not in ("optimizer_state", "scheduler_state")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dest)


def _task_pack(pred: np.ndarray, labels: Mapping[str, np.ndarray], g: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    packed = _pack_head(pred, labels, g, mask)
    tx, ty = axis_mae_px(decompose_controls(pred)["t"], labels["t"], mask)
    packed["E_tx"] = tx
    packed["E_ty"] = ty
    return packed


def _save_init(model, grid, seed_dir: Path, seed: int, spec) -> Dict[str, Any]:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_path = seed_dir / "checkpoints" / "resnet18_init.pt"
    torch.save({"model_state": state, "seed": seed, "reconstructed": False}, init_path)
    print("[phase18] forward Z_init (before optimizer)")
    z_init = extract_gap_grid(model, grid["images"])
    np.savez_compressed(
        seed_dir / "features" / "Z_init.npz",
        Z=z_init,
        fingerprint=np.asarray(fingerprint(spec)),
        seed=np.asarray(seed),
    )
    info = {
        "seed": seed,
        "state_sha256": _state_sha256(state),
        "init_ckpt_sha256": _file_sha256(init_path),
        "Z_init_sha256": _file_sha256(seed_dir / "features" / "Z_init.npz"),
        "fingerprint": fingerprint(spec),
        "environment": environment_versions(),
    }
    dump_json(seed_dir / "config" / "init_info.json", info)
    return info


def train_seed(seed: int, grid, split_codes: np.ndarray, split_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    seed_dir = _seed_dir(seed)
    run = _protocol_run(seed)
    train_data = gather(grid, split_codes, 0)
    val_data = gather(grid, split_codes, 1)
    best_path = seed_dir / "checkpoints" / "resnet18_best.pt"
    last_path = seed_dir / "checkpoints" / "resnet18_last.pt"
    history_path = seed_dir / "tables" / "history.csv"
    summary_path = seed_dir / "tables" / "train_summary.json"
    if summary_path.exists() and best_path.exists():
        print(f"[phase18] seed {seed}: training already finished")
        return {"seed": seed, "skipped_train": True, "summary": json.loads(summary_path.read_text(encoding="utf-8"))}

    seed_everything(seed)
    dev = device()
    amp = bool(dev.type == "cuda")
    model = build_model().to(dev)
    resume = last_path.exists() and not summary_path.exists()
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    best_val = math.inf
    if resume:
        payload = torch.load(last_path, map_location=dev, weights_only=False)
        if payload.get("fingerprint") != fingerprint(spec):
            raise RuntimeError("checkpoint geometry fingerprint mismatch")
        if payload.get("split_hash") != split_manifest["split_hash"]:
            raise RuntimeError("checkpoint split_hash mismatch")
        model.load_state_dict(payload["model_state"])
        start_epoch = int(payload.get("epoch") or 0)
        best_val = float(payload.get("best_val_mae_px", payload.get("val_mae_px", math.inf)))
        if history_path.exists():
            import csv as _csv

            with history_path.open(encoding="utf-8", newline="") as handle:
                history = list(_csv.DictReader(handle))
        init_info = json.loads((seed_dir / "config" / "init_info.json").read_text(encoding="utf-8"))
        print(f"[phase18] seed {seed}: resume epoch {start_epoch} best_val={best_val:.3f}")
    else:
        init_info = _save_init(model, grid, seed_dir, seed, spec)
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
        {
            **train_run_config(run),
            "seed": seed,
            "fingerprint": fingerprint(spec),
            "split_hash": split_manifest["split_hash"],
            "init": init_info,
            "resume": resume,
            "holdout": "unseen_2d_translation_identity_interpolation",
        },
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
            _plot_history(history, seed_dir / "figures" / "train_mae.png", f"phase18 seed {seed} MAE")
            print(
                f"[phase18] seed {seed} epoch {epoch}/{run.epochs} "
                f"train={train_errors['mae_px']:.3f} val={val_errors['mae_px']:.3f} px"
            )
            payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch,
                "fingerprint": fingerprint(spec),
                "split_hash": split_manifest["split_hash"],
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
        print(f"[phase18] seed {seed} TRAINING EXPLODED: {exc}")
        _write_csv(history_path, history)
    summary = {
        "seed": seed,
        "best_val_mae_px": best_val if math.isfinite(best_val) else None,
        "final_epoch": epoch,
        "stopped_reason": stopped_reason,
        "early_stop_mae_px": EARLY_STOP_MAE_PX,
        "patience": run.patience,
        "max_epochs": run.epochs,
        "selection": "validation MAE only; test translation identities never used",
        "split_hash": split_manifest["split_hash"],
        "fingerprint": fingerprint(spec),
        "init": init_info,
    }
    dump_json(summary_path, summary)
    if best_path.exists():
        slim = seed_dir / "checkpoints" / "resnet18_best_slim.pt"
        export_slim_best(best_path, slim)
    print(f"[phase18] seed {seed} stopped={stopped_reason} best_val={best_val}")
    return {"seed": seed, "summary": summary, "init": init_info}


def _evaluate_splits(model, grid, split_codes, labels, g) -> Dict[str, Any]:
    pred_grid = np.zeros((grid["P"].shape[0], grid["P"].shape[1], 6), dtype=np.float64)
    coord_by_split: Dict[str, Any] = {}
    for code, name in enumerate(SPLIT_NAMES):
        data = gather(grid, split_codes, code)
        pred = _predict(model, data["images"])
        pred_grid[data["shape_id"], data["translation_id"]] = pred
        coord_by_split[name] = coordinate_errors(pred, data["P"])
    pred_flat = pred_grid.reshape(-1, 6)
    tid_flat = np.tile(np.arange(split_codes.shape[1]), split_codes.shape[0])
    split_metrics = {}
    for code, name in enumerate(SPLIT_NAMES):
        mask = split_codes.reshape(-1) == code
        packed = _task_pack(pred_flat, labels, g, mask)
        axis = per_axis_report(decompose_controls(pred_flat)["t"], labels["t"], tid_flat, mask)
        split_metrics[name] = {**coord_by_split[name], **packed, "axis": axis}
    return {"pred_grid": pred_grid, "metrics": split_metrics}


def extract_trained(seed: int, grid, spec) -> Dict[str, Any]:
    seed_dir = _seed_dir(seed)
    z_path = seed_dir / "features" / "Z.npz"
    head_path = seed_dir / "features" / "head.npz"
    if z_path.exists() and head_path.exists():
        print(f"[phase18] seed {seed}: trained features exist")
        return {"seed": seed, "skipped": True}
    best_path = seed_dir / "checkpoints" / "resnet18_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"missing {best_path}")
    payload = torch.load(best_path, map_location=device(), weights_only=False)
    if payload.get("fingerprint") != fingerprint(spec):
        raise RuntimeError("checkpoint fingerprint mismatch")
    model = build_model().to(device())
    model.load_state_dict(payload["model_state"])
    z = extract_gap_grid(model, grid["images"])
    weight = model.fc.weight.detach().float().cpu().numpy()
    bias = model.fc.bias.detach().float().cpu().numpy()
    np.savez_compressed(z_path, Z=z, fingerprint=np.asarray(fingerprint(spec)), seed=np.asarray(seed))
    np.savez_compressed(head_path, W=weight, b=bias, fingerprint=np.asarray(fingerprint(spec)), seed=np.asarray(seed))
    info = {
        "seed": seed,
        "from_checkpoint": best_path.name,
        "ckpt_epoch": int(payload.get("epoch") or -1),
        "Z_sha256": _file_sha256(z_path),
        "test_gate_not_applied": True,
    }
    dump_json(seed_dir / "features" / "extract_info.json", info)
    return info


def _load_z_head(seed_dir: Path, spec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(seed_dir / "features" / "Z.npz", allow_pickle=False) as payload:
        if str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) != fingerprint(spec):
            raise RuntimeError("Z fingerprint mismatch")
        z = np.asarray(payload["Z"], dtype=np.float64)
    with np.load(seed_dir / "features" / "head.npz", allow_pickle=False) as payload:
        weight = np.asarray(payload["W"], dtype=np.float64)
        bias = np.asarray(payload["b"], dtype=np.float64)
    return z, weight, bias


def analyze_init(seed: int, split_codes: np.ndarray, labels, g) -> Dict[str, Any]:
    seed_dir = _seed_dir(seed)
    path = seed_dir / "features" / "Z_init.npz"
    if not path.exists():
        return {"seed": seed, "missing_Z_init": True}
    with np.load(path, allow_pickle=False) as payload:
        z = np.asarray(payload["Z"], dtype=np.float64)
    z_flat = z.reshape(-1, z.shape[-1])
    split_flat = split_codes.reshape(-1)
    masks = {name: split_flat == i for i, name in enumerate(SPLIT_NAMES)}
    raw_val = cross_probe_block(z_flat, labels["t"], g, masks["train"], masks["val"])
    raw_test = cross_probe_block(z_flat, labels["t"], g, masks["train"], masks["test"])
    info = {
        "seed": seed,
        "role": "secondary; not in Phase1.8 verdict",
        "trainfit_val_unseen_t": raw_val,
        "trainfit_test_unseen_t": raw_test,
        "note": "random CNN coarse position readout on unseen translation identities",
    }
    dump_json(seed_dir / "tables" / "init_probe.json", _to_py(info))
    return info


def analyze_trained(seed: int, grid, split_codes: np.ndarray, split_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    seed_dir = _seed_dir(seed)
    z, weight, bias = _load_z_head(seed_dir, spec)
    n_s, n_t, dim = z.shape
    z_flat = z.reshape(-1, dim)
    split_flat = split_codes.reshape(-1)
    sid = np.repeat(np.arange(n_s), n_t)
    tid = np.tile(np.arange(n_t), n_s)
    labels = {
        "P": grid["P"].reshape(-1, 6).astype(np.float64),
        "Q": grid["Q"].reshape(-1, 6).astype(np.float64),
        "t": grid["t"].reshape(-1, 2).astype(np.float64),
    }
    g = decompose_controls(labels["P"])["geometry_comp"]
    masks = {name: split_flat == i for i, name in enumerate(SPLIT_NAMES)}
    train_mask, val_mask, test_mask = masks["train"], masks["val"], masks["test"]
    mu = z_flat[train_mask].mean(0)
    train_centered = z_flat[train_mask] - mu
    t_train, g_train = labels["t"][train_mask], g[train_mask]
    t_mu, t_sig = t_train.mean(0), t_train.std(0) + 1e-12
    q_mu, q_sig = g_train.mean(0), g_train.std(0) + 1e-12
    _, u_bt = _fit_factor(train_centered, t_train, g_train, t_mu, t_sig, q_mu, q_sig)
    energy = _subspace_energy(train_centered, u_bt)
    pcs, pc_energy = _pc_spectrum(train_centered)
    flags = energy_flags(energy, float(pc_energy[:2].sum()))

    model = build_model().to(device())
    ckpt = torch.load(seed_dir / "checkpoints" / "resnet18_best.pt", map_location=device(), weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ev_model = _evaluate_splits(model, grid, split_codes, labels, g)

    raw = cross_probe_block(z_flat, labels["t"], g, val_mask, test_mask)
    const_t = constant_t_mae(labels["t"], val_mask, test_mask)
    true_cut = eval_cross_cut(z_flat, u_bt, mu, labels["t"], g, val_mask, test_mask, train_centered)
    train_diag = eval_cross_cut(z_flat, u_bt, mu, labels["t"], g, train_mask, test_mask, train_centered)
    ratio = ratios(true_cut["probe"], raw, const_t)
    t_hat_raw = probe_predict(z_flat, labels["t"], val_mask)
    t_hat_erased = probe_predict(true_cut["z_e"], labels["t"], val_mask)
    raw_axis = per_axis_report(t_hat_raw, labels["t"], tid, test_mask)
    erased_axis = per_axis_report(t_hat_erased, labels["t"], tid, test_mask)
    print(
        f"[phase18] seed {seed} backbone_test_P={ev_model['metrics']['test']['mae_px']:.3f} "
        f"R_t_OOD={ratio['R_t']:.3f} R_q_OOD={ratio['R_q']:.3f}"
    )

    rng = np.random.default_rng(ANALYSIS_SEED)
    train_tids = np.asarray(split_manifest["train_translation_ids"], dtype=np.int64)
    unique_t = np.zeros((len(train_tids), 2), dtype=np.float64)
    for i, t_id in enumerate(train_tids):
        unique_t[i] = labels["t"][tid == t_id][0]
    tid_to_pos = {int(t_id): i for i, t_id in enumerate(train_tids)}
    train_pos = np.array([tid_to_pos[int(v)] for v in tid[train_mask]], dtype=np.int64)
    perm_bases, perm_orders, perm_stats = [], [], {"energy": [], "probe_E_t": [], "probe_E_g": []}
    print(f"[phase18] seed {seed} identity-perm N={N_NULL} on 48 train translations")
    for i in range(N_NULL):
        order = rng.permutation(len(train_tids))
        fake_t = unique_t[order[train_pos]]
        _, basis_p = _fit_factor(train_centered, fake_t, g_train, t_mu, t_sig, q_mu, q_sig)
        ev = eval_cross_cut(z_flat, basis_p, mu, labels["t"], g, val_mask, test_mask, train_centered)
        perm_bases.append(basis_p)
        perm_orders.append(order)
        perm_stats["energy"].append(ev["energy"])
        perm_stats["probe_E_t"].append(ev["probe"]["t_mae_px"])
        perm_stats["probe_E_g"].append(ev["probe"]["g_mae_px"])
        if i % 100 == 0:
            print(f"[phase18] seed {seed} perm {i}/{N_NULL}")
    save_null_draws(
        seed_dir / "nulls" / "identity_perm.npz",
        "identity_perm_train48",
        perm_bases,
        perm_stats,
        perms=perm_orders,
    )

    print(f"[phase18] seed {seed} energy-matched rank-2 (strict, no fallback)")
    matched, match_e, match_info = sample_energy_matched_strict(
        train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, 500
    )
    match_stats = {"energy": [], "probe_E_t": [], "probe_E_g": []}
    if match_info["matched_ok"] and matched:
        for i, basis_m in enumerate(matched):
            ev = eval_cross_cut(z_flat, basis_m, mu, labels["t"], g, val_mask, test_mask, train_centered)
            match_stats["energy"].append(ev["energy"])
            match_stats["probe_E_t"].append(ev["probe"]["t_mae_px"])
            match_stats["probe_E_g"].append(ev["probe"]["g_mae_px"])
            if i % 100 == 0:
                print(f"[phase18] seed {seed} matched {i}/{len(matched)}")
        save_null_draws(seed_dir / "nulls" / "energy_matched.npz", "energy_matched_strict", matched, match_stats)
        match_block = {
            "info": match_info,
            "probe_E_t": summarize_null(true_cut["probe"]["t_mae_px"], match_stats["probe_E_t"], True),
            "probe_E_g": summarize_null(true_cut["probe"]["g_mae_px"], match_stats["probe_E_g"], True),
        }
    else:
        match_info["uninformative"] = True
        match_info["matched_ok"] = False
        save_null_draws(
            seed_dir / "nulls" / "energy_matched.npz",
            "energy_matched_uninformative",
            matched,
            {"energy": match_e, "probe_E_t": [], "probe_E_g": []},
            extra={"uninformative": [1]},
        )
        match_block = {"info": match_info, "note": "uninformative / unavailable; do not use p-value"}

    perm_block = {
        "n": N_NULL,
        "probe_E_t": summarize_null(true_cut["probe"]["t_mae_px"], perm_stats["probe_E_t"], True),
        "probe_E_g": summarize_null(true_cut["probe"]["g_mae_px"], perm_stats["probe_E_g"], True),
        "probe_E_g_preserve": summarize_null(true_cut["probe"]["g_mae_px"], perm_stats["probe_E_g"], False),
        "energy": summarize_null(energy, perm_stats["energy"], True),
    }
    np.savez_compressed(seed_dir / "features" / "B_t.npz", U=u_bt.astype(np.float32), mu=mu.astype(np.float32))
    payload = {
        "seed": seed,
        "split_hash": split_manifest["split_hash"],
        "fingerprint": fingerprint(spec),
        "bt_energy": energy,
        "energy_flags": flags,
        "backbone": ev_model["metrics"],
        "raw_valfit_test": raw,
        "constant_valfit_test_t_mae": const_t,
        "erased_valfit_test": true_cut["probe"],
        "erased_trainfit_test_diagnostic": train_diag["probe"],
        "R_t_OOD": ratio["R_t"],
        "R_q_OOD": ratio["R_q"],
        "raw_axis_test": raw_axis,
        "erased_axis_test": erased_axis,
        "nulls": {"permutation": perm_block, "energy_matched": match_block},
    }
    dump_json(seed_dir / "tables" / "bt_summary.json", _to_py(payload))
    dump_json(seed_dir / "tables" / "metrics.json", _to_py(ev_model["metrics"]))
    _plot_seed(seed_dir, seed, perm_stats, true_cut["probe"]["t_mae_px"], const_t, match_stats if match_info.get("matched_ok") else None)
    return payload


def _plot_seed(seed_dir: Path, seed: int, perm_stats, true_t: float, const_t: float, match_stats) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(perm_stats["probe_E_t"], bins=40, alpha=0.85)
    ax.axvline(true_t, color="C3", ls="--", label="真实 B_t val→test")
    ax.axvline(const_t, color="0.3", ls=":", label="常数")
    ax.set_title(f"seed {seed} identity-perm OOD t")
    ax.set_xlabel("px")
    ax.set_ylabel("次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(seed_dir / "figures" / "perm_ood_t.png", dpi=140)
    plt.close(fig)
    if match_stats and match_stats["probe_E_t"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.hist(match_stats["probe_E_t"], bins=40, alpha=0.85)
        ax.axvline(true_t, color="C3", ls="--", label="真实 B_t")
        ax.set_title(f"seed {seed} energy-matched OOD t")
        ax.set_xlabel("px")
        ax.legend()
        fig.tight_layout()
        fig.savefig(seed_dir / "figures" / "matched_ood_t.png", dpi=140)
        plt.close(fig)


def decide(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    backbone_ok = []
    raw_ok = []
    rt_ok = []
    rq_ok = []
    for row in rows:
        test = row["backbone"]["test"]
        const = float(row["constant_valfit_test_t_mae"])
        backbone_ok.append(bool(test["mae_px"] < 0.5 * const and test["E_t"] < 0.5 * const))
        raw_ok.append(bool(row["raw_valfit_test"]["t_mae_px"] < 0.5 * const))
        rt_ok.append(bool(row["R_t_OOD"] >= R_T_OOD_MIN))
        rq_ok.append(bool(row["R_q_OOD"] <= R_Q_OOD_MAX))
    n_bt = sum(rt_ok)
    perm_special = []
    for row in rows:
        perm = row["nulls"]["permutation"]["probe_E_t"]
        perm_special.append(bool(perm.get("p") is not None and perm["p"] <= 0.05))
    if all(backbone_ok) and all(raw_ok) and n_bt >= 2 and all(rq_ok) and sum(perm_special) >= 2:
        letter = "Generalized_selective_organization"
        note = (
            "train-estimated translation-associated linear organization generalizes to unseen "
            "2D translation identities within the interpolation regime."
        )
    elif not any(backbone_ok):
        letter = "No_factor_generalization"
        note = "backbone itself does not reasonably predict unseen translations; learning is limited to seen translation support."
    elif all(backbone_ok) and n_bt <= 1:
        letter = "No_factor_generalization"
        note = (
            "backbone can predict unseen translations, but train-estimated B_t is not a unified "
            "linear position direction that generalizes."
        )
    else:
        letter = "Partial_Mixed"
        note = "backbone interpolates unseen translation, but B_t selective erasure or geometry preservation is unstable across seeds."
    return {
        "verdict": letter,
        "note": note,
        "backbone_ok": backbone_ok,
        "raw_ok": raw_ok,
        "rt_ok": rt_ok,
        "rq_ok": rq_ok,
        "perm_special": perm_special,
    }


def write_handoff(decision: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Phase 1.8 Unseen Translation Interpolation",
        "",
        f"**Verdict: {decision['verdict']}**",
        "",
        decision["note"],
        "",
        "这是 unseen **2D translation-identity interpolation**，不是 unseen tx/ty 标量，也不是 extrapolation。",
        f"几何 fingerprint `{PROTOCOL_FP}`。split hash `{split_manifest['split_hash']}`。",
        "三 seed 全部 `weights=None` 重训。正式 assay 是 val→test；train-reprobe 只是 diagnostic。",
        "",
        "| seed | test P | test t | test q | test tx | test ty | R_t^OOD | R_q^OOD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = row["backbone"]["test"]
        lines.append(
            f"| {row['seed']} | {test['mae_px']:.3f} | {test['E_t']:.3f} | {test['E_q']:.3f} | "
            f"{test['E_tx']:.3f} | {test['E_ty']:.3f} | {row['R_t_OOD']:.3f} | {row['R_q_OOD']:.3f} |"
        )
    lines += [
        "",
        "不要开 unseen-shape、extrapolation、M1/M2。未 commit / push。",
        "",
    ]
    text = "\n".join(lines)
    (_cross_dir() / "HANDOFF.md").write_text(text, encoding="utf-8")
    (_root() / "HANDOFF.md").write_text(text, encoding="utf-8")
    print(text)


def run_seed(seed: int, grid=None, split_codes=None, split_manifest=None) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    if grid is None:
        grid = load_factorial_grid(spec)
    if split_codes is None or split_manifest is None:
        split_codes, split_manifest = load_split_codes()
    print(f"[phase18] ===== seed {seed} =====")
    train_info = train_seed(seed, grid, split_codes, split_manifest)
    extract_trained(seed, grid, spec)
    summary_bt = _seed_dir(seed) / "tables" / "bt_summary.json"
    if summary_bt.exists():
        trained = json.loads(summary_bt.read_text(encoding="utf-8"))
        print(f"[phase18] seed {seed}: analysis exists")
    else:
        trained = analyze_trained(seed, grid, split_codes, split_manifest)
    labels = {
        "P": grid["P"].reshape(-1, 6).astype(np.float64),
        "Q": grid["Q"].reshape(-1, 6).astype(np.float64),
        "t": grid["t"].reshape(-1, 2).astype(np.float64),
    }
    g = decompose_controls(labels["P"])["geometry_comp"]
    init = analyze_init(seed, split_codes, labels, g)
    return {"seed": seed, "train": train_info, "trained": trained, "init": init}


def summarize_if_ready() -> Optional[Dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        path = _seed_dir(seed) / "tables" / "bt_summary.json"
        if not path.exists():
            print(f"[phase18] waiting for seed {seed} analysis")
            return None
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    _, manifest = load_split_codes()
    decision = decide(rows)
    dump_json(_cross_dir() / "summary.json", _to_py({"decision": decision, "seeds": rows}))
    write_handoff(decision, rows, manifest)
    return {"decision": decision, "n": len(rows)}


def run_phase18(seed: Optional[int] = None) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    if fingerprint(spec) != PROTOCOL_FP:
        raise RuntimeError(f"fingerprint {fingerprint(spec)} != {PROTOCOL_FP}")
    split_manifest = freeze_split(spec, force=False)
    split_codes, split_manifest = load_split_codes()
    grid = load_factorial_grid(spec)
    seeds = (int(seed),) if seed is not None else SEEDS
    rows = [run_seed(item, grid, split_codes, split_manifest) for item in seeds]
    cross = summarize_if_ready()
    return {"seeds": [row["seed"] for row in rows], "split_hash": split_manifest["split_hash"], "cross": cross}


if __name__ == "__main__":
    run_phase18()
