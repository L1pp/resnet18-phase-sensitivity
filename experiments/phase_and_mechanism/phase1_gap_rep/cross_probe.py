"""Shared val→test linear re-probe helpers for Phase 1.7c and Phase 1.8.

Does not change the B_t estimator. Train-fitted probes are diagnostic only.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .audit import _erasure_z, _pc_spectrum, _principal_angles_deg, _qr_basis, _subspace_energy
from .bt_robustness import ENERGY_TOL_OK, MAX_DRAWS, N_NULL, _mae_px, _pvalue
from .common import COORD_SCALE


def probe_predict(z: np.ndarray, targets: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    design = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(design[fit_mask], targets[fit_mask], rcond=None)
    return design @ coef


def probe_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return _mae_px(pred[mask] - truth[mask])


def constant_t_mae(t: np.ndarray, fit_mask: np.ndarray, eval_mask: np.ndarray) -> float:
    mu = t[fit_mask].mean(axis=0)
    return _mae_px(t[eval_mask] - mu)


def axis_mae_px(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    diff = (pred[mask] - truth[mask]) * COORD_SCALE
    return float(np.mean(np.abs(diff[:, 0]))), float(np.mean(np.abs(diff[:, 1])))


def cross_probe_block(
    z: np.ndarray,
    t: np.ndarray,
    g: np.ndarray,
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    t_hat = probe_predict(z, t, fit_mask)
    g_hat = probe_predict(z, g, fit_mask)
    tx, ty = axis_mae_px(t_hat, t, eval_mask)
    return {
        "fit_n": int(fit_mask.sum()),
        "eval_n": int(eval_mask.sum()),
        "t_mae_px": probe_mae(t_hat, t, eval_mask),
        "t_tx_mae_px": tx,
        "t_ty_mae_px": ty,
        "g_mae_px": probe_mae(g_hat, g, eval_mask),
        "constant_t_mae_px": constant_t_mae(t, fit_mask, eval_mask),
        "fit_t_mae_px": probe_mae(t_hat, t, fit_mask),
        "fit_g_mae_px": probe_mae(g_hat, g, fit_mask),
    }


def ratios(erased: Mapping[str, Any], raw: Mapping[str, Any], constant_t: float) -> Dict[str, float]:
    r_t = float(erased["t_mae_px"]) / max(float(constant_t), 1e-12)
    r_q = float(erased["g_mae_px"]) / max(float(raw["g_mae_px"]), 1e-12)
    return {"R_t": r_t, "R_q": r_q}


def tx_ty_from_tid(tid: np.ndarray, n_ty: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    tid = np.asarray(tid, dtype=np.int64)
    return tid // int(n_ty), tid % int(n_ty)


def per_axis_report(
    pred_t: np.ndarray,
    true_t: np.ndarray,
    tid: np.ndarray,
    mask: np.ndarray,
    n_ty: int = 8,
) -> Dict[str, Any]:
    tx_idx, ty_idx = tx_ty_from_tid(tid[mask], n_ty)
    diff = (pred_t[mask] - true_t[mask]) * COORD_SCALE
    by_tx, by_ty = {}, {}
    for axis_name, ids, col in (("tx", tx_idx, 0), ("ty", ty_idx, 1)):
        store = by_tx if axis_name == "tx" else by_ty
        for value in np.unique(ids):
            sel = ids == value
            store[int(value)] = {
                "n": int(sel.sum()),
                "mae_px": float(np.mean(np.abs(diff[sel, col]))),
                "vec_mae_px": float(np.mean(np.abs(diff[sel]))),
            }
    return {
        "tx": by_tx,
        "ty": by_ty,
        "merged_t_mae_px": float(np.mean(np.abs(diff))),
        "tx_mae_px": float(np.mean(np.abs(diff[:, 0]))),
        "ty_mae_px": float(np.mean(np.abs(diff[:, 1]))),
    }


def eval_cross_cut(
    z_flat: np.ndarray,
    basis: np.ndarray,
    mu: np.ndarray,
    t: np.ndarray,
    g: np.ndarray,
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
    train_centered: np.ndarray,
) -> Dict[str, Any]:
    z_e = _erasure_z(z_flat, basis, mu, True)
    probe = cross_probe_block(z_e, t, g, fit_mask, eval_mask)
    return {
        "probe": probe,
        "energy": _subspace_energy(train_centered, basis),
        "z_e": z_e,
    }


def top2_pca_energy(train_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    pcs, pc_energy = _pc_spectrum(train_centered)
    return pcs, pc_energy, float(pc_energy[:2].sum())


def energy_flags(bt_energy: float, top2_energy: float) -> Dict[str, Any]:
    ratio = float(bt_energy) / max(float(top2_energy), 1e-12)
    return {
        "target_energy": float(bt_energy),
        "max_rank2_pca_energy": float(top2_energy),
        "target_over_max_rank2": ratio,
        "near_max_energy": bool(ratio > 0.95),
        "energy_null_degenerate": bool(ratio > 0.95),
    }


def basis_diversity_ok(bases: Sequence[np.ndarray], min_unique: int = 20, min_median_deg: float = 5.0) -> bool:
    if len(bases) < 2:
        return False
    hashes = set()
    for basis in bases:
        array = np.asarray(basis, dtype=np.float64)
        key = tuple(np.round(np.abs(array), 4).ravel().tolist())
        hashes.add(key)
    if len(hashes) < min(min_unique, len(bases)):
        return False
    rng = np.random.default_rng(0)
    n = len(bases)
    n_pairs = min(80, n // 2)
    if n_pairs < 8:
        return len(hashes) >= 2
    angles = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        deg = _principal_angles_deg(np.asarray(bases[i]), np.asarray(bases[j]))
        angles.append(float(np.mean(deg)))
    return float(np.median(angles)) >= min_median_deg


def sample_energy_matched_strict(
    train_centered: np.ndarray,
    rank: int,
    target: float,
    rng: np.random.Generator,
    pcs: np.ndarray,
    pc_energy: np.ndarray,
    n: int = N_NULL,
    min_ok: int = 500,
) -> Tuple[list, list, Dict[str, Any]]:
    from .audit import _pc_window_for_energy

    window = _pc_window_for_energy(pc_energy, rank, target)
    scale = max(target, 1e-12)
    matched, energies, rels = [], [], []
    for _ in range(MAX_DRAWS):
        basis = _qr_basis(pcs[:, :window] @ rng.normal(size=(window, rank)))[:, :rank]
        energy = _subspace_energy(train_centered, basis)
        err = abs(energy - target) / scale
        if err <= ENERGY_TOL_OK:
            matched.append(basis.copy())
            energies.append(energy)
            rels.append(err)
            if len(matched) >= n:
                break
    diverse = basis_diversity_ok(matched)
    mean_e = float(np.mean(energies)) if energies else 0.0
    mean_rel = abs(mean_e - target) / scale if energies else None
    flags = energy_flags(target, float(pc_energy[:2].sum()))
    matched_ok = bool(len(matched) >= min_ok and diverse and (mean_rel is not None and mean_rel <= ENERGY_TOL_OK))
    if flags["near_max_energy"] and not diverse:
        matched_ok = False
        flags["energy_null_degenerate"] = True
    info = {
        **flags,
        "pc_window": int(window),
        "n_collected": len(matched),
        "n_requested": int(n),
        "min_ok": int(min_ok),
        "mean_deleted_energy": mean_e,
        "mean_rel_error": mean_rel,
        "tol_requested": ENERGY_TOL_OK,
        "matched_ok": matched_ok,
        "diverse": bool(diverse),
        "uninformative": not matched_ok,
        "sampling": "strict_5pct_no_fallback",
    }
    return matched, energies, info


def summarize_null(true_val: float, samples: Sequence[float], larger: bool) -> Dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 0:
        return {"true": float(true_val), "n": 0, "p": None, "mean": None}
    return {
        "true": float(true_val),
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "p": _pvalue(true_val, array, larger=larger),
        "effect_true_minus_mean": float(true_val - array.mean()),
    }


def save_null_draws(path, kind: str, bases, stats: Mapping[str, Sequence[float]], extra=None, perms=None) -> None:
    payload = {
        "kind": np.asarray(kind),
        "energy": np.asarray(stats["energy"], dtype=np.float64),
        "probe_E_t": np.asarray(stats["probe_E_t"], dtype=np.float64),
        "probe_E_g": np.asarray(stats["probe_E_g"], dtype=np.float64),
        "bases": np.stack(bases).astype(np.float32) if len(bases) else np.zeros((0, 512, 2), dtype=np.float32),
    }
    if perms is not None and len(perms):
        payload["translation_id_perm"] = np.stack(perms).astype(np.int16)
    if extra:
        for key, value in extra.items():
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)
