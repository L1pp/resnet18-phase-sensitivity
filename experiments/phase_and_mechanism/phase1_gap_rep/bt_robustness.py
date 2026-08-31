"""Phase 1.6 robustness of rank-2 B_t. Frozen features only; writes phase1_6_bt_robustness/."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .analyze import _labels, _load_features, _load_split_codes
from .audit import (
    SPLIT_NAMES,
    _erasure_z,
    _frobenius_alignment,
    _functional_alignment,
    _histogram,
    _pc_spectrum,
    _pc_window_for_energy,
    _predict_p,
    _principal_angles_deg,
    _pvalue,
    _qr_basis,
    _readouts,
    _subspace_energy,
    _to_py,
)
from .common import (
    COORD_SCALE,
    RESULTS_ROOT,
    SEED,
    decompose_controls,
    dump_json,
    fingerprint,
    profile_spec,
    setup_matplotlib_chinese,
    train_run_spec,
)

N_NULL = 1000
N_FOLD = 8
ENERGY_TOL_STRICT = 0.02
ENERGY_TOL_OK = 0.05
MAX_DRAWS = 40000
REF = {
    "energy": 7.675116415539425,
    "test_head_E_t": 53.61653561097966,
    "test_head_E_q": 0.42685805673875665,
    "test_probe_E_t": 30.03045506419427,
    "test_probe_E_q": 0.4333008201306451,
    "angles_Bt_Wt_deg": [41.720941070238105, 54.59706780381266],
}
OUT_NAME = "phase1_6_bt_robustness"


def _out_dir() -> Path:
    root = RESULTS_ROOT / OUT_NAME
    for name in ("config", "tables", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _mae_px(diff: np.ndarray) -> float:
    return float(np.mean(np.abs(diff))) * COORD_SCALE


def _overlap_norm(a: np.ndarray, b: np.ndarray) -> float:
    qa, qb = _qr_basis(a), _qr_basis(b)
    denom = float(min(qa.shape[1], qb.shape[1]))
    return 0.0 if denom <= 0 else float(np.sum((qa.T @ qb) ** 2)) / denom


def _pack_head(pred: np.ndarray, labels: Mapping[str, np.ndarray], g: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    parts = decompose_controls(pred)
    return {
        "n": int(mask.sum()),
        "E_t": _mae_px(parts["t"][mask] - labels["t"][mask]),
        "E_q": _mae_px(parts["q"].reshape(pred.shape[0], 6)[mask] - labels["Q"][mask]),
        "E_P": _mae_px(pred[mask] - labels["P"][mask]),
        "E_g": float(np.mean(np.abs(parts["geometry_comp"][mask] - g[mask]))),
    }


def _fit_factor(zc, t, g, t_mu, t_sig, q_mu, q_sig) -> Tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([(t - t_mu) / t_sig, (g - q_mu) / q_sig], axis=1)
    beta, *_ = np.linalg.lstsq(design, zc, rcond=None)
    return beta, _qr_basis(beta[:2].T)


def _probe_fit(z_erased: np.ndarray, targets: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    design = np.concatenate([z_erased, np.ones((z_erased.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design[train_mask], targets[train_mask], rcond=None)
    return design @ coef


def _probe_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return _mae_px(pred[mask] - truth[mask])


def _eval_cut(z_flat, basis, mu, weight, bias, labels, g, masks) -> Dict[str, Any]:
    z_e = _erasure_z(z_flat, basis, mu, True)
    pred = _predict_p(z_e, weight, bias)
    head = {name: _pack_head(pred, labels, g, mask) for name, mask in masks.items()}
    train_mask = masks["train"]
    t_hat = _probe_fit(z_e, labels["t"], train_mask)
    g_hat = _probe_fit(z_e, g, train_mask)
    probe = {f"{name}_E_t": _probe_mae(t_hat, labels["t"], mask) for name, mask in masks.items()}
    probe.update({f"{name}_E_g": _probe_mae(g_hat, g, mask) for name, mask in masks.items()})
    return {"head": head, "probe": probe, "pred": pred}


def _sample_matched(train_centered, rank, target, rng, pcs, pc_energy, n, tol):
    window = _pc_window_for_energy(pc_energy, rank, target)
    scale = max(target, 1e-12)
    matched, energies, pool = [], [], []
    for _ in range(MAX_DRAWS):
        basis = _qr_basis(pcs[:, :window] @ rng.normal(size=(window, rank)))[:, :rank]
        energy = _subspace_energy(train_centered, basis)
        err = abs(energy - target) / scale
        pool.append((err, basis.copy(), energy))
        if err <= tol:
            matched.append(basis.copy())
            energies.append(energy)
            if len(matched) >= n:
                break
    if len(matched) < n:
        pool.sort(key=lambda item: item[0])
        matched = [item[1] for item in pool[:n]]
        energies = [item[2] for item in pool[:n]]
    mean_e = float(np.mean(energies)) if energies else 0.0
    rel = abs(mean_e - target) / scale
    info = {
        "pc_window": int(window),
        "n_collected": len(matched),
        "mean_deleted_energy": mean_e,
        "rel_error": rel,
        "tol_requested": tol,
        "matched_ok": bool(rel <= ENERGY_TOL_OK),
        "matched_strict": bool(rel <= ENERGY_TOL_STRICT),
    }
    return matched, energies, info


def _summarize_null(true_val: float, samples: Sequence[float], larger: bool) -> Dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    return {
        "true": float(true_val),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "p": _pvalue(true_val, array, larger=larger),
        "percentile": float(100.0 * np.mean(array < true_val) if larger else 100.0 * np.mean(array > true_val)),
    }


def _plot_hist(path: Path, samples, true_val, title, xlabel) -> str:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(samples, bins=40, alpha=0.85)
    ax.axvline(true_val, color="C3", ls="--", label="真实 B_t")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _write_cells(path, sid, tid, split, labels, g, pred0, pred_bt, energy) -> None:
    parts0 = decompose_controls(pred0)
    parts_b = decompose_controls(pred_bt)
    fields = [
        "shape_id", "translation_id", "split", "cut_type", "cut_rank", "deleted_energy",
        "true_tx", "true_ty", "pred_tx", "pred_ty", "error_t", "error_q", "error_P", "error_g",
    ]
    for i in range(4):
        fields.extend([f"true_g{i}", f"pred_g{i}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cut, pred, parts in (("baseline", pred0, parts0), ("B_t", pred_bt, parts_b)):
            q_pred = parts["q"].reshape(-1, 6)
            for i in range(len(sid)):
                row = {
                    "shape_id": int(sid[i]), "translation_id": int(tid[i]),
                    "split": SPLIT_NAMES[int(split[i])], "cut_type": cut,
                    "cut_rank": 0 if cut == "baseline" else 2,
                    "deleted_energy": 0.0 if cut == "baseline" else energy,
                    "true_tx": labels["t"][i, 0], "true_ty": labels["t"][i, 1],
                    "pred_tx": parts["t"][i, 0], "pred_ty": parts["t"][i, 1],
                    "error_t": float(np.mean(np.abs(parts["t"][i] - labels["t"][i]))) * COORD_SCALE,
                    "error_q": float(np.mean(np.abs(q_pred[i] - labels["Q"][i]))) * COORD_SCALE,
                    "error_P": float(np.mean(np.abs(pred[i] - labels["P"][i]))) * COORD_SCALE,
                    "error_g": float(np.mean(np.abs(parts["geometry_comp"][i] - g[i]))),
                }
                for k in range(4):
                    row[f"true_g{k}"] = g[i, k]
                    row[f"pred_g{k}"] = parts["geometry_comp"][i, k]
                writer.writerow(row)


def _decide(payload: Mapping[str, Any]) -> Dict[str, Any]:
    head = payload["true_cut"]["head"]["test"]
    probe = payload["true_cut"]["probe"]
    perm = payload["nulls"]["permutation"]
    matched = payload["nulls"]["energy_matched"]
    const_t = payload["constant_baseline"]["test_E_t"]
    baseline_q = payload["baseline"]["test"]["E_q"]
    folds = payload["shape_folds"]
    match_ok = bool(matched["info"]["matched_ok"])
    p_t_perm = perm["head_E_t"]["p"]
    p_t_match = matched["head_E_t"]["p"]
    p_probe_t_perm = perm["probe_E_t"]["p"]
    p_probe_t_match = matched["probe_E_t"]["p"]
    q_ok = head["E_q"] <= max(baseline_q * 1.5, baseline_q + 0.2)
    q_probe_ok = probe["test_E_g"] <= payload["baseline_probe"]["test_E_g"] + 0.05
    t_extreme = (
        p_t_perm <= 0.05 and p_probe_t_perm <= 0.05
        and (not match_ok or (p_t_match <= 0.05 and p_probe_t_match <= 0.05))
        and head["E_t"] > perm["head_E_t"]["mean"] * 1.5
        and probe["test_E_t"] > perm["probe_E_t"]["mean"] * 1.3
    )
    t_near_const = probe["test_E_t"] >= 0.8 * const_t
    angles = [fold["angles_deg"] for fold in folds]
    max_ang = max(max(item) for item in angles) if angles else 90.0
    mean_ang = float(np.mean([np.mean(item) for item in angles])) if angles else 90.0
    fold_q = [fold["head"]["test"]["E_q"] for fold in folds]
    fold_stable = mean_ang <= 20.0 and max_ang <= 35.0 and (max(fold_q) <= baseline_q + 0.3 if fold_q else False)
    fold_scatter = mean_ang > 25.0 or max_ang > 45.0 or ((max(fold_q) - min(fold_q) > 1.0) if fold_q else False)
    if (not match_ok) and t_extreme and q_ok and t_near_const:
        letter, note = "B", "平移破坏在 permutation 上极端，但能量匹配未达到 5% 容差。"
    elif t_extreme and q_ok and q_probe_ok and t_near_const and fold_stable and match_ok:
        letter, note = "A", "在当前表征和线性分析下，存在稳定的低维 translation-associated 组织。不是独立神经模块，也不是严格 disentanglement。"
    elif t_extreme and q_ok and fold_scatter:
        letter, note = "B", "全数据 B_t 选择性很强，但跨 shape 估计方向或不稳定性偏大。"
    else:
        letter, note = "C", "真实 B_t 在 permutation 或严格能量匹配下不够极端，或形状保住无法复现。"
    return {
        "letter": letter, "note": note, "t_extreme": t_extreme, "q_ok": q_ok,
        "t_near_const": t_near_const, "fold_stable": fold_stable, "match_ok": match_ok,
        "mean_fold_angle_deg": mean_ang, "max_fold_angle_deg": max_ang,
        "p_head_E_t_perm": p_t_perm, "p_head_E_t_matched": p_t_match,
        "p_probe_E_t_perm": p_probe_t_perm, "p_probe_E_t_matched": p_probe_t_match,
        "p_head_E_q_preserve_perm": perm["head_E_q_preserve"]["p"],
        "p_probe_E_g_preserve_perm": perm["probe_E_g_preserve"]["p"],
    }


def run_phase16(run_name: str = "adamw_l1_scratch") -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    out = _out_dir()
    z, weight, bias = _load_features(run.name, spec.name)
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

    definition = {
        "Z_c": "Z - mu_train", "q": "geometry_comp 4D",
        "X": "[tx, ty, q1, q2, q3, q4] standardized on train only",
        "fit": "OLS np.linalg.lstsq, no ridge",
        "B_t": "QR span of two translation coefficient rows, rank fixed at 2",
        "seed": SEED, "fingerprint": fingerprint(spec), "source_run": run.name,
        "rank": int(u_bt.shape[1]), "deleted_energy": energy,
        "angles_Bt_Wt_deg": angles_wt,
        "frobenius_Wt_on_Bt": _frobenius_alignment(w_t, u_bt),
        "frobenius_Wq_on_Bt": _frobenius_alignment(w_q, u_bt),
    }
    dump_json(out / "config" / "bt_definition.json", _to_py(definition))
    deltas = {
        "energy": abs(energy - REF["energy"]),
        "test_head_E_t": abs(true_cut["head"]["test"]["E_t"] - REF["test_head_E_t"]),
        "test_head_E_q": abs(true_cut["head"]["test"]["E_q"] - REF["test_head_E_q"]),
        "test_probe_E_t": abs(true_cut["probe"]["test_E_t"] - REF["test_probe_E_t"]),
        "test_probe_E_g_px": abs(true_cut["probe"]["test_E_g"] - REF["test_probe_E_q"]),
        "angle0": abs(angles_wt[0] - REF["angles_Bt_Wt_deg"][0]),
        "angle1": abs(angles_wt[1] - REF["angles_Bt_Wt_deg"][1]),
    }
    reproduce_ok = (
        int(u_bt.shape[1]) == 2 and deltas["energy"] < 0.05
        and deltas["test_head_E_t"] < 0.05 and deltas["test_head_E_q"] < 0.05
        and deltas["test_probe_E_t"] < 0.05 and deltas["test_probe_E_g_px"] < 0.05
        and deltas["angle0"] < 1.0 and deltas["angle1"] < 1.0
    )
    print(
        f"[phase16] energy={energy:.4f} head_t={true_cut['head']['test']['E_t']:.3f} "
        f"head_q={true_cut['head']['test']['E_q']:.3f} probe_t={true_cut['probe']['test_E_t']:.3f} "
        f"probe_g={true_cut['probe']['test_E_g']:.3f} ok={reproduce_ok}"
    )
    if not reproduce_ok:
        (out / "reproduce_fail.md").write_text(
            "# 复现失败\n\n" + "\n".join(f"- {k}: Δ={v:.4g}" for k, v in deltas.items()) + "\n",
            encoding="utf-8",
        )
        dump_json(out / "reproduce_fail.json", _to_py({"ok": False, "deltas": deltas, "got": true_cut["head"]["test"]}))
        print("[phase16] REPRODUCE FAIL — stopping")
        return {"ok": False, "deltas": deltas}

    rng = np.random.default_rng(SEED)
    unique_t = np.zeros((n_t, 2), dtype=np.float64)
    for t_id in range(n_t):
        unique_t[t_id] = labels["t"][tid == t_id][0]
    train_tid = tid[train_mask]
    perm_stats = {k: [] for k in ("energy", "head_E_t", "head_E_q", "probe_E_t", "probe_E_g", "F_t", "F_q")}
    print(f"[phase16] permutation N={N_NULL}")
    for i in range(N_NULL):
        if i % 100 == 0:
            print(f"[phase16] perm {i}/{N_NULL}")
        fake_t = unique_t[rng.permutation(n_t)[train_tid]]
        _, basis_p = _fit_factor(train_centered, fake_t, g_train, t_mu, t_sig, q_mu, q_sig)
        ev = _eval_cut(z_flat, basis_p, mu, weight, bias, labels, g, masks)
        perm_stats["energy"].append(_subspace_energy(train_centered, basis_p))
        perm_stats["head_E_t"].append(ev["head"]["test"]["E_t"])
        perm_stats["head_E_q"].append(ev["head"]["test"]["E_q"])
        perm_stats["probe_E_t"].append(ev["probe"]["test_E_t"])
        perm_stats["probe_E_g"].append(ev["probe"]["test_E_g"])
        perm_stats["F_t"].append(_functional_alignment(z_flat[test_mask] - mu, w_t, basis_p))
        perm_stats["F_q"].append(_functional_alignment(z_flat[test_mask] - mu, w_q, basis_p))

    pcs, pc_energy = _pc_spectrum(train_centered)
    print("[phase16] energy-matched rank-2")
    matched, match_e, match_info = _sample_matched(train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, ENERGY_TOL_STRICT)
    if not match_info["matched_strict"]:
        print(f"[phase16] 2% missed rel={match_info['rel_error']:.4f}, retry 5%")
        matched, match_e, match_info = _sample_matched(train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, ENERGY_TOL_OK)
    print(f"[phase16] matched rel={match_info['rel_error']:.4f} ok={match_info['matched_ok']}")
    match_stats = {k: [] for k in ("energy", "head_E_t", "head_E_q", "probe_E_t", "probe_E_g", "F_t", "F_q")}
    for i, basis_m in enumerate(matched):
        if i % 100 == 0:
            print(f"[phase16] matched {i}/{len(matched)}")
        ev = _eval_cut(z_flat, basis_m, mu, weight, bias, labels, g, masks)
        match_stats["energy"].append(_subspace_energy(train_centered, basis_m))
        match_stats["head_E_t"].append(ev["head"]["test"]["E_t"])
        match_stats["head_E_q"].append(ev["head"]["test"]["E_q"])
        match_stats["probe_E_t"].append(ev["probe"]["test_E_t"])
        match_stats["probe_E_g"].append(ev["probe"]["test_E_g"])
        match_stats["F_t"].append(_functional_alignment(z_flat[test_mask] - mu, w_t, basis_m))
        match_stats["F_q"].append(_functional_alignment(z_flat[test_mask] - mu, w_q, basis_m))

    F_t = _functional_alignment(z_flat[test_mask] - mu, w_t, u_bt)
    F_q = _functional_alignment(z_flat[test_mask] - mu, w_q, u_bt)

    print("[phase16] shape 8-fold")
    shuffled = np.random.default_rng(SEED).permutation(np.arange(n_s))
    folds = []
    for fold in range(N_FOLD):
        held = shuffled[fold * 8:(fold + 1) * 8]
        keep = np.ones(n_s, dtype=bool)
        keep[held] = False
        fold_mask = train_mask & keep[sid]
        _, u_f = _fit_factor(z_flat[fold_mask] - mu, labels["t"][fold_mask], g[fold_mask], t_mu, t_sig, q_mu, q_sig)
        ev = _eval_cut(z_flat, u_f, mu, weight, bias, labels, g, masks)
        folds.append({
            "fold": fold, "held_shapes": held.tolist(), "n_train_cells": int(fold_mask.sum()),
            "angles_deg": _principal_angles_deg(u_bt, u_f), "overlap_norm": _overlap_norm(u_bt, u_f),
            "energy": _subspace_energy(train_centered, u_f), "head": ev["head"], "probe": ev["probe"],
        })
        print(f"[phase16] fold {fold} angles={folds[-1]['angles_deg']}")

    print("[phase16] translation 8-fold")
    t_shuffled = np.random.default_rng(SEED + 1).permutation(np.arange(n_t))
    t_folds = []
    for fold in range(N_FOLD):
        held = t_shuffled[fold * 8:(fold + 1) * 8]
        keep = np.ones(n_t, dtype=bool)
        keep[held] = False
        fold_mask = train_mask & keep[tid]
        _, u_f = _fit_factor(z_flat[fold_mask] - mu, labels["t"][fold_mask], g[fold_mask], t_mu, t_sig, q_mu, q_sig)
        ev = _eval_cut(z_flat, u_f, mu, weight, bias, labels, g, masks)
        t_folds.append({
            "fold": fold, "held_translations": held.tolist(), "n_train_cells": int(fold_mask.sum()),
            "angles_deg": _principal_angles_deg(u_bt, u_f), "overlap_norm": _overlap_norm(u_bt, u_f),
            "head": ev["head"], "probe": ev["probe"],
        })

    z_no_wt = _erasure_z(z_flat, _qr_basis(w_t.T), mu, True)
    probe_t_nowt = _probe_fit(z_no_wt, labels["t"], train_mask)
    design = np.concatenate([z_no_wt, np.ones((z_no_wt.shape[0], 1))], axis=1)
    coef_t, *_ = np.linalg.lstsq(design[train_mask], labels["t"][train_mask], rcond=None)
    w_probe = coef_t[:dim].T
    bt_resid = u_bt - _qr_basis(w_t.T) @ (_qr_basis(w_t.T).T @ u_bt)
    bt_resid = _qr_basis(bt_resid) if np.linalg.norm(bt_resid) > 1e-8 else bt_resid
    wt_redundancy = {
        "angles_Bt_Wt_deg": angles_wt, "overlap_Bt_Wt": _overlap_norm(u_bt, w_t.T),
        "erasure_Wt_probe_test_E_t": _probe_mae(probe_t_nowt, labels["t"], test_mask),
        "erasure_Wt_head_test": _pack_head(_predict_p(z_no_wt, weight, bias), labels, g, test_mask),
        "new_probe_vs_Bt_deg": _principal_angles_deg(w_probe.T, u_bt),
        "new_probe_vs_Bt_residual_deg": _principal_angles_deg(w_probe.T, bt_resid) if bt_resid.size else None,
        "note": "Deleting W_t kills the current decoder; a new linear probe can still read t from leftover z.",
    }

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
        "histogram": {k: _histogram(v) for k, v in perm_stats.items()},
    }
    match_block = {
        "n": len(matched), "info": match_info,
        "energy": _summarize_null(energy, match_stats["energy"], True),
        "head_E_t": _summarize_null(true_cut["head"]["test"]["E_t"], match_stats["head_E_t"], True),
        "head_E_q": _summarize_null(true_cut["head"]["test"]["E_q"], match_stats["head_E_q"], True),
        "head_E_q_preserve": _summarize_null(true_cut["head"]["test"]["E_q"], match_stats["head_E_q"], False),
        "probe_E_t": _summarize_null(true_cut["probe"]["test_E_t"], match_stats["probe_E_t"], True),
        "probe_E_g": _summarize_null(true_cut["probe"]["test_E_g"], match_stats["probe_E_g"], True),
        "probe_E_g_preserve": _summarize_null(true_cut["probe"]["test_E_g"], match_stats["probe_E_g"], False),
        "F_t": _summarize_null(F_t, match_stats["F_t"], True),
        "F_q": _summarize_null(F_q, match_stats["F_q"], True),
        "histogram": {k: _histogram(v) for k, v in match_stats.items()},
    }
    payload = {
        "reproduce_ok": True, "deltas": deltas, "definition": definition,
        "baseline": baseline, "baseline_probe": baseline_probe,
        "constant_baseline": {"train_mean_t": t_mu.tolist(), "test_E_t": const_t["test"], "all": const_t},
        "true_cut": {
            "energy": energy, "head": true_cut["head"], "probe": true_cut["probe"],
            "F_t_from_Bt": F_t, "F_q_from_Bt": F_q, "angles_Bt_Wt_deg": angles_wt,
        },
        "nulls": {"permutation": perm_block, "energy_matched": match_block},
        "shape_folds": folds, "translation_folds": t_folds, "wt_redundancy": wt_redundancy,
    }
    payload["verdict"] = _decide(payload)

    fig_dir = out / "figures"
    figures = [
        _plot_hist(fig_dir / "perm_probe_t.png", perm_stats["probe_E_t"], true_cut["probe"]["test_E_t"], "Permutation null：translation re-probe MAE", "px"),
        _plot_hist(fig_dir / "matched_probe_t.png", match_stats["probe_E_t"], true_cut["probe"]["test_E_t"], "能量匹配随机：translation re-probe MAE", "px"),
        _plot_hist(fig_dir / "null_probe_g.png", perm_stats["probe_E_g"], true_cut["probe"]["test_E_g"], "Permutation null：geometry re-probe MAE", "scaled"),
    ]
    plt = setup_matplotlib_chinese()
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, [f["angles_deg"][0] for f in folds], marker="o", label="主角度 1")
    ax.plot(x, [f["angles_deg"][1] for f in folds], marker="s", label="主角度 2")
    ax.set_xlabel("shape fold"); ax.set_ylabel("度"); ax.set_title("离开 8 个 shape 后与全数据 B_t 的主角度")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(fig_dir / "fold_angles.png", dpi=140); plt.close(fig)
    figures.append("fold_angles.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, [f["head"]["test"]["E_t"] for f in folds], marker="o", label="原头位置")
    ax.plot(x, [f["head"]["test"]["E_q"] for f in folds], marker="s", label="原头形状")
    ax.axhline(true_cut["head"]["test"]["E_t"], color="C0", ls="--", alpha=0.5)
    ax.axhline(true_cut["head"]["test"]["E_q"], color="C1", ls="--", alpha=0.5)
    ax.set_xlabel("shape fold"); ax.set_ylabel("px"); ax.set_title("各 fold 删除后的 test MAE")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "fold_mae.png", dpi=140); plt.close(fig)
    figures.append("fold_mae.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(perm_stats["energy"], perm_stats["head_E_t"], s=8, alpha=0.35, label="perm")
    ax.scatter(match_stats["energy"], match_stats["head_E_t"], s=8, alpha=0.35, label="matched")
    ax.scatter([energy], [true_cut["head"]["test"]["E_t"]], c="C3", s=40, label="真实 B_t", zorder=3)
    ax.set_xlabel("删除能量"); ax.set_ylabel("原头位置 MAE (px)"); ax.set_title("删除能量 vs 位置损伤")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "energy_vs_t.png", dpi=140); plt.close(fig)
    figures.append("energy_vs_t.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(perm_stats["energy"], perm_stats["head_E_q"], s=8, alpha=0.35, label="perm")
    ax.scatter(match_stats["energy"], match_stats["head_E_q"], s=8, alpha=0.35, label="matched")
    ax.scatter([energy], [true_cut["head"]["test"]["E_q"]], c="C3", s=40, label="真实 B_t", zorder=3)
    ax.set_xlabel("删除能量"); ax.set_ylabel("原头形状 MAE (px)"); ax.set_title("删除能量 vs 形状损伤")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "energy_vs_q.png", dpi=140); plt.close(fig)
    figures.append("energy_vs_q.png")
    payload["figures"] = figures

    _write_cells(out / "tables" / "cells.csv", sid, tid, split_flat, labels, g, pred0, true_cut["pred"], energy)
    dump_json(out / "summary.json", _to_py(payload))
    v = payload["verdict"]
    decision = (
        f"# Phase1.6 结论（{v['letter']}）\n\n{v['note']}\n\n"
        f"## 复现\n\n- 能量 {energy:.4f}\n"
        f"- test 原头 位置 {true_cut['head']['test']['E_t']:.3f} / 形状 {true_cut['head']['test']['E_q']:.3f}\n"
        f"- test re-probe 位置 {true_cut['probe']['test_E_t']:.3f} / 几何 {true_cut['probe']['test_E_g']:.3f}\n"
        f"- 常数预测位置 {const_t['test']:.3f} px\n\n"
        f"## Null\n\n- perm 原头位置 p={perm_block['head_E_t']['p']:.4g} 均值 {perm_block['head_E_t']['mean']:.3f}\n"
        f"- perm re-probe 位置 p={perm_block['probe_E_t']['p']:.4g} 均值 {perm_block['probe_E_t']['mean']:.3f}\n"
        f"- perm 形状保住 p={perm_block['head_E_q_preserve']['p']:.4g}\n"
        f"- 能量匹配相对误差 {match_info['rel_error']:.4f} 合格={match_info['matched_ok']}\n"
        f"- matched 原头位置 p={match_block['head_E_t']['p']:.4g} 均值 {match_block['head_E_t']['mean']:.3f}\n\n"
        f"## Shape-fold\n\n- 平均主角度 {v['mean_fold_angle_deg']:.2f}° 最大 {v['max_fold_angle_deg']:.2f}°\n"
        f"- F_t←Bt={F_t:.4f} F_q←Bt={F_q:.4f}\n\n"
        f"正式结论 **{v['letter']}**。自主探索见 explore/。\n"
    )
    (out / "decision.md").write_text(decision, encoding="utf-8")
    print(decision)
    print(f"[phase16] wrote {out}")
    return payload


def bt_robust(run_name: str = "adamw_l1_scratch") -> Dict[str, Any]:
    return run_phase16(run_name)
