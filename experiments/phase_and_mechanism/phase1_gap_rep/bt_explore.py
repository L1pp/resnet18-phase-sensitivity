"""Autonomous follow-up: random-init GAP, per-shape B_t, t-swap, perm-vs-true angles."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .analyze import _labels, _load_features, _load_split_codes
from .audit import (
    SPLIT_NAMES,
    _erasure_z,
    _pc_spectrum,
    _predict_p,
    _principal_angles_deg,
    _pvalue,
    _qr_basis,
    _subspace_energy,
    _to_py,
)
from .bt_robustness import (
    _eval_cut,
    _fit_factor,
    _mae_px,
    _out_dir,
    _overlap_norm,
    _probe_fit,
    _probe_mae,
    _sample_matched,
)
from .common import (
    BATCH_SIZE,
    COORD_SCALE,
    SEED,
    build_model,
    decompose_controls,
    device,
    dump_json,
    gap_features,
    images_to_tensor,
    profile_spec,
    seed_everything,
    setup_matplotlib_chinese,
    train_run_spec,
)
from .generate_data import load_split


def _extract_gap(model, spec) -> np.ndarray:
    n_s, n_t = spec.n_shapes, spec.n_translations
    z = np.zeros((n_s, n_t, 512), dtype=np.float64)
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
                batch = images_to_tensor(images[start:start + BATCH_SIZE]).to(dev)
                feats.append(gap_features(model, batch).float().cpu().numpy())
            array = np.concatenate(feats, axis=0)
            z[sid, tid] = array
            filled[sid, tid] = True
            print(f"[explore] extract {split} n={len(images)}")
    if not bool(np.all(filled)):
        raise RuntimeError("incomplete random-init features")
    return z


def run_explore(run_name: str = "adamw_l1_scratch") -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    out = _out_dir() / "explore"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

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
    design = np.concatenate([(t_train - t_mu) / t_sig, (g_train - q_mu) / q_sig], axis=1)

    print("[explore] C-branch OLS / identity-plane diagnostics")
    t_std = (t_train - t_mu) / t_sig
    beta_t_only, *_ = np.linalg.lstsq(t_std, train_centered, rcond=None)
    u_t_only = _qr_basis(beta_t_only.T)
    ident_means = np.zeros((n_t, dim), dtype=np.float64)
    for t_id in range(n_t):
        mask = train_mask & (tid == t_id)
        ident_means[t_id] = z_flat[mask].mean(0) - mu
    ident_pcs, _ = _pc_spectrum(ident_means - ident_means.mean(0, keepdims=True))
    u_ident2 = ident_pcs[:, :2]
    ridge_rows = []
    xtx = design.T @ design
    xty = design.T @ train_centered
    for lam in (0.0, 1e-4, 1e-2, 1.0, 10.0, 100.0):
        beta_r = np.linalg.solve(xtx + lam * np.eye(design.shape[1]), xty)
        u_r = _qr_basis(beta_r[:2].T)
        ridge_rows.append({
            "lambda": lam,
            "angles_vs_ols_deg": _principal_angles_deg(u_r, u_bt),
            "overlap_vs_ols": _overlap_norm(u_r, u_bt),
            "energy": _subspace_energy(train_centered, u_r),
        })
    ols_diag = {
        "t_only_vs_joint_deg": _principal_angles_deg(u_t_only, u_bt),
        "t_only_overlap": _overlap_norm(u_t_only, u_bt),
        "t_only_energy": _subspace_energy(train_centered, u_t_only),
        "identity_mean_pc2_vs_Bt_deg": _principal_angles_deg(u_ident2, u_bt),
        "identity_mean_pc2_overlap": _overlap_norm(u_ident2, u_bt),
        "identity_mean_pc2_energy": _subspace_energy(train_centered, u_ident2),
        "ridge": ridge_rows,
        "note": (
            "Identity-perm keeps 64 translation clusters. Any 2D labeling of those identities "
            "can still hit the original head. t-only vs joint tests OLS sensitivity to q."
        ),
    }
    t_only_cut = _eval_cut(z_flat, u_t_only, mu, weight, bias, labels, g, masks)
    ols_diag["t_only_test_head_E_t"] = t_only_cut["head"]["test"]["E_t"]
    ols_diag["t_only_test_head_E_q"] = t_only_cut["head"]["test"]["E_q"]
    ols_diag["t_only_test_probe_E_t"] = t_only_cut["probe"]["test_E_t"]
    ols_diag["t_only_test_probe_E_g"] = t_only_cut["probe"]["test_E_g"]

    print("[explore] per-shape t-only subspaces")
    per_shape = []
    for s in range(n_s):
        mask = train_mask & (sid == s)
        t_s = labels["t"][mask]
        zc = z_flat[mask] - mu
        t_std = (t_s - t_mu) / t_sig
        beta, *_ = np.linalg.lstsq(t_std, zc, rcond=None)
        u_s = _qr_basis(beta.T)
        per_shape.append({
            "shape_id": s,
            "n": int(mask.sum()),
            "energy": _subspace_energy(zc, u_s),
            "angles_vs_global_deg": _principal_angles_deg(u_s, u_bt),
            "overlap_norm": _overlap_norm(u_s, u_bt),
        })
    angles0 = [row["angles_vs_global_deg"][0] for row in per_shape]
    angles1 = [row["angles_vs_global_deg"][1] for row in per_shape]
    per_shape_summary = {
        "mean_angle0": float(np.mean(angles0)),
        "mean_angle1": float(np.mean(angles1)),
        "median_angle0": float(np.median(angles0)),
        "median_angle1": float(np.median(angles1)),
        "max_angle1": float(np.max(angles1)),
        "mean_overlap": float(np.mean([row["overlap_norm"] for row in per_shape])),
        "rows": per_shape,
    }

    print("[explore] 50 identity-perm angles vs true B_t")
    rng = np.random.default_rng(SEED + 7)
    unique_t = np.zeros((n_t, 2), dtype=np.float64)
    for t_id in range(n_t):
        unique_t[t_id] = labels["t"][tid == t_id][0]
    train_tid = tid[train_mask]
    perm_angles = []
    for _ in range(50):
        fake_t = unique_t[rng.permutation(n_t)[train_tid]]
        _, u_p = _fit_factor(train_centered, fake_t, g_train, t_mu, t_sig, q_mu, q_sig)
        perm_angles.append(_principal_angles_deg(u_bt, u_p))
    perm_angle_summary = {
        "n": 50,
        "mean": [float(np.mean([a[0] for a in perm_angles])), float(np.mean([a[1] for a in perm_angles]))],
        "median": [float(np.median([a[0] for a in perm_angles])), float(np.median([a[1] for a in perm_angles]))],
    }

    print("[explore] feature-space t-swap")
    pred_swap_t, pred_keep_t, pred_swap_q = [], [], []
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
                zi_swap = z_flat[i] - ((z_flat[i] - mu) @ u_bt) @ u_bt.T + ((z_flat[j] - mu) @ u_bt) @ u_bt.T
                zj_swap = z_flat[j] - ((z_flat[j] - mu) @ u_bt) @ u_bt.T + ((z_flat[i] - mu) @ u_bt) @ u_bt.T
                pi = _predict_p(zi_swap[None], weight, bias)[0]
                pj = _predict_p(zj_swap[None], weight, bias)[0]
                ti = decompose_controls(pi[None])["t"][0]
                tj = decompose_controls(pj[None])["t"][0]
                qi = decompose_controls(pi[None])["q"].reshape(6)
                qj = decompose_controls(pj[None])["q"].reshape(6)
                pred_swap_t.append(float(np.mean(np.abs(ti - labels["t"][j]))) * COORD_SCALE)
                pred_swap_t.append(float(np.mean(np.abs(tj - labels["t"][i]))) * COORD_SCALE)
                pred_keep_t.append(float(np.mean(np.abs(ti - labels["t"][i]))) * COORD_SCALE)
                pred_swap_q.append(float(np.mean(np.abs(qi - labels["Q"][i]))) * COORD_SCALE)
                pred_swap_q.append(float(np.mean(np.abs(qj - labels["Q"][j]))) * COORD_SCALE)
                n_pairs += 1
    swap = {
        "n_pairs": n_pairs,
        "swap_t_mae": float(np.mean(pred_swap_t)) if pred_swap_t else None,
        "unswapped_t_mae": float(np.mean(pred_keep_t)) if pred_keep_t else None,
        "keep_q_mae": float(np.mean(pred_swap_q)) if pred_swap_q else None,
    }

    print("[explore] random-init ResNet18 GAP forward")
    seed_everything(SEED)
    model = build_model().to(device())
    z_rand = _extract_gap(model, spec)
    zr = z_rand.reshape(-1, dim)
    mu_r = zr[train_mask].mean(0)
    tc_r = zr[train_mask] - mu_r
    _, u_rand = _fit_factor(tc_r, t_train, g_train, t_mu, t_sig, q_mu, q_sig)
    energy_r = _subspace_energy(tc_r, u_rand)
    z_e = _erasure_z(zr, u_rand, mu_r, True)
    probe_t_r = _probe_fit(z_e, labels["t"], train_mask)
    probe_g_r = _probe_fit(z_e, g, train_mask)
    probe_t_r0 = _probe_fit(zr, labels["t"], train_mask)
    probe_g_r0 = _probe_fit(zr, g, train_mask)
    random_init = {
        "energy": energy_r,
        "overlap_vs_trained_Bt": _overlap_norm(u_rand, u_bt),
        "angles_vs_trained_Bt_deg": _principal_angles_deg(u_rand, u_bt),
        "baseline_probe_test_E_t": _probe_mae(probe_t_r0, labels["t"], test_mask),
        "baseline_probe_test_E_g": _probe_mae(probe_g_r0, g, test_mask),
        "erasure_probe_test_E_t": _probe_mae(probe_t_r, labels["t"], test_mask),
        "erasure_probe_test_E_g": _probe_mae(probe_g_r, g, test_mask),
        "constant_t": _mae_px(labels["t"][test_mask] - t_mu),
    }
    print("[explore] random-init 200 identity perms")
    rand_perm_probe_t = []
    for _ in range(200):
        fake_t = unique_t[rng.permutation(n_t)[train_tid]]
        _, u_p = _fit_factor(tc_r, fake_t, g_train, t_mu, t_sig, q_mu, q_sig)
        zep = _erasure_z(zr, u_p, mu_r, True)
        pt = _probe_fit(zep, labels["t"], train_mask)
        rand_perm_probe_t.append(_probe_mae(pt, labels["t"], test_mask))
    random_init["perm200_probe_E_t_mean"] = float(np.mean(rand_perm_probe_t))
    random_init["perm200_probe_E_t_p"] = _pvalue(random_init["erasure_probe_test_E_t"], rand_perm_probe_t, True)
    pcs, pc_energy = _pc_spectrum(tc_r)
    matched, _, info = _sample_matched(tc_r, 2, energy_r, rng, pcs, pc_energy, 200, 0.05)
    rand_match_t = []
    for basis in matched:
        zep = _erasure_z(zr, basis, mu_r, True)
        pt = _probe_fit(zep, labels["t"], train_mask)
        rand_match_t.append(_probe_mae(pt, labels["t"], test_mask))
    random_init["matched_info"] = info
    random_init["matched200_probe_E_t_mean"] = float(np.mean(rand_match_t)) if rand_match_t else None
    random_init["matched200_probe_E_t_p"] = (
        _pvalue(random_init["erasure_probe_test_E_t"], rand_match_t, True) if rand_match_t else None
    )

    np.savez_compressed(out / "Z_random.npz", Z=z_rand)
    payload = {
        "ols_identity_diagnostics": ols_diag,
        "per_shape_t_only": per_shape_summary,
        "perm_vs_true_Bt_angles": perm_angle_summary,
        "t_swap": swap,
        "random_init_gap": random_init,
        "judgment": "C-branch: diagnostics plus Q1/Q2/Q3 probes; no new backbone training",
    }
    dump_json(out / "summary.json", _to_py(payload))
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(angles0, bins=16, alpha=0.7, label="主角度 1")
    ax.hist(angles1, bins=16, alpha=0.7, label="主角度 2")
    ax.set_xlabel("度")
    ax.set_ylabel("shape 数")
    ax.set_title("各 shape 的 t-only 2D 相对全局 B_t")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "per_shape_angles.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist([a[0] for a in perm_angles], bins=16, alpha=0.7, label="主角度 1")
    ax.hist([a[1] for a in perm_angles], bins=16, alpha=0.7, label="主角度 2")
    ax.set_xlabel("度")
    ax.set_ylabel("次数")
    ax.set_title("identity-perm 的 B_t 相对真实 B_t")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "perm_vs_true_angles.png", dpi=140)
    plt.close(fig)
    print("[explore] ols", {k: ols_diag[k] for k in ols_diag if k not in ("ridge", "note")})
    print("[explore] random_init", {k: random_init[k] for k in random_init if k != "matched_info"})
    print("[explore] swap", swap)
    print("[explore] per-shape mean angles", per_shape_summary["mean_angle0"], per_shape_summary["mean_angle1"])
    print("[explore] perm vs true mean angles", perm_angle_summary["mean"])
    print(f"[explore] wrote {out}")
    return payload
