"""Phase 1.5b: compare translation-subspace cuts on frozen GAP features."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .analyze import _labels, _load_features, _load_split_codes
from .audit import (
    ENERGY_TOL,
    N_NULL,
    SPLIT_NAMES,
    _all_functional,
    _erasure_z,
    _frobenius_alignment,
    _haar_basis,
    _histogram,
    _linear_probe,
    _null_stats,
    _pack_errors,
    _pc_spectrum,
    _predict_p,
    _principal_angles_deg,
    _pvalue,
    _qr_basis,
    _readouts,
    _sample_energy_matched,
    _subspace_energy,
    _svd_components,
    _to_py,
    _train_effects,
)
from .common import (
    COORD_SCALE,
    SEED,
    decompose_controls,
    dirs,
    dump_json,
    fingerprint,
    profile_spec,
    run_dirs,
    setup_matplotlib_chinese,
    train_run_spec,
    write_run_metadata,
)
from .generate_data import load_split

KEY_NULLS = ("T_k1", "T_k2", "B_t", "W_t", "PC1")
N_NULL_OTHER = 400
N_BOOT = 1000


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    qa, qb = _qr_basis(a), _qr_basis(b)
    return float(np.sum((qa.T @ qb) ** 2))


def _load_shape_table(profile: str) -> Dict[str, np.ndarray]:
    path = dirs(profile)["tables"] / "shapes.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n = len(rows)
    out = {
        "theta_deg": np.zeros(n),
        "chord_px": np.zeros(n),
        "curve_sign": np.zeros(n),
        "alpha": np.zeros(n),
        "half_extent_px": np.zeros(n),
        "realized_curve_px": np.zeros(n),
    }
    for row in rows:
        sid = int(row["shape_id"])
        for key in out:
            out[key][sid] = float(row[key])
    return out


def _factor_bases(
    z_flat: np.ndarray, t: np.ndarray, g: np.ndarray, mask: np.ndarray, mu: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    t_sel, q_sel = t[mask], g[mask]
    zc = z_flat[mask] - mu
    t_mu, t_sig = t_sel.mean(0), t_sel.std(0) + 1e-12
    q_mu, q_sig = q_sel.mean(0), q_sel.std(0) + 1e-12
    design = np.concatenate([(t_sel - t_mu) / t_sig, (q_sel - q_mu) / q_sig], axis=1)
    beta, *_ = np.linalg.lstsq(design, zc, rcond=None)
    return _qr_basis(beta[:2].T), _qr_basis(beta[2:].T)


def _mae_tq(pred: np.ndarray, t_true: np.ndarray, q_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = pred.reshape(-1, 3, 2)
    t_hat = pts.mean(axis=1)
    q_hat = (pts - t_hat[:, None, :]).reshape(-1, 6)
    e_t = np.mean(np.abs(t_hat - t_true), axis=1) * COORD_SCALE
    e_q = np.mean(np.abs(q_hat - q_true), axis=1) * COORD_SCALE
    return e_t, e_q


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = N_BOOT) -> Dict[str, float]:
    means = np.empty(n, dtype=np.float64)
    size = len(values)
    for i in range(n):
        means[i] = float(values[rng.integers(0, size, size)].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def _collect_nulls(
    name: str,
    basis: np.ndarray,
    z_test: np.ndarray,
    mu: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    t_test: np.ndarray,
    q_test: np.ndarray,
    w_t: np.ndarray,
    w_q: np.ndarray,
    train_centered: np.ndarray,
    rng: np.random.Generator,
    pcs: np.ndarray,
    pc_energy: np.ndarray,
    n_null: int,
) -> Dict[str, Any]:
    rank = int(basis.shape[1])
    target = _subspace_energy(train_centered, basis)
    haar_et, haar_eq = [], []
    for _ in range(n_null):
        et, eq, _, _ = _null_stats(
            z_test, _haar_basis(z_test.shape[1], rank, rng), mu, weight, bias, t_test, q_test, w_t, w_q
        )
        haar_et.append(et)
        haar_eq.append(eq)
    matched, energies, info = _sample_energy_matched(
        train_centered, rank, target, rng, n=n_null, pcs=pcs, pc_energy=pc_energy
    )
    match_et, match_eq = [], []
    for item in matched:
        et, eq, _, _ = _null_stats(z_test, item, mu, weight, bias, t_test, q_test, w_t, w_q)
        match_et.append(et)
        match_eq.append(eq)
    print(f"[cuts] {name} nulls n={n_null} matched_rel={info['rel_error']:.3f} energy={target:.3f}")
    return {
        "n": n_null,
        "target_energy": target,
        "matched_info": info,
        "haar_mean_E_t": float(np.mean(haar_et)),
        "haar_mean_E_q": float(np.mean(haar_eq)),
        "matched_mean_E_t": float(np.mean(match_et)),
        "matched_mean_E_q": float(np.mean(match_eq)),
        "matched_std_E_t": float(np.std(match_et)),
        "matched_std_E_q": float(np.std(match_eq)),
        "matched_E_t": match_et,
        "matched_E_q": match_eq,
        "histogram": {"E_t": _histogram(match_et), "E_q": _histogram(match_eq)},
    }


def _plot_cut_bars(path: Path, rows: List[Mapping[str, Any]]) -> str:
    plt = setup_matplotlib_chinese()
    names = [row["name"] for row in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, key, title in ((axes[0], "E_t", "删该切法后的位置误差（test）"), (axes[1], "E_q", "删该切法后的形状误差（test）")):
        targeted = [row["test"][key] for row in rows]
        matched = [row["nulls"]["matched_mean_" + key] if row["nulls"] else np.nan for row in rows]
        ax.bar(x - 0.18, targeted, 0.36, label="定向删除")
        ax.bar(x + 0.18, matched, 0.36, label="能量匹配随机均值")
        ax.set_ylabel("px")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_pc1(path: Path, corr: Mapping[str, float]) -> str:
    plt = setup_matplotlib_chinese()
    keys = list(corr.keys())
    vals = [corr[key] for key in keys]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(keys, vals)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("与 PC1 分数的相关")
    ax.set_title("第一主成分在解释什么")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_angles(path: Path, names: List[str], bases: Mapping[str, np.ndarray]) -> str:
    plt = setup_matplotlib_chinese()
    n = len(names)
    mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            angles = _principal_angles_deg(bases[a], bases[b])
            mat[i, j] = float(np.mean(angles))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(mat, cmap="viridis_r", vmin=0.0, vmax=90.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("切法之间的平均主角度（度，越小越像）")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _write_cut_table(path: Path, rows: List[Mapping[str, Any]]) -> None:
    fields = [
        "name",
        "rank",
        "energy",
        "overlap_PC1",
        "overlap_PCkt",
        "overlap_Bt",
        "test_E_t",
        "test_E_q",
        "test_E_t_ci95",
        "test_E_q_ci95",
        "matched_E_t",
        "matched_E_q",
        "p_E_t",
        "p_E_q",
        "F_t",
        "F_q",
        "W_t_fro",
        "W_q_fro",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            nulls = row["nulls"] or {}
            boot = row["bootstrap_test"]
            writer.writerow(
                {
                    "name": row["name"],
                    "rank": row["rank"],
                    "energy": f"{row['energy']:.4f}",
                    "overlap_PC1": f"{row['overlap_PC1']:.4f}",
                    "overlap_PCkt": f"{row['overlap_top_rank']:.4f}",
                    "overlap_Bt": f"{row['overlap_Bt']:.4f}",
                    "test_E_t": f"{row['test']['E_t']:.4f}",
                    "test_E_q": f"{row['test']['E_q']:.4f}",
                    "test_E_t_ci95": f"{boot['E_t']['ci95_lo']:.3f}-{boot['E_t']['ci95_hi']:.3f}",
                    "test_E_q_ci95": f"{boot['E_q']['ci95_lo']:.3f}-{boot['E_q']['ci95_hi']:.3f}",
                    "matched_E_t": "" if not nulls else f"{nulls['matched_mean_E_t']:.4f}",
                    "matched_E_q": "" if not nulls else f"{nulls['matched_mean_E_q']:.4f}",
                    "p_E_t": "" if not nulls else f"{row['p_E_t']:.4g}",
                    "p_E_q": "" if not nulls else f"{row['p_E_q']:.4g}",
                    "F_t": f"{row['functional_test']['F_t_from_t']:.4f}",
                    "F_q": f"{row['functional_test']['F_q_from_t']:.4f}",
                    "W_t_fro": f"{row['frobenius']['W_t']:.4f}",
                    "W_q_fro": f"{row['frobenius']['W_q']:.4f}",
                }
            )


def cuts_run(run_name: str) -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    z, weight, bias = _load_features(run.name, spec.name)
    split_grid = _load_split_codes(spec.name)
    labels_grid = _labels(spec.name)
    shapes = _load_shape_table(spec.name)
    n_s, n_t, dim = z.shape
    z_flat = z.reshape(-1, dim)
    split_flat = split_grid.reshape(-1)
    sid = np.repeat(np.arange(n_s), n_t)
    labels_flat = {
        "P": labels_grid["P"].reshape(-1, 6),
        "Q": labels_grid["Q"].reshape(-1, 6),
        "t": labels_grid["t"].reshape(-1, 2),
    }
    masks = {name: split_flat == index for index, name in enumerate(SPLIT_NAMES)}
    train_mask = masks["train"]
    test_mask = masks["test"]
    mu = z_flat[train_mask].mean(axis=0)
    train_centered = z_flat[train_mask] - mu
    w_t, w_q = _readouts(weight)
    g_true = decompose_controls(labels_flat["P"])["geometry_comp"]

    t_effects = _train_effects(z, split_grid, axis=1, mu=mu)
    g_effects = _train_effects(z, split_grid, axis=0, mu=mu)
    t_basis, _, _ = _svd_components(t_effects)
    g_basis, _, _ = _svd_components(g_effects)
    pcs, pc_energy = _pc_spectrum(train_centered)
    u_bt, u_bq = _factor_bases(z_flat, labels_flat["t"], g_true, train_mask, mu)
    u_bt_val, _ = _factor_bases(z_flat, labels_flat["t"], g_true, masks["val"], z_flat[masks["val"]].mean(0))
    u_bt_test, _ = _factor_bases(z_flat, labels_flat["t"], g_true, test_mask, z_flat[test_mask].mean(0))

    bases = {
        "T_k1": t_basis[:, :1],
        "T_k2": t_basis[:, :2],
        "T_k3": t_basis[:, :3],
        "B_t": u_bt,
        "B_q": u_bq,
        "PC1": pcs[:, :1],
        "PC2": pcs[:, :2],
        "PC3": pcs[:, :3],
        "W_t": _qr_basis(w_t.T),
        "W_q": _qr_basis(w_q.T),
        "G_k1": g_basis[:, :1],
    }

    z_test = z_flat[test_mask]
    t_test = labels_flat["t"][test_mask]
    q_test = labels_flat["Q"][test_mask]
    rng = np.random.default_rng(SEED)
    pred0 = _predict_p(z_flat, weight, bias)
    baseline = {name: _pack_errors(pred0, labels_flat, mask) for name, mask in masks.items()}

    rows: List[Dict[str, Any]] = []
    for name, basis in bases.items():
        print(f"[cuts:{run.name}] erasure {name} rank={basis.shape[1]}")
        pred = _predict_p(_erasure_z(z_flat, basis, mu, True), weight, bias)
        e_t_cell, e_q_cell = _mae_tq(pred[test_mask], t_test, q_test)
        n_null = N_NULL if name in KEY_NULLS else N_NULL_OTHER
        do_null = name in KEY_NULLS or name in ("T_k3", "B_q", "PC3")
        packed = {split: _pack_errors(pred, labels_flat, mask) for split, mask in masks.items()}
        nulls = None
        p_t = p_q = None
        if do_null:
            nulls = _collect_nulls(
                name,
                basis,
                z_test,
                mu,
                weight,
                bias,
                t_test,
                q_test,
                w_t,
                w_q,
                train_centered,
                rng,
                pcs,
                pc_energy,
                n_null,
            )
            p_t = _pvalue(packed["test"]["E_t"], nulls["matched_E_t"])
            p_q = _pvalue(packed["test"]["E_q"], nulls["matched_E_q"])
            nulls.pop("matched_E_t", None)
            nulls.pop("matched_E_q", None)
        probe = {
            "t": _linear_probe(_erasure_z(z_flat, basis, mu, True), labels_flat["t"], train_mask, masks),
            "q_geom": _linear_probe(_erasure_z(z_flat, basis, mu, True), g_true, train_mask, masks),
        }
        rows.append(
            {
                "name": name,
                "rank": int(basis.shape[1]),
                "energy": _subspace_energy(train_centered, basis),
                "overlap_PC1": _overlap(basis, bases["PC1"]),
                "overlap_top_rank": _overlap(basis, pcs[:, : basis.shape[1]]),
                "overlap_Bt": _overlap(basis, u_bt),
                "overlap_Wt": _overlap(basis, bases["W_t"]),
                "train": packed["train"],
                "val": packed["val"],
                "test": packed["test"],
                "functional_test": _all_functional(z_flat[test_mask] - mu, w_t, w_q, basis, bases["B_q"]),
                "frobenius": {"W_t": _frobenius_alignment(w_t, basis), "W_q": _frobenius_alignment(w_q, basis)},
                "bootstrap_test": {"E_t": _bootstrap_mean(e_t_cell, rng), "E_q": _bootstrap_mean(e_q_cell, rng)},
                "probe_after": probe,
                "nulls": nulls,
                "p_E_t": p_t,
                "p_E_q": p_q,
            }
        )

    # Ink / PC1 interpretation
    train_data = load_split(spec, "train")
    images = np.asarray(train_data["images"], dtype=np.float64)
    if images.ndim == 4:
        ink = images.mean(axis=(1, 2, 3))
    else:
        ink = images.mean(axis=(1, 2))
    train_sid = train_data["shape_id"].astype(np.int64)
    train_tid = train_data["translation_id"].astype(np.int64)
    z_train_img = z[train_sid, train_tid]
    pc1_score = (z_train_img - mu) @ pcs[:, 0]
    t_train = np.asarray(train_data["t"], dtype=np.float64)
    t_norm = np.linalg.norm(t_train, axis=1)
    pc1_corr = {
        "ink": float(np.corrcoef(pc1_score, ink)[0, 1]),
        "t_norm": float(np.corrcoef(pc1_score, t_norm)[0, 1]),
        "tx": float(np.corrcoef(pc1_score, t_train[:, 0])[0, 1]),
        "ty": float(np.corrcoef(pc1_score, t_train[:, 1])[0, 1]),
    }
    g_pc1 = g_effects @ pcs[:, 0]
    for key, values in shapes.items():
        pc1_corr[f"shape_{key}"] = float(np.corrcoef(g_pc1, values)[0, 1])
    ink_by_shape = np.array([ink[train_sid == i].mean() if np.any(train_sid == i) else np.nan for i in range(n_s)])
    pc1_corr["shape_ink"] = float(np.corrcoef(g_pc1, ink_by_shape)[0, 1])

    stability = {
        "Bt_train_vs_val_deg": _principal_angles_deg(u_bt, u_bt_val),
        "Bt_train_vs_test_deg": _principal_angles_deg(u_bt, u_bt_test),
        "Bt_vs_Wt_deg": _principal_angles_deg(u_bt, bases["W_t"]),
        "Bt_vs_Tk1_deg": _principal_angles_deg(u_bt, bases["T_k1"]),
        "Bt_vs_PC1_deg": _principal_angles_deg(u_bt, bases["PC1"]),
        "Tk1_vs_PC1_deg": _principal_angles_deg(bases["T_k1"], bases["PC1"]),
        "Tk3_vs_PC3_deg": _principal_angles_deg(bases["T_k3"], bases["PC3"]),
        "Wt_vs_Tk1_deg": _principal_angles_deg(bases["W_t"], bases["T_k1"]),
    }

    out_dir = run_dirs(run.name)["root"] / "audit" / "cuts"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run": run.name,
        "fingerprint": fingerprint(spec),
        "note": (
            "Phase 1.5b cut comparison. All bases from train except B_t val/test used only for stability. "
            "Does not retrain. Does not write factorial/."
        ),
        "baseline": baseline,
        "pc_energy_top8": pc_energy[:8].tolist(),
        "pc1_correlations": pc1_corr,
        "stability_deg": stability,
        "cuts": rows,
        "n_null_key": N_NULL,
        "n_null_other": N_NULL_OTHER,
        "n_bootstrap": N_BOOT,
    }
    figures = [
        _plot_cut_bars(fig_dir / "cut_erasure_bars.png", rows),
        _plot_pc1(fig_dir / "pc1_correlations.png", pc1_corr),
        _plot_angles(fig_dir / "cut_angles.png", ["T_k1", "T_k2", "T_k3", "B_t", "PC1", "PC3", "W_t", "G_k1"], bases),
    ]
    summary["figures"] = figures
    _write_cut_table(tab_dir / "cuts.csv", rows)
    dump_json(out_dir / "summary.json", _to_py(summary))
    write_run_metadata(run, "cuts", {"out_dir": str(out_dir)})
    print(f"[cuts:{run.name}] wrote {out_dir}")
    return summary
