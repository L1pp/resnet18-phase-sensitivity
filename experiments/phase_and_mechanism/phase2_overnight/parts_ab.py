"""Part A/B: dense behavioral field and representation geometry on Phase 1.8 R18 slims."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from phase1_gap_rep.common import BATCH_SIZE, COORD_SCALE, decompose_controls, device, dump_json, images_to_tensor, setup_matplotlib_chinese
from phase1_gap_rep.train import _predict

from .ckpt import load_phase18_slim, phase18_slim_path
from .features import extract_stage_gaps
from .geometry import (
    additive_energies,
    cka,
    control_metrics,
    h_of_t,
    knn_overlap,
    local_jh,
    metric_geometry,
    pairwise_euclid,
    svd_ranks,
    tangent_principal_angle,
    tangent_rotation_summary,
)
from .protocol import N_DENSE, PHASE18_SEEDS, PHASE2_ROOT, dense_grid_px, freeze_protocol
from .render_dense import build_dense_cache, load_dense_cache

# protocol.grid_spacing lives in geometry
from .geometry import grid_spacing as _spacing

OUT = PHASE2_ROOT / "part_ab"


def _heat(arr: np.ndarray, path: Path, title: str, cmap: str = "coolwarm") -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(arr, origin="upper", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("ty index")
    ax.set_ylabel("tx index")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _mean_over_shapes(field: np.ndarray) -> np.ndarray:
    return field.reshape(field.shape[0], N_DENSE, N_DENSE, *field.shape[2:]).mean(axis=0)


def _quiver(vx: np.ndarray, vy: np.ndarray, path: Path, title: str, step: int = 2) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ys = np.arange(0, N_DENSE, step)
    xs = np.arange(0, N_DENSE, step)
    xx, yy = np.meshgrid(xs, ys)
    ax.quiver(xx, yy, vx[np.ix_(ys, xs)], vy[np.ix_(ys, xs)], angles="xy", scale_units="xy")
    ax.set_title(title)
    ax.set_xlabel("ty index")
    ax.set_ylabel("tx index")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _shape_tangent_panel(z: np.ndarray) -> Dict[str, Any]:
    """Per-shape tangent at grid center vs content-shared h(t)."""
    h_shared = h_of_t(z)
    jh_shared = local_jh(h_shared)
    c = N_DENSE // 2
    shared = np.stack([jh_shared["vx"][c, c], jh_shared["vy"][c, c]], axis=1)
    angles = []
    ratios = []
    for s in range(z.shape[0]):
        h_s = z[s] - z[s].mean(axis=0, keepdims=True)
        jh_s = local_jh(h_s)
        basis = np.stack([jh_s["vx"][c, c], jh_s["vy"][c, c]], axis=1)
        angles.append(tangent_principal_angle(shared, basis))
        ratios.append(float(jh_s["sigma2_over_sigma1"][c, c]))
    arr = np.asarray(angles, dtype=np.float64)
    mean_ang = float(arr.mean()) if arr.size else 0.0
    return {
        "n_shapes": int(z.shape[0]),
        "mean_angle_to_shared_deg": mean_ang,
        "max_angle_to_shared_deg": float(arr.max()) if arr.size else 0.0,
        "median_shape_sigma_ratio": float(np.median(ratios)) if ratios else 0.0,
        "content_shared_candidate": bool(mean_ang < 15.0),
        "shape_dependent_candidate": bool(mean_ang >= 25.0),
    }


def run_part_ab() -> Dict[str, Any]:
    proto = freeze_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    build_dense_cache()
    cache = load_dense_cache()
    images = cache["images"]  # memmap, do not copy 2.7GB
    n_s, n_g = int(images.shape[0]), int(images.shape[1])
    flat_img = images.reshape(n_s * n_g, images.shape[2], images.shape[3])
    P = cache["P"].reshape(n_s * n_g, 6)
    t_true = cache["t"].reshape(n_s * n_g, 2)
    t_grid = dense_grid_px()
    seed_rows: List[Dict[str, Any]] = []
    h_by_seed: Dict[int, Dict[str, np.ndarray]] = {}
    jac_by_seed: Dict[int, np.ndarray] = {}

    for seed in PHASE18_SEEDS:
        print(f"[phase2 A/B] seed {seed}")
        model = load_phase18_slim(seed)
        pred = _predict(model, flat_img)
        metrics, vec, pred_t, true_t = control_metrics(pred, P)
        # true_t from decompose should match t_true; use t_true (px)
        true_t = t_true
        pred_t = decompose_controls(pred)["t"] * COORD_SCALE
        vec = pred_t - true_t
        err = np.linalg.norm(vec, axis=-1)
        tx = np.abs(vec[:, 0])
        ty = np.abs(vec[:, 1])
        err_map = _mean_over_shapes(err.reshape(n_s, n_g))
        tx_map = _mean_over_shapes(tx.reshape(n_s, n_g))
        ty_map = _mean_over_shapes(ty.reshape(n_s, n_g))
        vx_map = _mean_over_shapes(vec[:, 0].reshape(n_s, n_g))
        vy_map = _mean_over_shapes(vec[:, 1].reshape(n_s, n_g))
        figdir = OUT / "figures" / f"seed_{seed}"
        _heat(err_map, figdir / "translation_error.png", f"seed {seed} |t err| px")
        _heat(tx_map, figdir / "tx_error.png", f"seed {seed} |tx err| px")
        _heat(ty_map, figdir / "ty_error.png", f"seed {seed} |ty err| px")
        _heat(vx_map, figdir / "vx_bias.png", f"seed {seed} pred_tx - true_tx")
        _heat(vy_map, figdir / "vy_bias.png", f"seed {seed} pred_ty - true_ty")
        _quiver(vx_map, vy_map, figdir / "t_error_vector_field.png", f"seed {seed} predicted_t - true_t")

        # behavioral Jacobian of mean predicted t field
        pred_t_grid = pred_t.reshape(n_s, n_g, 2).mean(axis=0).reshape(N_DENSE, N_DENSE, 2)
        sp = _spacing()
        jxx = np.zeros((N_DENSE, N_DENSE))
        jxy = np.zeros((N_DENSE, N_DENSE))
        jyx = np.zeros((N_DENSE, N_DENSE))
        jyy = np.zeros((N_DENSE, N_DENSE))
        jxx[1:-1] = (pred_t_grid[2:, :, 0] - pred_t_grid[:-2, :, 0]) / (2 * sp)
        jxy[:, 1:-1] = (pred_t_grid[:, 2:, 0] - pred_t_grid[:, :-2, 0]) / (2 * sp)
        jyx[1:-1] = (pred_t_grid[2:, :, 1] - pred_t_grid[:-2, :, 1]) / (2 * sp)
        jyy[:, 1:-1] = (pred_t_grid[:, 2:, 1] - pred_t_grid[:, :-2, 1]) / (2 * sp)
        det = jxx * jyy - jxy * jyx
        sigma1 = np.zeros((N_DENSE, N_DENSE))
        sigma2 = np.zeros((N_DENSE, N_DENSE))
        cond = np.zeros((N_DENSE, N_DENSE))
        for i in range(N_DENSE):
            for j in range(N_DENSE):
                s = np.linalg.svd(np.array([[jxx[i, j], jxy[i, j]], [jyx[i, j], jyy[i, j]]], dtype=np.float64), compute_uv=False)
                sigma1[i, j] = float(s[0]) if s.size else 0.0
                sigma2[i, j] = float(s[1]) if s.size > 1 else 0.0
                cond[i, j] = float(s[0] / s[1]) if s.size > 1 and s[1] > 1e-12 else np.inf
        _heat(jxx, figdir / "j_dtx_dtx.png", "d tx_pred / d tx")
        _heat(jxy, figdir / "j_dtx_dty.png", "d tx_pred / d ty")
        _heat(jyx, figdir / "j_dty_dtx.png", "d ty_pred / d tx")
        _heat(jyy, figdir / "j_dty_dty.png", "d ty_pred / d ty")
        _heat(det, figdir / "j_det.png", "det J")
        _heat(sigma1, figdir / "j_sigma1.png", "J_out sigma1")
        _heat(sigma2, figdir / "j_sigma2.png", "J_out sigma2")
        finite_cond = np.where(np.isfinite(cond), cond, np.nan)
        _heat(finite_cond, figdir / "j_cond.png", "J_out condition number")

        print(f"[phase2 A/B] seed {seed} extract layers")
        feats = extract_stage_gaps(model, flat_img)
        layer_out = {}
        h_by_seed[seed] = {}
        for name, zflat in feats.items():
            if zflat.size == 0 or zflat.shape[1] == 0:
                continue
            z = zflat.reshape(n_s, n_g, -1)
            en = additive_energies(z)
            h = h_of_t(z)
            jh = local_jh(h)
            rot = tangent_rotation_summary(jh)
            ranks = svd_ranks(h)
            met = metric_geometry(h, t_grid)
            layer_out[name] = {"energy": en, "svd": ranks, "tangent": rot, "metric": met}
            h_by_seed[seed][name] = h
            if name == "gap":
                _heat(jh["sigma1"], figdir / "h_sigma1.png", "h(t) sigma1")
                _heat(jh["sigma2"], figdir / "h_sigma2.png", "h(t) sigma2")
                _heat(jh["sigma2_over_sigma1"], figdir / "h_sigma_ratio.png", "h(t) sigma2/sigma1")
                finite_hcond = np.where(np.isfinite(jh["condition"]), jh["condition"], np.nan)
                _heat(finite_hcond, figdir / "h_condition.png", "h(t) local condition")
                layer_out[name]["shape_tangent"] = _shape_tangent_panel(z)
        row = {
            "seed": seed,
            "slim": str(phase18_slim_path(seed)),
            "behavioral": metrics,
            "jacobian_mean": {
                "dtx_dtx": float(np.nanmean(jxx)),
                "dtx_dty": float(np.nanmean(jxy)),
                "dty_dtx": float(np.nanmean(jyx)),
                "dty_dty": float(np.nanmean(jyy)),
                "det": float(np.nanmean(det)),
                "sigma1": float(np.nanmean(sigma1)),
                "sigma2": float(np.nanmean(sigma2)),
                "cond": float(np.nanmean(finite_cond)),
            },
            "layers": layer_out,
        }
        seed_rows.append(row)
        dump_json(OUT / "tables" / f"seed_{seed}.json", row)
        jac_by_seed[seed] = np.stack([jxx, jxy, jyx, jyy], axis=-1)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # B6 cross-seed: Gram / distances / kNN / behavioral Jacobian. Never compare raw axes.
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = a.reshape(-1).argsort().argsort().astype(np.float64)
        rb = b.reshape(-1).argsort().argsort().astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    cross = {}
    names = sorted(set.intersection(*[set(h_by_seed[s]) for s in PHASE18_SEEDS]))
    for name in names:
        hs = [h_by_seed[s][name] for s in PHASE18_SEEDS]
        dists = [pairwise_euclid(h) for h in hs]
        iu = np.triu_indices(dists[0].shape[0], k=1)
        cross[name] = {
            "cka_10_11": cka(hs[0], hs[1]),
            "cka_10_12": cka(hs[0], hs[2]),
            "cka_11_12": cka(hs[1], hs[2]),
            "dist_spearman_10_11": _spearman(dists[0][iu], dists[1][iu]),
            "dist_spearman_10_12": _spearman(dists[0][iu], dists[2][iu]),
            "dist_spearman_11_12": _spearman(dists[1][iu], dists[2][iu]),
            "knn_10_11": knn_overlap(dists[0], dists[1]),
            "knn_10_12": knn_overlap(dists[0], dists[2]),
            "knn_11_12": knn_overlap(dists[1], dists[2]),
        }
    jac_keys = list(PHASE18_SEEDS)
    jflat = {s: jac_by_seed[s].reshape(-1) for s in jac_keys}
    cross["behavioral_jacobian"] = {
        "spearman_10_11": _spearman(jflat[jac_keys[0]], jflat[jac_keys[1]]),
        "spearman_10_12": _spearman(jflat[jac_keys[0]], jflat[jac_keys[2]]),
        "spearman_11_12": _spearman(jflat[jac_keys[1]], jflat[jac_keys[2]]),
        "pearson_10_11": float(np.corrcoef(jflat[jac_keys[0]], jflat[jac_keys[1]])[0, 1]),
        "pearson_10_12": float(np.corrcoef(jflat[jac_keys[0]], jflat[jac_keys[2]])[0, 1]),
        "pearson_11_12": float(np.corrcoef(jflat[jac_keys[1]], jflat[jac_keys[2]])[0, 1]),
    }
    shape_dep = {int(row["seed"]): row["layers"].get("gap", {}).get("shape_tangent") for row in seed_rows}
    summary = {
        "protocol_hash": proto["protocol_hash"],
        "seeds": seed_rows,
        "cross_seed": cross,
        "b7_shape_tangent": shape_dep,
        "note": "Part A/B on Phase 1.8 ResNet18 slim; not G64.",
    }
    dump_json(OUT / "tables" / "summary.json", summary)
    lines = [
        "# Phase 2 Part A/B interim (local)",
        "",
        f"protocol `{proto['protocol_hash']}`。Phase 1.8 三 seed slim，不训练。",
        "",
        "| seed | t MAE px | tx | ty | q | det J mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        b = row["behavioral"]
        lines.append(
            f"| {row['seed']} | {b['t_mae_px']:.3f} | {b['tx_mae_px']:.3f} | {b['ty_mae_px']:.3f} | "
            f"{b['q_mae_px']:.3f} | {row['jacobian_mean']['det']:.3f} |"
        )
    gap0 = seed_rows[0]["layers"].get("gap", {})
    if gap0:
        lines += [
            "",
            "## GAP h(t) (seed 20260810)",
            f"- SVD ranks 80/90/95/99: {gap0['svd'].get('rank_80')} / {gap0['svd'].get('rank_90')} / {gap0['svd'].get('rank_95')} / {gap0['svd'].get('rank_99')}",
            f"- median sigma2/sigma1: {gap0['tangent']['median_sigma2_over_sigma1']:.3f}",
            f"- neighbor tangent angle mean: {gap0['tangent']['neighbor_angle_mean_deg']:.2f} deg",
            f"- curved candidate: {gap0['tangent']['curved_manifold_candidate']}",
            f"- metric pearson: {gap0['metric']['pearson_euclid']:.3f}",
        ]
    b7 = seed_rows[0]["layers"].get("gap", {}).get("shape_tangent") or {}
    if b7:
        lines += [
            "",
            "## B7 shape vs shared tangent (seed 20260810 GAP)",
            f"- mean angle to shared: {b7.get('mean_angle_to_shared_deg'):.2f} deg",
            f"- content-shared candidate: {b7.get('content_shared_candidate')}",
            f"- shape-dependent candidate: {b7.get('shape_dependent_candidate')}",
        ]
    gap_cross = cross.get("gap", {})
    if gap_cross:
        lines += [
            "",
            "## B6 cross-seed GAP",
            f"- CKA 10/11 / 10/12 / 11/12: {gap_cross.get('cka_10_11'):.3f} / {gap_cross.get('cka_10_12'):.3f} / {gap_cross.get('cka_11_12'):.3f}",
            f"- dist Spearman 10/11: {gap_cross.get('dist_spearman_10_11'):.3f}",
            f"- kNN overlap 10/11: {gap_cross.get('knn_10_11'):.3f}",
        ]
    (OUT / "INTERIM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[phase2 A/B] wrote {OUT}")
    return summary


if __name__ == "__main__":
    run_part_ab()
