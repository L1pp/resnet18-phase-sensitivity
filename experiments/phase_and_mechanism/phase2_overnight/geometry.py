"""Representation geometry: additive energy, local Jacobian of h(t), SVD, distances, CKA."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from phase1_gap_rep.common import COORD_SCALE, decompose_controls

from .protocol import N_DENSE, T_HIGH_PX, T_LOW_PX


def control_metrics(pred: np.ndarray, target: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dp = decompose_controls(pred)
    dt = decompose_controls(target)
    diff = pred - target
    t_err = np.linalg.norm(dp["t"] - dt["t"], axis=-1) * COORD_SCALE
    q_err = np.mean(np.abs(dp["q"] - dt["q"]), axis=(-1, -2)) * COORD_SCALE
    tx = np.abs(dp["t"][..., 0] - dt["t"][..., 0]) * COORD_SCALE
    ty = np.abs(dp["t"][..., 1] - dt["t"][..., 1]) * COORD_SCALE
    vec = (dp["t"] - dt["t"]) * COORD_SCALE
    metrics = {
        "control_mae_px": float(np.mean(np.abs(diff)) * COORD_SCALE),
        "control_rmse_px": float(np.sqrt(np.mean(diff**2)) * COORD_SCALE),
        "control_max_px": float(np.max(np.abs(diff)) * COORD_SCALE),
        "t_mae_px": float(np.mean(t_err)),
        "t_rmse_px": float(np.sqrt(np.mean(t_err**2))),
        "t_max_px": float(np.max(t_err)),
        "tx_mae_px": float(np.mean(tx)),
        "ty_mae_px": float(np.mean(ty)),
        "q_mae_px": float(np.mean(q_err)),
        "n": int(pred.shape[0]),
    }
    return metrics, vec, dp["t"] * COORD_SCALE, dt["t"] * COORD_SCALE


def additive_energies(z: np.ndarray) -> Dict[str, float]:
    """z: (n_shape, n_t, d)."""
    z = np.asarray(z, dtype=np.float64)
    mu = z.mean(axis=(0, 1), keepdims=True)
    g = z.mean(axis=1, keepdims=True) - mu
    h = z.mean(axis=0, keepdims=True) - mu
    r = z - mu - g - h
    def en(x):
        return float(np.mean(np.sum(x**2, axis=-1)))
    e_g, e_h, e_r = en(g), en(h), en(r)
    total = e_g + e_h + e_r
    return {
        "shape_energy": e_g,
        "position_energy": e_h,
        "interaction_energy": e_r,
        "total": total,
        "shape_frac": e_g / total if total else 0.0,
        "position_frac": e_h / total if total else 0.0,
        "interaction_frac": e_r / total if total else 0.0,
        "dim": int(z.shape[-1]),
    }


def h_of_t(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    mu = z.mean(axis=(0, 1), keepdims=True)
    return (z.mean(axis=0) - mu[0]).astype(np.float64)


def grid_spacing(n_dense: int = N_DENSE) -> float:
    return (T_HIGH_PX - T_LOW_PX) / (n_dense - 1)


def pairwise_euclid(x: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n) without materializing (n, n, d)."""
    x = np.asarray(x, dtype=np.float64)
    gram = x @ x.T
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


def infer_grid_n(n_t: int) -> int:
    side = int(round(np.sqrt(n_t)))
    if side * side != n_t:
        raise ValueError(f"n_t={n_t} is not a square grid")
    return side


def local_jh(h: np.ndarray, n_dense: int | None = None) -> Dict[str, np.ndarray]:
    """h: (n_t, d) in row-major (tx, ty) of an n_dense × n_dense grid."""
    d = h.shape[-1]
    n_dense = int(n_dense or infer_grid_n(h.shape[0]))
    h2 = h.reshape(n_dense, n_dense, d)
    sp = grid_spacing(n_dense)
    vx = np.zeros_like(h2)
    vy = np.zeros_like(h2)
    vx[1:-1] = (h2[2:] - h2[:-2]) / (2.0 * sp)
    vx[0] = (h2[1] - h2[0]) / sp
    vx[-1] = (h2[-1] - h2[-2]) / sp
    vy[:, 1:-1] = (h2[:, 2:] - h2[:, :-2]) / (2.0 * sp)
    vy[:, 0] = (h2[:, 1] - h2[:, 0]) / sp
    vy[:, -1] = (h2[:, -1] - h2[:, -2]) / sp
    # Jacobian in feature space is (d, 2); report SVD of [vx, vy] as d x 2
    sigma1 = np.zeros((n_dense, n_dense))
    sigma2 = np.zeros((n_dense, n_dense))
    ratio = np.zeros((n_dense, n_dense))
    cond = np.zeros((n_dense, n_dense))
    for i in range(n_dense):
        for j in range(n_dense):
            mat = np.stack([vx[i, j], vy[i, j]], axis=1)  # d x 2
            s = np.linalg.svd(mat, compute_uv=False)
            s1 = float(s[0]) if s.size else 0.0
            s2 = float(s[1]) if s.size > 1 else 0.0
            sigma1[i, j] = s1
            sigma2[i, j] = s2
            ratio[i, j] = s2 / s1 if s1 > 1e-12 else 0.0
            cond[i, j] = s1 / s2 if s2 > 1e-12 else np.inf
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma2_over_sigma1": ratio,
        "condition": cond,
        "vx": vx,
        "vy": vy,
    }


def tangent_principal_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Principal angle between two d x 2 bases."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.arccos(min(s.min(), 1.0))))


def tangent_rotation_summary(jh: Mapping[str, np.ndarray]) -> Dict[str, float]:
    vx, vy = jh["vx"], jh["vy"]
    n_dense = int(vx.shape[0])
    angles = []
    for i in range(n_dense):
        for j in range(n_dense):
            t0 = np.stack([vx[i, j], vy[i, j]], axis=1)
            if i + 1 < n_dense:
                t1 = np.stack([vx[i + 1, j], vy[i + 1, j]], axis=1)
                angles.append(tangent_principal_angle(t0, t1))
            if j + 1 < n_dense:
                t1 = np.stack([vx[i, j + 1], vy[i, j + 1]], axis=1)
                angles.append(tangent_principal_angle(t0, t1))
    arr = np.asarray(angles, dtype=np.float64)
    far = tangent_principal_angle(
        np.stack([vx[0, 0], vy[0, 0]], axis=1),
        np.stack([vx[-1, -1], vy[-1, -1]], axis=1),
    )
    med = float(np.median(jh["sigma2_over_sigma1"]))
    return {
        "neighbor_angle_mean_deg": float(arr.mean()) if arr.size else 0.0,
        "neighbor_angle_median_deg": float(np.median(arr)) if arr.size else 0.0,
        "neighbor_angle_max_deg": float(arr.max()) if arr.size else 0.0,
        "far_corner_angle_deg": far,
        "median_sigma2_over_sigma1": med,
        "curved_manifold_candidate": bool(med >= 0.25 and float(arr.mean()) >= 8.0),
    }


def svd_ranks(h: np.ndarray, fracs: Sequence[float] = (0.80, 0.90, 0.95, 0.99)) -> Dict[str, Any]:
    h = np.asarray(h, dtype=np.float64)
    hc = h - h.mean(axis=0, keepdims=True)
    s = np.linalg.svd(hc, compute_uv=False)
    energy = s**2
    total = float(energy.sum()) or 1.0
    cume = np.cumsum(energy) / total
    out = {"singular_values_top20": s[:20].tolist(), "n": int(h.shape[0]), "dim": int(h.shape[1])}
    for f in fracs:
        out[f"rank_{int(f * 100)}"] = int(np.searchsorted(cume, f) + 1)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def metric_geometry(h: np.ndarray, t_px: np.ndarray) -> Dict[str, float]:
    h = np.asarray(h, dtype=np.float64)
    t_px = np.asarray(t_px, dtype=np.float64)
    dz = pairwise_euclid(h)
    dt = pairwise_euclid(t_px)
    dman = np.abs(t_px[:, None, :] - t_px[None, :, :]).sum(-1)
    dx = np.abs(t_px[:, None, 0] - t_px[None, :, 0])
    dy = np.abs(t_px[:, None, 1] - t_px[None, :, 1])
    sep = dx + dy
    iu = np.triu_indices(h.shape[0], k=1)
    a, b = dz[iu], dt[iu]
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = _spearman(a, b)
    n_dense = infer_grid_n(h.shape[0])
    local = dt <= (2.5 * grid_spacing(n_dense))
    local[np.diag_indices(h.shape[0])] = False
    if local.any():
        lp = float(np.corrcoef(dz[local], dt[local])[0, 1])
    else:
        lp = float("nan")
    knn = 8
    preserve = []
    for i in range(h.shape[0]):
        true_nn = np.argsort(dt[i])[1 : knn + 1]
        feat_nn = np.argsort(dz[i])[1 : knn + 1]
        preserve.append(len(set(true_nn.tolist()) & set(feat_nn.tolist())) / knn)
    return {
        "pearson_euclid": pearson,
        "spearman_euclid": spearman,
        "pearson_manhattan": float(np.corrcoef(a, dman[iu])[0, 1]),
        "pearson_separable_xy": float(np.corrcoef(a, sep[iu])[0, 1]),
        "local_pearson": lp,
        "knn8_preservation": float(np.mean(preserve)),
        "distortion_mean": float(np.mean(np.abs(a / (b + 1e-8) - np.median(a / (b + 1e-8))))),
    }


def cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xtx = x @ x.T
    yty = y @ y.T
    hsic = float(np.sum(xtx * yty))
    n1 = float(np.sum(xtx * xtx))
    n2 = float(np.sum(yty * yty))
    return hsic / np.sqrt(n1 * n2 + 1e-12)


def knn_overlap(dx: np.ndarray, dy: np.ndarray, k: int = 8) -> float:
    acc = []
    for i in range(dx.shape[0]):
        a = set(np.argsort(dx[i])[1 : k + 1].tolist())
        b = set(np.argsort(dy[i])[1 : k + 1].tolist())
        acc.append(len(a & b) / k)
    return float(np.mean(acc))
