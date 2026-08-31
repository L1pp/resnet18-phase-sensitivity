"""Affine latent translation operators: ridge + reduced-rank. Numpy only."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .protocol import REDUCED_RANKS, RIDGE_ALPHAS


def apply_affine(z: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    return z @ matrix.T + bias


def compose_affine(
    m2: np.ndarray, b2: np.ndarray, m1: np.ndarray, b1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(M2, b2) ∘ (M1, b1) = (M2 M1, M2 b1 + b2)."""
    m2 = np.asarray(m2, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    return m2 @ m1, m2 @ b1 + b2


def normalized_error(pred: np.ndarray, target: np.ndarray) -> float:
    """E = E|pred-target|^2 / E|target - mean(target)|^2."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    resid = pred - target
    num = float(np.mean(np.sum(resid * resid, axis=-1)))
    centered = target - target.mean(axis=0, keepdims=True)
    den = float(np.mean(np.sum(centered * centered, axis=-1)))
    if den < 1e-12:
        return 0.0 if num < 1e-12 else float("inf")
    return num / den


def mean_sq_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    resid = pred - target
    return float(np.mean(np.sum(resid * resid, axis=-1)))


def truncate_rank(matrix: np.ndarray, rank: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if rank in (None, "full") or int(rank) >= min(matrix.shape):
        return matrix
    rank_i = int(rank)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank_i] * s[:rank_i]) @ vt[:rank_i]


def fit_affine_ridge(z: np.ndarray, zp: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fit zp ≈ M z + b. Regularize M only. z, zp: (n, d)."""
    z = np.asarray(z, dtype=np.float64)
    zp = np.asarray(zp, dtype=np.float64)
    n, dim = z.shape
    if zp.shape != (n, dim):
        raise ValueError(f"shape mismatch z={z.shape} zp={zp.shape}")
    design = np.concatenate([z, np.ones((n, 1), dtype=np.float64)], axis=1)
    gram = design.T @ design
    gram[:dim, :dim] = gram[:dim, :dim] + float(alpha) * np.eye(dim)
    try:
        weights = np.linalg.solve(gram, design.T @ zp)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram, design.T @ zp, rcond=None)[0]
    matrix = weights[:dim].T
    bias = weights[dim]
    return matrix, bias


def fit_affine_with_rank(
    z: np.ndarray, zp: np.ndarray, alpha: float, rank: Any
) -> Tuple[np.ndarray, np.ndarray]:
    matrix, bias = fit_affine_ridge(z, zp, alpha)
    return truncate_rank(matrix, rank), bias


def select_hyperparams(
    z_fit: np.ndarray,
    zp_fit: np.ndarray,
    z_val: np.ndarray,
    zp_val: np.ndarray,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    ranks: Sequence[Any] = REDUCED_RANKS,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {"e_val": float("inf"), "alpha": float(alphas[0]), "rank": "full"}
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        matrix_full, bias_full = fit_affine_ridge(z_fit, zp_fit, float(alpha))
        for rank in ranks:
            matrix = truncate_rank(matrix_full, rank)
            pred = apply_affine(z_val, matrix, bias_full)
            err = normalized_error(pred, zp_val)
            row = {"alpha": float(alpha), "rank": rank if rank == "full" else int(rank), "e_val": err}
            rows.append(row)
            if err < float(best["e_val"]):
                best = {
                    "e_val": err,
                    "alpha": float(alpha),
                    "rank": rank if rank == "full" else int(rank),
                    "matrix": matrix,
                    "bias": bias_full,
                }
    if "matrix" not in best:
        matrix, bias = fit_affine_ridge(z_fit, zp_fit, float(alphas[0]))
        best["matrix"] = matrix
        best["bias"] = bias
    best["grid"] = rows
    return best


def fit_operator(
    z_fit: np.ndarray,
    zp_fit: np.ndarray,
    z_val: np.ndarray,
    zp_val: np.ndarray,
    z_test: np.ndarray,
    zp_test: np.ndarray,
) -> Dict[str, Any]:
    chosen = select_hyperparams(z_fit, zp_fit, z_val, zp_val)
    matrix = np.asarray(chosen["matrix"], dtype=np.float64)
    bias = np.asarray(chosen["bias"], dtype=np.float64)
    pred_test = apply_affine(z_test, matrix, bias)
    pred_val = apply_affine(z_val, matrix, bias)
    return {
        "matrix": matrix,
        "bias": bias,
        "alpha": float(chosen["alpha"]),
        "rank": chosen["rank"],
        "e_val": float(normalized_error(pred_val, zp_val)),
        "e_test": float(normalized_error(pred_test, zp_test)),
        "mse_test": float(mean_sq_error(pred_test, zp_test)),
        "n_fit": int(z_fit.shape[0]),
        "n_val": int(z_val.shape[0]),
        "n_test": int(z_test.shape[0]),
        "dim": int(z_fit.shape[1]),
    }


def random_same_rank_operator(
    matrix: np.ndarray, bias: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = np.asarray(matrix, dtype=np.float64)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(s > 1e-8 * float(s[0]) if s.size else 0.0))
    rank = max(rank, 1)
    q1, _ = np.linalg.qr(rng.standard_normal((matrix.shape[0], rank)))
    q2, _ = np.linalg.qr(rng.standard_normal((matrix.shape[1], rank)))
    scale = float(np.mean(s[:rank])) if rank else 1.0
    rand_m = (q1 * scale) @ q2.T
    rand_b = rng.standard_normal(bias.shape) * (float(np.std(bias)) + 1e-6)
    return rand_m, rand_b


def shuffle_pairs(z: np.ndarray, zp: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(zp))
    return z, zp[perm]


def pack_operator(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "alpha": result["alpha"],
        "rank": result["rank"],
        "e_val": result["e_val"],
        "e_test": result["e_test"],
        "mse_test": result["mse_test"],
        "n_fit": result["n_fit"],
        "n_val": result["n_val"],
        "n_test": result["n_test"],
        "dim": result["dim"],
    }
