"""Known translation-group action on a latent space. Used only by tests and assay checks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from phase3_high_upside.operators import apply_affine


def _basis(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def homogeneous_shift(dx: float, dy: float, dim: int) -> np.ndarray:
    """h = [content..., x, y, 1]; x += dx, y += dy."""
    if dim < 4:
        raise ValueError("dim must be >= 4")
    hom = np.eye(dim, dtype=np.float64)
    hom[-3, -1] = float(dx)
    hom[-2, -1] = float(dy)
    return hom


def true_operator(dx: float, dy: float, dim: int = 8, basis_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    basis = _basis(dim, basis_seed)
    hom = homogeneous_shift(dx, dy, dim)
    matrix = basis @ hom @ basis.T
    return matrix, np.zeros(dim, dtype=np.float64)


def make_latent_field(
    n_shape: int,
    origins: np.ndarray,
    dim: int = 8,
    seed: int = 0,
    basis_seed: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    basis = _basis(dim, basis_seed)
    content = rng.standard_normal((n_shape, dim - 3))
    origins = np.asarray(origins, dtype=np.float64)
    n_t = len(origins)
    z = np.zeros((n_shape, n_t, dim), dtype=np.float64)
    for s in range(n_shape):
        for i, (tx, ty) in enumerate(origins):
            hom = np.concatenate([content[s], np.array([tx, ty, 1.0], dtype=np.float64)])
            z[s, i] = basis @ hom
    return {"z": z, "content": content, "origins": origins, "basis": basis, "basis_seed": basis_seed}


def pair_features(field: Dict[str, np.ndarray], dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    origins = field["origins"]
    dest = origins + np.array([dx, dy], dtype=np.float64)
    src, dst = [], []
    lookup = {(round(float(x), 6), round(float(y), 6)): i for i, (x, y) in enumerate(origins)}
    for i, (x, y) in enumerate(dest):
        key = (round(float(x), 6), round(float(y), 6))
        if key in lookup:
            src.append(i)
            dst.append(lookup[key])
    z = field["z"]
    if not src:
        matrix, bias = true_operator(dx, dy, z.shape[-1], int(field.get("basis_seed", 0)))
        flat = z.reshape(-1, z.shape[-1])
        return flat, apply_affine(flat, matrix, bias)
    return z[:, src].reshape(-1, z.shape[-1]), z[:, dst].reshape(-1, z.shape[-1])
