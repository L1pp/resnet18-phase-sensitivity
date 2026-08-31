"""Ridge readout. Never use unregularized 512-sample OLS."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


ALPHAS = np.logspace(-6, 4, 11)


def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    xty = x.T @ y
    d = xtx.shape[0]
    return np.linalg.solve(xtx + alpha * np.eye(d), xty)


def ridge_cv(x_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5, seed: int = 20260810) -> Tuple[float, np.ndarray]:
    x = _add_bias(np.asarray(x_train, dtype=np.float64))
    y = np.asarray(y_train, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, min(n_folds, n))
    best_a, best_err = float(ALPHAS[0]), np.inf
    for a in ALPHAS:
        errs = []
        for i, fold in enumerate(folds):
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            if mask.sum() < 2 or len(fold) < 1:
                continue
            w = _fit(x[mask], y[mask], float(a))
            pred = x[fold] @ w
            errs.append(np.mean(np.abs(pred - y[fold])))
        if not errs:
            continue
        err = float(np.mean(errs))
        if err < best_err:
            best_err, best_a = err, float(a)
    w = _fit(x, y, best_a)
    return best_a, w


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _add_bias(np.asarray(x, dtype=np.float64)) @ w


def position_readout(z_train: np.ndarray, t_train: np.ndarray, z_eval: np.ndarray, t_eval: np.ndarray) -> Dict[str, float]:
    alpha, w = ridge_cv(z_train, t_train)
    pred = ridge_predict(z_eval, w)
    err = np.linalg.norm(pred - t_eval, axis=-1)
    const = np.linalg.norm(t_eval - t_train.mean(0, keepdims=True), axis=-1)
    return {
        "alpha": alpha,
        "t_mae": float(np.mean(err)),
        "t_mae_px": float(np.mean(err) * (223.0 if t_eval.max() <= 1.5 else 1.0)),
        "constant_mae": float(np.mean(const)),
        "n_train": int(z_train.shape[0]),
        "n_eval": int(z_eval.shape[0]),
    }
