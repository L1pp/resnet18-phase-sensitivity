"""Group-law tests on fitted affine operators. Functional error only."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .operators import apply_affine, compose_affine, mean_sq_error, normalized_error
from .protocol import delta_key


def _get(ops: Mapping[str, Mapping[str, Any]], delta: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    blob = ops[delta_key(delta)]
    return np.asarray(blob["matrix"], dtype=np.float64), np.asarray(blob["bias"], dtype=np.float64)


def composition_error(
    ops: Mapping[str, Mapping[str, Any]],
    delta1: Sequence[float],
    delta2: Sequence[float],
    z: np.ndarray,
    z_target: np.ndarray,
) -> Dict[str, float]:
    """Compare M_{d1+d2} vs M_d2 ∘ M_d1 on held-out z -> z(t+d1+d2)."""
    key_sum = delta_key((float(delta1[0]) + float(delta2[0]), float(delta1[1]) + float(delta2[1])))
    if key_sum not in ops or delta_key(delta1) not in ops or delta_key(delta2) not in ops:
        return {"ok": 0.0, "missing": 1.0}
    m1, b1 = _get(ops, delta1)
    m2, b2 = _get(ops, delta2)
    m_sum, b_sum = _get(ops, (float(delta1[0]) + float(delta2[0]), float(delta1[1]) + float(delta2[1])))
    m_comp, b_comp = compose_affine(m2, b2, m1, b1)
    pred_direct = apply_affine(z, m_sum, b_sum)
    pred_comp = apply_affine(z, m_comp, b_comp)
    return {
        "ok": 1.0,
        "e_direct": float(normalized_error(pred_direct, z_target)),
        "e_composed": float(normalized_error(pred_comp, z_target)),
        "e_operator_disagree": float(normalized_error(pred_comp, pred_direct)),
        "mse_composed": float(mean_sq_error(pred_comp, z_target)),
    }


def inverse_error(
    ops: Mapping[str, Mapping[str, Any]],
    delta: Sequence[float],
    z: np.ndarray,
) -> Dict[str, float]:
    """Functional |M_{-d}(M_d z + b_d) + b_{-d} - z| / support variance. Not Frobenius."""
    neg = (-float(delta[0]), -float(delta[1]))
    if delta_key(delta) not in ops or delta_key(neg) not in ops:
        return {"ok": 0.0, "missing": 1.0}
    m, b = _get(ops, delta)
    mi, bi = _get(ops, neg)
    mid = apply_affine(z, m, b)
    back = apply_affine(mid, mi, bi)
    m_id, b_id = compose_affine(mi, bi, m, b)
    pred_id = apply_affine(z, m_id, b_id)
    return {
        "ok": 1.0,
        "e_roundtrip": float(normalized_error(back, z)),
        "e_composed_identity": float(normalized_error(pred_id, z)),
        "mse_roundtrip": float(mean_sq_error(back, z)),
    }


def commutativity_error(
    ops: Mapping[str, Mapping[str, Any]],
    dx: Sequence[float],
    dy: Sequence[float],
    z: np.ndarray,
) -> Dict[str, float]:
    """M_x M_y vs M_y M_x on support."""
    if delta_key(dx) not in ops or delta_key(dy) not in ops:
        return {"ok": 0.0, "missing": 1.0}
    mx, bx = _get(ops, dx)
    my, by = _get(ops, dy)
    m_xy, b_xy = compose_affine(mx, bx, my, by)
    m_yx, b_yx = compose_affine(my, by, mx, bx)
    pred_xy = apply_affine(z, m_xy, b_xy)
    pred_yx = apply_affine(z, m_yx, b_yx)
    return {
        "ok": 1.0,
        "e_xy_vs_yx": float(normalized_error(pred_xy, pred_yx)),
        "mse_xy_vs_yx": float(mean_sq_error(pred_xy, pred_yx)),
    }


def path_independence_error(
    ops: Mapping[str, Mapping[str, Any]],
    dx: Sequence[float],
    dy: Sequence[float],
    z: np.ndarray,
    z_target: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """x then y vs y then x, optionally vs true z(t+dx+dy)."""
    comm = commutativity_error(ops, dx, dy, z)
    if not comm.get("ok"):
        return comm
    mx, bx = _get(ops, dx)
    my, by = _get(ops, dy)
    via_xy = apply_affine(apply_affine(z, mx, bx), my, by)
    via_yx = apply_affine(apply_affine(z, my, by), mx, bx)
    out = {
        "ok": 1.0,
        "e_path_disagree": float(normalized_error(via_xy, via_yx)),
        "mse_path_disagree": float(mean_sq_error(via_xy, via_yx)),
    }
    if z_target is not None:
        out["e_xy_to_target"] = float(normalized_error(via_xy, z_target))
        out["e_yx_to_target"] = float(normalized_error(via_yx, z_target))
    return out


def repeated_composition_error(
    ops: Mapping[str, Mapping[str, Any]],
    unit: Sequence[float],
    times: int,
    z: np.ndarray,
    z_target: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """M_unit^times vs M_{times*unit}."""
    scaled = (float(unit[0]) * times, float(unit[1]) * times)
    if delta_key(unit) not in ops or delta_key(scaled) not in ops:
        return {"ok": 0.0, "missing": 1.0}
    mu, bu = _get(ops, unit)
    ms, bs = _get(ops, scaled)
    m_acc, b_acc = np.eye(mu.shape[0]), np.zeros(bu.shape[0])
    for _ in range(int(times)):
        m_acc, b_acc = compose_affine(mu, bu, m_acc, b_acc)
    pred_rep = apply_affine(z, m_acc, b_acc)
    pred_direct = apply_affine(z, ms, bs)
    out = {
        "ok": 1.0,
        "e_rep_vs_direct": float(normalized_error(pred_rep, pred_direct)),
        "mse_rep_vs_direct": float(mean_sq_error(pred_rep, pred_direct)),
    }
    if z_target is not None:
        out["e_rep_to_target"] = float(normalized_error(pred_rep, z_target))
        out["e_direct_to_target"] = float(normalized_error(pred_direct, z_target))
    return out


def frobenius_identity_gap(ops: Mapping[str, Mapping[str, Any]], delta: Sequence[float]) -> Dict[str, float]:
    """Diagnostic only. Do not use as the official inverse metric."""
    neg = (-float(delta[0]), -float(delta[1]))
    if delta_key(delta) not in ops or delta_key(neg) not in ops:
        return {"ok": 0.0}
    m, b = _get(ops, delta)
    mi, bi = _get(ops, neg)
    m_id, _b_id = compose_affine(mi, bi, m, b)
    ident = np.eye(m_id.shape[0])
    return {
        "ok": 1.0,
        "frobenius_m_minus_i": float(np.linalg.norm(m_id - ident)),
        "note": "diagnostic_only_not_official",
    }


DEFAULT_COMPOSITIONS: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
    ((1.0, 0.0), (1.0, 0.0)),
    ((2.0, 0.0), (2.0, 0.0)),
    ((4.0, 0.0), (4.0, 0.0)),
    ((0.0, 1.0), (0.0, 1.0)),
    ((1.0, 0.0), (0.0, 1.0)),
    ((2.0, 0.0), (0.0, 2.0)),
    ((4.0, 0.0), (0.0, -4.0)),
    ((0.5, 0.0), (0.5, 0.0)),
)

DEFAULT_INVERSES: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0), (2.0, 0.0), (4.0, 0.0), (8.0, 0.0),
    (0.0, 1.0), (0.0, 2.0), (0.5, 0.0), (1.0, 1.0),
)

DEFAULT_COMMUTE: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
    ((1.0, 0.0), (0.0, 1.0)),
    ((2.0, 0.0), (0.0, 2.0)),
    ((4.0, 0.0), (0.0, 4.0)),
    ((8.0, 0.0), (0.0, 8.0)),
)

DEFAULT_REPEATS: Tuple[Tuple[Tuple[float, float], int], ...] = (
    ((1.0, 0.0), 4),
    ((0.0, 1.0), 4),
    ((0.5, 0.0), 2),
    ((0.0, 0.5), 2),
    ((2.0, 0.0), 2),
)


def run_group_law_battery(
    ops: Mapping[str, Mapping[str, Any]],
    z: np.ndarray,
    targets: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    targets = targets or {}
    compositions = []
    for d1, d2 in DEFAULT_COMPOSITIONS:
        key = delta_key((d1[0] + d2[0], d1[1] + d2[1]))
        compositions.append(
            {
                "d1": list(d1),
                "d2": list(d2),
                **composition_error(ops, d1, d2, z, targets.get(key, apply_affine(z, *_get(ops, (d1[0] + d2[0], d1[1] + d2[1]))) if key in ops else z)),
            }
        )
    inverses = [{"delta": list(d), **inverse_error(ops, d, z)} for d in DEFAULT_INVERSES]
    commutes = [{"dx": list(a), "dy": list(b), **commutativity_error(ops, a, b, z)} for a, b in DEFAULT_COMMUTE]
    paths = []
    for a, b in DEFAULT_COMMUTE:
        key = delta_key((a[0] + b[0], a[1] + b[1]))
        paths.append(
            {
                "dx": list(a),
                "dy": list(b),
                **path_independence_error(ops, a, b, z, targets.get(key)),
            }
        )
    repeats = []
    for unit, times in DEFAULT_REPEATS:
        key = delta_key((unit[0] * times, unit[1] * times))
        repeats.append(
            {
                "unit": list(unit),
                "times": int(times),
                **repeated_composition_error(ops, unit, times, z, targets.get(key)),
            }
        )

    def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
        vals = [float(r[field]) for r in rows if r.get("ok") and field in r and r[field] is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    return {
        "composition": compositions,
        "inverse": inverses,
        "commutativity": commutes,
        "path_independence": paths,
        "repeated": repeats,
        "summary": {
            "mean_e_composed": _mean(compositions, "e_composed"),
            "mean_e_direct": _mean(compositions, "e_direct"),
            "mean_e_roundtrip": _mean(inverses, "e_roundtrip"),
            "mean_e_xy_vs_yx": _mean(commutes, "e_xy_vs_yx"),
            "mean_e_path_disagree": _mean(paths, "e_path_disagree"),
            "mean_e_rep_vs_direct": _mean(repeats, "e_rep_vs_direct"),
        },
    }
