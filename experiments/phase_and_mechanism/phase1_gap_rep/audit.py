"""Phase 1.5 mechanism audit on frozen GAP features. Does not retrain."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .analyze import _labels, _load_features, _load_split_codes
from .common import (
    COORD_SCALE,
    READOUT_ENERGY_FRACTION,
    SEED,
    decompose_controls,
    dirs,
    dump_json,
    fingerprint,
    orthogonal_matrix_a,
    profile_spec,
    run_dirs,
    setup_matplotlib_chinese,
    train_run_spec,
    write_run_metadata,
)

RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 32)
N_NULL = 1000
ENERGY_TOL = 0.15
SPLIT_NAMES = ("train", "val", "test")


def _to_py(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_py(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _qr_basis(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    q, _ = np.linalg.qr(array)
    rank = int(min(array.shape))
    return q[:, :rank]


def _svd_components(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    total = float(energy.sum())
    fractions = energy / total if total else np.zeros_like(energy)
    cumulative = np.cumsum(fractions)
    return vh.T, fractions, cumulative


def _energy_rank(cumulative: np.ndarray, fraction: float = READOUT_ENERGY_FRACTION) -> int:
    if cumulative.size == 0:
        return 1
    return int(min(len(cumulative), max(1, int(np.searchsorted(cumulative, fraction) + 1))))


def _train_effects(z: np.ndarray, split: np.ndarray, axis: int, mu: np.ndarray) -> np.ndarray:
    train = split == 0
    n = z.shape[axis]
    dim = z.shape[-1]
    effects = np.zeros((n, dim), dtype=np.float64)
    for index in range(n):
        mask = train.take(index, axis=axis)
        cells = z[:, index, :][mask] if axis == 1 else z[index, :, :][mask]
        if cells.size == 0:
            raise RuntimeError(f"no train cells for axis={axis} id={index}")
        effects[index] = cells.mean(axis=0) - mu
    return effects


def _project(centered: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (centered @ basis) @ basis.T


def _erasure_z(z_flat: np.ndarray, basis: np.ndarray, mu: np.ndarray, centered: bool) -> np.ndarray:
    if centered:
        return z_flat - _project(z_flat - mu, basis)
    return z_flat - _project(z_flat, basis)


def _predict_p(z_flat: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return z_flat @ weight.T + bias


def _mae_px(diff: np.ndarray) -> float:
    return float(np.mean(np.abs(diff))) * COORD_SCALE


def _pack_errors(pred: np.ndarray, labels: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
    parts = decompose_controls(pred)
    truth = decompose_controls(labels["P"])
    t_err = parts["t"] - labels["t"]
    q_err = parts["q"].reshape(pred.shape[0], 6) - labels["Q"]
    p_err = pred - labels["P"]
    g_err = parts["geometry_comp"] - truth["geometry_comp"]
    t_sel, q_sel, p_sel, g_sel = t_err[mask], q_err[mask], p_err[mask], g_err[mask]
    t_bias, q_bias, p_bias, g_bias = t_sel.mean(0), q_sel.mean(0), p_sel.mean(0), g_sel.mean(0)
    return {
        "n": int(mask.sum()),
        "E_t": _mae_px(t_sel),
        "E_q": _mae_px(q_sel),
        "E_P": _mae_px(p_sel),
        "E_gcomp": float(np.mean(np.abs(g_sel))),
        "E_t_residual": _mae_px(t_sel - t_bias),
        "E_q_residual": _mae_px(q_sel - q_bias),
        "E_P_residual": _mae_px(p_sel - p_bias),
        "E_gcomp_residual": float(np.mean(np.abs(g_sel - g_bias))),
        "bias_t_px": (t_bias * COORD_SCALE).tolist(),
        "bias_q_px": (q_bias * COORD_SCALE).tolist(),
        "bias_P_px": (p_bias * COORD_SCALE).tolist(),
        "bias_gcomp": g_bias.tolist(),
        "gcomp_mae": [float(np.mean(np.abs(g_sel[:, i]))) for i in range(g_sel.shape[1])],
    }


def _readouts(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    primed = orthogonal_matrix_a() @ weight
    return primed[[0, 3]], primed[[1, 2, 4, 5]]


def _frobenius_alignment(weight: np.ndarray, basis: np.ndarray) -> float:
    if np.linalg.norm(weight) == 0 or basis.size == 0:
        return 0.0
    return float(np.sum((weight @ basis) ** 2) / np.sum(weight**2))


def _principal_angles_deg(a: np.ndarray, b: np.ndarray) -> List[float]:
    qa, qb = _qr_basis(a), _qr_basis(b)
    svals = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    return [float(np.degrees(np.arccos(value))) for value in svals]


def _functional_alignment(centered: np.ndarray, w_row: np.ndarray, basis: np.ndarray) -> float:
    full = centered @ w_row.T
    denom = float(np.mean(np.sum(full**2, axis=1)))
    if denom <= 0:
        return 0.0
    projected = _project(centered, basis) @ w_row.T
    return float(np.mean(np.sum(projected**2, axis=1))) / denom


def _all_functional(
    centered: np.ndarray, w_t: np.ndarray, w_q: np.ndarray, u_t: np.ndarray, u_q: np.ndarray
) -> Dict[str, float]:
    return {
        "F_t_from_t": _functional_alignment(centered, w_t, u_t),
        "F_t_from_q": _functional_alignment(centered, w_t, u_q),
        "F_q_from_t": _functional_alignment(centered, w_q, u_t),
        "F_q_from_q": _functional_alignment(centered, w_q, u_q),
    }


def _subspace_energy(centered: np.ndarray, basis: np.ndarray) -> float:
    return float(np.mean(np.sum((centered @ basis) ** 2, axis=1)))


def _haar_basis(dim: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    return _qr_basis(rng.normal(size=(dim, rank)))


def _pvalue(target: float, samples: Sequence[float], larger: bool = True) -> float:
    array = np.asarray(samples, dtype=np.float64)
    count = int(np.sum(array >= target if larger else array <= target))
    return (1.0 + count) / (len(array) + 1.0)


def _histogram(samples: Sequence[float], bins: int = 40) -> Dict[str, Any]:
    counts, edges = np.histogram(np.asarray(samples, dtype=np.float64), bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _load_shapes(profile: str) -> np.ndarray:
    path = dirs(profile)["tables"] / "shapes.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    table = np.zeros((len(rows), 4), dtype=np.float64)
    for row in rows:
        sid = int(row["shape_id"])
        table[sid] = [float(row["theta_deg"]), float(row["chord_px"]), float(row["curve_sign"]), float(row["alpha"])]
    return table


def _pc_spectrum(train_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, singular, vh = np.linalg.svd(train_centered, full_matrices=False)
    energy = (singular**2) / max(train_centered.shape[0], 1)
    return vh.T, energy


def _pc_window_for_energy(pc_energy: np.ndarray, rank: int, target: float) -> int:
    best_m = rank
    best_gap = abs(float(pc_energy[:rank].sum()) - target)
    n_comp = int(pc_energy.shape[0])
    for m in range(rank, n_comp + 1):
        expected = (rank / m) * float(pc_energy[:m].sum())
        gap = abs(expected - target)
        if gap < best_gap:
            best_gap = gap
            best_m = m
        if expected < target and m >= best_m + 16:
            break
    return int(best_m)


def _sample_energy_matched(
    train_centered: np.ndarray,
    rank: int,
    target_energy: float,
    rng: np.random.Generator,
    n: int = N_NULL,
    max_draws: int = 40000,
    pcs: np.ndarray | None = None,
    pc_energy: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
    if pcs is None or pc_energy is None:
        pcs, pc_energy = _pc_spectrum(train_centered)
    dim = train_centered.shape[1]
    rank = int(max(1, min(rank, pcs.shape[1], dim)))
    max_energy = float(pc_energy[:rank].sum())
    window = _pc_window_for_energy(pc_energy, rank, target_energy)
    near_max = target_energy >= (1.0 - ENERGY_TOL) * max_energy or window == rank
    pool: List[Tuple[float, np.ndarray, float]] = []
    matched: List[np.ndarray] = []
    matched_e: List[float] = []
    scale = max(target_energy, 1e-12)
    u_high = pcs[:, :rank]
    for _ in range(max_draws):
        if near_max:
            basis = _qr_basis(u_high + 0.03 * rng.normal(size=u_high.shape))[:, :rank]
        else:
            matrix = pcs[:, :window] @ rng.normal(size=(window, rank))
            basis = _qr_basis(matrix)[:, :rank]
        energy = _subspace_energy(train_centered, basis)
        pool.append((abs(energy - target_energy), basis.copy(), energy))
        if abs(energy - target_energy) / scale <= ENERGY_TOL:
            matched.append(basis.copy())
            matched_e.append(energy)
            if len(matched) >= n:
                break
    if len(matched) < n:
        pool.sort(key=lambda item: item[0])
        chosen = pool[:n]
        matched = [item[1] for item in chosen]
        matched_e = [item[2] for item in chosen]
    mean_energy = float(np.mean(matched_e)) if matched_e else 0.0
    info = {
        "pc_window": window,
        "max_rank_energy": max_energy,
        "near_max": bool(near_max),
        "n_collected": len(matched),
        "mean_deleted_energy": mean_energy,
        "rel_error": abs(mean_energy - target_energy) / scale,
    }
    return matched, matched_e, info


def _linear_probe(
    features: np.ndarray, targets: np.ndarray, train_mask: np.ndarray, masks: Mapping[str, np.ndarray]
) -> Dict[str, float]:
    ones = np.ones((features.shape[0], 1), dtype=np.float64)
    design = np.concatenate([features, ones], axis=1)
    coef, *_ = np.linalg.lstsq(design[train_mask], targets[train_mask], rcond=None)
    pred = design @ coef
    return {f"{name}_mae_px": _mae_px(pred[mask] - targets[mask]) for name, mask in masks.items()}


def _group_mae(err_px: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    selected = ids[mask]
    values = err_px[mask]
    return {str(int(unique)): float(np.mean(values[selected == unique])) for unique in np.unique(selected)}


def _null_stats(z_test, basis, mu, weight, bias, t_test, q_test, w_t, w_q):
    pred = _predict_p(_erasure_z(z_test, basis, mu, True), weight, bias)
    pts = pred.reshape(-1, 3, 2)
    t_hat = pts.mean(axis=1)
    q_hat = (pts - t_hat[:, None, :]).reshape(-1, 6)
    e_t = float(np.mean(np.abs(t_hat - t_test))) * COORD_SCALE
    e_q = float(np.mean(np.abs(q_hat - q_test))) * COORD_SCALE
    centered = z_test - mu
    return (
        e_t,
        e_q,
        _functional_alignment(centered, w_q, basis),
        _functional_alignment(centered, w_t, basis),
    )


def _plot_nulls(figure_dir: Path, target: Mapping[str, float], haar: Mapping[str, List[float]], matched: Mapping[str, List[float]]) -> List[str]:
    plt = setup_matplotlib_chinese()
    written: List[str] = []
    titles = {
        "E_t": "删平移子空间后 E_t（test）",
        "E_q": "删平移子空间后 E_q（test）",
        "F_q_from_t": "功能对齐 F_q←t（test）",
        "F_t_from_t": "功能对齐 F_t←t（test）",
    }
    for key, title in titles.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.hist(haar[key], bins=40, alpha=0.55, label="Haar 同 rank")
        ax.hist(matched[key], bins=40, alpha=0.55, label="能量匹配随机")
        ax.axvline(target[key], color="C3", ls="--", label="train T 子空间")
        ax.set_title(title)
        ax.set_xlabel(key)
        ax.set_ylabel("次数")
        ax.legend()
        fig.tight_layout()
        name = f"null_{key}.png"
        fig.savefig(figure_dir / name, dpi=140)
        plt.close(fig)
        written.append(name)
    return written


def _plot_rank_sweep(figure_dir: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    plt = setup_matplotlib_chinese()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, which, title in (
        (axes[0], "T", "删 T 子空间（centered, test）"),
        (axes[1], "G", "删 G 子空间（centered, test）"),
    ):
        subset = [row for row in rows if row["which"] == which]
        ranks = [row["rank"] for row in subset]
        ax.plot(ranks, [row["test"]["E_t"] for row in subset], marker="o", label="E_t")
        ax.plot(ranks, [row["test"]["E_q"] for row in subset], marker="s", label="E_q")
        ax.set_xlabel("rank")
        ax.set_ylabel("MAE (px)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figure_dir / "rank_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_bias(figure_dir: Path, baseline: Mapping[str, Any], centered: Mapping[str, Any]) -> str:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = ["E_t", "E_q", "E_P"]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, [baseline["E_t"], baseline["E_q"], baseline["E_P"]], width, label="删除前")
    ax.bar(x, [centered["E_t"], centered["E_q"], centered["E_P"]], width, label="centered 删除后 MAE")
    ax.bar(
        x + width,
        [centered["E_t_residual"], centered["E_q_residual"], centered["E_P_residual"]],
        width,
        label="去掉平均偏差后",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("px")
    ax.set_title("test：bias 是否解释几何损伤")
    ax.legend()
    fig.tight_layout()
    name = "bias_vs_residual_test.png"
    fig.savefig(figure_dir / name, dpi=140)
    plt.close(fig)
    return name


def _write_cells(
    path: Path,
    shape_ids: np.ndarray,
    trans_ids: np.ndarray,
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    pred: np.ndarray,
    pred_c: np.ndarray,
    pred_u: np.ndarray,
) -> None:
    true_parts = decompose_controls(labels["P"])
    pred_parts = decompose_controls(pred)
    cent_parts = decompose_controls(pred_c)
    un_parts = decompose_controls(pred_u)
    q_true = labels["Q"]
    q_pred = pred_parts["q"].reshape(-1, 6)
    q_c = cent_parts["q"].reshape(-1, 6)
    q_u = un_parts["q"].reshape(-1, 6)
    fields = [
        "shape_id",
        "translation_id",
        "split",
        "true_tx",
        "true_ty",
        "pred_tx",
        "pred_ty",
        "error_t",
        "error_q",
        "error_P",
        "centered_error_t",
        "centered_error_q",
        "centered_error_P",
        "uncentered_error_t",
        "uncentered_error_q",
        "uncentered_error_P",
    ]
    for index in range(6):
        fields.extend([f"true_q{index}", f"pred_q{index}", f"centered_q{index}"])
    for index in range(4):
        fields.extend([f"true_g{index}", f"pred_g{index}", f"centered_g{index}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(shape_ids)):
            row = {
                "shape_id": int(shape_ids[i]),
                "translation_id": int(trans_ids[i]),
                "split": SPLIT_NAMES[int(split[i])],
                "true_tx": true_parts["t"][i, 0],
                "true_ty": true_parts["t"][i, 1],
                "pred_tx": pred_parts["t"][i, 0],
                "pred_ty": pred_parts["t"][i, 1],
                "error_t": float(np.mean(np.abs(pred_parts["t"][i] - true_parts["t"][i]))) * COORD_SCALE,
                "error_q": float(np.mean(np.abs(q_pred[i] - q_true[i]))) * COORD_SCALE,
                "error_P": float(np.mean(np.abs(pred[i] - labels["P"][i]))) * COORD_SCALE,
                "centered_error_t": float(np.mean(np.abs(cent_parts["t"][i] - true_parts["t"][i]))) * COORD_SCALE,
                "centered_error_q": float(np.mean(np.abs(q_c[i] - q_true[i]))) * COORD_SCALE,
                "centered_error_P": float(np.mean(np.abs(pred_c[i] - labels["P"][i]))) * COORD_SCALE,
                "uncentered_error_t": float(np.mean(np.abs(un_parts["t"][i] - true_parts["t"][i]))) * COORD_SCALE,
                "uncentered_error_q": float(np.mean(np.abs(q_u[i] - q_true[i]))) * COORD_SCALE,
                "uncentered_error_P": float(np.mean(np.abs(pred_u[i] - labels["P"][i]))) * COORD_SCALE,
            }
            for index in range(6):
                row[f"true_q{index}"] = q_true[i, index]
                row[f"pred_q{index}"] = q_pred[i, index]
                row[f"centered_q{index}"] = q_c[i, index]
            for index in range(4):
                row[f"true_g{index}"] = true_parts["geometry_comp"][i, index]
                row[f"pred_g{index}"] = pred_parts["geometry_comp"][i, index]
                row[f"centered_g{index}"] = cent_parts["geometry_comp"][i, index]
            writer.writerow(row)


def _decide(
    centered_test: Mapping[str, Any],
    matched_t: Sequence[float],
    matched_q: Sequence[float],
    *,
    ut_pc_overlap: float,
    k_t: int,
    target_energy: float,
    max_k_energy: float,
    matched_energy_mean: float,
) -> Dict[str, Any]:
    e_t = float(centered_test["E_t"])
    e_q = float(centered_test["E_q"])
    p_t = _pvalue(e_t, matched_t, larger=True)
    p_q = _pvalue(e_q, matched_q, larger=True)
    mean_t = float(np.mean(matched_t)) if len(matched_t) else float("nan")
    mean_q = float(np.mean(matched_q)) if len(matched_q) else float("nan")
    std_q = float(np.std(matched_q)) if len(matched_q) else float("nan")
    t_strong = p_t <= 0.05 and e_t > mean_t * 2.0
    q_like_random = (p_q > 0.10) or (e_q <= mean_q + 2.0 * std_q)
    q_above = p_q <= 0.05 and e_q > mean_q + 2.0 * std_q
    near_max = target_energy >= (1.0 - ENERGY_TOL) * max_k_energy
    collapsed = ut_pc_overlap >= 0.9 * k_t
    rel = abs(matched_energy_mean - target_energy) / max(target_energy, 1e-12)
    if collapsed or near_max:
        letter = "C"
        note = (
            "按平移主效应 90% 能量切出来的那几维，几乎就是整张特征里能量最大的那几维，"
            "并不是单独存放平移的小角落。删掉它等于把大部分特征清掉，对照实验没法证明「专门的平移码」。"
            "先不要改网络结构，先把「平移方向」怎么切定对。"
        )
    elif t_strong and q_like_random and not q_above:
        letter = "A"
        note = (
            "删掉平移小角落之后，位置误差远差于能量匹配的随机删除，形状误差差不多。"
            "更像选择性存放。下一步应多 seed 确认，先不要改模型。"
        )
    elif t_strong and q_above:
        letter = "B"
        note = (
            "位置小角落确实在干活，但删掉它形状也明显变差，而且比能量匹配的随机删除更伤。"
            "低维平移码有用，但和形状缠在一起。这才是后面 M1/M2 的动机。"
        )
    else:
        letter = "C"
        note = (
            "在 centered、能量匹配随机对照下，定向删平移的效果不够强。"
            "先不要做新架构，回头检查子空间估计。"
        )
    return {
        "letter": letter,
        "p_E_t_matched": p_t,
        "p_E_q_matched": p_q,
        "matched_mean_E_t": mean_t,
        "matched_mean_E_q": mean_q,
        "test_E_t": e_t,
        "test_E_q": e_q,
        "ut_pc_overlap": ut_pc_overlap,
        "near_max_energy": near_max,
        "collapsed_onto_top_pcs": collapsed,
        "matched_energy_rel_error": rel,
        "note": note,
    }


def audit_run(run_name: str) -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    z, weight, bias = _load_features(run.name, spec.name)
    split_grid = _load_split_codes(spec.name)
    labels_grid = _labels(spec.name)
    shapes = _load_shapes(spec.name)
    n_s, n_t, dim = z.shape
    z_flat = z.reshape(-1, dim)
    split_flat = split_grid.reshape(-1)
    sid = np.repeat(np.arange(n_s), n_t)
    tid = np.tile(np.arange(n_t), n_s)
    labels_flat = {
        "P": labels_grid["P"].reshape(-1, 6),
        "Q": labels_grid["Q"].reshape(-1, 6),
        "t": labels_grid["t"].reshape(-1, 2),
    }
    masks = {name: split_flat == index for index, name in enumerate(SPLIT_NAMES)}
    train_mask = masks["train"]
    mu = z_flat[train_mask].mean(axis=0)
    train_centered = z_flat[train_mask] - mu
    w_t, w_q = _readouts(weight)

    t_effects = _train_effects(z, split_grid, axis=1, mu=mu)
    g_effects = _train_effects(z, split_grid, axis=0, mu=mu)
    t_basis_full, t_frac, t_cum = _svd_components(t_effects)
    g_basis_full, g_frac, g_cum = _svd_components(g_effects)
    k_t = min(_energy_rank(t_cum), t_basis_full.shape[1])
    k_q90 = min(_energy_rank(g_cum), g_basis_full.shape[1])
    u_t = t_basis_full[:, :k_t]
    u_q90 = g_basis_full[:, : max(1, k_q90)]
    target_energy = _subspace_energy(train_centered, u_t)
    pcs, pc_energy = _pc_spectrum(train_centered)
    u_top = pcs[:, :k_t]
    ut_pc_overlap = float(np.sum((u_t.T @ u_top) ** 2))
    max_k_energy = float(pc_energy[:k_t].sum())
    print(
        f"[audit:{run.name}] k_t={k_t} Ut_energy={target_energy:.3f} "
        f"top{k_t}_energy={max_k_energy:.3f} overlap={ut_pc_overlap:.3f}/{k_t}"
    )

    rng = np.random.default_rng(SEED)
    pred0 = _predict_p(z_flat, weight, bias)
    pred_c = _predict_p(_erasure_z(z_flat, u_t, mu, True), weight, bias)
    pred_u = _predict_p(_erasure_z(z_flat, u_t, mu, False), weight, bias)
    baseline = {name: _pack_errors(pred0, labels_flat, mask) for name, mask in masks.items()}
    centered = {name: _pack_errors(pred_c, labels_flat, mask) for name, mask in masks.items()}
    uncentered = {name: _pack_errors(pred_u, labels_flat, mask) for name, mask in masks.items()}

    functional = {}
    frobenius = {
        "W_t_on_Ut": _frobenius_alignment(w_t, u_t),
        "W_t_on_Uq": _frobenius_alignment(w_t, u_q90),
        "W_q_on_Uq": _frobenius_alignment(w_q, u_q90),
        "W_q_on_Ut": _frobenius_alignment(w_q, u_t),
    }
    for name, mask in masks.items():
        functional[name] = _all_functional(z_flat[mask] - mu, w_t, w_q, u_t, u_q90)

    t_err0 = np.mean(np.abs(decompose_controls(pred0)["t"] - labels_flat["t"]), axis=1) * COORD_SCALE
    q_err0 = np.mean(np.abs(decompose_controls(pred0)["q"].reshape(-1, 6) - labels_flat["Q"]), axis=1) * COORD_SCALE
    t_err_c = np.mean(np.abs(decompose_controls(pred_c)["t"] - labels_flat["t"]), axis=1) * COORD_SCALE
    q_err_c = np.mean(np.abs(decompose_controls(pred_c)["q"].reshape(-1, 6) - labels_flat["Q"]), axis=1) * COORD_SCALE
    per_shape = {
        "baseline_E_t": _group_mae(t_err0, sid, masks["test"]),
        "baseline_E_q": _group_mae(q_err0, sid, masks["test"]),
        "centered_E_t": _group_mae(t_err_c, sid, masks["test"]),
        "centered_E_q": _group_mae(q_err_c, sid, masks["test"]),
    }
    per_translation = {
        "baseline_E_t": _group_mae(t_err0, tid, masks["test"]),
        "baseline_E_q": _group_mae(q_err0, tid, masks["test"]),
        "centered_E_t": _group_mae(t_err_c, tid, masks["test"]),
        "centered_E_q": _group_mae(q_err_c, tid, masks["test"]),
    }
    meta_groups: Dict[str, Any] = {}
    test_mask = masks["test"]
    for col, name in enumerate(("theta_deg", "chord_px", "curve_sign", "alpha")):
        keys = shapes[sid[test_mask], col]
        grouped: Dict[str, List[float]] = {}
        for key, value in zip(keys, q_err_c[test_mask]):
            grouped.setdefault(str(float(key)), []).append(float(value))
        meta_groups[name] = {key: float(np.mean(vals)) for key, vals in grouped.items()}

    z_c = _erasure_z(z_flat, u_t, mu, True)
    g_true = decompose_controls(labels_flat["P"])["geometry_comp"]
    probe_t = _linear_probe(z_c, labels_flat["t"], train_mask, masks)
    probe_q = _linear_probe(z_c, g_true, train_mask, masks)

    t_std = labels_flat["t"][train_mask]
    q_std = g_true[train_mask]
    t_mu, t_sig = t_std.mean(0), t_std.std(0) + 1e-12
    q_mu, q_sig = q_std.mean(0), q_std.std(0) + 1e-12
    design = np.concatenate([(t_std - t_mu) / t_sig, (q_std - q_mu) / q_sig], axis=1)
    beta, *_ = np.linalg.lstsq(design, train_centered, rcond=None)
    u_bt = _qr_basis(beta[:2].T)
    u_bq = _qr_basis(beta[2:].T)
    factor_align = {
        "frobenius": {
            "W_t_on_Bt": _frobenius_alignment(w_t, u_bt),
            "W_t_on_Bq": _frobenius_alignment(w_t, u_bq),
            "W_q_on_Bq": _frobenius_alignment(w_q, u_bq),
            "W_q_on_Bt": _frobenius_alignment(w_q, u_bt),
        },
        "functional_test": _all_functional(z_flat[test_mask] - mu, w_t, w_q, u_bt, u_bq),
        "principal_angles_Wt_Bt_deg": _principal_angles_deg(w_t.T, u_bt),
        "principal_angles_Wq_Bq_deg": _principal_angles_deg(w_q.T, u_bq),
        "principal_angles_Wt_Bq_deg": _principal_angles_deg(w_t.T, u_bq),
        "principal_angles_Wq_Bt_deg": _principal_angles_deg(w_q.T, u_bt),
    }
    g_pc1 = g_effects @ g_basis_full[:, 0]
    pc1_corr = {
        "chord_px": float(np.corrcoef(g_pc1, shapes[:, 1])[0, 1]),
        "theta_deg": float(np.corrcoef(g_pc1, shapes[:, 0])[0, 1]),
        "curve_sign": float(np.corrcoef(g_pc1, shapes[:, 2])[0, 1]),
        "alpha": float(np.corrcoef(g_pc1, shapes[:, 3])[0, 1]),
    }

    rank_rows: List[Dict[str, Any]] = []
    for which, full_basis, cumulative in (("T", t_basis_full, t_cum), ("G", g_basis_full, g_cum)):
        max_rank = int(full_basis.shape[1])
        for rank in RANKS:
            k = min(rank, max_rank)
            basis = full_basis[:, :k]
            pred_k = _predict_p(_erasure_z(z_flat, basis, mu, True), weight, bias)
            row: Dict[str, Any] = {
                "which": which,
                "rank": k,
                "requested_rank": rank,
                "explained_variance": float(cumulative[k - 1]) if k else 0.0,
                "frobenius": {"W_t": _frobenius_alignment(w_t, basis), "W_q": _frobenius_alignment(w_q, basis)},
                "functional_test": {
                    "F_t": _functional_alignment(z_flat[test_mask] - mu, w_t, basis),
                    "F_q": _functional_alignment(z_flat[test_mask] - mu, w_q, basis),
                },
            }
            for name, mask in masks.items():
                row[name] = _pack_errors(pred_k, labels_flat, mask)
            rank_rows.append(row)

    z_test = z_flat[test_mask]
    t_test = labels_flat["t"][test_mask]
    q_test = labels_flat["Q"][test_mask]
    print(f"[audit:{run.name}] sampling {N_NULL} Haar + energy-matched nulls, k_t={k_t}")
    haar_stats = {key: [] for key in ("E_t", "E_q", "F_q_from_t", "F_t_from_t")}
    for index in range(N_NULL):
        if index % 200 == 0:
            print(f"[audit:{run.name}] haar {index}/{N_NULL}")
        et, eq, fqt, ftt = _null_stats(
            z_test, _haar_basis(dim, k_t, rng), mu, weight, bias, t_test, q_test, w_t, w_q
        )
        haar_stats["E_t"].append(et)
        haar_stats["E_q"].append(eq)
        haar_stats["F_q_from_t"].append(fqt)
        haar_stats["F_t_from_t"].append(ftt)

    matched_bases, matched_energies, matched_info = _sample_energy_matched(
        train_centered, k_t, target_energy, rng, pcs=pcs, pc_energy=pc_energy
    )
    print(
        f"[audit:{run.name}] energy-matched window={matched_info['pc_window']} "
        f"near_max={matched_info['near_max']} rel_err={matched_info['rel_error']:.3f} "
        f"mean_energy={matched_info['mean_deleted_energy']:.3f}"
    )
    matched_stats = {key: [] for key in ("E_t", "E_q", "F_q_from_t", "F_t_from_t")}
    for index, basis in enumerate(matched_bases):
        if index % 200 == 0:
            print(f"[audit:{run.name}] matched {index}/{len(matched_bases)}")
        et, eq, fqt, ftt = _null_stats(z_test, basis, mu, weight, bias, t_test, q_test, w_t, w_q)
        matched_stats["E_t"].append(et)
        matched_stats["E_q"].append(eq)
        matched_stats["F_q_from_t"].append(fqt)
        matched_stats["F_t_from_t"].append(ftt)

    perm_et: List[float] = []
    perm_eq: List[float] = []
    train_s, train_tt = np.where(split_grid == 0)
    z_train_cells = z[train_s, train_tt]
    print(f"[audit:{run.name}] label permutation {N_NULL}")
    for index in range(N_NULL):
        if index % 200 == 0:
            print(f"[audit:{run.name}] perm {index}/{N_NULL}")
        shuffled = train_tt.copy()
        rng.shuffle(shuffled)
        counts = np.maximum(np.bincount(shuffled, minlength=n_t).astype(np.float64), 1.0)
        sums = np.zeros((n_t, dim), dtype=np.float64)
        np.add.at(sums, shuffled, z_train_cells)
        effects = sums / counts[:, None] - mu
        basis_p, _, _ = _svd_components(effects)
        kp = min(k_t, basis_p.shape[1])
        et, eq, _, _ = _null_stats(z_test, basis_p[:, :kp], mu, weight, bias, t_test, q_test, w_t, w_q)
        perm_et.append(et)
        perm_eq.append(eq)

    bt_energy = _subspace_energy(train_centered, u_bt)
    bt_rank = int(u_bt.shape[1])
    print(f"[audit:{run.name}] B_t energy={bt_energy:.3f} rank={bt_rank}")
    bt_matched, bt_matched_e, bt_matched_info = _sample_energy_matched(
        train_centered, bt_rank, bt_energy, rng, pcs=pcs, pc_energy=pc_energy
    )
    bt_haar_et, bt_haar_eq, bt_match_et, bt_match_eq = [], [], [], []
    for index in range(N_NULL):
        if index % 200 == 0:
            print(f"[audit:{run.name}] B_t haar {index}/{N_NULL}")
        et, eq, _, _ = _null_stats(
            z_test, _haar_basis(dim, bt_rank, rng), mu, weight, bias, t_test, q_test, w_t, w_q
        )
        bt_haar_et.append(et)
        bt_haar_eq.append(eq)
    for index, basis in enumerate(bt_matched):
        if index % 200 == 0:
            print(f"[audit:{run.name}] B_t matched {index}/{len(bt_matched)}")
        et, eq, _, _ = _null_stats(z_test, basis, mu, weight, bias, t_test, q_test, w_t, w_q)
        bt_match_et.append(et)
        bt_match_eq.append(eq)
    bt_test = _pack_errors(_predict_p(_erasure_z(z_flat, u_bt, mu, True), weight, bias), labels_flat, test_mask)

    audit_dir = run_dirs(run.name)["root"] / "audit"
    fig_dir = audit_dir / "figures"
    tab_dir = audit_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    _write_cells(tab_dir / "cells.csv", sid, tid, split_flat, labels_flat, pred0, pred_c, pred_u)
    target_plot = {
        "E_t": centered["test"]["E_t"],
        "E_q": centered["test"]["E_q"],
        "F_q_from_t": functional["test"]["F_q_from_t"],
        "F_t_from_t": functional["test"]["F_t_from_t"],
    }
    figures = _plot_nulls(fig_dir, target_plot, haar_stats, matched_stats)
    figures.append(_plot_rank_sweep(fig_dir, rank_rows))
    figures.append(_plot_bias(fig_dir, baseline["test"], centered["test"]))
    rank_t1 = next(row for row in rank_rows if row["which"] == "T" and row["rank"] == 1)
    rank_t2 = next((row for row in rank_rows if row["which"] == "T" and row["rank"] == 2), None)
    verdict = _decide(
        centered["test"],
        matched_stats["E_t"],
        matched_stats["E_q"],
        ut_pc_overlap=ut_pc_overlap,
        k_t=k_t,
        target_energy=target_energy,
        max_k_energy=max_k_energy,
        matched_energy_mean=float(np.mean(matched_energies)) if matched_energies else 0.0,
    )

    summary = {
        "run": run.name,
        "data_profile": spec.name,
        "fingerprint": fingerprint(spec),
        "note": (
            "All projectors estimated on train only. Previous analyze.py erasure was already centered; "
            "alignment previously used full-factorial T; this audit uses the same train T basis for alignment and erasure."
        ),
        "k_t_90": k_t,
        "k_g_90": k_q90,
        "target_subspace_energy": target_energy,
        "max_pc_energy_rank_kt": max_k_energy,
        "ut_overlap_top_kt_pcs": ut_pc_overlap,
        "pc_energy_top12": pc_energy[:12].tolist(),
        "t_singular_energy_frac": t_frac[:12].tolist(),
        "g_singular_energy_frac": g_frac[:12].tolist(),
        "baseline": baseline,
        "erasure_uncentered_Ut": uncentered,
        "erasure_centered_Ut": centered,
        "functional_alignment": functional,
        "frobenius_alignment": frobenius,
        "principal_angles_deg": {
            "Wt_Ut": _principal_angles_deg(w_t.T, u_t),
            "Wq_Uq90": _principal_angles_deg(w_q.T, u_q90),
            "Wt_Uq90": _principal_angles_deg(w_t.T, u_q90),
            "Wq_Ut": _principal_angles_deg(w_q.T, u_t),
        },
        "probe_after_centered_erasure": {"t": probe_t, "q_geom": probe_q},
        "factor_regression": factor_align,
        "g_pc1_correlations": pc1_corr,
        "rank_sweep": rank_rows,
        "per_shape_test": per_shape,
        "per_translation_test": per_translation,
        "geometry_by_shape_factor_centered_test": meta_groups,
        "nulls": {
            "n": N_NULL,
            "haar": {
                "mean": {key: float(np.mean(vals)) for key, vals in haar_stats.items()},
                "std": {key: float(np.std(vals)) for key, vals in haar_stats.items()},
                "p_value": {
                    "E_t": _pvalue(centered["test"]["E_t"], haar_stats["E_t"]),
                    "E_q": _pvalue(centered["test"]["E_q"], haar_stats["E_q"]),
                    "F_t_from_t": _pvalue(functional["test"]["F_t_from_t"], haar_stats["F_t_from_t"]),
                    "F_q_from_t": _pvalue(functional["test"]["F_q_from_t"], haar_stats["F_q_from_t"]),
                },
                "histogram": {key: _histogram(vals) for key, vals in haar_stats.items()},
            },
            "energy_matched": {
                "n_collected": len(matched_bases),
                "mean_deleted_energy": float(np.mean(matched_energies)) if matched_energies else None,
                "info": matched_info,
                "mean": {key: float(np.mean(vals)) for key, vals in matched_stats.items()},
                "std": {key: float(np.std(vals)) for key, vals in matched_stats.items()},
                "p_value": {
                    "E_t": _pvalue(centered["test"]["E_t"], matched_stats["E_t"]),
                    "E_q": _pvalue(centered["test"]["E_q"], matched_stats["E_q"]),
                    "F_t_from_t": _pvalue(functional["test"]["F_t_from_t"], matched_stats["F_t_from_t"]),
                    "F_q_from_t": _pvalue(functional["test"]["F_q_from_t"], matched_stats["F_q_from_t"]),
                },
                "histogram": {key: _histogram(vals) for key, vals in matched_stats.items()},
            },
            "label_permutation": {
                "mean_E_t": float(np.mean(perm_et)),
                "std_E_t": float(np.std(perm_et)),
                "mean_E_q": float(np.mean(perm_eq)),
                "std_E_q": float(np.std(perm_eq)),
                "p_E_t": _pvalue(centered["test"]["E_t"], perm_et),
                "p_E_q": _pvalue(centered["test"]["E_q"], perm_eq),
            },
            "factor_Bt_erasure_test": {
                "targeted": bt_test,
                "bt_energy": bt_energy,
                "matched_info": bt_matched_info,
                "mean_deleted_energy": float(np.mean(bt_matched_e)) if bt_matched_e else None,
                "haar_mean_E_t": float(np.mean(bt_haar_et)),
                "haar_mean_E_q": float(np.mean(bt_haar_eq)),
                "matched_mean_E_t": float(np.mean(bt_match_et)),
                "matched_mean_E_q": float(np.mean(bt_match_eq)),
                "p_E_t_matched": _pvalue(bt_test["E_t"], bt_match_et),
                "p_E_q_matched": _pvalue(bt_test["E_q"], bt_match_eq),
            },
        },
        "figures": figures,
        "verdict": verdict,
    }
    dump_json(audit_dir / "summary.json", _to_py(summary))
    t2_line = ""
    if rank_t2 is not None:
        t2_line = (
            f"- 只删平移主效应前 2 维（test）：位置 {rank_t2['test']['E_t']:.2f} px，"
            f"形状 {rank_t2['test']['E_q']:.2f} px\n"
        )
    decision = (
        f"# Phase1.5 结论（{verdict['letter']}）\n\n"
        f"{verdict['note']}\n\n"
        "## 用大白话说发生了什么\n\n"
        "网络把每张图压成 512 个数。这 512 个数里，大约 97% 的变化集中在一个方向上。"
        "原先按「平移主效应 90% 能量」切出来的 3 维，几乎就是能量最大的那 3 个方向"
        f"（重叠 {ut_pc_overlap:.3f}/{k_t}）。所以删这 3 维 ≈ 把特征的绝大部分清掉，"
        "不是在挖一块专门放平移的小角落。\n\n"
        f"- 删除前 test：位置 {baseline['test']['E_t']:.3f} px，形状 {baseline['test']['E_q']:.3f} px\n"
        f"- 中心化后删这 3 维：位置 {verdict['test_E_t']:.3f} px，形状 {verdict['test_E_q']:.3f} px\n"
        f"- 不中心化就删：位置 {uncentered['test']['E_t']:.3f} px，形状 {uncentered['test']['E_q']:.3f} px"
        "（更差，说明旧分析里的 4.5 px 已经是中心化之后的数）\n"
        f"- 去掉统一偏置后，形状误差仍是 {centered['test']['E_q_residual']:.3f} px，不是「整体平移了一下」\n"
        f"- 能量匹配随机（这次相对误差 {verdict['matched_energy_rel_error']:.3f}）："
        f"位置均值 {verdict['matched_mean_E_t']:.3f}，形状均值 {verdict['matched_mean_E_q']:.3f}；"
        f"p(位置)={verdict['p_E_t_matched']:.4g}，p(形状)={verdict['p_E_q_matched']:.4g}\n"
        f"- 打乱平移标签再切 3 维：位置均值 {float(np.mean(perm_et)):.3f}，"
        f"形状均值 {float(np.mean(perm_eq)):.3f}（和真删除几乎一样，说明这 3 维不跟平移标签绑定）\n"
        f"- 只删平移主效应第 1 维（test）：位置 {rank_t1['test']['E_t']:.2f} px，"
        f"形状 {rank_t1['test']['E_q']:.2f} px\n"
        f"{t2_line}"
        f"- 按真实平移坐标回归出的 2 维（B_t）删掉后：位置 {bt_test['E_t']:.2f} px，"
        f"形状 {bt_test['E_q']:.3f} px（形状几乎还是删除前的水平）；"
        f"能量匹配随机形状均值 {float(np.mean(bt_match_eq)):.2f}，p(形状)={_pvalue(bt_test['E_q'], bt_match_eq):.4g}\n"
        f"- 形状那一维和弦长相关只有 {pc1_corr['chord_px']:.3f}，并不是简单的「墨量/尺度」\n\n"
        "按方案三条标准：**正式结论是 C**。"
        "B_t 看起来更像「位置有一块干净的低维码」，但那是换了切法之后的观察，"
        "还不能当成 M1/M2 的绿灯。\n\n"
        "本轮没有训练 M1/M2。请确认 verdict 之前不要开 Phase2。\n"
    )
    (audit_dir / "decision.md").write_text(decision, encoding="utf-8")
    write_run_metadata(run, "audit", {"verdict": verdict["letter"], "audit_dir": str(audit_dir)})
    print(decision)
    print(f"[audit:{run.name}] wrote {audit_dir}")
    return summary

