"""ANOVA / SVD / readout alignment / translation-subspace erasure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .common import (
    ANALYSIS_GATE_MAE_PX,
    COORD_SCALE,
    ENERGY_RANK_FRACTIONS,
    RANDOM_ERASURE_REPEATS,
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
from .generate_data import load_split


def _load_features(run_name: str, data_profile: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = profile_spec(data_profile)
    z_path = run_dirs(run_name)["features"] / "Z.npz"
    head_path = run_dirs(run_name)["features"] / "head.npz"
    if not z_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"missing features in {run_dirs(run_name)['features']}; run extract --run {run_name}")
    with np.load(z_path, allow_pickle=False) as payload:
        if str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) != fingerprint(spec):
            raise RuntimeError("Z fingerprint mismatch")
        z = np.asarray(payload["Z"], dtype=np.float64)
    with np.load(head_path, allow_pickle=False) as payload:
        if str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) != fingerprint(spec):
            raise RuntimeError("head fingerprint mismatch")
        weight = np.asarray(payload["W"], dtype=np.float64)
        bias = np.asarray(payload["b"], dtype=np.float64)
    return z, weight, bias


def _load_split_codes(profile: str) -> np.ndarray:
    spec = profile_spec(profile)
    path = dirs(spec)["data"] / "split.npz"
    with np.load(path, allow_pickle=False) as payload:
        if str(np.asarray(payload["fingerprint"]).reshape(-1)[0]) != fingerprint(spec):
            raise RuntimeError("split fingerprint mismatch")
        return np.asarray(payload["split_codes"])


def _labels(profile: str) -> Dict[str, np.ndarray]:
    spec = profile_spec(profile)
    p = np.zeros((spec.n_shapes, spec.n_translations, 6), dtype=np.float64)
    q = np.zeros((spec.n_shapes, spec.n_translations, 6), dtype=np.float64)
    t = np.zeros((spec.n_shapes, spec.n_translations, 2), dtype=np.float64)
    for split in ("train", "val", "test"):
        data = load_split(spec, split)
        sid = data["shape_id"].astype(np.int64)
        tid = data["translation_id"].astype(np.int64)
        p[sid, tid] = data["P"]
        q[sid, tid] = data["Q"]
        t[sid, tid] = data["t"]
    return {"P": p, "Q": q, "t": t}


def anova(z: np.ndarray) -> Dict[str, Any]:
    mu = z.mean(axis=(0, 1), keepdims=False)
    t_effect = z.mean(axis=0) - mu
    g_effect = z.mean(axis=1) - mu
    residual = z - mu - t_effect[None, :, :] - g_effect[:, None, :]
    energy_t = float(np.mean(np.sum(t_effect**2, axis=-1)))
    energy_g = float(np.mean(np.sum(g_effect**2, axis=-1)))
    energy_i = float(np.mean(np.sum(residual**2, axis=-1)))
    energy_total = float(np.mean(np.sum((z - mu) ** 2, axis=-1)))
    reconstructed = energy_t + energy_g + energy_i
    return {
        "mu": mu,
        "T": t_effect,
        "G": g_effect,
        "I": residual,
        "energy": {
            "definition": "mean over factorial cells of squared Euclidean norm; for the balanced 64x64 design E_total = E_T + E_G + E_I",
            "E_T": energy_t,
            "E_G": energy_g,
            "E_I": energy_i,
            "E_total": energy_total,
            "E_T_frac": energy_t / energy_total if energy_total else 0.0,
            "E_G_frac": energy_g / energy_total if energy_total else 0.0,
            "E_I_frac": energy_i / energy_total if energy_total else 0.0,
            "reconstruction_abs_err": abs(energy_total - reconstructed),
        },
    }


def svd_spectrum(matrix: np.ndarray) -> Dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    total = float(energy.sum())
    fractions = energy / total if total else np.zeros_like(energy)
    cumulative = np.cumsum(fractions)
    ranks = {}
    n_comp = int(len(cumulative))
    for frac in ENERGY_RANK_FRACTIONS:
        if total:
            ranks[f"rank_{int(frac * 100)}"] = int(min(n_comp, np.searchsorted(cumulative, frac) + 1))
        else:
            ranks[f"rank_{int(frac * 100)}"] = 0
    k90 = int(min(n_comp, np.searchsorted(cumulative, READOUT_ENERGY_FRACTION) + 1)) if total else 0
    k90 = max(1, min(k90, vh.shape[0]))
    return {
        "singular_values": singular,
        "energy_fractions": fractions,
        "cumulative": cumulative,
        "ranks": ranks,
        "k90": k90,
        "basis": vh[:k90].T,
    }


def _frobenius_alignment(weight: np.ndarray, basis: np.ndarray) -> float:
    if np.linalg.norm(weight) == 0:
        return 0.0
    projected = weight @ basis
    return float(np.sum(projected**2) / np.sum(weight**2))


def _principal_angles_deg(a: np.ndarray, b: np.ndarray) -> List[float]:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    svals = np.linalg.svd(qa.T @ qb, compute_uv=False)
    svals = np.clip(svals, 0.0, 1.0)
    return [float(np.degrees(np.arccos(value))) for value in svals]


def _train_translation_basis(z: np.ndarray, split_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    train = split_codes == 0
    mu = z[train].mean(axis=0)
    n_t = z.shape[1]
    effects = np.zeros((n_t, z.shape[-1]), dtype=np.float64)
    for t_id in range(n_t):
        mask = train[:, t_id]
        effects[t_id] = z[mask, t_id].mean(axis=0) - mu
    spectrum = svd_spectrum(effects)
    return mu, spectrum["basis"], int(spectrum["k90"])


def _random_basis(dim: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=(dim, rank))
    q, _ = np.linalg.qr(noise)
    return q[:, :rank]


def _predict_from_z(z_vec: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return z_vec @ weight.T + bias


def _split_errors(pred: np.ndarray, labels: Dict[str, np.ndarray]) -> Dict[str, float]:
    parts = decompose_controls(pred)
    t_err = np.abs(parts["t"] - labels["t"])
    q_err = np.abs(parts["q"].reshape(pred.shape[0], 6) - labels["Q"])
    p_err = np.abs(pred - labels["P"])
    geom = decompose_controls(labels["P"])
    g_err = np.abs(parts["geometry_comp"] - geom["geometry_comp"])
    return {
        "coord_mae_px": float(p_err.mean()) * COORD_SCALE,
        "translation_mae_px": float(t_err.mean()) * COORD_SCALE,
        "relative_geometry_mae_px": float(q_err.mean()) * COORD_SCALE,
        "geometry_comp_mae": float(g_err.mean()),
    }


def _linear_probe(features: np.ndarray, targets: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> Dict[str, float]:
    ones = np.ones((features.shape[0], 1), dtype=np.float64)
    design = np.concatenate([features, ones], axis=1)
    coef, *_ = np.linalg.lstsq(design[train_mask], targets[train_mask], rcond=None)
    pred = design @ coef
    out = {}
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        err = np.abs(pred[mask] - targets[mask]).mean() * COORD_SCALE
        out[f"{name}_translation_mae_px"] = float(err)
    return out


def _plot_spectra(figure_dir: Path, t_spec: Dict[str, Any], g_spec: Dict[str, Any]) -> List[str]:
    plt = setup_matplotlib_chinese()
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, spec, title in (
        (axes[0], t_spec, "translation 主效应奇异值"),
        (axes[1], g_spec, "geometry 主效应奇异值"),
    ):
        ax.plot(np.arange(1, len(spec["singular_values"]) + 1), spec["singular_values"], marker="o", ms=3)
        ax.set_xlabel("成分")
        ax.set_ylabel("奇异值")
        ax.set_title(title)
        ax.set_yscale("log")
    fig.tight_layout()
    path = figure_dir / "svd_singular_values.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(np.arange(1, len(t_spec["cumulative"]) + 1), t_spec["cumulative"], label="translation T")
    ax.plot(np.arange(1, len(g_spec["cumulative"]) + 1), g_spec["cumulative"], label="geometry G")
    for frac in ENERGY_RANK_FRACTIONS:
        ax.axhline(frac, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("维数")
    ax.set_ylabel("累计解释能量")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("累计变差能量（不要用二维 PCA 图判断分解）")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "svd_cumulative_energy.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path.name)
    return written


def analyze(profile: str = "factorial") -> Dict[str, Any]:
    raise RuntimeError(
        "legacy analyze writes to factorial wreckage; "
        "use: python phase1_gap_rep/run.py analyze --run adamw_l1_scratch"
    )


def analyze_run(run_name: str) -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    d = run_dirs(run.name)
    z, weight, bias = _load_features(run.name, spec.name)
    split_codes = _load_split_codes(spec.name)
    labels = _labels(spec.name)
    factors = anova(z)
    t_spec = svd_spectrum(factors["T"])
    g_spec = svd_spectrum(factors["G"])
    matrix_a = orthogonal_matrix_a()
    weight_prime = matrix_a @ weight
    w_t = weight_prime[[0, 3]]
    w_q = weight_prime[[1, 2, 4, 5]]
    r_t = t_spec["basis"]
    r_q = g_spec["basis"]
    alignment = {
        "k_t_90": t_spec["k90"],
        "k_q_90": g_spec["k90"],
        "W_t_on_P_t": _frobenius_alignment(w_t, r_t),
        "W_t_on_P_q": _frobenius_alignment(w_t, r_q),
        "W_q_on_P_q": _frobenius_alignment(w_q, r_q),
        "W_q_on_P_t": _frobenius_alignment(w_q, r_t),
        "principal_angles_Wt_Rt_deg": _principal_angles_deg(w_t.T, r_t),
        "principal_angles_Wq_Rq_deg": _principal_angles_deg(w_q.T, r_q),
        "principal_angles_Wt_Rq_deg": _principal_angles_deg(w_t.T, r_q),
        "principal_angles_Wq_Rt_deg": _principal_angles_deg(w_q.T, r_t),
    }

    mu_train, p_t_basis, k_t = _train_translation_basis(z, split_codes)
    z_flat = z.reshape(-1, 512)
    centered = z_flat - mu_train
    z_no_t = mu_train + centered - (centered @ p_t_basis) @ p_t_basis.T
    pred_base = _predict_from_z(z_flat, weight, bias).reshape(spec.n_shapes, spec.n_translations, 6)
    pred_no_t = _predict_from_z(z_no_t, weight, bias).reshape(spec.n_shapes, spec.n_translations, 6)

    rng = np.random.default_rng(SEED + 17)
    random_rows: List[Dict[str, float]] = []
    for repeat in range(RANDOM_ERASURE_REPEATS):
        basis = _random_basis(512, k_t, rng)
        z_rand = mu_train + centered - (centered @ basis) @ basis.T
        pred_rand = _predict_from_z(z_rand, weight, bias).reshape(spec.n_shapes, spec.n_translations, 6)
        random_rows.append(_split_errors(pred_rand.reshape(-1, 6), {key: value.reshape(-1, value.shape[-1]) for key, value in labels.items()}))

    labels_flat = {key: value.reshape(-1, value.shape[-1]) for key, value in labels.items()}
    before = _split_errors(pred_base.reshape(-1, 6), labels_flat)
    after = _split_errors(pred_no_t.reshape(-1, 6), labels_flat)
    random_mean = {key: float(np.mean([row[key] for row in random_rows])) for key in before}
    random_std = {key: float(np.std([row[key] for row in random_rows])) for key in before}

    train_mask = (split_codes == 0).reshape(-1)
    val_mask = (split_codes == 1).reshape(-1)
    test_mask = (split_codes == 2).reshape(-1)
    probe = _linear_probe(z_no_t, labels_flat["t"], train_mask, val_mask, test_mask)

    figures = _plot_spectra(d["figures"], t_spec, g_spec)
    summary = {
        "run": run.name,
        "data_profile": spec.name,
        "fingerprint": fingerprint(spec),
        "loss": f"MSE + {run.l1_weight} * L1",
        "energy": factors["energy"],
        "translation_ranks": t_spec["ranks"],
        "geometry_ranks": g_spec["ranks"],
        "readout_alignment": alignment,
        "erasure": {
            "subspace_from": "train-only translation main effect SVD, rank = 90% energy",
            "k_t": k_t,
            "before": before,
            "translation_erasure": after,
            "random_erasure_mean": random_mean,
            "random_erasure_std": random_std,
            "random_repeats": RANDOM_ERASURE_REPEATS,
        },
        "linear_probe_z_no_t_to_translation": probe,
        "analysis_gate_mae_px": ANALYSIS_GATE_MAE_PX,
        "figures": figures,
        "note": (
            "1 px is the analysis gate, not the training early-stop. "
            "Energy uses the full balanced factorial. P_t for erasure is estimated from train cells only. "
            "This representation was trained with MSE + 0.25 L1."
        ),
    }
    dump_json(d["tables"] / "summary.json", summary)
    write_run_metadata(run, "analyze", {"summary_keys": list(summary)})
    print(
        f"[analyze:{run.name}] E_T={factors['energy']['E_T_frac']:.3f} "
        f"E_G={factors['energy']['E_G_frac']:.3f} E_I={factors['energy']['E_I_frac']:.3f}"
    )
    print(f"[analyze:{run.name}] T ranks={t_spec['ranks']} G ranks={g_spec['ranks']}")
    print(
        f"[analyze:{run.name}] align Wt-Pt={alignment['W_t_on_P_t']:.3f} Wt-Pq={alignment['W_t_on_P_q']:.3f} "
        f"Wq-Pq={alignment['W_q_on_P_q']:.3f} Wq-Pt={alignment['W_q_on_P_t']:.3f}"
    )
    print(
        f"[analyze:{run.name}] t MAE px before={before['translation_mae_px']:.3f} "
        f"erase_t={after['translation_mae_px']:.3f} random={random_mean['translation_mae_px']:.3f}"
    )
    print(
        f"[analyze:{run.name}] geometry MAE px before={before['relative_geometry_mae_px']:.3f} "
        f"erase_t={after['relative_geometry_mae_px']:.3f} random={random_mean['relative_geometry_mae_px']:.3f}"
    )
    return summary
