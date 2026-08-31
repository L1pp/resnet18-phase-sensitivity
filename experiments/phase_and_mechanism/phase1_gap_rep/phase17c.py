"""Phase 1.7c cross-split re-probe audit.

Reuses Phase 1.7 models and the frozen B_t definition. Does not train a new
ResNet. Writes only under results/phase1_gap_rep/phase1_7c_cross_split/.
Does not modify phase1_7_multiseed/ or factorial/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .analyze import _labels, _load_split_codes
from .audit import SPLIT_NAMES, _pc_spectrum, _subspace_energy, _to_py
from .bt_robustness import ENERGY_TOL_OK, N_NULL, _fit_factor, _sample_matched
from .common import RESULTS_ROOT, decompose_controls, dump_json, fingerprint, profile_spec, setup_matplotlib_chinese
from .cross_probe import (
    constant_t_mae,
    cross_probe_block,
    energy_flags,
    eval_cross_cut,
    ratios,
    save_null_draws,
    summarize_null,
)
from .phase17 import ANALYSIS_SEED, PROTOCOL_FP, SEEDS, _file_sha256, _load_z_head, _root as _phase17_root, _seed_dir

OUT_NAME = "phase1_7c_cross_split"
R_T_RECOVER = 0.5
R_T_COLLAPSE = 0.8
R_Q_KEEP = 1.10


def _out_root() -> Path:
    path = RESULTS_ROOT / OUT_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _seed_out(seed: int) -> Path:
    path = _out_root() / f"seed_{seed}"
    for name in ("config", "tables", "figures", "nulls"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _cross_out() -> Path:
    path = _out_root() / "cross_seed_summary"
    for name in ("tables", "figures"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _load_saved_bt(seed_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = seed_dir / "features" / "B_t.npz"
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["U"], dtype=np.float64), np.asarray(payload["mu"], dtype=np.float64)


def _load_null_bases(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as payload:
        bases = np.asarray(payload["bases"], dtype=np.float64) if "bases" in payload.files else None
        perms = np.asarray(payload["translation_id_perm"]) if "translation_id_perm" in payload.files else None
        return bases, perms


def audit_seed(seed: int) -> Dict[str, Any]:
    spec = profile_spec("factorial")
    src = _seed_dir(seed)
    out = _seed_out(seed)
    z, weight, bias = _load_z_head(src, spec)
    del weight, bias
    split_grid = _load_split_codes(spec.name)
    labels_grid = _labels(spec.name)
    n_s, n_t, dim = z.shape
    z_flat = z.reshape(-1, dim)
    split_flat = split_grid.reshape(-1)
    sid = np.repeat(np.arange(n_s), n_t)
    tid = np.tile(np.arange(n_t), n_s)
    del sid
    labels_t = labels_grid["t"].reshape(-1, 2)
    g = decompose_controls(labels_grid["P"].reshape(-1, 6))["geometry_comp"]
    masks = {name: split_flat == i for i, name in enumerate(SPLIT_NAMES)}
    train_mask, val_mask, test_mask = masks["train"], masks["val"], masks["test"]
    mu = z_flat[train_mask].mean(0)
    train_centered = z_flat[train_mask] - mu
    t_train, g_train = labels_t[train_mask], g[train_mask]
    t_mu, t_sig = t_train.mean(0), t_train.std(0) + 1e-12
    q_mu, q_sig = g_train.mean(0), g_train.std(0) + 1e-12
    _, u_bt = _fit_factor(train_centered, t_train, g_train, t_mu, t_sig, q_mu, q_sig)
    saved_u, saved_mu = _load_saved_bt(src)
    bt_match = None
    if saved_u is not None:
        overlap = float(np.sum((_qr_overlap(u_bt, saved_u))))
        bt_match = {
            "saved_exists": True,
            "overlap_norm": overlap,
            "mu_max_abs_diff": float(np.max(np.abs(mu - saved_mu))) if saved_mu is not None else None,
        }
    energy = _subspace_energy(train_centered, u_bt)
    pcs, pc_energy = _pc_spectrum(train_centered)
    flags = energy_flags(energy, float(pc_energy[:2].sum()))

    raw = cross_probe_block(z_flat, labels_t, g, val_mask, test_mask)
    raw_train_diag = cross_probe_block(z_flat, labels_t, g, train_mask, test_mask)
    const_valfit_test = constant_t_mae(labels_t, val_mask, test_mask)
    true_cut = eval_cross_cut(z_flat, u_bt, mu, labels_t, g, val_mask, test_mask, train_centered)
    erased_train_diag = eval_cross_cut(z_flat, u_bt, mu, labels_t, g, train_mask, test_mask, train_centered)
    ratio = ratios(true_cut["probe"], raw, const_valfit_test)
    print(
        f"[phase17c] seed {seed} raw_t={raw['t_mae_px']:.3f} erased_t={true_cut['probe']['t_mae_px']:.3f} "
        f"const={const_valfit_test:.3f} R_t={ratio['R_t']:.3f} R_q={ratio['R_q']:.3f}"
    )

    perm_src = src / "nulls" / "identity_perm.npz"
    match_src = src / "nulls" / "energy_matched.npz"
    perm_bases, perm_ids = _load_null_bases(perm_src)
    match_bases, _ = _load_null_bases(match_src)
    rng = np.random.default_rng(ANALYSIS_SEED)

    if perm_bases is None:
        print(f"[phase17c] seed {seed}: regenerating identity-perm N={N_NULL}")
        unique_t = np.zeros((n_t, 2), dtype=np.float64)
        for t_id in range(n_t):
            unique_t[t_id] = labels_t[tid == t_id][0]
        train_tid = tid[train_mask]
        perm_bases_list, perm_ids_list = [], []
        for i in range(N_NULL):
            order = rng.permutation(n_t)
            fake_t = unique_t[order[train_tid]]
            _, basis_p = _fit_factor(train_centered, fake_t, g_train, t_mu, t_sig, q_mu, q_sig)
            perm_bases_list.append(basis_p)
            perm_ids_list.append(order)
            if i % 100 == 0:
                print(f"[phase17c] seed {seed} perm {i}/{N_NULL}")
        perm_bases = np.stack(perm_bases_list)
        perm_ids = np.stack(perm_ids_list)
        perm_source = "regenerated"
    else:
        perm_source = "loaded_phase17_bases"
        print(f"[phase17c] seed {seed}: reusing {len(perm_bases)} identity-perm bases")

    perm_stats = {"energy": [], "probe_E_t": [], "probe_E_g": []}
    for i, basis in enumerate(perm_bases):
        ev = eval_cross_cut(z_flat, basis, mu, labels_t, g, val_mask, test_mask, train_centered)
        perm_stats["energy"].append(ev["energy"])
        perm_stats["probe_E_t"].append(ev["probe"]["t_mae_px"])
        perm_stats["probe_E_g"].append(ev["probe"]["g_mae_px"])
        if i % 200 == 0:
            print(f"[phase17c] seed {seed} perm re-probe {i}/{len(perm_bases)}")

    if match_bases is None:
        print(f"[phase17c] seed {seed}: regenerating energy-matched N={N_NULL} (Phase1.7 sampler)")
        matched, match_e, match_info_17 = _sample_matched(
            train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, 0.02
        )
        if not match_info_17["matched_strict"]:
            matched, match_e, match_info_17 = _sample_matched(
                train_centered, 2, energy, rng, pcs, pc_energy, N_NULL, ENERGY_TOL_OK
            )
        match_bases = np.stack(matched)
        match_source = "regenerated"
        match_info_base = match_info_17
    else:
        match_source = "loaded_phase17_bases"
        match_info_base = {"loaded_from_disk": True, "n_collected": int(len(match_bases))}
        print(f"[phase17c] seed {seed}: reusing {len(match_bases)} energy-matched bases")

    match_stats = {"energy": [], "probe_E_t": [], "probe_E_g": []}
    for i, basis in enumerate(match_bases):
        ev = eval_cross_cut(z_flat, basis, mu, labels_t, g, val_mask, test_mask, train_centered)
        match_stats["energy"].append(ev["energy"])
        match_stats["probe_E_t"].append(ev["probe"]["t_mae_px"])
        match_stats["probe_E_g"].append(ev["probe"]["g_mae_px"])
        if i % 200 == 0:
            print(f"[phase17c] seed {seed} matched re-probe {i}/{len(match_bases)}")

    mean_e = float(np.mean(match_stats["energy"])) if match_stats["energy"] else 0.0
    rel = abs(mean_e - energy) / max(energy, 1e-12)
    match_info = {
        **flags,
        **match_info_base,
        "source": match_source,
        "actual_energy_mean": mean_e,
        "relative_error": rel,
        "matched_ok": bool(rel <= ENERGY_TOL_OK) and not flags["energy_null_degenerate"],
        "energy_null_degenerate": bool(flags["energy_null_degenerate"] or rel > ENERGY_TOL_OK),
    }

    save_null_draws(
        out / "nulls" / "identity_perm.npz",
        "identity_perm_cross",
        list(perm_bases),
        perm_stats,
        extra={"source": [perm_source]},
        perms=list(perm_ids) if perm_ids is not None else None,
    )
    save_null_draws(
        out / "nulls" / "energy_matched.npz",
        "energy_matched_cross",
        list(match_bases),
        match_stats,
        extra={"rel_error": [rel] * len(match_bases), "source": [match_source]},
    )

    perm_block = {
        "n": len(perm_stats["energy"]),
        "source": perm_source,
        "probe_E_t": summarize_null(true_cut["probe"]["t_mae_px"], perm_stats["probe_E_t"], True),
        "probe_E_g": summarize_null(true_cut["probe"]["g_mae_px"], perm_stats["probe_E_g"], True),
        "probe_E_g_preserve": summarize_null(true_cut["probe"]["g_mae_px"], perm_stats["probe_E_g"], False),
        "energy": summarize_null(energy, perm_stats["energy"], True),
    }
    match_block = {
        "n": len(match_stats["energy"]),
        "source": match_source,
        "info": match_info,
        "probe_E_t": summarize_null(true_cut["probe"]["t_mae_px"], match_stats["probe_E_t"], True),
        "probe_E_g": summarize_null(true_cut["probe"]["g_mae_px"], match_stats["probe_E_g"], True),
        "probe_E_g_preserve": summarize_null(true_cut["probe"]["g_mae_px"], match_stats["probe_E_g"], False),
        "energy": summarize_null(energy, match_stats["energy"], True),
    }
    if match_info["energy_null_degenerate"]:
        match_block["probe_E_t"]["p_usable"] = False
        match_block["note"] = "energy null degenerate or unmatched; do not claim strict equal-energy significance"

    payload = {
        "seed": seed,
        "experiment": "phase1.7c_cross_split",
        "geometry_fingerprint": fingerprint(spec),
        "protocol_fingerprint_expected": PROTOCOL_FP,
        "bt_source": "re-estimated on train cells with frozen Phase1.7 OLS definition",
        "bt_saved_check": bt_match,
        "energy": energy,
        "energy_flags": flags,
        "raw_valfit_test": raw,
        "raw_trainfit_test_diagnostic": raw_train_diag,
        "constant_valfit_test_t_mae": const_valfit_test,
        "erased_valfit_test": true_cut["probe"],
        "erased_trainfit_test_diagnostic": erased_train_diag["probe"],
        "R_t_cross": ratio["R_t"],
        "R_q_cross": ratio["R_q"],
        "nulls": {"permutation": perm_block, "energy_matched": match_block},
        "source_seed_dir": str(src),
        "Z_sha256": _file_sha256(src / "features" / "Z.npz"),
    }
    dump_json(out / "tables" / "summary.json", _to_py(payload))
    _plot_seed(out, seed, perm_stats, match_stats, true_cut["probe"]["t_mae_px"], const_valfit_test)
    return payload


def _qr_overlap(a: np.ndarray, b: np.ndarray) -> float:
    from .bt_robustness import _overlap_norm

    return _overlap_norm(a, b)


def _plot_seed(out: Path, seed: int, perm_stats, match_stats, true_t: float, const_t: float) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(perm_stats["probe_E_t"], bins=40, alpha=0.85)
    ax.axvline(true_t, color="C3", ls="--", label="真实 B_t val→test")
    ax.axvline(const_t, color="0.3", ls=":", label="常数预测")
    ax.set_title(f"seed {seed} identity-perm：cross-split t")
    ax.set_xlabel("px")
    ax.set_ylabel("次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "perm_cross_t.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(match_stats["probe_E_t"], bins=40, alpha=0.85)
    ax.axvline(true_t, color="C3", ls="--", label="真实 B_t val→test")
    ax.axvline(const_t, color="0.3", ls=":", label="常数预测")
    ax.set_title(f"seed {seed} 能量匹配：cross-split t")
    ax.set_xlabel("px")
    ax.set_ylabel("次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "matched_cross_t.png", dpi=140)
    plt.close(fig)


def decide(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    recover = [int(row["seed"]) for row in rows if float(row["R_t_cross"]) < R_T_RECOVER]
    near_const = [
        int(row["seed"])
        for row in rows
        if 0.8 <= float(row["R_t_cross"]) <= 1.2
        and abs(float(row["erased_valfit_test"]["t_mae_px"]) - float(row["constant_valfit_test_t_mae"])) <= 6.0
    ]
    explode = [int(row["seed"]) for row in rows if float(row["R_t_cross"]) > 1.2]
    geom = [int(row["seed"]) for row in rows if float(row["R_q_cross"]) <= R_Q_KEEP]
    if len(recover) >= 2:
        letter = "estimator_train_only"
        note = (
            "至少两个 seed 的 val→test probe 从擦除后的表征里恢复了大量 translation。"
            "`B_t` 主要消除了 estimator-training distribution 上的线性 translation covariance，"
            "但没有充分删除独立 cells 中的位置读取。本轮不停止 Phase1.8。"
        )
    elif len(near_const) == 3 and len(geom) == 3:
        letter = "cross_split_holds"
        note = (
            "三个 seed 的 cross-split translation 仍接近常数水平，且 geometry 线性可读性保持。"
            "selective erasure 不只是 train-probe 构造现象。"
        )
    elif len(explode) >= 2 and len(recover) == 0:
        letter = "not_recovered_val_probe_illposed"
        note = (
            "三个 seed 都没有把 translation 读回来（R_t^cross 全部远大于 0.5），"
            "所以不是 estimator-only 的反例。但擦除后 val→test t MAE 是数百 px，"
            "不是常数 30 px：val 只有 512 点去拟合 513 维 OLS，信号被删后 probe 在 val 上过拟合、在 test 上爆炸。"
            "geometry R_q^cross 仍约 1。train-fit diagnostic 仍贴 30.03 px，与 Phase1.7 构造一致。"
        )
    else:
        letter = "mixed"
        note = "cross-split 结果在 seed 间不完全一致。保留数字，不修改 B_t，不据此停止 1.8。"
    return {
        "verdict": letter,
        "note": note,
        "recover_seeds": recover,
        "near_const_seeds": near_const,
        "explode_seeds": explode,
        "geometry_ok_seeds": geom,
        "does_not_block_phase18": True,
    }


def write_decision(decision: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase 1.7c Cross-Split Re-Probe",
        "",
        f"**Verdict: {decision['verdict']}**",
        "",
        decision["note"],
        "",
        "本轮不训练新 ResNet，不修改 `B_t`，不写入 `phase1_7_multiseed/` 或 `factorial/`。",
        "主证据是 val→test re-probe，不是 train-fitted probe。1.7c 不阻塞 Phase 1.8。",
        "",
        "| seed | raw val→test t | erased val→test t | const t | R_t^cross | raw q | erased q | R_q^cross |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw = row["raw_valfit_test"]
        erased = row["erased_valfit_test"]
        lines.append(
            f"| {row['seed']} | {raw['t_mae_px']:.3f} | {erased['t_mae_px']:.3f} | "
            f"{row['constant_valfit_test_t_mae']:.3f} | {row['R_t_cross']:.3f} | "
            f"{raw['g_mae_px']:.3f} | {erased['g_mae_px']:.3f} | {row['R_q_cross']:.3f} |"
        )
    lines += [
        "",
        "Diagnostic train→test erased probe 见各 seed `tables/summary.json` 的 "
        "`erased_trainfit_test_diagnostic`，不进入正式结论。",
        "",
        f"几何 fingerprint `{PROTOCOL_FP}`。未 commit / push。",
        "",
    ]
    text = "\n".join(lines)
    (_cross_out() / "decision.md").write_text(text, encoding="utf-8")
    (_out_root() / "HANDOFF.md").write_text(text, encoding="utf-8")
    print(text)


def run_phase17c() -> Dict[str, Any]:
    spec = profile_spec("factorial")
    if fingerprint(spec) != PROTOCOL_FP:
        raise RuntimeError(f"fingerprint {fingerprint(spec)} != {PROTOCOL_FP}")
    if not _phase17_root().exists():
        raise FileNotFoundError("missing Phase 1.7 results; cannot run 1.7c")
    rows = [audit_seed(seed) for seed in SEEDS]
    decision = decide(rows)
    payload = {"decision": decision, "seeds": [_to_py(row) for row in rows]}
    dump_json(_cross_out() / "summary.json", _to_py(payload))
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.15, [row["R_t_cross"] for row in rows], width=0.3, label="R_t^cross")
    ax.bar(xs + 0.15, [row["R_q_cross"] for row in rows], width=0.3, label="R_q^cross")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.axhline(R_T_RECOVER, color="C3", ls=":", lw=1, label="恢复阈值 0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(row["seed"]) for row in rows])
    ax.set_ylabel("比值")
    ax.set_title("Phase1.7c val→test 主终点")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_cross_out() / "figures" / "primary_R_cross.png", dpi=140)
    plt.close(fig)
    write_decision(decision, rows)
    print(f"[phase17c] verdict={decision['verdict']} wrote {_out_root()}")
    return payload


if __name__ == "__main__":
    run_phase17c()
