"""Figures for the local mechanism probe. Always setup Chinese fonts first."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from phase1_gap_rep.common import setup_matplotlib_chinese

from .const import RESULTS_ROOT, STAGES


def _figdir() -> Path:
    d = RESULTS_ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_fft(payload: Mapping[str, Any]) -> None:
    plt = setup_matplotlib_chinese()
    figdir = _figdir()
    for fam_name, fam in payload.items():
        plot_blob = fam.get("_plot") or {}
        stages = list(fam.get("stages") or {})
        steps = []
        if stages:
            steps = [int(s) for s in (fam["stages"][stages[0]].get("steps") or {}) if str(s).isdigit()]
            steps.sort()
        if not steps:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
        ax = axes[0]
        for stage in stages:
            rec0 = plot_blob.get(f"{stage}_{steps[0]}")
            rec1 = plot_blob.get(f"{stage}_1")
            src = rec1 or rec0
            if src is None:
                continue
            prof = src.get("delta_from_step0", src) if rec1 is not None else src
            if isinstance(prof, dict) and "radial" in prof:
                profile = prof["radial"]["profile"]
            else:
                profile = src["radial"]["profile"]
            ax.plot(profile, label=stage, lw=1.6)
        ax.set_xlabel("径向波数 r")
        ax.set_ylabel("平均功率")
        ax.set_title(f"{fam_name} step1 Δu 径向谱")
        ax.legend(fontsize=8)
        ax = axes[1]
        x = np.arange(len(stages))
        peak = []
        low = []
        for stage in stages:
            rec = ((fam.get("stages") or {}).get(stage) or {}).get("steps") or {}
            s1 = rec.get("1") or rec.get("1.0")
            if not s1:
                peak.append(0.0)
                low.append(0.0)
                continue
            fft = (s1.get("delta_from_step0") or s1).get("fft") or s1.get("fft") or {}
            peak.append(float(fft.get("peak_frac") or 0.0))
            low.append(float(fft.get("low_freq_rle2_frac") or 0.0))
        ax.bar(x - 0.18, peak, 0.36, label="peak_frac")
        ax.bar(x + 0.18, low, 0.36, label="low_r≤2")
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_ylabel("能量占比")
        ax.set_title(f"{fam_name} step1 Δu 能量集中度")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"s1_{fam_name}_radial.png", dpi=140)
        plt.close(fig)

        n_st = max(len(stages), 1)
        fig, axes = plt.subplots(1, n_st, figsize=(4.0 * n_st, 3.6), squeeze=False)
        for i, stage in enumerate(stages):
            rec = plot_blob.get(f"{stage}_1")
            ax = axes[0][i]
            if rec is None:
                ax.set_axis_off()
                continue
            img = rec.get("_delta_log_power")
            if img is None:
                img = rec.get("log_power")
            im = ax.imshow(np.asarray(img), origin="lower", cmap="magma")
            ax.set_title(f"{stage} step1 log|FFT Δu|")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figdir / f"s1_{fam_name}_step1_heatmap.png", dpi=140)
        plt.close(fig)


def plot_rv(table: Mapping[str, Any]) -> None:
    plt = setup_matplotlib_chinese()
    figdir = _figdir()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, machine, title in (
        (axes[0], "amd_2d", "AMD 2D R(v)"),
        (axes[1], "local_2d", "本机 2D R(v)"),
    ):
        blob = table.get(machine) or {}
        xs = list(STAGES)
        unit = [float((blob.get(s) or {}).get("amp_unit") or np.nan) for s in xs]
        adam = [float((blob.get(s) or {}).get("amp_adam") or np.nan) for s in xs]
        rnd = [float((blob.get(s) or {}).get("rand_median") or np.nan) for s in xs]
        x = np.arange(len(xs))
        ax.bar(x - 0.25, unit, 0.25, label="unit-g")
        ax.bar(x, adam, 0.25, label="AdamW")
        ax.bar(x + 0.25, rnd, 0.25, label="random median")
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(xs)
        ax.set_ylabel("R(v)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "s4_rv_existing.png", dpi=140)
    plt.close(fig)


def plot_opt_audit(table: Mapping[str, Any]) -> None:
    plt = setup_matplotlib_chinese()
    figdir = _figdir()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    rows = table.get("rows") or []
    labels = [f"{r['stage']}/{r['opt']}" for r in rows]
    s1 = [float(r.get("box_step1") or np.nan) for r in rows]
    s10 = [float(r.get("box_step10") or np.nan) for r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, s1, 0.36, label="step1 box")
    ax.bar(x + 0.18, s10, 0.36, label="step10 box")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean_box_mae_px")
    ax.set_title("2D 10-step optimizer 审计")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "s2_opt_audit.png", dpi=140)
    plt.close(fig)


def plot_tangent(table: Mapping[str, Any]) -> None:
    plt = setup_matplotlib_chinese()
    figdir = _figdir()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    stages = list(table.get("stages") or STAGES)
    gfrac = [float(((table.get("stages") or {}).get(s) or {}).get("energy_g_topk") or np.nan) for s in stages]
    afrac = [float(((table.get("stages") or {}).get(s) or {}).get("energy_adam_topk") or np.nan) for s in stages]
    ffrac = [float(((table.get("stages") or {}).get(s) or {}).get("energy_df_topk") or np.nan) for s in stages]
    x = np.arange(len(stages))
    ax.bar(x - 0.25, gfrac, 0.25, label="g in top-k")
    ax.bar(x, afrac, 0.25, label="Adam Δθ in top-k")
    ax.bar(x + 0.25, ffrac, 0.25, label="Δf_step1 in J top-k")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("能量比例")
    ax.set_ylim(0, 1.05)
    ax.set_title("support-Hessian top-k 投影")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "s3_tangent.png", dpi=140)
    plt.close(fig)


def plot_mlp_phase(table: Mapping[str, Any]) -> None:
    plt = setup_matplotlib_chinese()
    figdir = _figdir()
    rows = table.get("rows") or []
    arches = []
    supports = []
    for r in rows:
        if r.get("arch") not in arches:
            arches.append(r["arch"])
        if r.get("support") not in supports:
            supports.append(r["support"])
    stages = ("last", "full")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, stage in zip(axes, stages):
        mat = np.full((len(arches), len(supports)), np.nan)
        for r in rows:
            if r.get("stage") != stage:
                continue
            i = arches.index(r["arch"])
            j = supports.index(r["support"])
            mat[i, j] = float(r.get("box_ratio") or np.nan)
        im = ax.imshow(mat, origin="upper", cmap="YlOrRd")
        ax.set_xticks(range(len(supports)))
        ax.set_xticklabels(supports)
        ax.set_yticks(range(len(arches)))
        ax.set_yticklabels(arches)
        ax.set_title(f"MLP box_ratio  {stage}")
        for i in range(len(arches)):
            for j in range(len(supports)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figdir / "l3_mlp_phase.png", dpi=140)
    plt.close(fig)
