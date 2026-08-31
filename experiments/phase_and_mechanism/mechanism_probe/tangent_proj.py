"""S3: support-loss Hessian top-k via finite-difference HVP. No full NTK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from error_extension.field_metrics import load_field_npz
from error_extension.freeze import enter_train
from functional_drift.blob2d import corners4_pack, train_anchor_mae_px
from phase1_gap_rep.common import seed_everything
from today_shortcycle.calib import mae_px

from .blob_io import (
    add_flat,
    cosine,
    device,
    load_blob,
    official_eval,
    pack_grads,
    pack_params,
    predict_px,
    prepare_stage,
    restore_state,
    rms,
    snapshot_state,
    support_loss,
    trainable,
    write_json,
)
from .const import (
    ADAMW_BETAS,
    CODE_REV,
    DRIFT_ROOT,
    FD_REL_RMS,
    HESS_ITERS,
    HESS_K,
    LR_USED,
    RESULTS_ROOT,
    SEED,
    STAGES,
    WD,
)


def _choose_eps(theta: torch.Tensor, vec: torch.Tensor) -> float:
    n = max(int(theta.numel()), 1)
    rms_theta = float(torch.sqrt(torch.mean(theta.float() ** 2)).item())
    if not np.isfinite(rms_theta) or rms_theta <= 0:
        rms_theta = 1.0
    vn = float(torch.linalg.vector_norm(vec.float()).item())
    if vn < 1e-12:
        vn = 1.0
    # rms(eps * v/||v||) / rms(theta) ≈ FD_REL_RMS
    return float(FD_REL_RMS * rms_theta * np.sqrt(n) * vn)


def _grad_flat(model: nn.Module, images: np.ndarray, xy_norm: np.ndarray) -> torch.Tensor:
    enter_train(model)
    params = trainable(model)
    for p in params:
        if p.grad is not None:
            p.grad = None
    loss = support_loss(model, images, xy_norm)
    loss.backward()
    return pack_grads(params).detach()


def _hvp_fd(
    model: nn.Module,
    snap: Dict[str, torch.Tensor],
    stage: str,
    images: np.ndarray,
    xy_norm: np.ndarray,
    vec: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    params = trainable(model)
    restore_state(model, snap)
    prepare_stage(model, stage)
    add_flat(params, vec, float(eps))
    g_plus = _grad_flat(model, images, xy_norm).cpu()
    restore_state(model, snap)
    prepare_stage(model, stage)
    add_flat(trainable(model), vec, -float(eps))
    g_minus = _grad_flat(model, images, xy_norm).cpu()
    restore_state(model, snap)
    prepare_stage(model, stage)
    return (g_plus - g_minus) / (2.0 * float(eps))


def _qr_cols(mat: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


def _topk_hessian(
    model: nn.Module,
    snap: Dict[str, torch.Tensor],
    stage: str,
    images: np.ndarray,
    xy_norm: np.ndarray,
    k: int,
    niter: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    params = trainable(model)
    theta = pack_params(params).detach().cpu()
    n = int(theta.numel())
    k = int(min(k, n, 16))
    rng = torch.Generator().manual_seed(SEED)
    v = torch.randn(n, k, generator=rng, dtype=torch.float32)
    v = _qr_cols(v)
    dummy = torch.zeros(n, dtype=torch.float32)
    dummy[0] = 1.0
    eps0 = _choose_eps(theta, dummy)
    for it in range(int(niter)):
        w = torch.zeros_like(v)
        for j in range(k):
            col = v[:, j].contiguous()
            eps = _choose_eps(theta, col)
            hv = _hvp_fd(model, snap, stage, images, xy_norm, col, eps)
            w[:, j] = hv
        v = _qr_cols(w)
        print(f"    lanczos-ish iter {it+1}/{niter}", flush=True)
    w = torch.zeros_like(v)
    for j in range(k):
        col = v[:, j].contiguous()
        eps = _choose_eps(theta, col)
        w[:, j] = _hvp_fd(model, snap, stage, images, xy_norm, col, eps)
    a = v.T @ w
    evals, evecs = torch.linalg.eigh(0.5 * (a + a.T))
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    v = v @ evecs[:, order]
    return v.cpu().numpy(), evals.cpu().numpy(), float(eps0)


def _energy_in_span(vec: np.ndarray, basis: np.ndarray) -> float:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim == 1:
        b = b[:, None]
    q, _ = np.linalg.qr(b, mode="reduced")
    proj = q.T @ x
    nrm = float(np.dot(x, x))
    if nrm < 1e-18:
        return float("nan")
    return float(np.dot(proj, proj) / nrm)


def _load_df_step1(stage: str) -> Optional[np.ndarray]:
    local = RESULTS_ROOT / "s2_opt_audit" / f"{stage}__adamw_reset" / "traj"
    f0 = local / "step_0000" / "field.npz"
    f1 = local / "step_0001" / "field.npz"
    if not (f0.exists() and f1.exists()):
        f0 = DRIFT_ROOT / "amd" / stage / "traj" / "step_0000" / "field.npz"
        f1 = DRIFT_ROOT / "amd" / stage / "traj" / "step_0001" / "field.npz"
    if not (f0.exists() and f1.exists()):
        return None
    a = load_field_npz(f0)
    b = load_field_npz(f1)
    return np.asarray(b["pred"], dtype=np.float64) - np.asarray(a["pred"], dtype=np.float64)


def _jvp_field(
    model: nn.Module,
    snap: Dict[str, torch.Tensor],
    stage: str,
    vec: torch.Tensor,
    eps: float,
    f0: np.ndarray,
    s0: np.ndarray,
    support_images: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    restore_state(model, snap)
    prepare_stage(model, stage)
    add_flat(trainable(model), vec, float(eps))
    f1 = np.asarray(official_eval(model, name="jvp")["pred"], dtype=np.float64)
    s1 = predict_px(model, support_images)
    restore_state(model, snap)
    prepare_stage(model, stage)
    return (f1 - f0) / float(eps), (s1 - s0) / float(eps)


def run_stage(stage: str, corners: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out = RESULTS_ROOT / "s3_tangent" / stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))

    seed_everything(SEED)
    model = load_blob()
    prepare_stage(model, stage)
    snap = snapshot_state(model)
    images = corners["images"]
    xy = corners["xy_norm"]
    true_s = corners["true_px"]
    f0 = np.asarray(official_eval(model, name=f"{stage}_s3_f0")["pred"], dtype=np.float64)
    s0 = predict_px(model, images)

    g = _grad_flat(model, images, xy).cpu()
    restore_state(model, snap)
    prepare_stage(model, stage)
    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=LR_USED, weight_decay=WD, betas=ADAMW_BETAS)
    for p, gcomp in zip(params, _split_like(g, params)):
        p.grad = gcomp.to(device=p.device, dtype=p.dtype)
    opt.step()
    adam_delta = (pack_params(trainable(model)).detach().cpu() - pack_params_from_snap(snap, model)).cpu()
    restore_state(model, snap)
    prepare_stage(model, stage)

    print(f"[S3] {stage} hessian top-{HESS_K}", flush=True)
    basis, evals, eps0 = _topk_hessian(model, snap, stage, images, xy, HESS_K, HESS_ITERS)

    df = _load_df_step1(stage)
    modes = []
    rvs = []
    theta = pack_params(trainable(model)).detach().cpu()
    for j in range(basis.shape[1]):
        vec = torch.from_numpy(np.asarray(basis[:, j], dtype=np.float32))
        vn = float(torch.linalg.vector_norm(vec).item())
        if vn < 1e-12:
            continue
        vec = vec / vn
        eps = _choose_eps(theta, vec)
        d_dense, d_sup = _jvp_field(model, snap, stage, vec, eps, f0, s0, images)
        rv = rms(d_dense) / max(rms(d_sup), 1e-12)
        rvs.append(float(rv))
        modes.append(d_dense.reshape(-1))
        print(f"    mode {j} R(v)={rv:.3f} eval={float(evals[j]):.4g}", flush=True)

    mode_mat = np.stack(modes, axis=1) if modes else np.zeros((f0.size, 1))
    df_energy = float("nan")
    if df is not None and modes:
        df_energy = _energy_in_span(df.reshape(-1), mode_mat)

    rec = {
        "code_rev": CODE_REV,
        "stage": stage,
        "n_trainable": int(basis.shape[0]),
        "k": int(basis.shape[1]),
        "eps0": eps0,
        "hessian_evals_desc": [float(x) for x in evals],
        "energy_g_topk": _energy_in_span(g.numpy(), basis),
        "energy_adam_topk": _energy_in_span(adam_delta.numpy(), basis),
        "energy_df_topk": df_energy,
        "R_v_modes": rvs,
        "R_v_modes_median": float(np.median(rvs)) if rvs else float("nan"),
        "R_v_modes_max": float(np.max(rvs)) if rvs else float("nan"),
        "df_source": "s2_or_amd_step1",
        "anchor0": float(mae_px(s0, true_s)),
    }
    write_json(out / "summary.json", rec)
    return rec


def pack_params_from_snap(snap: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
    chunks = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        chunks.append(snap[name].detach().reshape(-1).float().cpu())
    return torch.cat(chunks)


def _split_like(flat: torch.Tensor, params: List[nn.Parameter]) -> List[torch.Tensor]:
    out = []
    offset = 0
    cpu = flat.detach().cpu()
    for p in params:
        n = p.numel()
        out.append(cpu[offset : offset + n].reshape_as(p).clone())
        offset += n
    return out


def run() -> Dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    corners = corners4_pack()
    stages = {}
    for stage in STAGES:
        print(f"[S3] stage={stage}", flush=True)
        stages[stage] = run_stage(stage, corners)
    table = {"code_rev": CODE_REV, "stages": stages}
    write_json(RESULTS_ROOT / "s3_tangent" / "summary.json", table)
    from .plots import plot_tangent

    plot_tangent(table)
    print("[S3] done", flush=True)
    return table
