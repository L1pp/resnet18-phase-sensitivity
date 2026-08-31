"""Local GTX 1060 slice: universal translation operator + 4-model layer-wise.

No training. Does not write Phase 1/2 directories. Slim grid so 6GB / 7700K can finish.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from phase1_gap_rep.common import IMAGE_SIZE, device, setup_matplotlib_chinese
from phase2_overnight.features import extract_stage_gaps

from .ckpt_locator import ModelRef
from .group_law import commutativity_error, composition_error, inverse_error
from .io_util import dump_json, load_json
from .model_io import build_random_twin, load_ref
from .operators import apply_affine, fit_affine_ridge, mean_sq_error, normalized_error, pack_operator, truncate_rank
from .primitives import random_params, render_family
from .protocol import (
    EVAL_SHAPE_IDS,
    PHASE3_ROOT,
    SEED_RANDOM_INIT,
    T_HIGH_PX,
    T_LOW_PX,
    delta_key,
    in_safe_box,
)
from .render_bezier import render_at

OUT = PHASE3_ROOT / "local_1060_cat1"
SEED = 20260816
N_ID = 8
N_ORIGIN_1D = 5
BATCH_SIZE = 16
STAGES = ["layer1", "layer2", "layer3", "layer4", "gap"]
CONTENTS = ("noise", "blob", "line", "quadratic", "polygon")
DELTAS: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (2.0, 0.0),
    (-2.0, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
    (4.0, 0.0),
    (0.0, 4.0),
    (1.0, 1.0),
)
COMPOSE_PAIRS = (
    ((1.0, 0.0), (1.0, 0.0)),
    ((2.0, 0.0), (2.0, 0.0)),
    ((1.0, 0.0), (0.0, 1.0)),
)
INVERSES = ((1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0))
COMMUTE = (((1.0, 0.0), (0.0, 1.0)), ((2.0, 0.0), (0.0, 2.0)))
BEZIER_IDS = tuple(int(x) for x in EVAL_SHAPE_IDS[:N_ID])
PATCH = 32
LOCAL_ALPHAS: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
LOCAL_RANKS: Tuple[Any, ...] = (64, 128, "full")


def origins_px() -> np.ndarray:
    xs = np.linspace(67.0, 156.0, N_ORIGIN_1D)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx, yy], axis=-1).reshape(-1, 2)


def origin_split(n_origin: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_origin).astype(int).tolist()
    n_fit, n_val = 15, 5
    return {"fit": perm[:n_fit], "val": perm[n_fit : n_fit + n_val], "test": perm[n_fit + n_val :]}


def fit_operator(
    z_fit: np.ndarray,
    zp_fit: np.ndarray,
    z_val: np.ndarray,
    zp_val: np.ndarray,
    z_test: np.ndarray,
    zp_test: np.ndarray,
) -> Dict[str, Any]:
    """Narrower ridge/rank grid than Track A so 1060/7700K can finish."""
    best: Dict[str, Any] = {"e_val": float("inf"), "alpha": float(LOCAL_ALPHAS[0]), "rank": "full"}
    for alpha in LOCAL_ALPHAS:
        matrix_full, bias_full = fit_affine_ridge(z_fit, zp_fit, float(alpha))
        for rank in LOCAL_RANKS:
            matrix = truncate_rank(matrix_full, rank)
            err = normalized_error(apply_affine(z_val, matrix, bias_full), zp_val)
            if err < float(best["e_val"]):
                best = {
                    "e_val": err,
                    "alpha": float(alpha),
                    "rank": rank if rank == "full" else int(rank),
                    "matrix": matrix,
                    "bias": bias_full,
                }
    matrix = np.asarray(best["matrix"], dtype=np.float64)
    bias = np.asarray(best["bias"], dtype=np.float64)
    pred_test = apply_affine(z_test, matrix, bias)
    pred_val = apply_affine(z_val, matrix, bias)
    return {
        "matrix": matrix,
        "bias": bias,
        "alpha": float(best["alpha"]),
        "rank": best["rank"],
        "e_val": float(normalized_error(pred_val, zp_val)),
        "e_test": float(normalized_error(pred_test, zp_test)),
        "mse_test": float(mean_sq_error(pred_test, zp_test)),
        "n_fit": int(z_fit.shape[0]),
        "n_val": int(z_val.shape[0]),
        "n_test": int(z_test.shape[0]),
        "dim": int(z_fit.shape[1]),
    }


def _phase2(name: str, run_dir: str, arch: str) -> ModelRef:
    path = (
        PHASE3_ROOT.parent
        / "phase2_overnight_discovery"
        / "models"
        / run_dir
        / "checkpoints"
        / "best_slim.pt"
    )
    return ModelRef(
        name=name,
        path=str(path),
        arch=arch,
        variant="standard",
        seed=20260820,
        kind="phase2",
        role="primary",
        found=path.is_file(),
    )


def model_panel() -> List[Tuple[str, Any]]:
    return [
        ("r18_g64_good", _phase2("r18_g64", "resnet18__G64__20260820", "resnet18")),
        ("densenet_g64_good", _phase2("dn_g64", "densenet121__G64__20260820", "densenet121")),
        ("r18_g9_bad", _phase2("r18_g9", "resnet18__G9__20260820", "resnet18")),
        ("random_init", None),
    ]


def freeze_identities() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((N_ID, PATCH, PATCH)).astype(np.float32)
    params = {fam: [random_params(fam, rng, IMAGE_SIZE) for _ in range(N_ID)] for fam in CONTENTS if fam != "noise"}
    return {"bezier_ids": list(BEZIER_IDS), "noise": noise, "params": params}


def render_content(kind: str, ident: int, xy: np.ndarray, frozen: Mapping[str, Any]) -> np.ndarray:
    cx, cy = float(xy[0]), float(xy[1])
    if kind == "bezier":
        return render_at(int(frozen["bezier_ids"][ident]), xy)
    if kind == "noise":
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        patch = frozen["noise"][ident]
        scaled = np.clip(127.5 + 60.0 * patch, 0, 255).astype(np.uint8)
        x0 = int(round(cx - PATCH / 2.0))
        y0 = int(round(cy - PATCH / 2.0))
        x1, y1 = x0 + PATCH, y0 + PATCH
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(IMAGE_SIZE, x1), min(IMAGE_SIZE, y1)
        psx, psy = xs0 - x0, ys0 - y0
        canvas[ys0:ys1, xs0:xs1] = scaled[psy : psy + (ys1 - ys0), psx : psx + (xs1 - xs0)]
        return canvas
    params = frozen["params"][kind][ident]
    return render_family(kind, cx, cy, params, IMAGE_SIZE)


def extract_store(
    model,
    kind: str,
    frozen: Mapping[str, Any],
    origins: np.ndarray,
    stages: Sequence[str] = STAGES,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    n_s, n_o = N_ID, len(origins)
    feats: Dict[str, Dict[str, np.ndarray]] = {}
    valid: Dict[str, np.ndarray] = {}
    jobs_spec = [("src", (0.0, 0.0))] + [(delta_key(d), d) for d in DELTAS]
    for dkey, delta in jobs_spec:
        dest = origins + np.asarray(delta, dtype=np.float64).reshape(1, 2)
        ok = in_safe_box(dest)
        if dkey != "src":
            ok = ok & in_safe_box(origins)
        valid[dkey] = ok
        jobs = [(si, oi, dest[oi]) for si in range(n_s) for oi in range(n_o) if ok[oi]]
        bucket: Dict[str, np.ndarray] = {}
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]
            images = np.stack([render_content(kind, si, xy, frozen) for si, _oi, xy in chunk], axis=0)
            extracted = extract_stage_gaps(model, images, batch_size=len(chunk), stages=list(stages))
            if not bucket:
                for name, arr in extracted.items():
                    bucket[name] = np.full((n_s, n_o, arr.shape[-1]), np.nan, dtype=np.float32)
            for local, (si, oi, _xy) in enumerate(chunk):
                for name, arr in extracted.items():
                    bucket[name][si, oi] = arr[local]
        feats[dkey] = bucket
        print(f"    extracted {kind} {dkey} n={len(jobs)}", flush=True)
    return {"features": feats, "valid": {k: np.asarray(v, dtype=bool) for k, v in valid.items()}, "origins": origins}


def _pairs(store: Mapping[str, Any], stage: str, dkey: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = store["features"]["src"][stage]
    dst = store["features"][dkey][stage]
    ok = np.asarray(store["valid"][dkey], dtype=bool)
    z, zp, oids, valid = [], [], [], []
    n_s, n_o, _ = src.shape
    for si in range(n_s):
        for oi in range(n_o):
            good = bool(ok[oi]) and np.isfinite(src[si, oi]).all() and np.isfinite(dst[si, oi]).all()
            z.append(src[si, oi])
            zp.append(dst[si, oi])
            oids.append(oi)
            valid.append(good)
    return (
        np.asarray(z, dtype=np.float64),
        np.asarray(zp, dtype=np.float64),
        np.asarray(oids, dtype=int),
        np.asarray(valid, dtype=bool),
    )


def _split(
    z: np.ndarray, zp: np.ndarray, oids: np.ndarray, valid: np.ndarray, o_split: Mapping[str, Sequence[int]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name in ("fit", "val", "test"):
        mask = valid & np.isin(oids, np.asarray(o_split[name], dtype=int))
        out[name] = (z[mask], zp[mask])
    return out


def fit_stage(store: Mapping[str, Any], stage: str, o_split: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    ops: Dict[str, Any] = {}
    packed: Dict[str, Any] = {}
    for delta in DELTAS:
        dkey = delta_key(delta)
        if stage not in store["features"]["src"]:
            packed[dkey] = {"skipped": True, "reason": "missing_stage"}
            continue
        z, zp, oids, valid = _pairs(store, stage, dkey)
        splits = _split(z, zp, oids, valid, o_split)
        if min(len(splits["fit"][0]), len(splits["val"][0]), len(splits["test"][0])) < 6:
            packed[dkey] = {"skipped": True, "reason": "too_few_pairs"}
            continue
        result = fit_operator(*splits["fit"], *splits["val"], *splits["test"])
        ops[dkey] = {"matrix": result["matrix"], "bias": result["bias"], **pack_operator(result)}
        packed[dkey] = {**pack_operator(result), "delta": [float(delta[0]), float(delta[1])]}
    return {"ops": ops, "packed": packed}


def test_z(store: Mapping[str, Any], stage: str, o_split: Mapping[str, Sequence[int]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    src = store["features"]["src"][stage]
    o_test = set(int(x) for x in o_split["test"])
    zs = []
    n_s, n_o, _ = src.shape
    for si in range(n_s):
        for oi in range(n_o):
            if oi in o_test and np.isfinite(src[si, oi]).all():
                zs.append(src[si, oi])
    z = np.asarray(zs, dtype=np.float64) if zs else np.zeros((0, src.shape[-1]))
    targets: Dict[str, np.ndarray] = {}
    for delta in DELTAS:
        dkey = delta_key(delta)
        dst = store["features"][dkey][stage]
        ok = np.asarray(store["valid"][dkey], dtype=bool)
        rows, src_rows = [], []
        for si in range(n_s):
            for oi in range(n_o):
                if oi not in o_test or not ok[oi]:
                    continue
                if np.isfinite(src[si, oi]).all() and np.isfinite(dst[si, oi]).all():
                    src_rows.append(src[si, oi])
                    rows.append(dst[si, oi])
        if rows:
            targets[dkey] = np.asarray(rows, dtype=np.float64)
            targets[dkey + "__src"] = np.asarray(src_rows, dtype=np.float64)
    return z, targets


def slim_group_law(ops: Mapping[str, Any], z: np.ndarray, targets: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    compositions = []
    for d1, d2 in COMPOSE_PAIRS:
        key = delta_key((d1[0] + d2[0], d1[1] + d2[1]))
        tgt = targets.get(key, z)
        compositions.append({"d1": list(d1), "d2": list(d2), **composition_error(ops, d1, d2, z, tgt)})
    inverses = [{"delta": list(d), **inverse_error(ops, d, z)} for d in INVERSES]
    commutes = [{"dx": list(a), "dy": list(b), **commutativity_error(ops, a, b, z)} for a, b in COMMUTE]

    def _mean(rows: Sequence[Mapping[str, Any]], field: str):
        vals = [float(r[field]) for r in rows if r.get("ok") and field in r and r[field] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "composition": compositions,
        "inverse": inverses,
        "commutativity": commutes,
        "summary": {
            "mean_e_composed": _mean(compositions, "e_composed"),
            "mean_e_direct": _mean(compositions, "e_direct"),
            "mean_e_roundtrip": _mean(inverses, "e_roundtrip"),
            "mean_e_xy_vs_yx": _mean(commutes, "e_xy_vs_yx"),
        },
    }


def save_ops(path: Path, fitted: Mapping[str, Any]) -> None:
    if not fitted["ops"]:
        return
    np.savez_compressed(
        path,
        **{f"{k}_M": v["matrix"] for k, v in fitted["ops"].items()},
        **{f"{k}_b": v["bias"] for k, v in fitted["ops"].items()},
    )


def load_model(tag: str, ref: ModelRef | None):
    if tag == "random_init":
        model, _ = build_random_twin("resnet18", seed=SEED_RANDOM_INIT)
        return model
    if ref is None or not ref.found:
        raise FileNotFoundError(tag)
    return load_ref(ref)


def mean_e(packed: Mapping[str, Any]) -> float | None:
    vals = [v["e_test"] for v in packed.values() if isinstance(v, dict) and "e_test" in v]
    return float(np.mean(vals)) if vals else None


def cross_apply(
    ops_src: Mapping[str, Any],
    store_dst: Mapping[str, Any],
    stage: str,
    o_split: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    rows = []
    for delta in DELTAS:
        dkey = delta_key(delta)
        if dkey not in ops_src:
            continue
        z, zp, oids, valid = _pairs(store_dst, stage, dkey)
        splits = _split(z, zp, oids, valid, o_split)
        z_t, zp_t = splits["test"]
        if len(z_t) < 4:
            continue
        pred = apply_affine(z_t, ops_src[dkey]["matrix"], ops_src[dkey]["bias"])
        rows.append({"delta": dkey, "e_cross": float(normalized_error(pred, zp_t)), "n": int(len(z_t))})
    return {
        "per_delta": rows,
        "mean_e_cross": float(np.mean([r["e_cross"] for r in rows])) if rows else None,
    }


def plot_layerwise(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for row in rows:
        stages = [s["stage"] for s in row["stages"] if s.get("mean_e_test") is not None]
        ys = [s["mean_e_test"] for s in row["stages"] if s.get("mean_e_test") is not None]
        if stages:
            ax.plot(stages, ys, marker="o", label=row["model"])
    ax.set_ylabel("mean held-out E_Δ")
    ax.set_title("Layer-wise translation operator 误差（浅层→GAP）")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cross(matrix: np.ndarray, labels: Sequence[str], path: Path) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("应用到的内容 B")
    ax.set_ylabel("拟合 operator 的内容 A")
    ax.set_title(r"$M_\Delta^{A}$ 交叉应用到 $z^{B}$ 的误差")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_layerwise(frozen: Mapping[str, Any], origins: np.ndarray, o_split: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    out_dir = OUT / "layerwise"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, ref in model_panel():
        print(f"[layerwise] {tag} cuda={device()}", flush=True)
        model_dir = out_dir / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        summary_path = model_dir / "summary.json"
        if summary_path.is_file():
            row = load_json(summary_path)
            print(f"  skip existing {tag}", flush=True)
            rows.append(row)
            continue
        try:
            model = load_model(tag, ref)
            store = extract_store(model, "bezier", frozen, origins)
            stage_rows = []
            for stage in STAGES:
                if stage not in store["features"]["src"]:
                    continue
                fitted = fit_stage(store, stage, o_split)
                z, targets = test_z(store, stage, o_split)
                law = slim_group_law(fitted["ops"], z, targets) if fitted["ops"] and len(z) else {"summary": {}}
                dump_json(model_dir / f"{stage}_operators.json", fitted["packed"])
                dump_json(model_dir / f"{stage}_group_law.json", law)
                save_ops(model_dir / f"{stage}_matrices.npz", fitted)
                stage_rows.append(
                    {
                        "stage": stage,
                        "mean_e_test": mean_e(fitted["packed"]),
                        "group_law": law.get("summary"),
                        "n_ops": len(fitted["ops"]),
                    }
                )
            row = {"model": tag, "ok": True, "stages": stage_rows}
        except Exception as exc:
            row = {"model": tag, "ok": False, "error": str(exc), "tb": traceback.format_exc()[-2000:], "stages": []}
            print(f"  FAIL {tag}: {exc}", flush=True)
        dump_json(model_dir / "summary.json", row)
        rows.append(row)
    dump_json(out_dir / "panel.json", {"rows": rows})
    plot_layerwise(rows, out_dir / "figures" / "e_by_stage.png")
    return {"rows": rows}


def run_universal(frozen: Mapping[str, Any], origins: np.ndarray, o_split: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    out_dir = OUT / "universal"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = _phase2("r18_g64", "resnet18__G64__20260820", "resnet18")
    model = load_model("r18_g64_good", ref)
    stores: Dict[str, Any] = {}
    fitted_gap: Dict[str, Any] = {}
    for kind in CONTENTS:
        print(f"[universal] content={kind}", flush=True)
        store = extract_store(model, kind, frozen, origins, stages=["gap"])
        stores[kind] = store
        fitted = fit_stage(store, "gap", o_split)
        z, targets = test_z(store, "gap", o_split)
        law = slim_group_law(fitted["ops"], z, targets) if fitted["ops"] and len(z) else {"summary": {}}
        cdir = out_dir / kind
        cdir.mkdir(parents=True, exist_ok=True)
        dump_json(cdir / "operators.json", fitted["packed"])
        dump_json(cdir / "group_law.json", law)
        save_ops(cdir / "matrices.npz", fitted)
        fitted_gap[kind] = fitted
        dump_json(cdir / "summary.json", {"mean_e_test": mean_e(fitted["packed"]), "group_law": law.get("summary")})
    labels = list(CONTENTS)
    mat = np.full((len(labels), len(labels)), np.nan)
    cross_rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            blob = cross_apply(fitted_gap[a]["ops"], stores[b], "gap", o_split)
            mat[i, j] = blob["mean_e_cross"] if blob["mean_e_cross"] is not None else np.nan
            cross_rows.append({"fit_on": a, "apply_to": b, **blob})
    dump_json(out_dir / "cross.json", {"rows": cross_rows})
    plot_cross(mat, labels, out_dir / "figures" / "cross_heatmap.png")
    noise_to_geo = [
        r
        for r in cross_rows
        if r["fit_on"] == "noise" and r["apply_to"] in {"blob", "line", "quadratic", "polygon"}
    ]
    return {
        "self_e": {k: mean_e(fitted_gap[k]["packed"]) for k in labels},
        "cross": cross_rows,
        "noise_to_semantic": noise_to_geo,
    }


def write_handoff(layerwise: Mapping[str, Any], universal: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    lines = [
        "# 本机 1060：universal operator 与 layer-wise",
        "",
        f"时间：{datetime.now(timezone.utc).isoformat()}",
        f"设备：{device()}",
        "协议：slim 8 identities × 5×5 origins × 11 Δ。不是 Track A 全网格。",
        "",
        "## Layer-wise",
        "",
    ]
    for row in layerwise.get("rows", []):
        lines.append(f"- `{row.get('model')}` ok={row.get('ok')}")
        for st in row.get("stages") or []:
            lines.append(f"  - {st['stage']}: mean E_Δ={st.get('mean_e_test')}  composed={((st.get('group_law') or {}).get('mean_e_composed'))}")
    lines += ["", "## Universal operator（R18 G64 GAP）", ""]
    for k, v in (universal.get("self_e") or {}).items():
        lines.append(f"- 在 `{k}` 上拟合：mean E_Δ={v}")
    lines.append("")
    lines.append("noise 拟合的 M 交叉应用到语义/几何内容：")
    for r in universal.get("noise_to_semantic") or []:
        lines.append(f"- noise → {r['apply_to']}: mean E_cross={r.get('mean_e_cross')}")
    lines += [
        "",
        "需求结论（描述性，单协议 slim）：若 noise→Bezier/blob 的 E_cross 接近该内容自拟合 E_Δ，则 operator 更像 network-level 而不是数据分布特定。",
        "",
        "未做：A10 的 7+8 spectroscopy；全 33×33 cache。",
    ]
    (OUT / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_json(OUT / "protocol.json", protocol)


def run() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    origins = origins_px()
    o_split = origin_split(len(origins))
    frozen = freeze_identities()
    np.savez_compressed(OUT / "frozen_noise.npz", noise=frozen["noise"])
    protocol = {
        "seed": SEED,
        "n_id": N_ID,
        "n_origin": int(len(origins)),
        "deltas": [list(d) for d in DELTAS],
        "t_low": T_LOW_PX,
        "t_high": T_HIGH_PX,
        "contents": list(CONTENTS),
        "stages": list(STAGES),
        "origin_split": o_split,
        "bezier_ids": list(BEZIER_IDS),
        "device": str(device()),
        "note": "slim local 1060 protocol; not Track A",
    }
    dump_json(OUT / "protocol.json", protocol)
    print("device", device(), flush=True)
    layerwise = run_layerwise(frozen, origins, o_split)
    universal = run_universal(frozen, origins, o_split)
    dump_json(OUT / "summary.json", {"layerwise": layerwise, "universal": {"self_e": universal.get("self_e"), "noise_to_semantic": universal.get("noise_to_semantic")}})
    write_handoff(layerwise, universal, protocol)
    return {"out": str(OUT)}


if __name__ == "__main__":
    run()
