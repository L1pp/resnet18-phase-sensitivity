"""Track B: controlled symmetry-breaking ladder on 96x96 blobs."""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from phase1_gap_rep.common import setup_matplotlib_chinese

from .controlled_cnn import build_controlled, variant_spec
from .equivariance import measure_defects
from .io_util import dump_json
from .primitives import make_sample
from .protocol import IMAGE_SIZE_B, PHASE3_ROOT, SEED_TRACK_B, TRACK_B_REGIMES, TRACK_B_VARIANTS, record_failure
from .trainer import load_best, mae_px, predict_xy, train_xy

OUT = PHASE3_ROOT / "track_b"
SPARSE_XY = (0.25, 0.50, 0.75)


def _blob_set(n: int, seed: int, regime: str) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = np.zeros((n, IMAGE_SIZE_B, IMAGE_SIZE_B), dtype=np.uint8)
    xy = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        if regime == "sparse":
            u = float(rng.choice(SPARSE_XY))
            v = float(rng.choice(SPARSE_XY))
            cx = u * (IMAGE_SIZE_B - 1)
            cy = v * (IMAGE_SIZE_B - 1)
            sample = make_sample("blob", rng, image_size=IMAGE_SIZE_B, center=(cx, cy))
        else:
            sample = make_sample("blob", rng, image_size=IMAGE_SIZE_B)
        images[i] = sample["image"]
        xy[i] = sample["xy_norm"]
    return images, xy


def _dense_test(n: int = 800, seed: int = 20260840) -> Tuple[np.ndarray, np.ndarray]:
    return _blob_set(n, seed, "dense")


def _should_add_seed(rows: List[Dict[str, Any]]) -> bool:
    by = {r["variant"]: r for r in rows if r.get("ok") and r.get("regime") == "dense"}
    s0 = by.get("S0", {})
    s3 = by.get("S3", {})
    s2 = by.get("S2", {})
    s0_mae = s0.get("test_mae_px")
    recover = s3.get("test_mae_px") if s3.get("test_mae_px") is not None else s2.get("test_mae_px")
    return s0_mae is not None and recover is not None and s0_mae > 12.0 and recover < 6.0


def _plot_ladder(rows: List[Dict[str, Any]], path) -> None:
    plt = setup_matplotlib_chinese()
    dense = [r for r in rows if r.get("ok") and r.get("regime") == "dense" and r.get("seed") == SEED_TRACK_B]
    if not dense:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar([r["variant"] for r in dense], [r["test_mae_px"] for r in dense])
    ax.set_ylabel("dense test MAE (px)")
    ax.set_title("Track B symmetry-breaking ladder（单 seed dense）")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_one(variant: str, regime: str, seed: int) -> Dict[str, Any]:
    name = f"{variant}__{regime}__{seed}"
    out = OUT / "models" / name
    train_img, train_xy = _blob_set(2048 if regime == "dense" else 576, seed, regime)
    val_img, val_xy = _blob_set(256, seed + 17, "dense")
    test_img, test_xy = _dense_test()
    model = build_controlled(variant)
    summary = train_xy(
        model,
        train_img,
        train_xy,
        val_img,
        val_xy,
        out,
        seed=seed,
        arch="controlled_cnn",
        image_size=IMAGE_SIZE_B,
        steps=3000,
        mode="R3",
        amp=False,
        extra={"variant": variant, "regime": regime, "spec": variant_spec(variant)},
    )
    model = load_best(build_controlled(variant), out)
    test_mae = mae_px(predict_xy(model, test_img), test_xy, IMAGE_SIZE_B)
    eq = measure_defects(model, test_img[:64], shifts=(1, 2, 4), batch_size=8)
    dump_json(out / "tables" / "dense_test.json", {"test_mae_px": test_mae, "n": int(len(test_img))})
    dump_json(out / "tables" / "equivariance.json", eq)
    return {
        "ok": True,
        "name": name,
        "variant": variant,
        "regime": regime,
        "seed": seed,
        "val_mae_px": summary["best_val_mae_px"],
        "test_mae_px": test_mae,
        "mean_d_eq": eq.get("mean_d_eq"),
        "mean_d_gap": eq.get("mean_d_gap"),
        "n_params": variant_spec(variant)["n_params"],
    }


def run_track_b() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for variant in TRACK_B_VARIANTS:
        for regime in TRACK_B_REGIMES:
            try:
                rows.append(run_one(variant, regime, SEED_TRACK_B))
            except Exception as exc:
                record_failure(f"track_b_{variant}_{regime}", str(exc), {"tb": traceback.format_exc()[-3000:]})
                rows.append({"ok": False, "variant": variant, "regime": regime, "error": str(exc)})
    if _should_add_seed(rows):
        for variant in TRACK_B_VARIANTS:
            try:
                rows.append(run_one(variant, "dense", SEED_TRACK_B + 1))
            except Exception as exc:
                record_failure(f"track_b_{variant}_seed2", str(exc), {"tb": traceback.format_exc()[-2000:]})
    _plot_ladder(rows, OUT / "figures" / "ladder_dense.png")
    flag = None
    dense = {r["variant"]: r for r in rows if r.get("ok") and r.get("regime") == "dense" and r.get("seed") == SEED_TRACK_B}
    if dense.get("S0") and dense.get("S3") and dense["S0"]["test_mae_px"] > 12 and dense["S3"]["test_mae_px"] < 5:
        flag = "HIGH_UPSIDE: absolute coordinate computation tracks controlled translation-symmetry breaking"
    payload = {"rows": rows, "high_upside_flag": flag}
    dump_json(OUT / "panel.json", payload)
    return payload
