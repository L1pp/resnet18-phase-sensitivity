"""Track C: leave-one-family-out content-independent coordinates."""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from phase1_gap_rep.common import setup_matplotlib_chinese

from .backbones_xy import build_xy_backbone
from .io_util import dump_json
from .primitives import FAMILIES, make_batch, make_sample
from .protocol import IMAGE_SIZE_C, PHASE3_ROOT, PRIMITIVE_FAMILIES, SEED_TRACK_C, record_failure
from .trainer import load_best, mae_px, predict_xy, train_xy

OUT = PHASE3_ROOT / "track_c"


def _family_batch(families: Sequence[str], n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = make_batch(families, n, seed, image_size=IMAGE_SIZE_C, size_mode="absolute")
    return blob["images"], blob["xy_norm"], blob["family"]


def _error_heatmap(pred: np.ndarray, target: np.ndarray, path) -> None:
    plt = setup_matplotlib_chinese()
    scale = float(IMAGE_SIZE_C - 1)
    err = np.linalg.norm((pred - target) * scale, axis=-1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    hb = ax.hexbin(target[:, 0], target[:, 1], C=err, gridsize=18, reduce_C_function=np.mean)
    ax.set_xlabel("x norm")
    ax.set_ylabel("y norm")
    ax.set_title("未见 family 的位置误差热图 (px)")
    fig.colorbar(hb, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_holdout(unseen: str, seed: int) -> Dict[str, Any]:
    seen = [f for f in PRIMITIVE_FAMILIES if f != unseen]
    name = f"lofo_{unseen}__{seed}"
    out = OUT / "models" / name
    tr_img, tr_xy, _ = _family_batch(seen, 4000, seed)
    va_img, va_xy, _ = _family_batch(seen, 512, seed + 1)
    te_seen_img, te_seen_xy, _ = _family_batch(seen, 800, seed + 2)
    te_un_img, te_un_xy, _ = _family_batch([unseen], 800, seed + 3)
    model, _ = build_xy_backbone("resnet18")
    summary = train_xy(
        model,
        tr_img,
        tr_xy,
        va_img,
        va_xy,
        out,
        seed=seed,
        arch="resnet18",
        image_size=IMAGE_SIZE_C,
        steps=4000,
        mode="R3",
        amp=True,
        extra={"unseen_family": unseen, "seen_families": seen},
    )
    model = load_best(build_xy_backbone("resnet18")[0], out)
    seen_mae = mae_px(predict_xy(model, te_seen_img), te_seen_xy, IMAGE_SIZE_C)
    unseen_mae = mae_px(predict_xy(model, te_un_img), te_un_xy, IMAGE_SIZE_C)
    pred_u = predict_xy(model, te_un_img)
    _error_heatmap(pred_u, te_un_xy, out / "figures" / "unseen_heatmap.png")
    dump_json(
        out / "tables" / "transfer.json",
        {"seen_mae_px": seen_mae, "unseen_mae_px": unseen_mae, "unseen_family": unseen},
    )
    return {
        "ok": True,
        "name": name,
        "unseen": unseen,
        "seen_mae_px": seen_mae,
        "unseen_mae_px": unseen_mae,
        "val_mae_px": summary["best_val_mae_px"],
    }


def run_single_family(seed: int) -> Dict[str, Any]:
    name = f"single_quadratic__{seed}"
    out = OUT / "models" / name
    tr_img, tr_xy, _ = _family_batch(["quadratic"], 4000, seed)
    va_img, va_xy, _ = _family_batch(["quadratic"], 512, seed + 1)
    model, _ = build_xy_backbone("resnet18")
    train_xy(model, tr_img, tr_xy, va_img, va_xy, out, seed=seed, arch="resnet18", image_size=IMAGE_SIZE_C, steps=4000, mode="R3", amp=True, extra={"train_family": "quadratic"})
    model = load_best(build_xy_backbone("resnet18")[0], out)
    rows = []
    for family in FAMILIES:
        img, xy, _ = _family_batch([family], 600, seed + 9)
        rows.append({"family": family, "mae_px": mae_px(predict_xy(model, img), xy, IMAGE_SIZE_C)})
    dump_json(out / "tables" / "zero_shot.json", {"rows": rows})
    return {"ok": True, "name": name, "rows": rows}


def run_all6(seed: int) -> Dict[str, Any]:
    name = f"all6__{seed}"
    out = OUT / "models" / name
    tr_img, tr_xy, _ = _family_batch(PRIMITIVE_FAMILIES, 4800, seed)
    va_img, va_xy, _ = _family_batch(PRIMITIVE_FAMILIES, 512, seed + 1)
    model, _ = build_xy_backbone("resnet18")
    summary = train_xy(model, tr_img, tr_xy, va_img, va_xy, out, seed=seed, arch="resnet18", image_size=IMAGE_SIZE_C, steps=4000, mode="R3", amp=True, extra={"families": list(PRIMITIVE_FAMILIES)})
    return {"ok": True, "name": name, "val_mae_px": summary["best_val_mae_px"]}


def run_track_c() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for family in PRIMITIVE_FAMILIES:
        try:
            rows.append(run_holdout(family, SEED_TRACK_C))
        except Exception as exc:
            record_failure(f"track_c_lofo_{family}", str(exc), {"tb": traceback.format_exc()[-3000:]})
            rows.append({"ok": False, "unseen": family, "error": str(exc)})
    try:
        all6 = run_all6(SEED_TRACK_C)
    except Exception as exc:
        record_failure("track_c_all6", str(exc), {"tb": traceback.format_exc()[-2000:]})
        all6 = {"ok": False, "error": str(exc)}
    single = None
    strong = [r for r in rows if r.get("ok") and r.get("unseen_mae_px") is not None and r["unseen_mae_px"] < 8.0]
    if len(strong) >= 3:
        try:
            single = run_single_family(SEED_TRACK_C)
        except Exception as exc:
            record_failure("track_c_single_quadratic", str(exc), {"tb": traceback.format_exc()[-2000:]})
            single = {"ok": False, "error": str(exc)}
    unseen_ok = [r["unseen_mae_px"] for r in rows if r.get("ok")]
    flag = None
    if unseen_ok and float(np.median(unseen_ok)) < 6.0:
        flag = "HIGH_UPSIDE: content-independent coordinate readout (zero-shot family)"
    payload = {"lofo": rows, "all6": all6, "single_family": single, "high_upside_flag": flag}
    dump_json(OUT / "panel.json", payload)
    return payload
