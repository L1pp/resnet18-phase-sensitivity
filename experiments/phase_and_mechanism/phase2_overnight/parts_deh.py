"""Parts D/E/F/G/H: R18 density, extrapolation, equal steps, dense eval, unseen shapes."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phase1_gap_rep.common import dump_json, setup_matplotlib_chinese

from .protocol import PHASE2_ROOT, SEED_ATLAS, load_protocol, record_failure
from .train_regime import dense_eval_model, n_steps_ref, train_regime

OUT = PHASE2_ROOT / "part_deh"
R18_REGIMES = ("G64", "G32", "G16", "G9", "G4", "C16", "C9", "C4")


def run_parts_deh() -> Dict[str, Any]:
    proto = load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    steps = n_steps_ref()
    rows: List[Dict[str, Any]] = []
    for regime in R18_REGIMES:
        try:
            tr = train_regime("resnet18", regime, SEED_ATLAS, steps=steps)
            ev = dense_eval_model("resnet18", regime, SEED_ATLAS)
            rows.append({"regime": regime, "train": tr, "dense": ev, "ok": True})
        except Exception as exc:
            rows.append({"regime": regime, "ok": False, "error": str(exc)})
            record_failure(f"r18_{regime}", str(exc))
            print(f"[phase2 D/E] FAIL {regime}: {exc}")
    dump_json(OUT / "r18_regimes.json", rows)

    # Part H: separate run_name so we do not overwrite G64/G16 atlas models
    h = proto["shape_split_h"]
    unseen = list(h["val"]) + list(h["test"])
    h_rows = []
    for name, regime, sids in (
        ("H1_Shape64Pos", "G64", h["train"]),
        ("H2_Shape16Pos", "G16", h["train"]),
    ):
        run_name = f"resnet18__{name}__{SEED_ATLAS}"
        try:
            tr = train_regime("resnet18", regime, SEED_ATLAS, shape_ids=sids, steps=steps, run_name=run_name)
            ev = dense_eval_model("resnet18", regime, SEED_ATLAS, run_name=run_name, eval_shape_ids=unseen)
            h_rows.append({"name": name, "train": tr, "dense": ev, "ok": True})
        except Exception as exc:
            h_rows.append({"name": name, "ok": False, "error": str(exc)})
            record_failure(name, str(exc))
    dump_json(OUT / "shape_generalization.json", h_rows)

    # G plots
    plt = setup_matplotlib_chinese()
    npos = {"G64": 64, "G32": 32, "G16": 16, "G9": 9, "G4": 4}
    xs, ys = [], []
    for row in rows:
        if row.get("ok") and row["regime"] in npos:
            xs.append(npos[row["regime"]])
            ys.append(row["dense"]["interpolation"].get("t_mae_px", np.nan))
    if xs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("training positions")
        ax.set_ylabel("dense interpolation t MAE px")
        ax.set_title("Part G: MAE vs number of training positions")
        fig.savefig(OUT / "mae_vs_n_positions.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _pair(a: str, b: str, fname: str, title: str) -> None:
        ra = next((r for r in rows if r.get("ok") and r["regime"] == a), None)
        rb = next((r for r in rows if r.get("ok") and r["regime"] == b), None)
        if not ra or not rb:
            return
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        labels = ["interpolation", "in_hull", "outside_hull"]
        xa = [ra["dense"].get(k, {}).get("t_mae_px", np.nan) for k in labels]
        xb = [rb["dense"].get(k, {}).get("t_mae_px", np.nan) for k in labels]
        idx = np.arange(len(labels))
        ax.bar(idx - 0.18, xa, 0.36, label=a)
        ax.bar(idx + 0.18, xb, 0.36, label=b)
        ax.set_xticks(idx, labels)
        ax.set_ylabel("t MAE px")
        ax.set_title(title)
        ax.legend()
        fig.savefig(OUT / fname, dpi=140, bbox_inches="tight")
        plt.close(fig)

    _pair("G16", "C16", "g16_vs_c16.png", "G16 spread vs C16 central")
    _pair("G9", "C9", "g9_vs_c9.png", "G9 spread vs C9 central")
    _pair("G4", "C4", "g4_vs_c4.png", "G4 corners vs C4 central")

    ok_rows = [r for r in rows if r.get("ok")]
    if ok_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for row in ok_rows:
            bins = row["dense"].get("mae_vs_nearest_train") or []
            if bins:
                ax.plot([(b["lo"] + b["hi"]) / 2 for b in bins], [b["t_mae_px"] for b in bins], marker="o", label=row["regime"])
        ax.set_xlabel("nearest training translation (px)")
        ax.set_ylabel("t MAE px")
        ax.set_title("Part G: error vs nearest train distance")
        ax.legend(fontsize=8)
        fig.savefig(OUT / "mae_vs_nearest_train.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for row in ok_rows:
            bins = row["dense"].get("mae_vs_hull_distance") or []
            if bins:
                ax.plot([(b["lo"] + b["hi"]) / 2 for b in bins], [b["t_mae_px"] for b in bins], marker="o", label=row["regime"])
        ax.set_xlabel("distance to train hull (px)")
        ax.set_ylabel("t MAE px")
        ax.set_title("Part G: error vs hull distance")
        ax.legend(fontsize=8)
        fig.savefig(OUT / "mae_vs_hull_distance.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    return {"regimes": rows, "h": h_rows, "n_steps_ref": steps}
