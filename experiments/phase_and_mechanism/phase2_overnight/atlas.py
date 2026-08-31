"""CNN architecture atlas: 5 extra families x Full64/Sparse9/Central9 + init correlation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phase1_gap_rep.common import dump_json, setup_matplotlib_chinese

from .protocol import ARCH_ATLAS, PHASE2_ROOT, SEED_ATLAS, record_failure
from .train_regime import dense_eval_model, model_dir, n_steps_ref, train_regime

OUT = PHASE2_ROOT / "atlas"
ATLAS_REGIMES = ("G64", "G9", "C9")  # Full64 / Sparse9 / Central9


def run_atlas() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    steps = n_steps_ref()
    rows: List[Dict[str, Any]] = []
    for arch in ARCH_ATLAS:
        for regime in ATLAS_REGIMES:
            if arch == "resnet18":
                # trained in Part D/E
                path = model_dir(arch, regime, SEED_ATLAS) / "tables" / "dense_eval.json"
                init_p = model_dir(arch, regime, SEED_ATLAS) / "tables" / "init_probe.json"
                rec: Dict[str, Any] = {"arch": arch, "regime": regime, "reused_r18": True}
                if path.exists():
                    rec["dense"] = __import__("json").loads(path.read_text(encoding="utf-8"))
                    rec["ok"] = True
                else:
                    try:
                        rec["train"] = train_regime(arch, regime, SEED_ATLAS, steps=steps)
                        rec["dense"] = dense_eval_model(arch, regime, SEED_ATLAS)
                        rec["ok"] = True
                    except Exception as exc:
                        rec.update({"ok": False, "error": str(exc)})
                if init_p.exists():
                    rec["init_probe"] = __import__("json").loads(init_p.read_text(encoding="utf-8"))
                rows.append(rec)
                continue
            try:
                tr = train_regime(arch, regime, SEED_ATLAS, steps=steps)
                ev = dense_eval_model(arch, regime, SEED_ATLAS)
                rows.append({"arch": arch, "regime": regime, "train": tr, "dense": ev, "init_probe": tr.get("init_probe"), "ok": True})
            except Exception as exc:
                rows.append({"arch": arch, "regime": regime, "ok": False, "error": str(exc)})
                record_failure(f"atlas_{arch}_{regime}", str(exc))
                print(f"[atlas] FAIL {arch} {regime}: {exc}")
    dump_json(OUT / "atlas_rows.json", rows)

    # init vs sparse/extrap scatter
    plt = setup_matplotlib_chinese()
    xs, ys_s, ys_c, labels = [], [], [], []
    for arch in ARCH_ATLAS:
        s9 = next((r for r in rows if r.get("ok") and r["arch"] == arch and r["regime"] == "G9"), None)
        c9 = next((r for r in rows if r.get("ok") and r["arch"] == arch and r["regime"] == "C9"), None)
        if not s9:
            continue
        init = (s9.get("init_probe") or s9.get("train", {}).get("init_probe") or {})
        e_init = init.get("t_mae_px", init.get("t_mae"))
        e_sp = s9.get("dense", {}).get("interpolation", {}).get("t_mae_px")
        e_ex = (c9 or {}).get("dense", {}).get("outside_hull", {}).get("t_mae_px") if c9 else None
        if e_init is None or e_sp is None:
            continue
        xs.append(float(e_init))
        ys_s.append(float(e_sp))
        ys_c.append(float(e_ex) if e_ex is not None else np.nan)
        labels.append(arch)
    if xs:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.scatter(xs, ys_s)
        for x, y, lab in zip(xs, ys_s, labels):
            ax.annotate(lab, (x, y), fontsize=8)
        ax.set_xlabel("E_init t MAE px")
        ax.set_ylabel("Sparse9 interpolation t MAE px")
        ax.set_title("Init scaffold vs sparse interpolation")
        fig.savefig(OUT / "init_vs_sparse.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        if np.isfinite(ys_c).any():
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            ax.scatter(xs, ys_c)
            for x, y, lab in zip(xs, ys_c, labels):
                ax.annotate(lab, (x, y), fontsize=8)
            ax.set_xlabel("E_init t MAE px")
            ax.set_ylabel("Central9 outside-hull t MAE px")
            ax.set_title("Init scaffold vs extrapolation")
            fig.savefig(OUT / "init_vs_extrap.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
    return {"rows": rows}
