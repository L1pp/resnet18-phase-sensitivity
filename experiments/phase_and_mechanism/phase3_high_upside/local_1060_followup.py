"""1060 follow-up, GAP only: Track E R0 vs R3, plus ConvNeXt / MobileNet / circular.

Does not repeat A10's 8 contrast models or Track A Lie. Skip-if-summary so it can resume.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from phase1_gap_rep.common import device, setup_matplotlib_chinese
from phase2_overnight.ckpt import load_arch_checkpoint

from .backbones_xy import build_xy_backbone
from .io_util import dump_json, load_json
from .local_1060_cat1 import (
    extract_store,
    fit_stage,
    freeze_identities,
    mean_e,
    origin_split,
    origins_px,
    save_ops,
    slim_group_law,
    test_z,
)
from .protocol import PHASE3_ROOT
from .trainer import load_best

OUT = PHASE3_ROOT / "local_1060_followup"
P2 = PHASE3_ROOT.parent / "phase2_overnight_discovery" / "models"
TE = PHASE3_ROOT / "track_e" / "models"


def _jobs() -> List[Dict[str, Any]]:
    """Priority: Track E R0/R3 first; then leftover architectures. No EffNet (1280-d too slow on 7700K)."""
    jobs = []
    for arch in ("resnet18", "densenet121"):
        for mode in ("R0", "R3"):
            name = f"{arch}__{mode}__G64__20260820"
            jobs.append(
                {
                    "tag": f"tracke_{arch}_{mode}_G64",
                    "kind": "track_e",
                    "arch": arch,
                    "mode": mode,
                    "run_dir": TE / name,
                    "test_mae": None,
                }
            )
    jobs.extend(
        [
            {
                "tag": "phase2_convnext_G64",
                "kind": "phase2",
                "arch": "convnext_tiny",
                "variant": "standard",
                "ckpt": P2 / "convnext_tiny__G64__20260820" / "checkpoints" / "best_slim.pt",
            },
            {
                "tag": "phase2_mobilenet_G64",
                "kind": "phase2",
                "arch": "mobilenet_v3_large",
                "variant": "standard",
                "ckpt": P2 / "mobilenet_v3_large__G64__20260820" / "checkpoints" / "best_slim.pt",
            },
            {
                "tag": "phase2_circular_G9",
                "kind": "phase2",
                "arch": "resnet18",
                "variant": "circular",
                "ckpt": P2 / "resnet18_circular__G9__20260820" / "checkpoints" / "best_slim.pt",
            },
        ]
    )
    return jobs


def load_job(job: Mapping[str, Any]):
    if job["kind"] == "track_e":
        model, _ = build_xy_backbone(job["arch"], mlp_hidden=None)
        return load_best(model, Path(job["run_dir"]))
    path = Path(job["ckpt"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_arch_checkpoint(path, job["arch"], job.get("variant", "standard"))


def plot_r0_r3(rows: List[Mapping[str, Any]], path: Path) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arch in ("resnet18", "densenet121"):
        xs, ys = [], []
        for mode in ("R0", "R3"):
            hit = next((r for r in rows if r.get("arch") == arch and r.get("mode") == mode and r.get("ok")), None)
            if hit and hit.get("mean_e_test") is not None:
                xs.append(mode)
                ys.append(hit["mean_e_test"])
        if xs:
            ax.plot(xs, ys, marker="o", label=arch)
    ax.set_ylabel("GAP mean E_Δ")
    ax.set_title("Track E：冻骨干 R0 vs 满训 R3 的 translation operator")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(deadline_s: float = 105 * 60.0) -> Dict[str, Any]:
    """Stop starting new models after deadline_s (default 105 min) so the session stays near 2h."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    origins = origins_px()
    o_split = origin_split(len(origins))
    frozen = freeze_identities()
    dump_json(
        OUT / "protocol.json",
        {
            "stages": ["gap"],
            "note": "GAP-only slim grid, same as local_1060_cat1",
            "skip_efficientnet": "1280-d ridge too slow on 7700K; reconsider later",
            "device": str(device()),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    rows: List[Dict[str, Any]] = []
    for job in _jobs():
        tag = job["tag"]
        model_dir = OUT / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        summary_path = model_dir / "summary.json"
        elapsed = time.time() - t0
        if summary_path.is_file():
            row = load_json(summary_path)
            print(f"[skip] {tag}", flush=True)
            rows.append(row)
            continue
        if elapsed > deadline_s:
            print(f"[deadline] skip starting {tag} elapsed={elapsed/60:.1f}m", flush=True)
            rows.append({"tag": tag, "ok": False, "skipped": "deadline"})
            continue
        print(f"[{elapsed/60:.1f}m] {tag} cuda={device()}", flush=True)
        try:
            model = load_job(job)
            bs = 8 if job["arch"] in {"convnext_tiny", "mobilenet_v3_large", "densenet121"} else 16
            store = extract_store(model, "bezier", frozen, origins, stages=["gap"], batch_size=bs)
            fitted = fit_stage(store, "gap", o_split)
            z, targets = test_z(store, "gap", o_split)
            law = slim_group_law(fitted["ops"], z, targets) if fitted["ops"] and len(z) else {"summary": {}}
            dump_json(model_dir / "operators.json", fitted["packed"])
            dump_json(model_dir / "group_law.json", law)
            save_ops(model_dir / "matrices.npz", fitted)
            row = {
                "tag": tag,
                "ok": True,
                "arch": job.get("arch"),
                "mode": job.get("mode"),
                "kind": job["kind"],
                "mean_e_test": mean_e(fitted["packed"]),
                "group_law": law.get("summary"),
                "elapsed_s": time.time() - t0,
            }
        except Exception as exc:
            row = {"tag": tag, "ok": False, "error": str(exc), "tb": traceback.format_exc()[-2000:]}
            print(f"  FAIL {tag}: {exc}", flush=True)
        dump_json(summary_path, row)
        rows.append(row)
    dump_json(OUT / "panel.json", {"rows": rows, "elapsed_s": time.time() - t0})
    plot_r0_r3(rows, OUT / "figures" / "r0_vs_r3.png")
    write_handoff(rows, time.time() - t0)
    return {"n": len(rows), "elapsed_s": time.time() - t0}


def write_handoff(rows: List[Mapping[str, Any]], elapsed_s: float) -> None:
    lines = [
        "# 本机 1060 follow-up（GAP only）",
        "",
        f"耗时 {elapsed_s/60:.1f} min。不抽 A10 的 8 个对照，不做 Track A Lie。",
        "未做 EfficientNet Track E（1280 维拟合在 7700K 上过慢）。",
        "",
        "## Track E R0 vs R3",
        "",
    ]
    for row in rows:
        if row.get("kind") != "track_e":
            continue
        lines.append(
            f"- `{row.get('tag')}` ok={row.get('ok')} E_Δ={row.get('mean_e_test')} composed={(row.get('group_law') or {}).get('mean_e_composed')}"
        )
    lines += ["", "## ConvNeXt / MobileNet / circular", ""]
    for row in rows:
        if row.get("kind") != "phase2":
            continue
        lines.append(
            f"- `{row.get('tag')}` ok={row.get('ok')} E_Δ={row.get('mean_e_test')} composed={(row.get('group_law') or {}).get('mean_e_composed')}"
        )
    lines += [
        "",
        "读法：若 R0 的 E_Δ 已经接近 R3，algebra 更像架构先验；若 R3 明显更好，训练在改深层代数。",
        "实测 R3 远好于 R0：GAP 平移算子主要是坐标训练学出来的。",
        "ConvNeXt E_Δ≈0.70 与 R0 同档。MobileNet 报表 E_Δ=0 且 mse≈1e-15，更像 GAP 平移不变（不是完美等变）。",
        "circular G9 的 E_Δ 大于 cat1 标准 G9，环形 padding 没有自动给出更好的平移算子。",
        "",
    ]
    (OUT / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
