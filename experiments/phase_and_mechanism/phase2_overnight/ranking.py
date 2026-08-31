"""Write DISCOVERY_RANKING.md and cloud-side HANDOFF (training chapters only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from phase1_gap_rep.common import dump_json

from .protocol import PHASE2_ROOT, load_protocol


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cloud_handoff(progress: Dict[str, Any]) -> None:
    proto = load_protocol()
    deh = _load(PHASE2_ROOT / "part_deh" / "r18_regimes.json") or []
    atlas = _load(PHASE2_ROOT / "atlas" / "atlas_rows.json") or []
    branches = _load(PHASE2_ROOT / "branches" / "original_branches.json") or {}
    ranking: List[str] = [
        "# DISCOVERY_RANKING",
        "",
        "A/B（连续场 / 表征几何）在**本地** `part_ab/`，本文件只根据云端新训练排序。没出现的现象不包装。",
        "",
    ]
    # crude auto bullets from interpolation numbers
    goods = []
    for row in deh:
        if not row.get("ok"):
            continue
        inter = row.get("dense", {}).get("interpolation", {}).get("t_mae_px")
        extra = row.get("dense", {}).get("outside_hull", {}).get("t_mae_px")
        goods.append((row["regime"], inter, extra))
    ranking.append("## R18 density / extrap")
    for regime, inter, extra in goods:
        ranking.append(f"- `{regime}` interp={inter} outside={extra}")
    ranking.append("")
    ranking.append("## Atlas")
    for row in atlas:
        if not row.get("ok"):
            ranking.append(f"- FAIL {row.get('arch')} {row.get('regime')}: {row.get('error')}")
            continue
        inter = row.get("dense", {}).get("interpolation", {}).get("t_mae_px")
        ranking.append(f"- {row.get('arch')} {row.get('regime')} interp={inter}")
    (PHASE2_ROOT / "DISCOVERY_RANKING.md").write_text("\n".join(ranking) + "\n", encoding="utf-8")

    lines = [
        "# AI_HANDOFF_PHASE2_OVERNIGHT（云端训练章）",
        "",
        f"protocol `{proto['protocol_hash']}`。Part A/B 在本地，不要在本文件里假装做过 1.8 稠密场。",
        "",
        "## 1. Executive summary",
        "见 DISCOVERY_RANKING.md。拷回后与 `part_ab/` 合并。",
        "",
        "## 5. Sampling-density boundary",
        json.dumps(deh, indent=2, default=str)[:4000],
        "",
        "## 6–8. Atlas / branches",
        f"atlas rows={len(atlas)} branches={branches.get('ran', branches)}",
        "",
        "## 10. Recommended next step",
        "完成 overnight 后停止。不要自动再开实验。不要 commit/push。",
        "",
        f"progress={json.dumps(progress, default=str)[:2000]}",
    ]
    (PHASE2_ROOT / "AI_HANDOFF_PHASE2_OVERNIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_json(PHASE2_ROOT / "cloud_progress_snapshot.json", progress)
