"""Write ranking + HANDOFF from fallen JSON only. Never invent numbers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .io_util import dump_json, load_json
from .protocol import PHASE3_ROOT


def _load(rel: str) -> Optional[Dict[str, Any]]:
    path = PHASE3_ROOT / rel
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def _field(block: Optional[Dict[str, Any]], *keys: str, default: str = "not_run") -> Any:
    if block is None:
        return default
    cur: Any = block
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    return cur


def write_ranking() -> str:
    a = _load("track_a/panel.json")
    b = _load("track_b/panel.json")
    c = _load("track_c/panel.json")
    d = _load("track_d/panel.json")
    e = _load("track_e/panel.json")
    f = _load("track_f/panel.json")
    items: List[Dict[str, Any]] = []

    def add(title: str, flag: Any, observation: str, upside: str) -> None:
        items.append(
            {
                "title": title,
                "flag": flag if flag not in (None, "not_run") else None,
                "observation": observation,
                "upside": upside,
            }
        )

    add(
        "H1 spontaneous approximate translation group law",
        _field(a, "rows", default=None),
        f"Track A panel rows={len(a.get('rows', [])) if a else 'not_run'}",
        "Very High",
    )
    add(
        "H2 coordinate ability follows symmetry-breaking ladder",
        _field(b, "high_upside_flag"),
        f"Track B flag={_field(b, 'high_upside_flag')}",
        "Very High",
    )
    add(
        "H3 zero-shot cross-content coordinate system",
        _field(c, "high_upside_flag"),
        f"Track C flag={_field(c, 'high_upside_flag')}",
        "High",
    )
    add(
        "H4 cross-resolution normalized coordinate system",
        _field(d, "high_upside_flag"),
        f"Track D flag={_field(d, 'high_upside_flag')}",
        "High",
    )
    add(
        "H5 random architecture position aptitude vs trained",
        "see track_e/panel.json",
        f"Track E rows={len(e.get('rows', [])) if e else 'not_run'}",
        "Medium",
    )
    add(
        "H6 translation special vs general continuous transforms",
        _field(f, "ok"),
        f"Track F={f if f else 'not_run'}",
        "Medium",
    )

    lines = [
        "# HIGH_UPSIDE_DISCOVERY_RANKING",
        "",
        "按科学 upside 排序，不按 MAE。数字只来自已落盘 JSON。没跑到的写 not_run。",
        "",
    ]
    for i, item in enumerate(items, 1):
        lines += [
            f"## {i}. {item['title']}",
            "",
            f"- Observation: {item['observation']}",
            f"- Effect size: 见对应 track panel.json",
            f"- Controls passed: 见 track 内 controls / S0 / shuffled",
            f"- Known prior-work collision: Kayhan boundary / NFT / equivariance-by-contrast / random-feature literature",
            f"- What is actually new here: vanilla CNN + ordinary coordinate supervision, no group loss",
            f"- Alternative explanation: readout trick / boundary cue / sampling phase / assay leakage",
            f"- Replication status: 本轮 discovery，单协议",
            f"- Scientific upside: {item['upside']}",
            f"- Cheapest decisive follow-up: 预注册多 seed 重复该 track 的主指标",
            f"- Flag: {item['flag']}",
            "",
        ]
    text = "\n".join(lines) + "\n"
    path = PHASE3_ROOT / "HIGH_UPSIDE_DISCOVERY_RANKING.md"
    path.write_text(text, encoding="utf-8")
    return text


def write_handoff() -> str:
    a = _load("track_a/panel.json")
    b = _load("track_b/panel.json")
    c = _load("track_c/panel.json")
    d = _load("track_d/panel.json")
    e = _load("track_e/panel.json")
    f = _load("track_f/panel.json")
    pre = _load("config/preflight.json")
    answers = {
        "1_latent_translation_operator": a if a else "not_run",
        "2_group_law": _load("track_a/panel.json"),
        "3_symmetry_breaking_predicts_coords": b if b else "not_run",
        "4_zero_shot_family": c if c else "not_run",
        "5_cross_resolution": d if d else "not_run",
        "6_random_frozen": e if e else "not_run",
        "7_feature_learning_gain": e if e else "not_run",
        "8_arch_prior_predicts_final": e if e else "not_run",
        "9_best_next_mainline": "choose from HIGH_UPSIDE flags; if none, H2 ladder or H1 negative is still informative",
        "10_drop": "do not revive B_t / OLS / CKA-as-mainline / circular-ResNet-as-S0",
    }
    lines = [
        "# AI_HANDOFF_PHASE3_HIGH_UPSIDE",
        "",
        "云端自动稿。数字以 JSON 为准。禁止编造 not_run 项。不要 commit/push。",
        "",
        f"preflight: {pre.get('protocol_hash') if pre else 'not_run'}",
        "",
        "## 十问",
        "",
        f"1. 是否发现 latent translation operator？ `{answers['1_latent_translation_operator'] if isinstance(answers['1_latent_translation_operator'], str) else 'see track_a/panel.json'}`",
        f"2. 是否近似满足 composition/inverse/commutativity？ 见各模型 `track_a/*/group_law.json`",
        f"3. controlled symmetry breaking 是否预测 coordinate ability？ `{_field(b, 'high_upside_flag')}`",
        f"4. coordinate head 能否 zero-shot 跨 primitive family？ `{_field(c, 'high_upside_flag')}`",
        f"5. 224-trained coordinate system 能否跨 resolution？ `{_field(d, 'high_upside_flag')}`",
        f"6. random frozen CNN 到底能做到什么程度？ 见 `track_e/panel.json` R0/R1",
        f"7. feature learning 增益有多大？ 比较同架构 R0 vs R3",
        f"8. architecture prior 是否预测最终表现？ 3 架构只做探索相关",
        f"9. 哪一个现象最值得升级成下一轮正式研究主线？ 优先已打 HIGH_UPSIDE 的 H1/H2",
        f"10. 哪些方向应放弃？ {answers['10_drop']}",
        "",
        "不要宣称严格数学群表示。",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (PHASE3_ROOT / "AI_HANDOFF_PHASE3_HIGH_UPSIDE.md").write_text(text, encoding="utf-8")
    dump_json(PHASE3_ROOT / "handoff_answers.json", {k: (v if isinstance(v, (str, int, float, bool)) or v is None else "see_panel") for k, v in answers.items()})
    return text
