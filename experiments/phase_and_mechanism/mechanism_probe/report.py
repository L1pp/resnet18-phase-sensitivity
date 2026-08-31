"""Write step-1 summary + a short follow-up note for the planning AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .const import CODE_REV, PACK_ROOT, RESULTS_ROOT
from .blob_io import write_json


def _load(rel: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_ROOT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def _fft_line(fam: Dict[str, Any], stage: str, step: str, *, delta: bool = True) -> str:
    rec = ((fam.get("stages") or {}).get(stage) or {}).get("steps") or {}
    blob = rec.get(str(step)) or {}
    src = blob.get("delta_from_step0") if delta and blob.get("delta_from_step0") else blob
    fft = (src or {}).get("fft") or {}
    return (
        f"{_fmt(fft.get('peak_frac'), 3)} / {_fmt(fft.get('low_freq_rle2_frac'), 3)}"
    )


def build_markdown() -> str:
    env = _load("env.json") or {}
    s1 = _load("s1_fft.json") or {}
    s2 = _load("s2_opt_audit/summary.json") or {}
    s3 = _load("s3_tangent/summary.json") or {}
    s4 = _load("s4_rv_existing.json") or {}
    l3 = _load("l3_mlp/summary.json") or {}
    fact = _load("s2_opt_audit/adamw_reset_fact.json") or {}

    lines: List[str] = []
    a = lines.append
    a("# 第一步结果：本机短时机制判别")
    a("")
    a(f"`code_rev={CODE_REV}`。A10 未开。只使用本机 1060 / CPU。")
    a("")
    a(f"环境：`{env.get('executable')}`  torch `{env.get('torch')}`  GPU `{env.get('device_name')}`")
    a("")
    a("## 0. 一句话")
    a("")
    a("已有 2D 轨迹的 step1 shock 是 **宽谱、三档同形、主要由 AdamW 预处理造成** 的瞬时现象；**最终固化** 才分叉（head 回场，l4/full 把 shock 写进函数）。unit-g / 随机方向 / 锐 Hessian 模的 R(v) **都不** 支持 `head ≪ l4`。Adam 一步在 l4/full 上几乎不落在锐 Hessian 子空间（能量 0.7%），更像走进 support-loss 的平坦方向。约束实验（L2）和 optimizer×plasticity（L1）都值得开；完整 catapult 联合轨迹（L4）优先级下调。")
    a("")
    a("## 1. S2a 事实（无新训练）")
    a("")
    a(str(fact.get("fact") or "sparse FT 新建 AdamW，slim 无 optimizer state。"))
    a("现有百像素 step1 就是 **AdamW reset** 下的结果。")
    a("")
    a("## 2. S1 FFT（已有场，affine-removed `u` 与 Δu）")
    a("")
    a("表内数字：step1 **Δu** 的 `peak_frac` / `low_freq_r≤2_frac`。")
    a("")
    a("| 家族 | head | l4 | full |")
    a("|---|---|---|---|")
    amd = s1.get("amd_2d") or {}
    a10 = s1.get("a10_6d") or {}
    mlp = s1.get("mlp") or {}
    a(f"| AMD 2D | {_fft_line(amd, 'head', '1')} | {_fft_line(amd, 'l4', '1')} | {_fft_line(amd, 'full', '1')} |")
    a(f"| A10 6D | {_fft_line(a10, 'head', '1')} | {_fft_line(a10, 'l4', '1')} | {_fft_line(a10, 'full', '1')} |")
    a("")
    a("step1 谱相似度（功率 cosine）：")
    a("")
    a("| 家族 | step | head vs l4 | head vs full | l4 vs full |")
    a("|---|---|---|---|---|")
    for name, fam in (("AMD 2D", amd), ("A10 6D", a10)):
        cmp1 = (fam.get("comparisons") or {}).get("1") or {}
        a(
            f"| {name} | 1 | "
            f"{_fmt((cmp1.get('head_vs_l4') or {}).get('cosine_delta_power') or (cmp1.get('head_vs_l4') or {}).get('cosine_power'))} | "
            f"{_fmt((cmp1.get('head_vs_full') or {}).get('cosine_delta_power') or (cmp1.get('head_vs_full') or {}).get('cosine_power'))} | "
            f"{_fmt((cmp1.get('l4_vs_full') or {}).get('cosine_delta_power') or (cmp1.get('l4_vs_full') or {}).get('cosine_power'))} |"
        )
        keys = sorted((fam.get("comparisons") or {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        if keys:
            last = keys[-1]
            cmpl = (fam.get("comparisons") or {}).get(last) or {}
            a(
                f"| {name} | {last} | "
                f"{_fmt((cmpl.get('head_vs_l4') or {}).get('cosine_power'))} | "
                f"{_fmt((cmpl.get('head_vs_full') or {}).get('cosine_power'))} | "
                f"{_fmt((cmpl.get('l4_vs_full') or {}).get('cosine_power'))} |"
            )
    a("")
    a("MLP 对照（已有 F3 轨迹）step1 Δu peak/low：")
    a("")
    a("| run | peak/low |")
    a("|---|---|")
    for key in ("linear_full", "mlp_s_full", "mlp_w_full", "mlp_w_last"):
        a(f"| {key} | {_fft_line(mlp, key, '1')} |")
    a("")
    a("图：`results/mechanism_probe/local/figures/s1_*.png`")
    a("")
    a("## 3. S4a 已有切线的 R(v)")
    a("")
    a("| 机器 | 档 | unit-g | Adam | random median | Adam d_box |")
    a("|---|---|---:|---:|---:|---:|")
    for machine in ("amd_2d", "local_2d", "a10_6d"):
        blob = s4.get(machine) or {}
        for stage, rec in blob.items():
            if not isinstance(rec, dict):
                continue
            a(
                f"| {machine} | {stage} | {_fmt(rec.get('amp_unit'))} | "
                f"{_fmt(rec.get('amp_adam'))} | {_fmt(rec.get('rand_median'))} | "
                f"{_fmt(rec.get('d_box_adam'), 1)} |"
            )
    a("")
    a(str(s4.get("ranking_note") or ""))
    a("")
    a("## 4. S2 10-step optimizer 审计（本机 ResNet，闸门 blob_G64）")
    a("")
    a("| stage | opt | box0 | box1 | box10 | anchor1 |")
    a("|---|---|---:|---:|---:|---:|")
    for rec in s2.get("rows") or []:
        a(
            f"| {rec.get('stage')} | {rec.get('opt')} | {_fmt(rec.get('box_step0'), 2)} | "
            f"{_fmt(rec.get('box_step1'), 2)} | {_fmt(rec.get('box_step10'), 2)} | "
            f"{_fmt(rec.get('anchor_step1'), 2)} |"
        )
    a("")
    a("keep-state 是代理：不能从 slim 恢复预训练 Adam 矩，改为同冻结档 30×G64 dense 热身再切四角。head 热身较轻（box≈5.6）；**full 热身 box 可到 178，代理不干净。长实验不要再扫 keep-state。**")
    a("")
    a("## 5. S3 support-Hessian top-k")
    a("")
    a("| stage | k | g∈top-k | AdamΔθ∈top-k | Δf_step1∈J(top-k) | R(v) median / max |")
    a("|---|---:|---:|---:|---:|---:|")
    for stage, rec in (s3.get("stages") or {}).items():
        a(
            f"| {stage} | {rec.get('k')} | {_fmt(rec.get('energy_g_topk'))} | "
            f"{_fmt(rec.get('energy_adam_topk'))} | {_fmt(rec.get('energy_df_topk'))} | "
            f"{_fmt(rec.get('R_v_modes_median'))} / {_fmt(rec.get('R_v_modes_max'))} |"
        )
    a("")
    a("判读：l4/full 的 **Adam Δθ 不在锐 Hessian 里**（约 0.7% / 1.9%）。k=16 的 Δf 能量接近 1 是因为 top-16 已含小特征值，不能当成“shock 在锐模”。head 近零模 R(v) 达数百，高增益在 support Hessian 核。不要单独引用 `energy_df_topk`。")
    a("")
    a("## 6. L3 MLP capacity × support")
    a("")
    a("| arch | stage | support | n | pre box | sparse box | ratio |")
    a("|---|---|---|---:|---:|---:|---:|")
    for rec in l3.get("rows") or []:
        a(
            f"| {rec.get('arch')} | {rec.get('stage')} | {rec.get('support')} | "
            f"{rec.get('n_support')} | {_fmt(rec.get('pretrain_box'), 4)} | "
            f"{_fmt(rec.get('sparse_box'), 4)} | {_fmt(rec.get('box_ratio'), 2)} |"
        )
    a("")
    a("## 7. 给规划 AI 的后续极简方案")
    a("")
    a("见仓库根 [`FOLLOWUP_MINIMAL_mechanism_probe.md`](FOLLOWUP_MINIMAL_mechanism_probe.md)，以及 `results/mechanism_probe/local/FOLLOWUP_MINIMAL.md`。")
    a("")
    return "\n".join(lines) + "\n"


def build_followup() -> str:
    s2 = _load("s2_opt_audit/summary.json") or {}
    s3 = _load("s3_tangent/summary.json") or {}
    s1 = _load("s1_fft.json") or {}

    def _s2_shock(opt: str) -> str:
        rows = [r for r in (s2.get("rows") or []) if r.get("opt") == opt and r.get("stage") == "l4"]
        if not rows:
            return "—"
        return f"l4 step1 box={_fmt(rows[0].get('box_step1'), 1)}"

    l4 = (s3.get("stages") or {}).get("l4") or {}
    amd = s1.get("amd_2d") or {}
    cmp1 = (amd.get("comparisons") or {}).get("1") or {}
    head_l4 = _fmt((cmp1.get("head_vs_l4") or {}).get("cosine_delta_power") or (cmp1.get("head_vs_l4") or {}).get("cosine_power"))

    return f"""# 后续步骤（极简，供规划 AI）

本机第一步已完成（`{CODE_REV}`）。A10/AMD 仍关机。不要重跑 error_extension / functional_drift。

## 本机已经说明的事

- 稀疏 FT = **AdamW reset**。slim 无动量。百像素 step1 不是继承预训练 Adam 矩。
- 2D step1：三档频谱很像（head vs l4 Δu cosine≈{head_l4}），peak_frac 约 0.08–0.16，**不是**少数 spatial frequency 上的伪影。终点谱分叉：head vs l4 cosine≈0.41，l4 vs full 仍≈0.96。
- R(v) 在 unit-g / 随机 / 锐 Hessian 模上均为 ~0.7–1.1，**head 不小于 l4**。
- S2：AdamW reset 复现 AMD（head step1=108.2，l4≈95，full≈114）。**SGD 几乎打掉 shock**（l4/full step1 box≈4.4，与 step0 的 4.55 相当）。0.1×LR 把 shock 压到 ~9–13 px。
- S3：l4/full 的 Adam Δθ 落在 support-Hessian top-16 的能量只有 **0.7% / 1.9%**。锐方向不是高增益方向。head 的近零特征值方向 R(v) 可达数百——高增益在 **support Hessian 核**，不在锐模。
- L3：`mlp_w`+`full`+四角 19.6×；同一预训练下 last-layer 稳住（ratio 0.42）；G9 降到 3.7×，G16/G64 不再崩。linear 稳住。drift ~ plasticity / 约束丰富度。

keep-state 代理（同冻结档 30×G64 dense 热身）对 l4/full **不干净**：full 热身就把 box 拉到 178。长实验不要再扫 keep-state。

## 建议下一轮

1. **主押 L2 minimal constraint rescue（AMD 2D）**  
   `blob_G64 → sparse l4/full`，额外 0/1/2/4/8 个 off-support 点：random vs 按高增益/近核方向的 `|J v|` 选点。  
   成功：2–4 个选中点把终点几十 px 压到几 px。

2. **次优先 L1 optimizer × plasticity（AMD 2D）**  
   S2 已显示 optimizer 管 transient。L1 用来确认 **SGD 压 shock 之后，l4 最终是否仍 consolidation**。若 SGD 终点仍 drift，则两控制量分离成立。A10 6D 只做 SGD vs AdamW × 三档 × 1 个 LR。

3. **L4 catapult 联合轨迹：降级**  
   锐 Hessian 不是 shock 载体。不要为 L4 单独开机。若 L2 需要沿 step 看核方向能量，再附带记。

4. **不要做**  
   重训 G9/五档/impulse/满 NTK；2 维与 6 维混 load；A10 余量加戏；keep-state 轴。

## 机器

- AMD：重传 `blob_G64` + 2 维栈，先 L2 后 L1。
- A10：第二窗口 6 维复制；8h 一实例一件事。
- 本机：L3 已完成。不再训 ResNet 长轨迹。
"""


def run() -> Dict[str, Any]:
    md = build_markdown()
    follow = build_followup()
    (RESULTS_ROOT / "STEP1_REPORT.md").write_text(md, encoding="utf-8")
    (RESULTS_ROOT / "FOLLOWUP_MINIMAL.md").write_text(follow, encoding="utf-8")
    (PACK_ROOT / "实验结果汇总_mechanism_probe_step1_20260819.md").write_text(md, encoding="utf-8")
    (PACK_ROOT / "FOLLOWUP_MINIMAL_mechanism_probe.md").write_text(follow, encoding="utf-8")
    write_json(RESULTS_ROOT / "step1_index.json", {"report": str(RESULTS_ROOT / "STEP1_REPORT.md")})
    print("[report] wrote STEP1_REPORT.md and FOLLOWUP_MINIMAL.md", flush=True)
    return {"ok": True}
