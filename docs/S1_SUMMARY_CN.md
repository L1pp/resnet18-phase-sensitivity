# S1 相关实验结果汇总

> 证据截点：2026-08-22 07:10 PDT（2026-08-22 14:10 UTC）  
> 文档状态：已纳入用户确认完成的最新实验和低成本 S1 最终审计链。当前低成本机器可读证据为 `analysis_verification=VERIFIED`、`evidence_ledger=VERIFIED`；这表示本轮执行/分析审计已收口，**不表示 A10 clean formal aggregate 已由 `inconclusive` 改成 `passed`**。  
> 操作边界：本次只读取、核对和汇总已有产物，没有启动、停止或续跑任何训练/审计任务，也没有修改现有结果树。

## 1. 技术摘要

1. **Neural Affine clean reproduction 已完成 A10 正式运行和回传，但正式协议结论仍是 `inconclusive`。** corrected vanilla 三个 seed 均满足协议的 anchor、raw full-box 和 affine-removed 数值门槛：anchor MAE 为 `0.090804–0.108438 px`，raw full-box MAE 为 `43.361880–53.403000 px`，affine-removed MAE 为 `19.890558–28.040853 px`。正式独立审计因 checkpoint 重前向与已保存全场预测的最大差 `0.006992–0.014236 px` 超过 `0.001 px` 门槛而失败，因此 aggregate 保持 `inconclusive / independent_audit_failed`。
2. **后续补充审计定位了上述预测差的具体计算路径。** 只增加 deterministic flags 时，六个 vanilla run 的差值不变；把预处理顺序改为与保存预测一致的“CPU 上 `uint8 -> float32 -> /255`，再传 CUDA”后，corrected 与 legacy vanilla 共六项均以 `0` 最大预测差通过。该补充材料明确声明 `does_not_replace_formal_aggregate=true`，所以不能把它改写成“正式审计已通过”。
3. **clean controls 已落盘。** closed-form identity 的 raw MAE 为 `6.84e-14 px`；单 seed explicit-xy MLP 的 raw MAE 为 `1.377552 px`；三 seed CoordConv control 的 raw full-box MAE 为 `29.517438–37.542947 px`。这些结果与 corrected vanilla 采用同一 clean 协议框架，但 closed-form、explicit-xy、CoordConv 和 vanilla 是不同预测器/架构，不应合并成同一总体。
4. **低成本并行主流水线中的 AA/P32 六组 60-epoch 新训练已完成并独立验证。** 三 seed 的 baseline 最终 dense MAE 均值为 `0.265284 px`，antialias 为 `0.500679 px`；逐 seed 的 `AA - baseline` 为 `+0.411833、-0.027603、+0.321956 px`。最终 P32 amplitude 的三组差均为正，均值从 baseline `0.049090 px` 到 antialias `0.100534 px`。
5. **低成本 clean S1 单向快照的 A/C 已在两次失败尝试后通过最终独立分析校验。** 当前 `analysis_verification.json` 顶层与 `s1_snapshot` 均为 `VERIFIED`，`failures=[]`、`differences=[]`；snapshot A 的 anchor-affine MAE 为 `0.082981–0.102154 px`，snapshot C 的 degree-2 residual MAE 为 `9.919246–23.550801 px`。此前 `snapshot_field_shape` 和 A/C mismatch 的两份 `AUDIT_HALT` 仍作为失败历史保留，不再代表当前状态。
6. **本轮执行/审计任务已经收口，但两个结论层必须分开。** 低成本 AA/P32、A/B/C 和后写入 S1 snapshot A/C 已独立 verified；D 仍只是历史法医自审。A10 clean formal aggregate 仍是 `inconclusive / independent_audit_failed`，本文件不给出额外 promotion、机制或因果判断。

## 2. 范围、目录与证据等级

### 2.1 本轮 S1 主线和辅助目录

| 目录 | 本文件中的角色 | 当前使用方式 |
|---|---|---|
| `neural_affine_s1_clean/` | clean-room 代码、协议、包 manifest | 用于协议和实现边界；不把本地代码目录当作运行结果 |
| `s1clean_returns/a10_20260821/` | A10 正式回传包与 supplementary audit | 正式 aggregate、controls、审计状态和归档完整性的主要来源 |
| `s1_parallel_lowcost_20260821/` | AA/P32、A/B/C、D forensic、S1 单向快照入口 | 独立 verified 的 AA/P32、A/B/C 与后写入 S1 snapshot A/C，和仅自审完成的 D 分开记录 |
| `s1clean_local_smoke_vanilla/`、`s1clean_local_smoke_coordconv/` | 本机 20-step 运行烟测 | 只证明入口可运行，不作为正式结果 |
| `s1clean_supervisor_logs/` | A10 launcher 重连与启动记录 | 只作执行过程证据 |
| `s1clean_supplementary_tools/` | deterministic/parity re-audit 工具 | 只作 supplementary 诊断方法来源 |
| `s1clean_test_xtjg80at/` | 名称含 s1，但当前读取时目录枚举返回 `Access is denied` | 未用于任何数值或状态判断；本文件不对其内容作断言 |

### 2.2 名称含 `S1/s1`、但不属于本轮 clean S1 的历史产物

以下目录或文件只是历史阶段编号、seed 简写或文件扩展名，已在既有总档案中记录，本文件不重复并入当前 S1：

- `results/phase3_high_upside_discovery/track_b/models/S1__dense__20260833/`
- `results/phase3_high_upside_discovery/track_b/models/S1__sparse__20260833/`
- `results/spatial_field_dynamics/cycle/s1_G9/`
- `results/mechanism_probe/local/s1_fft.json` 及相应 `s1_*.png`
- `results/error_extension/local/q1_controls_vs_s16.json`，其中 `s16` 是 seed `20260816` 的简写
- `download_and_push.ps1`，其中 `.ps1` 只是 PowerShell 扩展名

### 2.3 证据等级

本文件按以下顺序使用证据：

1. 运行期 `metrics.json`、`summary.json`、`aggregate_decision.json`、`history.csv`、predictions、checkpoint 和 manifest；
2. 独立 audit 输出与 supplementary audit 输出；
3. sealed 分析 CSV/JSON 及其独立 verification/ledger；
4. 汇总报告和规划文档，仅用于状态、范围或索引，不用旧叙述覆盖新原始产物。

本次定量结果适合使用表格而不是合并图表：S1 clean 是 A10 上的 2D blob/corners4 协议，AA/P32 是本机 GTX 1060 上的另一套 1D dense protocol，A/B/C 又包含历史 field 和 cached 6D translation-only readout。把这些数值画在同一坐标系中会掩盖协议差异。

## 3. Neural Affine clean reproduction

### 3.1 固定协议与运行环境

正式协议来自 `neural_affine_s1_clean/protocol.json`：

| 项目 | 固定值 |
|---|---|
| task | `neural_affine_2d_clean_blob` |
| seeds | `20260816、20260817、20260818` |
| 图像与坐标 | `224×224`，`coord_scale=223`，safe box `[59,164] px` |
| renderer | blob，`sigma=6`，`supersample=4`，PIL LANCZOS，`uint8/L` |
| support | `8×8` anchor grid 的四角 tids `[0,7,56,63]`；4 个训练/anchor 点 |
| dense evaluation | `41×41=1681` 点，tx-major/ty-minor |
| optimization | 3000 steps，batch 64，AdamW，lr `1e-3`，weight decay `1e-4`，`MSE + 0.25×L1`，cosine eta_min `1e-5` |
| numeric mode | fp32，`amp=false` |
| 正式门槛 | anchor `<=0.25 px`；raw full-box `>=20 px`；affine-removed `>=10 px`；prediction replay `<=0.001 px`；metric diff `<=1e-6` |

正式 GPU 环境由 aggregate 记录为 NVIDIA A10、CUDA 12.8、PyTorch `2.10.0+cu128`、torchvision `0.25.0+cu128`、Python `3.12.13`。全部正式与诊断运行总耗时为 `2373.814 s`。CPU preflight/cache 只用于准备和跨环境缓存核对，不是 GPU 实验结果。

执行器在正式运行前核对 package manifest、CPU preflight、A10 gate 和 cache hash；先运行 closed-form/explicit-xy baselines，再运行 corrected vanilla、legacy exact-init replay、CoordConv control，最后用独立 evaluator 从协议重建 dense/support images 并审计各 run。

### 3.2 corrected vanilla 三 seed 的正式结果

下表三项都以平均欧氏坐标误差、单位 px 记录。`raw full-box` 是 CNN 在 1681 点上的原始预测误差；`affine-removed` 是对原始全场误差做 affine 分解后的残差量，不等于后文“由四个 anchor predictions 构造一个新 affine predictor”的误差。

| seed | anchor MAE | raw full-box MAE | affine-removed MAE | 协议 `valid` | 协议 `positive` | formal audit | checkpoint 重前向 vs saved prediction 最大差 |
|---:|---:|---:|---:|---|---|---|---:|
| 20260816 | 0.108438 | 53.403000 | 28.040853 | true | true | failed | 0.014236 |
| 20260817 | 0.090804 | 43.361880 | 19.890558 | true | true | failed | 0.008361 |
| 20260818 | 0.102926 | 45.794585 | 26.903877 | true | true | failed | 0.006992 |
| 三 seed 均值 | 0.100722 | 47.519822 | 24.945096 | 3/3 | 3/3 | 0/3 | — |

正式 `aggregate_decision.json` 的结论字段是：

- `verdict = inconclusive`
- `reason = independent_audit_failed`
- `diagnostic_complete = true`

formal audit 中的 `metric_comparison` 比较的是 primary metrics 与从已保存 `predictions.npz` 重算的 metrics，该项通过；正式失败门是 checkpoint 独立 CUDA 重前向与 saved predictions 的逐点最大差超过 `0.001 px`。独立 CUDA 重前向得到的 metrics 与 primary metrics 仍有小差异，不能误写成“全部小于 `1e-6`”：corrected vanilla 三 seed 的 absolute delta 分别为 anchor `1.012087e-4 / 5.301466e-4 / 8.748691e-4`、raw `1.099074e-5 / 3.478357e-5 / 1.616182e-5`、residual `7.515990e-7 / 3.355932e-5 / 2.312965e-5`。因此“数值门槛 3/3 命中”和“正式 aggregate inconclusive”必须同时保留。

### 3.3 supplementary preprocess-parity 审计

supplementary audit 对 corrected vanilla 和 legacy exact vanilla 共六项执行了两种复核：

| run | formal / deterministic 最大预测差 | independent-forward vs primary metrics absolute delta（anchor / raw / residual） | CPU preprocess parity 最大预测差 | parity 状态 |
|---|---:|---:|---:|---|
| corrected vanilla s16 | 0.014235556 | `1.012087e-4 / 1.099074e-5 / 7.515990e-7` | 0 | passed |
| corrected vanilla s17 | 0.008360565 | `5.301466e-4 / 3.478357e-5 / 3.355932e-5` | 0 | passed |
| corrected vanilla s18 | 0.006991506 | `8.748691e-4 / 1.616182e-5 / 2.312965e-5` | 0 | passed |
| legacy exact vanilla s16 | 0.015697658 | `5.998800e-4 / 5.623987e-6 / 7.174920e-6` | 0 | passed |
| legacy exact vanilla s17 | 0.007589638 | `1.500076e-4 / 1.966146e-5 / 7.776792e-6` | 0 | passed |
| legacy exact vanilla s18 | 0.012507617 | `2.866004e-4 / 3.163398e-7 / 3.565401e-5` | 0 | passed |

补充材料记录的差异来源是 float32 预处理顺序：formal evaluator 在 CUDA transfer 后执行 `uint8 -> float32 -> /255`，保存预测的路径在 CPU 上先完成转换和除法再传 CUDA。只改 deterministic flags 不改变差值；按保存路径做 parity re-forward 时六项都精确复现。

边界：该诊断说明现有六项 saved predictions 与哪条预处理路径一致，但 supplementary summary 明确写有 `does_not_replace_formal_aggregate=true`。本文件据此记录“已定位并复现差异”，不把正式 `inconclusive` 改写成 `passed`。

### 3.4 baselines 与架构/历史诊断 controls

| predictor / run | seed | anchor MAE | raw full-box MAE | affine-removed / residual MAE | audit /边界 |
|---|---:|---:|---:|---:|---|
| closed-form identity | — | — | `6.84e-14` | `1.13e-13` | analytical baseline |
| explicit-xy MLP | 20260816 | 0.002670 | 1.377552 | 0.600517 | 单 seed、3000 steps |
| corrected CoordConv | 20260816 | 0.106561 | 31.812789 | 22.616700 | formal independent audit passed；architecture control |
| corrected CoordConv | 20260817 | 0.129414 | 29.517438 | 18.141536 | formal independent audit passed；architecture control |
| corrected CoordConv | 20260818 | 0.077996 | 37.542947 | 22.938406 | formal independent audit passed；architecture control |
| legacy exact vanilla | 20260816 | 0.082796 | 53.212541 | 25.062596 | exact-init diagnostic only |
| legacy exact vanilla | 20260817 | 0.145289 | 50.651440 | 28.846836 | exact-init diagnostic only |
| legacy exact vanilla | 20260818 | 0.078954 | 47.053782 | 22.584411 | exact-init diagnostic only |

三 seed 均值仅作描述：corrected CoordConv 的 anchor/raw/residual 分别为 `0.104657 / 32.957725 / 21.232214 px`；legacy exact vanilla 为 `0.102347 / 50.305921 / 25.497948 px`。legacy exact replay 的三 seed raw 和 residual 均未落入相应历史值的 5% tolerance，只有 anchor 的绝对差门槛通过，因此它不替代 corrected clean primary。

### 3.5 完整性与可重放边界

- clean code package manifest 实际列出 26 个文件；本次逐项重算 26/26 的存在性、字节数和 SHA-256 均通过，`MANIFEST.sha256` 也为 26 行；aggregate 为 `a86efeb8bf987bdec5c3919db60dc2c1050444dda2491a725aea03dae9373e37`。此前阶段性版本写成 24 个，属于汇总计数错误，已在本版修正。
- A10 formal return archive 本地大小 `1,245,156,324 bytes`；本次重算 SHA-256 为 `0b29ce1c71089a7f8398182b9f6f5cc13b6f8e5ed1b9784775476387cfc526af`，与声明一致。
- 送入低成本目录的 primary S1 manifest 为 `COMPLETE`，声明 15 个文件；本次逐项重算存在性、字节数和 SHA-256，失败数为 0。其 source manifest SHA-256 为 `2cc0509b826b53433a943f3b98879931028e3dbcf08c717cd7393fd65c7c618d`。
- supplementary tar 本地大小 `24,747 bytes`，本次重算 SHA-256 为 `f08ea77f1b56db4b93c7b7cf340dd79158b34cf65a794bfe4e2d9006f3fe86de`。checkout 中展开了 summary 和 manifest；manifest 内其他 nested audit/context/tool 文件仍在 tar 中，本次没有逐项解包重算，不能声称对每个 nested 文件都完成了本机独立复放。

### 3.6 本机 smoke 不是正式实验结果

| 目录 | variant | steps / batch | best anchor MAE | final anchor MAE | elapsed | 用途 |
|---|---|---|---:|---:|---:|---|
| `s1clean_local_smoke_vanilla/` | vanilla | 20 / 8 | 96.2936 | 100.5390 | 3.11 s | 入口/训练/保存路径 smoke |
| `s1clean_local_smoke_coordconv/` | CoordConv | 20 / 8 | 94.1587 | 94.1587 | 3.22 s | architecture path smoke |

两个 smoke 的 `protocol_hash` 与正式协议相同，但 `code_hash=5741b18d9fbd45f4d9aa4a7bdb154f4c1295ba13ae58c7a1a353c61875a3d303`，不同于 A10 正式包 aggregate/code hash `a86efeb8bf987bdec5c3919db60dc2c1050444dda2491a725aea03dae9373e37`；训练设置也分别是 `20 steps / batch 8 / anchor_every 10` 与正式的 `3000 / 64 / 100`。因此 smoke 只证明流程和早期训练状态，不能与正式表格合并。

## 4. 低成本并行实验与审计

### 4.1 主流水线状态与后写入 S1 快照

低成本主流水线的 `state/pipeline.json` 在 `2026-08-21T17:51:48Z` 记录 `COMPLETED / DONE`。其中 AA/P32 与 A/B/C 已独立 verified；D 的历史法医审计为 `AUDIT_COMPLETE`，但 ledger 标注 `SELF_AUDIT_NO_SECOND_REVIEW`。第一版 `LOW_COST_PARALLEL_FINAL.md` 在 `17:52:46Z` 生成时同时记录：

- `S1 单向快照 = SKIPPED_NO_COMPLETE_S1_INPUT`
- `S1 快照 A/C = SKIPPED_NO_SNAPSHOT`

之后的时间线是：

| UTC 时间 | 后写入事件 | 当前证据状态 |
|---|---|---|
| 2026-08-22 01:26:48 | `s1_inbox/.../manifest.json` 完成 | `COMPLETE`，三 corrected vanilla + supplementary，共 15 个文件 |
| 2026-08-22 01:28:04 | `snapshots/s1_snapshot_status.json` | `SNAPSHOTTED`，manifest SHA `2cc050...` |
| 2026-08-22 01:34:26 | `sealed/s1_snapshot_analysis/summary.json` | `SEALED`，A 24 rows、C 3 records |
| 2026-08-22 02:06:25 | `auditor/analysis_verification.json`、error 与 halt 记录 | S1 snapshot 为 `REJECTED`；三个 seed 均报 `snapshot_field_shape` |
| 2026-08-22 02:47:14 | 第二次 `AUDIT_HALT` | 39 项失败：36 项 snapshot A mismatch、3 项 snapshot C degree-0 missing |
| 2026-08-22 02:49–02:54 | analyzer/auditor 更新、重新冻结并重跑 A/C | `run_snapshot_fields.py`、`auditor.py` 更新；A 24 rows、C 3 records 重新 `SEALED` |
| 2026-08-22 03:36:12 | 最新 `auditor/analysis_verification.json` | 顶层及 A/B/C/S1 snapshot 均 `VERIFIED`；`failures=[]`、S1 `differences=[]` |
| 2026-08-22 03:36:47 | `auditor/aggregate_verification.json` | `VERIFIED`，6/6 runs，`failures=[]` |
| 2026-08-22 03:38:46–03:38:48 | ledger、protected checks、final report 和 code-freeze verification | ledger `VERIFIED`；protected root `UNCHANGED`；code-freeze verification `VERIFIED` |

当前机器可读终态以时间更晚的 `analysis_verification.json` 和 `evidence_ledger.json` 为准：S1 snapshot 的 3 个 manifest-locked fields、A 24 行和 C 3 records 均已独立复算并通过，ledger 也以 `VERIFIED / failures=[]` 收口。`verify-analysis_error.json` 与两份 `AUDIT_HALT` 是早期失败尝试的保留产物；不能删除其历史，也不能用它们覆盖较晚的成功验证。由于 analyzer/auditor 在第二次失败后发生更新并重新冻结，最终 `VERIFIED` 只适用于更新后的冻结审计代码，不应把前后尝试静默视为同一次运行。

最终输入形状核对显示，三个 `predictions.npz` 都包含 `pred_px`、`true_px` 两个 `(1681,2)` float64 dense 字段，以及 `anchor_pred_px`、`anchor_true_px` 两个 `(4,2)` float64 anchor 字段；snapshot manifest 的 15/15 文件字节数与 SHA-256 均匹配。这是当前 shape gate 已通过的直接证据，但不据此反推第一次失败的唯一原因。

低成本 `LOW_COST_PARALLEL_FINAL.md` 已于 `2026-08-22T03:38:47.993697Z` 重新生成，旧版的 `SKIPPED` 已变成 `S1 单向快照=SCANNED`、`S1 快照 A/C 执行=SEALED`；但该 Markdown 正文仍没有显式列出 S1 snapshot 的 independent `VERIFIED`。因此本汇总用最新审计 JSON 与 ledger 认定独立校验状态，同时保留最终 Markdown 的字段粒度边界。最新 `code_freeze_verification.json` 为 `VERIFIED / differences=[]`，其中被验证文件的 SHA-256 为 `2f3b536afc1ea04aebc6ecd2de711fc99bfd3c4dbace83f62ee70865400dc0ef`；它与 ledger 中 auditor freeze 的哈希不是同一对象，不能互相比较。

### 4.2 AA/P32 三 seed × baseline/antialias

新训练协议：本机 NVIDIA GeForce GTX 1060 6 GB、PyTorch `2.12.0+cu126`；seeds `20260810–20260812`；baseline 与 antialias 各 60 epochs；batch 64；AdamW lr `1e-3`、weight decay `1e-4`；AMP true；dense set 为 `x=32..191` 的 160 点。全部六个 run 的 execution 为 sealed，独立 verification 为 verified。

#### 60-epoch 最终值

| seed | baseline dense MAE | antialias dense MAE | AA − baseline | baseline P32 amp | antialias P32 amp | AA − baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 20260810 | 0.261211 | 0.673045 | +0.411833 | 0.040831 | 0.161927 | +0.121096 |
| 20260811 | 0.306219 | 0.278616 | -0.027603 | 0.064436 | 0.070642 | +0.006206 |
| 20260812 | 0.228421 | 0.550377 | +0.321956 | 0.042003 | 0.069033 | +0.027030 |
| 三 seed 均值 | 0.265284 | 0.500679 | +0.235395 | 0.049090 | 0.100534 | +0.051444 |

其他 aggregate 均值：baseline / antialias 的 dense RMSE 为 `0.323029 / 0.542230 px`，dense bias 为 `0.056428 / -0.476720 px`；对应 paired difference 均值分别为 `+0.219202 / -0.533148 px`。

paired difference 的样本 SD：dense MAE `0.232154 px`，dense RMSE `0.247838 px`，dense bias `0.460311 px`，P32 amplitude `0.061212 px`。`n=3`，这里只记录描述性统计，没有做显著性或因果推断。

#### 训练轨迹的若干固定 epoch 读数

| seed | baseline ep20 / ep40 / ep60 MAE | antialias ep20 / ep40 / ep60 MAE |
|---:|---:|---:|
| 20260810 | 3.025860 / 3.455282 / 0.261211 | 0.352705 / 4.941415 / 0.673045 |
| 20260811 | 3.687857 / 0.275589 / 0.306219 | 4.111154 / 1.419781 / 0.278616 |
| 20260812 | 1.050921 / 0.787376 / 0.228421 | 29.685242 / 2.410310 / 0.550377 |

三 seed 的曲线均存在明显非单调波动。历史 seed `20260810` 的 baseline epoch40→60 大幅下降在新 run 中存在；同一固定比较在 seed `20260811` 没有下降，在 seed `20260812` 有较小下降。epoch20 时 antialias 只在 seed `20260810` 低于 baseline；最终 P32 amplitude 则三 seed 都高于各自 baseline。以上是固定 epoch 读数，不把“转变发生时间”扩展成未定义的自动判定。

### 4.3 A：anchor-implied extension

#### 原低成本批次：历史 Neural Affine fields，已 sealed/verified

这里的 affine、bilinear、RBF 都是用网络在四个角点的 predictions 拟合/插值后，再外推到 1681 点得到的**新解析 predictor**；它们不是 CNN 原始 dense output。

| seed | raw CNN | anchor affine | anchor bilinear | thin-plate RBF | anchor constant | anchor nearest | centroid | template match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260816 | 43.1164 | 0.0695 | 0.0714 | 0.0718 | 41.1674 | 39.4563 | 0.5303 | 0.5303 |
| 20260817 | 43.7550 | 0.0956 | 0.0956 | 0.0956 | 41.1675 | 39.4651 | 0.5303 | 0.5303 |
| 20260818 | 61.8617 | 0.0696 | 0.0696 | 0.0697 | 41.1674 | 39.4778 | 0.5303 | 0.5303 |

#### 后写入 clean S1 snapshot：重算后独立校验通过

| seed | raw CNN | anchor affine | anchor bilinear | thin-plate RBF | anchor constant | anchor nearest | centroid | template match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260816 | 53.403000 | 0.101045 | 0.101679 | 0.101843 | 41.167471 | 39.506051 | 0.530309 | 0.530305 |
| 20260817 | 43.361880 | 0.082981 | 0.083052 | 0.083066 | 41.167429 | 39.453146 | 0.530309 | 0.530305 |
| 20260818 | 45.794585 | 0.102154 | 0.102177 | 0.102181 | 41.167475 | 39.498572 | 0.530309 | 0.530305 |

该表来自同一份 source manifest SHA `2cc050...` 锁定的 predictions。第二次失败后，A 分析重新生成 24 行；最新 `analysis_verification.json` 独立复算得到同一组数值，S1 analysis 的 `differences=[] / status=VERIFIED`。因此本表替换此前阶段性版本中的未验证 A 数值；早期表值及 mismatch 仍由两份 `AUDIT_HALT` 的时间线说明，但不再作为当前结果。

### 4.4 B：endpoint feature ridge

原低成本批次的 B 已 sealed 并由独立分析审计验证。其输出包含 360 个 ridge 结果行、72 个 fold manifest 行；split 为四个 interlaced parity folds，每次一个 fold calibration、其余三个 fold test。下表是本汇总从 sealed CSV 按 input 聚合四个 outer folds 后，在已测试 layer/dimension 中取最低 mean test MAE 的描述性索引；它不是新的预注册 headline。

| 输入/seed | 最低 mean test MAE | 对应 layer / dimension | 边界 |
|---|---:|---|---|
| historical Neural Affine 2D / 20260816 | 8.59 | layer2 / raw | 2D endpoint features |
| historical Neural Affine 2D / 20260817 | 3.17 | layer4 / raw | 2D endpoint features |
| historical Neural Affine 2D / 20260818 | 6.09 | layer4 / raw | 2D endpoint features |
| cached 6D / phase2_G64 | 0.65 | GAP / raw | translation-only readout |
| cached 6D / unfreeze_g64_corners4_full | 1.44 | GAP / raw | translation-only readout |
| cached 6D / random_init_s20260816 | 5.68 | GAP / raw | translation-only readout |

cached 6D features 的 shape 为 `13448×512`，每个 translation 重复 8 次；2D endpoint 和 cached 6D translation-only readout 已分别封存，不能把上表六行当作同一总体或直接比较架构优劣。

### 4.5 C：field morphology

原低成本历史 field morphology 已 sealed/verified：

| seed | affine residual 低频能量占比（radius <= 2） | degree 0 residual MAE / explained | degree 1 residual MAE / explained | degree 2 residual MAE / explained | one-pixel slices |
|---:|---:|---:|---:|---:|---|
| 20260816 | 0.158320 | `32.514380 / 0.375203` | `25.710954 / 0.608829` | `23.487612 / 0.665980` | sealed |
| 20260817 | 0.436182 | `24.287638 / 0.660233` | `19.163168 / 0.776191` | `14.510293 / 0.875433` | sealed |
| 20260818 | 0.234460 | `35.259747 / 0.630596` | `30.361333 / 0.732780` | `26.077975 / 0.795223` | sealed |

后写入 clean S1 snapshot morphology（最新独立校验通过）：

| seed | affine residual 低频能量占比（radius <= 2） | degree 0 residual MAE / explained | degree 1 residual MAE / explained | degree 2 residual MAE / explained | one-pixel slices |
|---:|---:|---:|---:|---:|---|
| 20260816 | 0.203772 | `31.563299 / 0.592975` | `27.201240 / 0.678778` | `23.550801 / 0.751242` | unavailable from frozen field only |
| 20260817 | 0.637059 | `20.362073 / 0.750755` | `17.085707 / 0.810742` | `9.919246 / 0.933585` | unavailable from frozen field only |
| 20260818 | 0.241341 | `27.844256 / 0.591452` | `22.247696 / 0.721540` | `19.048704 / 0.792306` | unavailable from frozen field only |

第二张表与 snapshot A 使用同一批 manifest-locked 输入；当前 independent `verify-analysis` 已复算并以 `differences=[] / VERIFIED` 收口。三个 seed 的 `one_pixel_slices` 仍明确为 `UNAVAILABLE_FROM_FROZEN_FIELD_ONLY`，所以该项没有数值结果，不能因整段 verification 通过而补写或推断。

### 4.6 D：历史 AA/P32 证据链法医审计

D 的状态为 `AUDIT_COMPLETE`，且明确 `training_is_not_gated_by_this_audit=true`；evidence ledger 对其审查等级标为 `SELF_AUDIT_NO_SECOND_REVIEW`。它重建的是历史单 seed 链，不验证也不阻止六组新训练，也不能写成独立 verification passed。

- 历史协议：seed `20260810`、60 epochs、batch 64、ReduceLROnPlateau factor `0.5` / patience `4`、dense 160 点。
- P32 定义：先对像素误差做 x 方向线性去趋势，乘 Hann window，取最接近 `1/32 cycles/pixel` 的 rFFT bin，再除以 `window.sum()/2`。
- 历史 20-epoch 对照表中 baseline MAE/P32 为 `2.863214 / 0.152341 px`，antialias 为 `0.358741 / 0.070102 px`。
- baseline 历史 checkpoints `ep10、ep20、ep40、ep60、best` 的 stored 与 recomputed MAE/RMSE/bias/P32 差均为 0；记录到 13 条 scheduler/LR 转换事件。
- 该历史读数与本轮六组新训练的 current histories 分开保存；不能用 D 的单 seed 结果覆盖三 seed aggregate。

## 5. 最终状态、失败历史和不可横比边界

1. **本轮任务已完成：** 用户已确认最新实验完成；低成本 `analysis_verification`、aggregate verification 和 evidence ledger 均为 `VERIFIED`，失败列表为空，protected comparison 为 `UNCHANGED`。这只关闭当前执行/分析审计任务，不改变下列科学证据边界。
2. **两次失败审计必须保留：** `AUDIT_HALT_20260822T020755_6632.json` 记录三项 `snapshot_field_shape`；`AUDIT_HALT_20260822T024920_14364.json` 记录 36 项 A mismatch 和 3 项 C degree missing。当前 `verify-analysis_error.json` 也是第二次失败留下的旧文件，时间早于最终成功的 `analysis_verification.json`，不能把它当成当前 verdict。
3. **最终成功属于更新后的审计代码：** 第二次失败后，`analysis/run_snapshot_fields.py` 与 `auditor/auditor.py` 更新并重新冻结；当前 `VERIFIED` 适用于这套冻结版本。报告保留前后时间线，不把失败和成功静默合并。
4. **低成本 final Markdown 的字段粒度较粗：** 2026-08-22 03:38 UTC 版只显示 S1 snapshot `SCANNED`、A/C `SEALED`，没有显式写 independent `VERIFIED`；最终独立状态应读取较新的机器可读 `analysis_verification.json`、`report_manifest.json` 和 ledger。
5. **formal 与 supplementary 状态并存：** A10 formal aggregate 为 `inconclusive`；supplementary parity 为 6/6 exact replay；低成本 snapshot A/C 又是另一条独立分析验证。后二者都不自动改写 formal verdict。
6. **n=3 只支持描述：** corrected vanilla、CoordConv 和 AA/P32 都只有三 seed；本文件不做显著性、总体分布或机制因果推断。
7. **不同机器/协议不合并：** A10 2D clean、GTX 1060 AA/P32、历史 A10 fields、cached 6D features、local smoke 均保持分列。
8. **anchor-derived predictor 不等于 CNN dense predictor：** A 中 `anchor affine/bilinear/RBF` 使用四个角点 predictions 构造新的解析函数，不能拿其约 `0.07–0.102 px` 误写成 CNN 原始全场误差。
9. **local smoke 不进入科学结果：** 20-step smoke 的训练目标只是运行路径，不与 3000-step A10 正式结果比较。
10. **AI 证据边界：** 本项目相关实验代码、执行、分析和本汇总均由 AI 完成，尚未经过完整人工审计；尽管已有 manifest/hash、独立 evaluator 和审计轨，仍不能保证所有实验结果均无误。

## 6. 本次收口与未来更新触发条件

截至本版证据截点，本次用户所指的最新实验和低成本 S1 审计已完成，没有仍在运行的本轮结果需要预留占位。当前收口字段为：

- S1 snapshot input：15/15 manifest-locked files，三个 dense fields；
- snapshot A/C execution：`SEALED`，A 24 rows、C 3 records，`rejected=[]`；
- independent analysis：`VERIFIED`，S1 `differences=[]`；
- AA/P32 aggregate：`VERIFIED`，6/6 runs；
- evidence ledger：`VERIFIED / failures=[]`；
- protected comparison：`UNCHANGED`；
- clean formal aggregate：仍为 `inconclusive / independent_audit_failed`。

只有出现新的 formal aggregate、重新签发的独立审计、补齐的 one-pixel slices 或新的 S1 实验产物时，才需要继续更新本文件；不能仅因低成本 snapshot 分析已通过，就把 A10 formal clean 改写为 passed。

## 7. 关键证据入口

### Neural Affine clean

- `neural_affine_s1_clean/protocol.json`
- `neural_affine_s1_clean/package_manifest.json`
- `neural_affine_s1_clean/s1clean/runner.py`
- `neural_affine_s1_clean/s1clean/independent_eval.py`
- `s1clean_returns/a10_20260821/neural_affine_s1_clean_return_20260821T155018Z.tar.gz`
- `s1clean_returns/a10_20260821/supplementary/SUPPLEMENTARY_SUMMARY.json`
- `s1clean_returns/a10_20260821/supplementary/SUPPLEMENTARY_MANIFEST.json`
- `s1_parallel_lowcost_20260821/s1_inbox/s1_corrected_vanilla_primary_20260821/run/aggregate_decision.json`
- `s1_parallel_lowcost_20260821/s1_inbox/s1_corrected_vanilla_primary_20260821/manifest.json`

### 低成本并行

- `s1_parallel_lowcost_20260821/config.json`
- `s1_parallel_lowcost_20260821/state/pipeline.json`
- `s1_parallel_lowcost_20260821/reports/LOW_COST_PARALLEL_FINAL.md`
- `s1_parallel_lowcost_20260821/reports/report_manifest.json`
- `s1_parallel_lowcost_20260821/sealed/aa_p32_aggregate/summary.json`
- `s1_parallel_lowcost_20260821/sealed/analysis_A/summary.json`
- `s1_parallel_lowcost_20260821/sealed/analysis_B/summary.json`
- `s1_parallel_lowcost_20260821/sealed/analysis_B/ridge_results.csv`
- `s1_parallel_lowcost_20260821/sealed/analysis_C/summary.json`
- `s1_parallel_lowcost_20260821/forensics/D/history_audit.json`
- `s1_parallel_lowcost_20260821/auditor/aggregate_verification.json`
- `s1_parallel_lowcost_20260821/auditor/analysis_verification.json`
- `s1_parallel_lowcost_20260821/auditor/verify-analysis_error.json`
- `s1_parallel_lowcost_20260821/auditor/archive/AUDIT_HALT_20260822T020755_6632.json`
- `s1_parallel_lowcost_20260821/auditor/archive/AUDIT_HALT_20260822T024920_14364.json`
- `s1_parallel_lowcost_20260821/auditor/evidence_ledger.json`
- `s1_parallel_lowcost_20260821/manifests/code_freeze_verification.json`
- `s1_parallel_lowcost_20260821/manifests/protected_comparison.json`
- `s1_parallel_lowcost_20260821/snapshots/s1_snapshot_status.json`
- `s1_parallel_lowcost_20260821/snapshots/s1/s1_corrected_vanilla_primary_20260821_2cc0509b826b/snapshot_manifest.json`
- `s1_parallel_lowcost_20260821/sealed/s1_snapshot_analysis/summary.json`
- `s1_parallel_lowcost_20260821/sealed/s1_snapshot_analysis/analysis_A.csv`

## 8. 版本记录

- 2026-08-21 PDT：建立阶段性汇总；纳入 A10 clean formal/supplementary、低成本原批次 AA/P32 与 A/B/C/D，以及后写入 clean S1 snapshot A/C；同步记录后续独立 `verify-analysis` 因三项 `snapshot_field_shape` 被拒绝并写入 `AUDIT_HALT`，整体仍未闭合。
- 2026-08-22 PDT：用户确认最新实验完成；纳入第二次 A/C mismatch 审计失败、审计代码更新与重新冻结、最终 `analysis_verification/evidence_ledger=VERIFIED`；用最终独立复算值替换 snapshot A 的未验证旧表，确认 snapshot C 数值通过；将 clean package manifest 计数从错误的 24 修正为现场核验的 26/26。A10 formal verdict 保持 `inconclusive`。
