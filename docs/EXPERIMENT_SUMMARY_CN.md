# 已做实验汇总（截至 2026-08-26）

> 本文是供后续分析使用的事实型原始实验档案。内容按实验协议与可比性分组，记录设置、测量值、控制、缺失项和解释边界；不提供研究结论、方向排序或后续建议。

## 0. 文档范围、证据与复核状态

### 0.1 收录范围

- 时间范围：仓库中已有实验至 2026-08-26 的 Track B B1–B8 正式矩阵、B6 后验机制诊断，以及 Track A MiniCNN 验证性实验。
- 收录内容：正式协议、关键训练与评测设置、主要读数、零值或较大误差等结果、后来复算不一致的结果、控制实验、随机种子、架构与优化器、影响可比性的协议差异、未执行或因预设条件未满足而跳过的步骤。
- 不收录内容：设备性能测试、吞吐/带宽 benchmark、GPU endurance、存储与下载测试、压缩包名称、SHA、逐文件数量、云实例进程状态、上传与解压日志、可由证据路径直接恢复的机器绝对路径。设备环境只在识别科学实验协议时记录。
- 数值单位默认为像素；“±”仅在原实验确实汇总多个 seed 或样本统计时使用。

### 0.2 AI 生成与可信度说明

本项目的实验规划细化、代码编写、数据生成、训练执行、结果整理和既有分析均由 AI 完成。用户未对全部代码、数据划分、训练过程和统计分析进行逐行人工审计。因此：

1. 本文中的数值表示仓库现有产物或依据现有产物复算得到的记录，不等同于已经独立复现的事实。
2. 本轮整理检查了协议、源码、JSON/CSV/NPZ 字段和部分场的复算一致性，但没有对每个训练脚本进行独立实现，也没有重新训练全部模型。
3. 若汇总文档与落盘指标冲突，本文优先记录运行期产物；若运行期产物之间仍冲突，则并列记录冲突，不择一解释。
4. 远端“瘦包”若只有汇总 JSON、没有权重或 dense field，只能核对字段与汇总值，不能在本机重新前向验证。
5. 旧文档中的自动分类与方向优先级字段不作为本文证据。

### 0.3 2026-08-20 后的独立复核升级

自 2026-08-20 后新执行、并在本文列为正式结果的训练或零训练分析主线，均配置了与主执行路径分离的只读复核智能体，或独立 evaluator/auditor。复核覆盖冻结协议、代码与输入身份、support/query 标签边界、落盘预测、指标重算、晋级门和最终聚合。相较只读取主执行智能体自报结果，这些实验的证据可信度更高；实际记录也表明复核不是形式步骤：S1 clean 的正式重前向门被判失败，S1 snapshot 的两次中间审计被叫停，NTK v1 因标签边界问题在产生科学结果前终止。

这种“更高可信度”是相对的、程序性的，不等同于独立实验室复现、人工同行评审或结果必然正确。复核智能体仍属于 AI 系统，可能与执行智能体共享模型局限；hash/manifest 只能绑定字节与流程身份，不能单独证明训练与科学解释正确。本文因此逐项保留 `passed`、`failed`、`inconclusive`、`incomplete`、`supplementary` 等原状态。低成本 D 项明确为 `SELF_AUDIT_NO_SECOND_REVIEW`，仅作历史法医记录，不纳入“新增正式实验均有独立复核”的范围；补充复核也不覆盖或改写既有 formal verdict。

2026-08-24–26 的 Track B 正式矩阵具有逐 cell reviewer、正式 aggregate、本机数值复算和三个分离的报告层复核回执；B6 三项后验诊断与 Track A MiniCNN 也在执行任务中由独立复核智能体检查。后两者的实验根没有另存 machine-readable reviewer receipt，因此其复核证据固化程度低于 Track B 正式矩阵，本文在相应小节单列这一边界。

### 0.4 证据优先级

| 层级 | 证据 | 本文使用方式 |
|---|---|---|
| E1 | 运行期 metrics/summary/train_summary/history、场 NPZ、数据 manifest、checkpoint | 优先记录；场与标签齐全时可复算 |
| E2 | 由 E1 产物生成的表格、图和本机复算 JSON | 记录复算口径；与 E1 冲突时并列 |
| E3 | 协议与源码中的预注册设置、阈值、分支条件 | 用于说明实验意图和实际执行边界 |
| E4 | AI_HANDOFF、阶段汇总、旧版总汇 | 用于定位；其中解释性文字不直接转录 |
| E5 | README 中 planned 字段、未落盘的口头状态 | 不作为已执行实验结果 |

## 1. 平台、任务与统一读数

### 1.1 实验平台

| 平台 | 已记录环境 | 本文涉及的工作 |
|---|---|---|
| 本机 | Windows 10；i7-7700K；GTX 1060 6 GB；Python 3.12.10；PyTorch 2.12.0+cu126 | 早期训练、CPU/短时复算、局部算子测量、MLP 与机制探针 |
| A10 | NVIDIA A10 22 GB；CUDA 12.8；Python 3.12.13；PyTorch 2.10.0+cu128 | Phase 2/3、6D Spatial Field、短周期、6D drift/constraint/rescue |
| AMD | AMD gfx1100 约 48 GB；ROCm 7.2；Python 3.12.3；PyTorch 2.10.0+rocm7.2.4 | 2D Spatial Field、短周期、2D drift/constraint/rescue；另有一项 6D G32 |

上述环境来自 <code>MACHINE_ASSETS.md</code> 与各实验的 env 记录。历史 Phase 2 大多使用 AMP；Spatial Field、today shortcycle、error extension、functional drift、mechanism probe、constraint removal 和 SGD rescue 均记录为 fp32、<code>amp=false</code>。Phase 2 EfficientNet recovery 使用 fp32 和梯度裁剪，与原始 AMP Atlas 不属于同一训练协议。

### 1.2 合成任务与坐标

- 主空间任务在 224×224 画布上合成渲染；常用 <code>coord_scale=223</code>。
- Spatial Field 系列的安全盒为坐标 [59,164]×[59,164]；官方 dense box 为每轴 41 点，共 1681 个位置。
- 6D 任务输出二次 Bézier 三个控制点的六个坐标；2D 任务只输出目标位置 x、y。两者的 MAE 维数、数据内容和头部维度不同，本文始终分表。
- 跨分辨率任务以目标画布像素报告误差。A10 all6 使用 s(L)=(L−1)/223 缩放安全盒；AMD Task 2a 使用固定 [59,164] 网格，L=160 时部分点位于画布外。二者不能以同一列直接比较。
- 相对坐标、绝对坐标、Bézier 控制点、整条曲线误差和 stroke IoU 是不同测量对象。

### 1.3 常用指标

| 名称 | 定义或口径 |
|---|---|
| control MAE / t MAE | 预测控制点或平移坐标的平均绝对误差 |
| q MAE | 去除平移后几何坐标的平均绝对误差 |
| mean_box_mae_px / box | 安全盒 41×41 点上的平均像素 MAE；Spatial Field 系列主读数 |
| interp / outside | 相对训练支持域凸包的内部或外部读数；其点集随支持几何改变 |
| u | 对误差场逐输出维拟合并移除仿射分量后的残差 |
| anchor MAE | 当前训练支持点上的误差；支持点数与任务维度必须同时读取 |
| composed/direct/roundtrip/commutator | 特征平移算子的复合、直接、往返及交换子读数 |
| curve RMSE / stroke IoU | Bézier 曲线采样距离与栅格笔画重叠；不等同于控制点 MAE |

### 1.4 全局可比性规则

以下差异出现时，本文只在各自协议内列数，不作裸数排序：

- 2D xy 与 6D Bézier；
- val、test、interp、outside、全安全盒 dense；
- 不同画布或不同跨分辨率坐标 chart；
- equal optimizer steps 与 equal epochs / equal unique images；
- AMP 与 fp32、是否梯度裁剪、是否恢复训练；
- 冻结 backbone 后拟合线性头与端到端 CNN；
- 不同 primitive family、不同可见性/OOB 分布；
- 单 seed、三 seed 和跨样本统计；
- load-best checkpoint、最后一步和中途 probe；
- 仅有汇总 JSON 的远端瘦包与本机可重新前向的完整产物。

## 2. 实验索引

| 实验族 | 时间/阶段 | 任务维度 | 主要变化 | 主要证据入口 |
|---|---|---:|---|---|
| stride / antialias / no-GAP | 早期 | 2D | 下采样与 GAP | <code>results/main</code>、<code>results/baseline_60ep</code> |
| Bézier inverse / minimal / OOB | 早期 | 8D/6D | 控制点反演、曲线阶数与出画布 | <code>results/bezier_inverse</code>、<code>results/quadratic_bezier_minimal</code>、<code>results/bezier_oob</code> |
| Phase 1–1.8 | 2026-08-10 起 | 6D | 64×64 因子、GAP 子空间、跨 split 探针 | <code>results/phase1_gap_rep</code> |
| Phase 2 | discovery/recovery | 6D | 位置密度、架构、padding、内容、层表示 | <code>results/phase2_overnight_discovery</code> |
| Phase 3 / spectroscopy | 后续批次 | 2D/6D | 特征算子、内容、分辨率、随机先验 | <code>results/phase3_high_upside_discovery</code> |
| AMD Batch 1/2 | 2026-08-16–17 | 2D/6D | 群变换、位置监督、transplant、curriculum | <code>results/amd_batch1_discovery</code>、<code>results/amd_batch2_discovery</code> |
| Spatial Field Dynamics | 2026-08-17 | 2D/6D | 训练时长、basin、hysteresis、支持几何、replay | <code>results/spatial_field_dynamics</code> |
| today shortcycle | 2026-08-17–18 | 2D/6D | calibration、Frozen G64、神经仿射、torus | <code>results/today_shortcycle</code> |
| error extension / drift | 2026-08-18 | 2D/6D | seed 场、解冻层级、首步与轨迹 | <code>results/error_extension</code>、<code>results/functional_drift</code> |
| mechanism probe | 2026-08-19 | 2D/6D | FFT、优化器、Hessian、MLP 容量 | <code>results/mechanism_probe</code> |
| constraint removal | 2026-08-19–20 | 2D/6D | SGD、支持几何、NTK-null | <code>results/constraint_removal</code> |
| SGD Anchor Rescue | 2026-08-20 | 2D/6D | 预注册 SGD 候选、40k 延长、dense 轨迹 | <code>results/sgd_anchor_rescue</code> |
| S0 clean-room closeout | 2026-08-20–21 | 2D/6D | 已保存场复评、partial、BN、OLS、kernel 审计 | <code>closeout_sprint_clean/results</code> |
| Neural Affine S1 clean | 2026-08-21 | 2D | 独立代码路径、four corners、三 seed、独立重前向 | <code>neural_affine_s1_clean</code>、<code>s1clean_returns/a10_20260821</code> |
| 低成本并行实验与 S1 snapshot | 2026-08-21–22 | 1D/2D/6D | AA/P32 三 seed、anchor extension、ridge、morphology | <code>s1_parallel_lowcost_20260821</code> |
| Neural Affine NTK / frozen feature | 2026-08-22–23 | 2D | 轴向分析、exact empirical NTK、head-only、场差分 | <code>remote_results/20260822</code>、<code>results/neural_affine_crossover_clean</code> |
| support-density × plasticity | 2026-08-23 | 2D | corners4/G9/G16/G64 × head/full/frozen，三 seed | <code>handoff/20260823_crossover_results</code> |
| Track A MiniCNN 验证 | 2026-08-25 | 2D | finite-zero、true-valid、torus-circular 边界机制验证 | <code>results/track_a_minicnn_v1</code> |
| Track B function selection | 2026-08-25–26 | 2D | B1–B8 正式矩阵与 B6 后验分解 | <code>track_b</code> |

## 3. stride、GAP 与早期位置回归

### 3.1 约 20 epoch 的三架构比较

设置：224×224；水平线长度 21 px、厚度 3 px、y=112，x 范围 30.5–193.5。训练 offset=[0.1,0.3,0.5,0.7,0.9]，验证 offset=[0.05,0.25,0.45,0.65,0.85]；dense 为 x=32…191 的 160 个整数位置，覆盖 5 个 32-px 周期。训练/验证/dense 样本数 810/810/160；seed=20260810；batch=64；AdamW；CUDA AMP；训练与验证坐标无重合。dense 表记录：

| 模型 | 参数量 | dense MAE | dense RMSE | 相邻步长 MAE / 1 px | 非单调比例 | FFT P32 幅值 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 11.177 M | 2.863 | 3.065 | 0.313 | 0.0063 | 0.152 |
| antialias | 11.177 M | 0.359 | 0.434 | 0.133 | 0 | 0.070 |
| no-GAP | 17.600 M | 0.566 | 0.668 | 0.232 | 0 | 0.016 |

证据：<code>results/main/tables/model_comparison.csv</code>。参数量、下采样和聚合方式同时变化的项目不用于单因素因果归因。

### 3.2 baseline 延长训练

同一 baseline 在不同 checkpoint 的 dense 读数：

| checkpoint | dense MAE | P32 residue peak-to-peak | FFT P32 |
|---|---:|---:|---:|
| epoch 10 | 2.889 | 2.549 | 0.847 |
| epoch 20 | 3.026 | 2.595 | 1.066 |
| epoch 40 | 3.455 | 0.806 | 0.258 |
| epoch 60 | 0.261 | 0.404 | 0.041 |
| best 字段对应 checkpoint | 0.200 | 0.408 | 0.068 |

证据：<code>results/baseline_60ep/tables/dense_summary.csv</code>。20 epoch 表与 60 epoch 表不是同一训练预算。

## 4. Bézier 反演

### 4.1 旧三次 Bézier

模型接收渲染图，ResNet18 从头训练，GAP 后 Linear 512→8，输出三次 Bézier 的四个控制点；单 seed=20260810。数据量 train/val/ID/Hole/Extra=8000/1000/1000/1000/1000；224 画布、4× supersampling、LANCZOS、stroke=3、curve samples=256。AdamW 1e−3、weight decay 1e−4、SmoothL1 beta=0.02、ReduceLROnPlateau、AMP；best epoch=30。

| split | overall coordinate MAE | endpoint MAE | interior-control MAE | curve-t MAE |
|---|---:|---:|---:|---:|
| ID | 7.551 | 3.484 | 11.618 | 3.410 |
| Hole | 6.992 | 3.038 | 10.946 | 3.356 |
| Extra | 10.939 | 4.343 | 17.535 | 4.775 |

Extra 的超出训练支持范围坐标 MAE=21.249，correct-side fraction 约 0.503。局部数值 Jacobian 检查为 600/600 满秩。该检查只覆盖采样点附近的局部可逆性，不是全局唯一性检查。证据：<code>results/bezier_inverse/full/tables/summary.json</code>。

### 4.2 受限二次 Bézier minimal

设置：seed=20260810；train/val/test=10000/256/512；40 epoch；batch=64；ResNet18 从头训练、GAP、Linear 512→6；AdamW 1e−3、weight decay 1e−4、cosine 到 1e−5；MSE；CUDA AMP；仅训练集使用 ±12 px augmentation；限制为曲线完全位于画布内。记录的 best epoch=38，last epoch=40。

| 读数 | 值 |
|---|---:|
| test control MAE | 0.728 |
| endpoint MAE | 0.572 |
| interior MAE | 1.039 |
| curve RMSE | 0.678 |
| Jacobian 满秩 | 64/64 |
| Jacobian condition median | 7.72 |
| 16 个逆优化样本：best control distance mean | 0.992 |
| 重启间离散度 | 17.521 |
| 距离 ≤1 px 比例 | 0.625 |

逆优化依赖初值和有限重启；上述 16 个样本与单 seed 不代表全分布。证据：该实验的 <code>summary.json</code> 与逆优化表。

### 4.3 OOB 二次与三次

两项均使用 seed=20260810、batch=64、AdamW，损失为 MSE+0.25×L1，并从 40 epoch 续至 100 epoch；记录的 best epoch 分别为 100 和 98。

| 曲线 | n | in-frame / OOB | control MAE | endpoint | interior | curve RMSE | stroke IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 二次 | 1000 | 333 / 667 | 1.796 | 0.904 | 3.580 | 1.595 | 0.578 |
| 三次 | 1000 | 1000 / 0 | 4.286 | 1.739 | 6.833 | 2.908 | 0.450 |

二次样本中，in-frame control MAE=1.428，OOB control MAE=1.980。control MAE 与 curve RMSE 的 Pearson 分别为 0.972、0.916；control MAE>3 且 IoU≥0.5 的比例分别为 2.0%、18.1%。二次与三次的 OOB 分布不同，因此本表记录两个数据分布，不能作为仅改变曲线阶数的控制。

## 5. Phase 1：64×64 因子与冻结 GAP

### 5.1 数据、模型和训练 run

- 数据为 64 个 shape identity × 64 个 translation identity，共 4096 个组合；划分 3072/512/512。
- val/test 是未在 train 中出现的 shape–translation 配对重组；shape 与 translation identity 本身来自同一全集，不是全新 shape 或全新 translation identity。
- 模型为从头训练 ResNet18、GAP、Linear 512→6；无 augmentation、无 z-score。
- shape 因子为角度 4×尺度 4×curve-sign 2×alpha 2；translation 为 8×8。

| run | 优化设置 | best epoch | test control MAE | 备注 |
|---|---|---:|---:|---|
| factorial AdamW | AdamW，MSE | 188 | 2.254 | 初版 |
| AdamW+L1 resume | 从旧 checkpoint 续训并新建 optimizer | 117 | 1.276 | resume |
| AdamW+L1 scratch | AdamW 1e−3→1e−5 cosine，MSE+0.25L1，100 epoch | 95 | 0.536 | resume=false |
| SGD 1.0 | SGD，lr=1.0 | 1 | NaN | 首 epoch 出现非有限值 |
| SGD fixed | SGD，lr=0.1，gradient clipping | 37 | 2.321 | 有限值 |

scratch run 的 train/val/test MAE=0.398/0.549/0.536；test RMSE=0.718，endpoint/interior MAE=0.493/0.623。证据：<code>results/phase1_gap_rep/adamw_l1_scratch</code>。

### 5.2 初版 ANOVA、SVD 与 rank-3 erasure

对冻结 GAP 表征的初版分解记录：

| 项 | 数值 |
|---|---:|
| translation 能量占比 | 2.088% |
| shape 能量占比 | 96.733% |
| interaction 能量占比 | 1.179% |
| translation SVD 80/90/95% rank | 2 / 3 / 5 |
| shape SVD 80/90/95% rank | 1 / 1 / 1 |

旧 rank-3 translation erasure 的 before→after t probe 为 0.257→29.886；energy-matched random 对照报告 0.910±0.353。Phase 1.5 随后检查发现旧 rank-3 子空间与主方差方向高度重合，因而下节分别记录能量、监督子空间和重探针。

### 5.3 Phase 1.5：旧 rank-3 切法的测量

| 测量 | 数值 |
|---|---:|
| centered 总能量 | 64.736 |
| PC1 能量 | 62.965（97.26%） |
| 旧 T-rank3 删除能量 | 62.801 |
| 旧 T-rank3 与前三主成分 overlap | 2.975 / 3 |
| 旧切法后，原头 t / q | 29.893 / 4.491 |
| permutation 后，原头 t / q | 28.188 / 4.694 |
| 旧切法后，重新拟合 probe t / q | 29.906 / 0.915 |
| PC1 与平移坐标相关 | 近 0 |
| 仅删除 PC1 后 t / q | 0.520 / 2.826 |

这些行使用同一冻结 Z，但“原头”和“重新拟合 probe”是不同读出。

### 5.4 Phase 1.5b：监督 rank-2 B_t

B_t 的构造：在 train 上把标准化 2D translation 与 4D geometry 联合回归到 centered GAP，再对 translation 对应的两行做 QR，得到 rank=2 子空间。

| 处理 | t MAE | q MAE |
|---|---:|---:|
| 删除 B_t，原训练头 | 53.617 | 0.427 |
| energy-matched random rank-2，原训练头 | 18.064 | 21.076 |
| 删除 B_t，重新拟合线性 probe | 30.030 | 0.433 |
| 删除原头 W_t 对应方向，再拟合 GAP probe | 0.472 | — |

随机对照经验尾部计数对应 p=0.001（t 与 q 各自的脚本尾部口径为 1/1000）。B_t 与两个比较子空间的主角为 41.7°、54.6°。

### 5.5 Phase 1.6：置换和 shape-fold

| 项 | 数值 |
|---|---:|
| 原头 true B_t 删除后的 t | 53.62 |
| identity permutation 均值；经验 p | 51.91；0.50 |
| re-probe true B_t 删除后的 t | 30.03 |
| permutation 均值；经验 p | 5.85；0.001 |
| random rank-2 均值；经验 p | 2.37；0.001 |
| 删除前后 geometry q | 0.427 / 0.433 |
| shape-fold 子空间角均值 / 最大值 | 1.02° / 2.59° |

“原头”与“re-probe”仍为不同读出；经验 p 值来自有限置换/随机样本。

### 5.6 Phase 1.7：三 seed 与擦除复算冲突

seed=20260810 复用既有 scratch checkpoint；20260811、20260812 为新训练。

| seed | test control MAE | t MAE | q MAE | B_t 删除能量比例 |
|---:|---:|---:|---:|---:|
| 20260810 | 0.536 | 0.312 | 0.438 | 11.9% |
| 20260811 | 0.569 | 0.371 | 0.431 | 92.5% |
| 20260812 | 0.569 | 0.327 | 0.478 | 76.0% |
| mean±sample SD | 0.558±0.019 | 0.337±0.030 | 0.449±0.026 | — |

运行时 <code>bt_summary.json</code>、per-seed summary 与 master 表把三个 seed 的 B_t erasure 后 re-probe t 记为 30.030 左右。另一次依据当前 Z、B_t 与标签的代码复算记录约 0.27，且相关源码中存在常数 30.030；目前没有找到独立 runtime 结果来替代正式表。两套数值的来源不同且冲突，本文不将该 erasure 数字用于可解释比较。seed 20260811 的 energy-matched random 记录 <code>matched_ok=false</code>，相对能量误差约 7%。

### 5.7 Phase 1.7c：val→test 高维线性探针

每个 split 为 512 样本；含 bias 的 OLS 有 513 个参数。val 拟合后在 test 上：

| seed | raw t | erased t | raw q | erased q | constant t |
|---:|---:|---:|---:|---:|---:|
| 20260810 | 3.458 | 445.456 | 8.868 | 9.659 | 30.030 |
| 20260811 | 7.307 | 612.267 | 13.348 | 11.522 | 30.030 |
| 20260812 | 10.017 | 325.489 | 7.041 | 6.799 | 30.030 |

由于样本数小于含 bias 参数数，该 probe 是高维、跨 split、病态读出；数值只按这一协议记录。

### 5.8 Phase 1.8：unseen translation identity

划分为 train 48 identities×64=3072，val/test 各 8 identities×64=512；val identities 为 9,12,22,26,37,43,49,54，test identities 为 11,17,21,30,34,41,44,51。三个 <code>train_config</code> 均记录 <code>resume=true</code>，但 <code>resume_from=null</code>，且 train_summary 没有对应 resume 记录；因此该 metadata 不能单独证明实际恢复训练。

| seed | test control MAE | t MAE | q MAE |
|---:|---:|---:|---:|
| 20260810 | 0.638 | 0.419 | 0.499 |
| 20260811 | 0.532 | 0.378 | 0.377 |
| 20260812 | 0.543 | 0.401 | 0.369 |

constant t 约为 22.5。val→test assay：

| seed | raw t | erased t | R_t | R_q |
|---:|---:|---:|---:|---:|
| 20260810 | 31.26 | 138.04 | 6.14 | 0.389 |
| 20260811 | 8.74 | 58.39 | 2.59 | 1.333 |
| 20260812 | 11.93 | 168.58 | 7.49 | 1.194 |

该 assay 仍使用 512×513 OLS。identity-permutation 的特殊匹配为 0/3；seed 20260811 的匹配记录异常，seed 20260812 没有有效 energy-null 样本。

## 6. Phase 2：位置覆盖、架构与恢复训练

### 6.1 Part A/B：三 seed 的密集行为与表征相似度

dense 集为 32 shapes×41×41=53,792 个样本。

| seed | t MAE | q MAE | mean det(J) |
|---:|---:|---:|---:|
| 20260810 | 1.597 | 1.329 | 0.908 |
| 20260811 | 1.524 | 1.226 | 0.915 |
| 20260812 | 1.410 | 1.212 | 0.911 |

跨 seed 冻结 GAP 比较：

| seed 对 | linear CKA | pairwise-distance Spearman | kNN overlap | behavioral-Jacobian Pearson |
|---|---:|---:|---:|---:|
| 10–11 | 0.109 | 0.112 | 0.306 | 0.992 |
| 10–12 | 0.056 | 0.139 | 0.204 | 0.989 |
| 11–12 | 0.006 | 0.035 | 0.067 | 0.988 |

CKA、距离、kNN 与 behavioral Jacobian 测量的对象不同；本表不把它们合成为单一分数。

### 6.2 Discovery 协议

- seed=20260820；每个 run 5600 optimizer steps；batch=64；AdamW+cosine；损失 MSE+0.25×L1；原 discovery 多数使用 CUDA AMP；按 val 保存 checkpoint。
- 5600 steps 相同，但 epoch 数和 unique images 不同。G9 与 C9 只有 504 个 unique 训练图像，其它网格的 unique image 数也随支持密度变化。
- interp/outside 根据各 run 自己的支持域定义，支持域改变时点集也改变。

### 6.3 ResNet18 位置密度与覆盖

| regime | dense interp t MAE | dense outside t MAE |
|---|---:|---:|
| G64 | 1.213 | — |
| G32 | 2.963 | 3.421 |
| G16 | 3.709 | — |
| G9 | 7.566 | — |
| G4 | 17.157 | — |
| C16 | 2.995 | 12.443 |
| C9 | 2.350 | 17.108 |
| C4 | 3.302 | 33.694 |

“C9 interp=2.350”与“G9=7.584”的 interp 点集不同；全 box 指标是在后续 Spatial Field 协议中统一加入的。

### 6.4 架构 Atlas

表内为 dense interpolation t MAE，训练超参相同但架构容量、归一化和实现细节不同。

| 架构 | G64 | G9 | C9 |
|---|---:|---:|---:|
| DenseNet121 | 0.600 | 4.259 | 1.977 |
| ResNet18 | 1.214 | 7.584 | 2.350 |
| ResNet50 | 1.492 | 7.945 | 3.305 |
| EfficientNet-B0（原 AMP） | step 395 数值爆炸 | 41.736 | 11.068 |
| MobileNetV3 | 41.113 | 41.817 | 11.017 |
| ConvNeXt-Tiny | 45.174 | 70.444 | 41.634 |

EfficientNet G64 的原 AMP run 未产生与其它有限值 run 同口径的完成读数。

### 6.5 padding、antialias 与 CoordConv

| ResNet18 变体 | G9 interp | C9 interp |
|---|---:|---:|
| standard | 7.584 | 2.350 |
| antialias | 8.415 | 0.749 |
| circular padding | 59.781 | 8.996 |
| circular+antialias | 22.235 | 0.918 |
| CoordConv | 7.439 | 未运行 |

这些变体修改了输入通道或下采样/padding 路径；表内是同训练预算的结果记录。

### 6.6 shape generalization、层表示与轨迹

shape generalization 的 dense cache 只包含预设 16 个 unseen shapes 中的 7 个：

| 项 | t MAE | q MAE |
|---|---:|---:|
| H1 G64 | 2.047 | 1.958 |
| H2 G16 | 4.018 | 3.727 |

GAP 层分解：

| 架构 | shape | position | interaction | metric Pearson | rank90 |
|---|---:|---:|---:|---:|---:|
| ResNet18 G64 | 0.329 | 0.616 | 0.055 | 0.952 | 2 |
| ResNet50 G64 | 0.455 | 0.518 | 0.028 | 0.956 | 2 |
| ConvNeXt G64 | 0.029 | 0.128 | 0.842 | −0.015 | 61 |

ResNet18 G64 seed=20260813 的训练轨迹：

| checkpoint | t MAE | positive determinant fraction | metric Pearson | rank90 |
|---|---:|---:|---:|---:|
| init | 164.3 | 0.29 | 0.65 | — |
| epoch 10 | 17.7 | — | 约 0.99 | — |
| epoch 100 | 1.15 | 0.56 | 0.99 | 3 |

该轨迹没有另存与 discovery 完全相同的 normal dense_eval 表。

### 6.7 Recovery

EfficientNet-B0 G64 recovery 改用 fp32、gradient clipping=1.0、5600 steps；best val=0.195，dense overall/interp=1.418/1.421，读数均有限。它与 Atlas 中的 AMP run 协议不同。

其它 recovery 分支：

| 分支 | val 或设置 | interp | outside |
|---|---|---:|---:|
| LeftHalf | 5600 steps | 2.353 | 24.989 |
| C16 CoordConv | seed=20260820 | 3.237 | 12.249 |
| C16 circular | seed=20260820 | 22.979 | 48.249 |
| G2x | val=2.01 | 无 interp 点 | 30.549 |
| G2y | val=3.08 | 无 interp 点 | 27.757 |

G2x/G2y 的支持几何使 dense 点均被标为 outside。

### 6.8 line / arc 内容对照

同为 ResNet18-G64，但输出为 2D 位置：

| 训练内容 | own-content MAE | cross-content MAE |
|---|---:|---:|
| line | 3.147 | 8.447 |
| arc | 0.647 | 8.033 |

旧 <code>init_probe</code> 中 t/constant 为 0，是因为 <code>train['t']</code> 未填充；该字段不作为位置读数。own/cross 表来自另一个已填充标签的评测。

## 7. Phase 3：算子、内容、分辨率与初始化

### 7.1 Track A：冻结特征上的平移仿射

对冻结 GAP 拟合平移算子。表中 mean E、composed、direct、roundtrip、commutator 均为相应脚本的算子误差；head 为原任务 MAE。

| 模型 | mean E | composed | direct | roundtrip | commutator | head |
|---|---:|---:|---:|---:|---:|---:|
| Phase1.8 seed10 | 0.0460 | 0.0526 | 0.0505 | 0.0273 | 0.0227 | 0.91 |
| Phase1.8 seed11 | 0.1507 | 0.2281 | 0.2095 | 0.1016 | 0.1153 | 0.77 |
| Phase1.8 seed12 | 0.1489 | 0.2977 | 0.2524 | 0.1192 | 0.0469 | 0.81 |
| DenseNet G64 | 0.00464 | 0.00519 | 0.00469 | 0.00340 | 0.00159 | 5.02 |
| ResNet18 G64 | 0.00940 | 0.0106 | 0.00999 | 0.00501 | 0.00127 | 0.60 |
| EfficientNet fp32 G64 | 0.0184 | 0.0185 | 0.0181 | 0.0149 | 0.00120 | 0.84 |
| random ResNet18 | 0.0480 | 0.0572 | 0.0530 | 0.0362 | 0.00392 | 未训练头 |

### 7.2 Track B：对称性破坏阶梯

受控 CNN：96×96 blob，约 1.1 M 参数，seed=20260833。dense/sparse 分支各自记录：

| 阶段 | dense task | dense D_eq | sparse task | sparse D_eq |
|---|---:|---:|---:|---:|
| S0 | 15.09 | 0 | 15.07 | 0 |
| S1 | 0.1508 | 9.4e−5 | 0.1507 | 6.9e−6 |
| S2 | 0.1508 | 1.4e−3 | 0.1508 | 8.6e−4 |
| S3 | 15.07 | 1.4e−3 | 未完成 | 未完成 |

初版 Track B panel 的条目均记录 <code>ok=false</code>，错误为 “numpy.ndarray object is not callable”；后续 controlled CNN 产生了上表的部分结果。S3 sparse 缺完整结果，S4–S6 未运行；原 panel 含未更新的 planned/stale 字段，本文不抄录。

### 7.3 Track C：primitive family 留一法

输出为 2D xy。每行模型未在训练中看到该 primitive family：

| 留出 family | unseen test | seen test | val |
|---|---:|---:|---:|
| line | 2.32 | 2.41 | 2.32 |
| circle | 1.87 | 2.33 | 2.35 |
| arc | 11.01 | 0.575 | 0.532 |
| quadratic | 9.56 | 1.42 | 1.43 |
| polygon | 2.29 | 2.11 | 2.09 |
| blob | 5.86 | 2.53 | 2.29 |

六个 unseen test 的中位数为 4.09；all6 模型 val=2.10。single-quadratic 模型在 quadratic/ blob/line/circle/arc/polygon 上的测试 MAE 分别为 1.49/5.68/6.33/6.70/12.00/32.12。留一法与 single-family 模型的训练集不同。

### 7.4 Track D：跨分辨率零样本

每档 n=600。relative 与 absolute 是两个模型/坐标定义。

| L | relative MAE（px） | absolute MAE（px） | relative slope x/y | absolute slope x/y |
|---:|---:|---:|---|---|
| 160 | 16.70 | 14.94 | 1.363 / 1.389 | 1.343 / 1.372 |
| 192 | 7.72 | 7.39 | — | — |
| 224 | 1.84 | 1.84 | — | — |
| 256 | 8.35 | 9.00 | — | — |
| 320 | 29.28 | 36.27 | — | — |

原始归一化 MAE：L160 relative/absolute=0.1051/0.0939；L192=0.0404/0.0387；L224=0.00827；L256=0.0327/0.0353；L320=0.0918/0.1137。

### 7.5 Track E：随机初始化到分阶段训练

R0/R1 训练 2000 steps，R2/R3 训练 5600 steps；R0/R1 冻结 backbone，R2 只训练最后 stage+head，R3 全量解冻。G64 train/val=3584/512，G9=504/72；test 是从 dense cache 固定抽取的 512 点，不是 val 集。表内为 val/test MAE。

| regime / 架构 | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| G64 ResNet18 | 12.18/11.25 | 10.54/10.31 | 2.26/2.99 | 0.477/1.00 |
| G64 DenseNet | 9.33/9.24 | 7.16/7.79 | 1.87/2.81 | 0.230/0.797 |
| G64 EfficientNet | 45.65/40.83 | 45.65/40.85 | 5.30/5.27 | 0.153/3.35 |
| G9 ResNet18 | 54.59/40.39 | 38.09/27.99 | 1.22/8.45 | 0.249/12.92 |
| G9 DenseNet | 52.13/38.47 | 10.78/10.24 | 0.358/8.66 | 0.087/8.38 |
| G9 EfficientNet | 约 56.50/41.95 | 约 56.50/41.95 | 约 56.50/41.95 | 约 56.50/41.95 |

阶段间训练预算和冻结状态按 Track E 协议变化；R0 与 R3 不是只改变训练步数的单因素对照。实现中模型在 <code>seed_everything(seed)</code> 调用前创建，因此记录的 seed 控制后续 DataLoader/训练过程，但不能证明各 run 使用完全相同的初始权重。

### 7.6 Track F：平移、旋转与尺度联合头

seed=20260834；ResNet18 输出 tx、ty、theta、log-scale；AMP=true；train/val/test=3000/400/600。val composite=57.98；test translation MAE=10.57 px、rotation MAE=46.56°、scale absolute error=0.0153。三个分量单位不同，composite 依赖脚本中的加权定义。模型和新 4D head 同样在设 seed 前创建，因此 seed 字段不能证明其初始化已固定。

## 8. Operator spectroscopy 与本机局部复算

### 8.1 A10 已有权重谱

| 模型 | mean E | composed | commutator | unseen op | val | dense/interp |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet G64 | 0.0046 | 0.0052 | 0.024 | 1.29 | 0.19 | 0.60 |
| ResNet18 G64 | 0.0094 | 0.011 | 0.030 | 1.49 | 0.52 | 1.21 |
| EfficientNet fp32 G64 | 0.018 | 0.018 | 0.028 | 1.24 | 0.19 | 1.42 |
| ResNet18 G32 | 0.011 | 0.012 | 0.024 | 1.93 | 0.53 | 2.96 |
| ResNet18 G16 | 0.0079 | 0.0091 | 0.028 | 1.48 | 0.73 | 3.72 |
| DenseNet G9 | 0.0083 | 0.011 | 0.024 | 1.15 | 0.37 | 4.26 |
| ResNet18 G9 | 0.011 | 0.013 | 0.028 | 1.74 | 1.00 | 7.58 |
| ResNet18-AA G9 | 0.0045 | 0.0047 | 0.026 | 1.54 | 0.60 | 8.42 |
| ResNet18 C9 | 0.0094 | 0.014 | 0.026 | 1.32 | 0.78 | 2.35 |
| ResNet18 G4 | 0.0042 | 0.0052 | 0.023 | 1.75 | 1.80 | 17.20 |
| EfficientNet G9 | 1.008 | 1.007 | 0.040 | 1.00 | 39.8 | 41.7 |
| random ResNet18 | 0.048 | 0.057 | 0.041 | 1.62 | — | — |

另含 Phase1.8 三 seed：mean E=0.046/0.151/0.149，composed=0.053/0.228/0.298，commutator=0.006/0.020/0.022，unseen op=2.30/1.83/1.74。

在 n=11 的混合模型表中：

| 变量与 dense | Spearman（p） | Pearson |
|---|---:|---:|
| anchor val | 0.782（0.00447） | 0.931 |
| mean E | −0.055（0.873） | 0.916 |
| composed | 0.064（0.853） | 0.916 |
| unseen op | 0.100（0.770） | −0.321 |
| roundtrip | −0.009（未单列 p） | 0.917 |
| commutator | 0.055（未单列 p） | 0.715 |

n=11 混合了架构、支持密度和部分训练协议；Pearson 与 Spearman 的差异保留为统计读数。

### 8.2 Intertwiner 配对

| 源→目标 | train | unseen | identity baseline |
|---|---:|---:|---:|
| Phase1.8 seed10→11 | 0.0147 | 0.039 | 0.065 |
| ResNet18 G64→DenseNet G64 | 0.0054 | 0.013 | — |
| ResNet18 G64→EfficientNet G64 | 0.0024 | 0.0021 | — |
| ResNet18 G64→random ResNet18 | 0.0072 | 0.051 | 0.045 |
| ResNet18 G64→ResNet18 G9 | 0.0024 | 0.0055 | 0.040 |

### 8.3 本机 GTX 1060 的局部层测量

本机 slim 评测使用 8 identities×5×5 origins×11 deltas；与 A10 全量 spectroscopy 的样本数不同。

| 模型 | layer1 mean E | GAP mean E |
|---|---:|---:|
| ResNet18 G64 | 0.0013 | 0.012 |
| DenseNet G64 | 0.0010 | 0.0022 |
| ResNet18 G9 | 0.0015 | 0.009 |
| random ResNet18 | 0.0066 | 0.073 |

跨内容算子矩阵，行为单位按本机脚本；行是拟合算子的内容，列是被移动内容：

| fit＼eval | noise | blob | line | quadratic | polygon |
|---|---:|---:|---:|---:|---:|
| noise | 0.033 | 0.477 | 0.751 | 0.475 | 0.151 |
| blob | 0.290 | 0.015 | 0.274 | 0.535 | 0.243 |
| line | 0.603 | 0.057 | 0.012 | 0.070 | 0.670 |
| quadratic | 0.476 | 0.097 | 0.052 | 0.012 | 0.615 |
| polygon | 0.092 | 0.117 | 0.235 | 0.204 | 0.011 |

Track E 本机 GAP 的 R0→R3：

| 模型 | R0 mean E / composed | R3 mean E / composed |
|---|---|---|
| ResNet18 G64 | 0.639 / 0.699 | 0.013 / 0.018 |
| DenseNet G64 | 0.421 / 0.455 | 0.004 / 0.006 |

补测：ConvNeXt G64=0.705/0.764；MobileNet G64=0/0，但对应特征 MSE 约 1e−15，记录为近常量特征条件下的零值；circular G9=0.171/0.172。EfficientNet 本机局部层测量未执行。

## 9. AMD Batch 1（ROCm）

### 9.1 平台与前置 run

该批实际运行于 AMD gfx1100/ROCm。seed=20260816；ResNet18；5600 steps；fp32。前置模型训练/验证样本为 1792/256，translation identities 为 56/8。val MAE 的 best/final 字段为 0.558/0.565；true-shift composed/direct=0.0039/0.0036。

### 9.2 多种变换读数

以下为各脚本自己的目标单位，平移、角度、log-scale 不合并：

| 内容/变换 | 任务读数 | extra 读数 | composed 或附加读数 |
|---|---:|---:|---:|
| polygon translation T2 | 2.146 | 25.79 | 0.0019 |
| blob translation T2 | 2.522 | 30.51 | 0.0014 |
| polygon SO(2)，普通损失 | 108.94° | 165.54° | — |
| polygon SO(2)，circular loss | 97.50° | 13.54° | 与上一行损失不同 |
| blob SO(2) | 85.08° | 168.75° | — |
| polygon scale | 0.00494 log-s | 0.066 | — |
| blob scale | 0.145 log-s | 0.231 | — |

SO(2) 的普通损失与 circular loss 是不同训练目标，不能只用 extra 角误差做同协议比较。

### 9.3 无位置监督任务

| 任务 | 自身 val/task loss | composed |
|---|---:|---:|
| coordinate regression | 0.609 | 0.0031 |
| classification | 0 | 0.010 |
| reconstruction | 0.0038 | 0.028 |
| SSL | 2.67 | 0.066 |

自身损失的定义不同；classification 的 0、reconstruction 的 0.0038 和坐标 MAE 不同单位。

### 9.4 小型架构/约束梯子

| 变体 | val | composed |
|---|---:|---:|
| P0 | 1.77 | 0.158 |
| circular | 1.78 | 0.115 |
| antialias | 1.34 | 0.201 |
| toroidal | 1.75 | 0.126 |
| stride1 | 34.53 | 约 0 |
| negative-control | 34.75 | 约 0 |

P0 约 0.70 M 参数；stride1 与 negative-control 使用 batch=32。旧 <code>group_law direct0</code> 的 target 为空，本文只记录 shifted/composed 指标。该批协议条目 D 未运行。

## 10. Batch 2（目录沿用 AMD 名称，实际为 A10 CUDA）

首轮存在实现问题，以下使用 <code>a10_fixes</code> 后的结果。seed=20260816；fp32、<code>amp=false</code>、batch=64。不同条目存在 2D 与 6D 输出头。

### 10.1 Frozen-backbone transplant

只训练约 3k–6k 参数的小头；随机标定 n=128；1600 steps。

| 源→目标 | val | dense |
|---|---:|---:|
| ResNet18→DenseNet，G64 | 40.15 | 37.57 |
| DenseNet→ResNet18，G64 | 254.58 | 255.80 |
| ResNet18 G64→G9 | 50.87 | 39.30 |

这是冻结骨干与新小头的读数，不是目标架构端到端重新训练。

### 10.2 Resolution calibration

原模型 zero-shot：224 relative=2.00，320 relative=31.05，320 slope=0.66。对 320 做 1200-step calibration 后：

| 解冻范围 | 320 MAE | 320 slope | 回测 224 MAE |
|---|---:|---:|---:|
| head | 19.28 | 0.95 | 55.55 |
| layer4 | 6.34 | 0.99 | 53.98 |
| full | 6.43 | 0.99 | 58.81 |

320 calibration 与 224 zero-shot 使用不同训练状态；表中同时保留 320 与回测 224。

### 10.3 Content transfer

每项 1200 steps：

| 源/目标设置 | head | layer4 | full |
|---|---:|---:|---:|
| all6 | 12.47 | 7.22 | 8.76 |
| arc | 11.71 | 9.77 | 11.09 |
| quadratic | 7.50 | 5.07 | 5.14 |
| polygon negative-control | 28.67 | 13.80 | 9.86 |

各行的源内容、目标内容或基线不同，数字只在对应配置内读取。

### 10.4 Algebra regularizer

仅 G9；2800 steps；72 个平行四边形约束：

| 设置 | val | dense | exact/algebra 字段 |
|---|---:|---:|---:|
| no regularizer | 1.65 | 5.05 | 0.74 |
| weight 0.1 | 1.32 | 5.24 | 0.58 |

### 10.5 Curriculum

每档 2800 steps；R3；6D 输出头。

| 顺序 | 阶段 | val | dense/interp |
|---|---|---:|---:|
| forward | G16 | 1.43 | 3.46 |
| forward | G32 | 0.72 | 1.70 |
| forward | G64 | 0.61 | 1.14 |
| reverse | G32 | 0.70 | 1.37 |
| reverse | G16 | 0.65 | 2.16 |
| reverse | G9 | 0.61 | 3.00 |

reverse 从原始 G64 checkpoint 开始，不是从 forward G64 终点继续。

## 11. Spatial Field Dynamics

### 11.1 统一协议

- <code>code_rev=spatial_field_20260817</code>；fp32、<code>amp=false</code>。
- 224 画布、<code>coord_scale=223</code>、安全盒 [59,164]；官方 box 为 41×41 点。
- A10 主要运行 6D Bézier；AMD 的 content、relative、cross-resolution 主要为 2D xy，另有 6D G32 grokking。
- 默认 AdamW 1e−3、weight decay 1e−4、cosine 到 1e−5、batch=64；每个阶段重新建立 optimizer。
- 按 val 保存并在官方 dense 评测前加载 <code>best_slim</code>。少数 probe 在当前/最后状态运行，本文标为 probe，不与 official 裸比。

### 11.2 A10 grokking 时间轨迹

| run | steps | train | val | official box | 说明 |
|---|---:|---:|---:|---:|---|
| G9 long | 56,000 | 0.0315 | 0.3294 | 6.5606 | init 与旧 G9 hash 不同 |
| G16 long | 28,000 | 0.0669 | 0.3795 | 3.4804 | — |
| scratch G9 | 8,000 | 0.2394 | 1.0322 | 4.9899 | matched-basin 起点之一 |

G9 long 的 official checkpoint 序列：

| steps | box |
|---:|---:|
| 5,600 | 7.24 |
| 11,200 | 8.03 |
| 28,000 | 6.48 |
| 56,000 | 6.56 |

记录中的非 official probe 使用不同 checkpoint 状态或子集，未并入此表。

### 11.3 Matched G9 basin

每项训练 8000 steps；终点均使用 G9 支持：

| 初始化 | train | official box |
|---|---:|---:|
| scratch | 0.239 | 4.9899 |
| G16→G9 | <0.4 | 3.9169 |
| G32→G9 | <0.4 | 4.2097 |
| G64→G9 | <0.4 | 4.2741 |
| forward-G64→G9 | <0.4 | 4.1669 |

初始化 checkpoint 的训练历史不同；该表的支持几何和 8000-step 预算相同。

### 11.4 Hysteresis

| loop | 档位 | up box | down box | up−down |
|---|---|---:|---:|---:|
| 1 | G16 | 3.4276 | 2.9056 | 0.5220 |
| 1 | G32 | 2.6060 | 2.1195 | 0.4865 |
| 2 | G16 | 3.5411 | 2.3377 | 1.2034 |
| 2 | G32 | 2.5547 | 2.2152 | 0.3395 |

第二轮没有 <code>G9_up_2</code> 产物；up/down 来自顺序训练路径，不是独立随机初始化。

### 11.5 支持几何

| 支持集 | 支持点数 | hull 情况 | box | interp |
|---|---:|---|---:|---:|
| maximin9 | 9 | 二维覆盖 | 4.5559 | 4.5658 |
| random9 | 9 | 随机 | 5.1048 | 4.5909 |
| boundary8+center | 9 | 边界+中心 | 5.6460 | 4.7951 |
| cross5 | 5 | 十字 | 9.7705 | 7.1760 |
| diagonal8 | 8 | hull 面积为 0 | 25.8883 | 2.0068 |

diagonal8 的 interp 只覆盖退化线段附近，而 box 覆盖完整二维安全盒。

### 11.6 Replay 剂量与 cycle

| replay 设置 | 旧样本占比 | box |
|---|---:|---:|
| p0（无再训练基线） | 0 | 4.9194 |
| p0p1 | 0.0998% | 4.4207 |
| p1 | 0.97% | 3.3199 |
| p5 | 4.89% | 1.8131 |

cycle：

| 阶段 | 支持 | box |
|---|---|---:|
| s1 | G9 | 4.5773 |
| s2 | G64 | 0.7904 |
| s3 | G9 | 2.6587 |

### 11.7 AMD content→quadratic G9

以下均为 2D xy，目标为 quadratic G9：

| 初始化内容 | box |
|---|---:|
| quadratic scratch | 5.5129 |
| blob | 5.7888 |
| line | 5.8264 |
| polygon | 6.3219 |
| arc transfer | 10.73 |
| arc scratch 对照 | 11.37 |

### 11.8 AMD 224↔320

2D blob 任务，使用 relative 像素口径：

| 路径 | box |
|---|---:|
| scratch 320 G9 | 45.2107 |
| 224 G64→320 G9 | 21.6692 |
| 224 G64→320 G64 | 9.5014 |
| 320 G9→224 G9 | 21.3763 |

该表与 A10 all6 的 6D、缩放安全盒协议不同。

### 11.9 Relative 坐标与锚点校准

| 设置 | box |
|---|---:|
| 未校准 train-pairs 模型 | 135.498 |
| 1 anchor bias calibration | 1.7785 |
| 3 anchors | 1.8838 |
| 4 anchors | 1.8582 |

这里拟合的是相对模型输出到绝对坐标的校准参数；未校准与校准后使用同一模型。

### 11.10 Acquisition / forgetting

上行 acquisition 的 box 从 5.6356 降至 1.4798；支持点 n=9→11 的最大相邻下降为 1.32。下行序列从 1.72 增至 2.34；未生成 <code>down_n64</code>。

### 11.11 AMD 6D G32 长轨迹

| steps | official box |
|---:|---:|
| 5,600 | 3.1297 |
| 11,200 | 3.0875 |
| 28,000 | 3.5301 |

该项是 AMD 上从零训练的 6D G32，不是 A10 Phase 2 checkpoint 的跨机延续。

### 11.12 产物边界

Spatial Field 的部分远端回传只有 JSON、图或 slim，不包含完整权重、训练缓存或 41×41 场；这些条目不能在本机统一重新前向。A10 shortcycle 后回传了 27 个 field NPZ；AMD 的若干旧 shortcycle 条目最初仅有机侧 Markdown，后续各轮是否有场须按各实验目录单独判断。

## 12. Today Shortcycle（2026-08-17–18）

### 12.1 协议

该轮使用 fp32、<code>amp=false</code>。训练项通常使用 AdamW 1e−3、weight decay 1e−4、cosine 到 1e−5、batch=64。A10 与 AMD 任务按现有资产分配，不能把两机结果视作跨机复现：

- A10 all6 是 Track C 的 2D 混合 primitive 模型，跨分辨率网格按 s(L) 缩放。
- A10 Frozen G64 OLS 使用 Phase 2 的 6D Bézier G64 backbone。
- A10 Neural Affine 使用单一 blob、2D translation，新训 3000 steps。
- AMD relative、quad、blob 和 torus 均为 2D，但模型和训练目标不同。

### 12.2 A10 all6 跨分辨率

每档含 8 个 eval appearances×41×41；raw、known-scale+bias 与 full affine：

| L | s(L) | raw | s(L)+bias | affine |
|---:|---:|---:|---:|---:|
| 160 | 0.7130 | 11.355 | 2.118 | 1.608 |
| 192 | 0.8565 | 5.058 | 2.286 | 1.658 |
| 224 | 1.0000 | 1.587 | 1.751 | 1.747 |
| 256 | 1.1435 | 4.282 | 3.957 | 2.295 |
| 320 | 1.4305 | 13.674 | 10.254 | 5.688 |

单独再对预测乘 s(L) 的读数为 30.150/18.883/1.587/28.062/105.371；模型输出本来已经以目标画布像素表示，因此该列与“已知尺度加 bias”不是同一变换。

AMD Task 2a 使用固定网格，仅作本协议内记录：

| L | raw | L/224+bias | affine | 网格是否 OOB |
|---:|---:|---:|---:|---|
| 192 | 12.715 | 10.100 | 10.014 | 否 |
| 224 | 4.506 | 4.282 | 4.277 | 否 |
| 256 | 15.392 | 11.506 | 6.041 | 否 |
| 320 | 51.801 | 25.588 | 16.818 | 否 |

L=160 的固定 [59,164] 网格部分出画布，未列入比较。

### 12.3 Relative 与 absolute calibration

AMD 2D、224 画布：

| 模型 | raw box | bias-only | affine |
|---|---:|---:|---:|
| relative/train_pairs | 135.498 | 1.786 | 1.749 |
| absolute quad_G64 | 1.984 | 1.912 | 1.849 |

relative 模型的偏置向量范数为 135.496 px。64 个单 anchor calibration 的 dense 均值/标准差为 1.823/0.068，64/64 小于 2 px。按 appearance 使用同一 global bias 后，appearance 0=0.65、appearance 5=4.41；appearance 5 使用自身 bias 时为 0.39。其余 appearance 约为 1.3–1.7。

### 12.4 Frozen G64 + OLS

冻结 Phase 2 ResNet18 G64 backbone，使用 GAP 特征闭式拟合 6D 控制点：

| run 名 | 实际 support | support t MAE | exact | box | hull area |
|---|---|---:|---:|---:|---:|
| ols_G64 | 64 tids | 0.318 | 0.643 | 1.192 | 11025 |
| ols_G9 | G9 9 点 | 0.094 | 0.270 | 3.174 | 11025 |
| ols_G4 | 四角 0,7,56,63 | 约 0 | 0.168 | 3.712 | 11025 |
| ols_3nc | 0,7,56 | 约 0 | 0.148 | 2.767 | 5512.5 |

<code>ols_G4</code> 是 frozen G64+四角 OLS，不是 Phase 2 G4 CNN。G4 CNN checkpoint 当时缺失，Task 5 对该模型跳过。OLS 设计矩阵数值条件数很大：G9 约 2.9e14，G64 约 7.8e14；支持样本数分别为 G64 3584、G9 504、G4 224、3nc 168。

### 12.5 Neural Affine controls

单一 blob、2D translation、3000 steps、fp32。四角 hull 覆盖整个测试盒。

| 模型/支持 | seed | train anchor | box | u |
|---|---:|---:|---:|---:|
| closed-form affine / corners4 | — | — | 约 0 | 约 0 |
| 显式 x,y MLP / corners4 | — | 0.0041 | 3.539 | 1.110 |
| vanilla CNN / corners4 | 20260816 | 0.108 | 43.12 | 30.94 |
| vanilla CNN / corners4 | 20260817 | 0.098 | 43.75 | 23.79 |
| vanilla CNN / corners4 | 20260818 | 0.076 | 61.86 | 26.31 |
| CoordConv / corners4 | — | 0.054 | 28.44 | 16.72 |
| vanilla CNN / noncollinear3 | — | 0.050 | 42.12 | 21.35 |
| vanilla CNN / collinear3 | — | 0.063 | 44.51 | 37.36 |
| vanilla CNN / cross5 | — | 0.134 | 33.84 | 26.60 |

closed-form、显式坐标 MLP、CoordConv 与 vanilla CNN 的输入表示和参数化不同；它们是控制设置，不是只改变优化器的复现。

### 12.6 已有 6D 场的统一重放

本节为 A10 已有权重重新前向，无新训练：

| run | box | hull area | exact | in-hull | outside | u |
|---|---:|---:|---:|---:|---:|---:|
| Phase2 G9 | 7.566 | 11025 | 0.167 | 7.566 | — | 7.273 |
| geometry maximin9 | 4.556 | 11025 | 0.408 | 4.556 | — | 4.542 |
| geometry random9 | 5.105 | 9112.5 | 0.481 | 4.585 | 7.359 | 4.577 |
| boundary8+center | 5.646 | 9225 | — | 4.795 | 9.380 | 5.570 |
| cross5 | 9.770 | 5512.5 | — | 7.176 | 12.138 | 8.157 |
| diagonal8 | 25.888 | 0 | 0.576 | 1.937 | 26.487 | 21.764 |
| basin scratch G9 | 4.990 | 11025 | 0.513 | 4.990 | — | 4.880 |
| basin G64→G9 | 4.274 | 11025 | 0.261 | 4.274 | — | 3.599 |
| grok G9 10× | 6.561 | 11025 | 0.089 | 6.561 | — | 6.251 |

diagonal8 的 hull area=0，in-hull 仅对应退化支持线附近；其 1.937 与全 box 25.888 不是同一域。

### 12.7 Torus 与同 9 节点训练

AMD torus：

| 评测 | MAE |
|---|---:|
| 历史 ladder val | 约 1.75 |
| 当前 G64 val，wrapped=true | 1.7505 |
| 同 split，wrapped=false | 1.7505 |
| dense 8 appearances×41×41 | 51.2879 |

安全盒内 wrap 是恒等映射，且模型卷积使用 <code>padding_mode=zeros</code>。val 与 dense 的样本域不同。

同一 G9 9 节点、quadratic、4000 steps：

| run | support MAE | 8 个未见 appearance | 同 appearance 41×41 |
|---|---:|---:|---:|
| absolute-9 | 0.204 | 85.002 | 29.95 |
| tree+anchor | 0.269 | 32.078 | 24.49 |
| cycles+anchor | 0.271 | 32.467 | 23.45 |

Task 7 与 Task 4 的内容、模型来源、训练点和 appearance 评测不同，不是后者的多 seed 重复。

## 13. Error Extension（2026-08-18）

### 13.1 协议与三 seed 场

训练使用 AdamW 1e−3、cosine、batch=64、fp32；6D G9 从零训练 8000 steps，6D 解冻 4000 steps，2D 新训/解冻 3000 steps；seeds=20260816/17/18。

对每个 seed 的误差场移除仿射分量后比较 u：

| 数据来源 | pairwise corr(u) | 中位数 |
|---|---|---:|
| A10 Neural Affine 2D 三 seed | 0.606–0.769 | 0.671 |
| A10 6D G9：16–17 / 16–18 / 17–18 | 0.590 / 0.909 / 0.652 | 0.652 |
| AMD 独立 2D 三 seed | 0.729 / 0.840 / 0.735 | 0.735 |
| A10 与 AMD 2D 同 seed 配对 | 0.716 / 0.738 / 0.675 | 0.716 |

原协议把中位数 ≥0.7、<0.3 和中间区间映射到脚本标签。本文保留阈值与读数，不转录标签。

A10 6D G9：

| seed | train anchor | val anchor | box | u |
|---:|---:|---:|---:|---:|
| 20260816 | 0.239 | 1.032 | 4.990 | 4.880 |
| 20260817 | 0.339 | 0.748 | 5.377 | 5.188 |
| 20260818 | 0.160 | 0.951 | 4.987 | 4.888 |

AMD 2D：

| seed | train anchor | box | u |
|---:|---:|---:|---:|
| 20260816 | 0.175 | 55.57 | 31.64 |
| 20260817 | 0.165 | 43.62 | 25.11 |
| 20260818 | 0.143 | 47.21 | 29.30 |

### 13.2 从同一 G64 起点独立解冻

A10 6D，Phase 2 G64→corners4：

| 解冻范围 | train anchor | val anchor | box | u |
|---|---:|---:|---:|---:|
| head | 0.591 | 0.635 | 1.331 | 1.230 |
| layer4 | 0.136 | 0.905 | 12.645 | 6.422 |
| layer3+4 | 0.164 | 0.732 | 14.842 | 8.616 |
| layer2–4 | 0.172 | 0.931 | 16.383 | 8.618 |
| full | 0.472 | 1.475 | 11.519 | 7.747 |

A10 6D，Phase 2 G64→G9：

| 解冻范围 | train anchor | val anchor | box | u |
|---|---:|---:|---:|---:|
| head | 0.441 | 0.474 | 1.231 | 1.212 |
| layer4 | 0.187 | 0.700 | 3.827 | 2.850 |
| layer3+4 | 0.349 | 0.824 | 4.500 | 4.348 |
| layer2–4 | 0.233 | 0.989 | 5.286 | 4.837 |
| full | 0.389 | 1.042 | 4.577 | 3.721 |

AMD 2D，blob G64→corners4：

| 解冻范围 | train anchor | last-train | box | u |
|---|---:|---:|---:|---:|
| head | 0.111 | 0.119 | 4.588 | 4.319 |
| layer4 | 0.074 | 0.076 | 54.131 | 32.007 |
| layer3+4 | 0.078 | 0.078 | 48.083 | 22.482 |
| layer2–4 | 0.094 | 0.127 | 53.024 | 24.673 |
| full | 0.065 | 0.065 | 70.345 | 27.588 |

每档从同一原始 checkpoint 独立开始，不是逐档继续；A10 6D 与 AMD 2D 分属不同任务。

### 13.3 Label impulse 与 GAP ridge

AMD seed16 共生成 12/12 个 label-impulse G 场。邻格方向 cosine 中位数=0.794；预注册阈值=0.5。+x/−x 反对称 cosine 中位数=−0.049；预注册阈值=0.5，四个 corner tid 分别为 0.075、−0.173、0.274、−0.310。由于第二个前置条件未满足，协议规定的 seed17 impulse 与双锚 superposition 未执行。

ridge 目标均为 sparse CNN 的 u，support 上拟合 Pearson 约为 1：

| 维度/目标 | 特征 dump | 全场 Pearson | holdout |
|---|---|---:|---:|
| 2D corners4 seed16 | random-init | 0.461 | 0.461 |
| 2D corners4 seed16 | sparse CNN GAP | −0.209 | −0.213 |
| 2D corners4 seed16 | frozen blob G64 GAP | −0.015 | −0.017 |
| 6D scratch G9 seed16 | random-init | 0.058 | 0.056 |
| 6D scratch G9 seed16 | Phase2 G9 | 0.047 | 0.046 |
| 6D scratch G9 seed16 | Phase2 G64 | 0.194 | 0.193 |
| 6D scratch G9 seed16 | scratch G9 自身 GAP | −0.367 | −0.368 |
| 6D scratch G9 seed16 | corners-full GAP | 0.112 | 0.111 |

这些是有限支持上的 ridge 线性插值相关。由于 impulse 的反对称前置条件未满足，未执行的 seed17/superposition 不能作为零效应数据。

## 14. Functional Drift（2026-08-18）

### 14.1 协议边界

所有轨迹从原始未解冻 checkpoint 开始。远端 slim 不含原 optimizer state，因此 functional drift 使用新建 AdamW optimizer；“step 1”包含该 optimizer 的首次矩估计。A10 官方场为 32×41×41 的 6D，AMD 为 1×41×41 的 2D。

### 14.2 切线与第 1 步

| 平台/阶段 | unit-gradient amplification | Δbox，GD | Δbox，Adam | 对照 cosine |
|---|---:|---:|---:|---:|
| AMD head | 0.8134 | 7.1295 | 103.6284 | Adam vs step1=1.0000 |
| AMD layer4 | 0.7573 | −0.2494 | 95.6854 | Adam vs step1=0.9998 |
| AMD full | 0.7578 | −0.2425 | 113.7299 | Adam vs step1=0.9997 |
| A10 head | 未记录 | 0.0093 | 20.0159 | GD vs step1=0.5989 |
| A10 layer4 | 未记录 | 10.4743 | 23.7849 | GD vs step1=−0.3112 |
| A10 full | 未记录 | 11.9632 | 24.1012 | GD vs step1=−0.3072 |

GD 与 Adam 的步长归一化和预条件不同；负 Δbox 表示该定义下的一步变化方向，不表示终点结果。

### 14.3 轨迹的 step0、step1、step10 与终点

AMD 2D：

| 阶段 | step0 box | step1 | step10 | endpoint box | endpoint anchor |
|---|---:|---:|---:|---:|---:|
| head | 4.5507 | 108.1791 | 34.4136 | 4.6555 | 0.1193 |
| layer4 | 4.5507 | 91.7767 | 23.1736 | 54.1368 | 0.0762 |
| full | 4.5507 | 110.9141 | 115.0757 | 70.3450 | 0.0654 |

A10 6D：

| 阶段 | step0 box | step1 | step10 | endpoint box | endpoint anchor |
|---|---:|---:|---:|---:|---:|
| head | 1.2129 | 36.1578 | 10.8532 | 1.3035 | 0.5039 |
| layer4 | 1.2129 | 6.6267 | 58.1263 | 12.6449 | 0.1359 |
| full | 1.2129 | 6.4976 | 98.2580 | 11.5256 | 0.1029 |

仅报告终点会遗漏首步与中间峰值，因此四个 checkpoint 均保留。

### 14.4 MLP dense→sparse

本机 2D MLP/linear 控制：

| 架构 | 解冻 | weight decay | dense pretrain u | sparse endpoint u | endpoint/pretrain | sparse anchor |
|---|---|---:|---:|---:|---:|---:|
| linear | full | 1e−4 | 0.0057 | 0.0001 | 0.02 | 0.0002 |
| linear | last | 1e−4 | 0.0057 | 0.0001 | 0.02 | 0.0002 |
| small MLP | full | 1e−4 | 0.1381 | 0.2658 | 1.93 | 0.0047 |
| small MLP | last | 1e−4 | 0.2623 | 0.7803 | 2.97 | 0.0026 |
| wide MLP | full | 1e−4 | 0.0841 | 1.6472 | 19.60 | 0.0150 |
| wide MLP | full | 0 | 0.0998 | 1.6781 | 16.81 | 0.0244 |
| wide MLP | last | 1e−4 | 0.3337 | 0.6902 | 2.07 | 0.0034 |

各行的 pretrain 值随架构、解冻范围和 weight decay 配置变化；ratio 使用同行起点。

## 15. Mechanism Probe（2026-08-19）

### 15.1 范围

该轮主要复用 functional-drift 场做频谱和线性化分析。本机新增 10-step optimizer 审计、support-Hessian top-k 与 MLP capacity×support；没有在 GTX 1060 上启动新的数千步 ResNet 长轨迹。所有 ResNet 10-step 项从同一 blob G64 2D checkpoint 开始。

### 15.2 Δu 频谱

step1 的 affine-removed Δu；每格为 peak-frequency fraction / low-frequency r≤2 fraction：

| 平台 | head | layer4 | full |
|---|---|---|---|
| AMD 2D | 0.077 / 0.264 | 0.113 / 0.263 | 0.147 / 0.209 |
| A10 6D | 0.813 / 0.946 | 0.547 / 0.839 | 0.388 / 0.755 |

不同解冻范围之间的功率谱 cosine：

| 平台/step | head–layer4 | head–full | layer4–full |
|---|---:|---:|---:|
| AMD step1 | 0.934 | 0.837 | 0.961 |
| AMD endpoint | 0.410 | 0.481 | 0.958 |
| A10 step1 | 0.984 | 0.937 | 0.983 |
| A10 endpoint | 0.530 | 0.682 | 0.923 |

MLP 对照的 step1 peak/low：linear full=0.028/0.107；small MLP full=0.102/0.609；wide MLP full=0.154/0.677；wide MLP last=0.086/0.417。

### 15.3 10-step optimizer 审计

| 解冻范围 | optimizer | box0 | box1 | box10 | anchor1 |
|---|---|---:|---:|---:|---:|
| head | AdamW reset | 4.55 | 108.18 | 34.41 | 127.02 |
| head | SGD | 4.55 | 11.68 | 6.14 | 10.81 |
| head | AdamW small-lr | 4.55 | 13.25 | 7.74 | 12.54 |
| head | keep-state proxy | 4.55 | 7.88 | 10.87 | 9.37 |
| layer4 | AdamW reset | 4.55 | 95.46 | 23.39 | 131.79 |
| layer4 | SGD | 4.55 | 4.40 | 6.17 | 1.55 |
| layer4 | AdamW small-lr | 4.55 | 8.67 | 6.59 | 13.51 |
| layer4 | keep-state proxy | 4.55 | 35.22 | 11.38 | 43.32 |
| full | AdamW reset | 4.55 | 114.21 | 115.55 | 138.94 |
| full | SGD | 4.55 | 4.38 | 6.13 | 1.63 |
| full | AdamW small-lr | 4.55 | 9.93 | 8.23 | 14.92 |
| full | keep-state proxy | 4.55 | 175.11 | 81.59 | 171.48 |

keep-state 项不是原 optimizer state 恢复：slim 不含该状态，脚本先在 G64 dense 上热身 30 steps，再切换 corners4。full 热身期间 box 可到约 178，因此该代理与其它三种 optimizer 不是干净的单因素比较。

### 15.4 方向放大率 R(v)

| 来源/范围 | unit-gradient | Adam | random median | Adam Δbox |
|---|---:|---:|---:|---:|
| AMD head | 0.813 | 0.835 | 0.865 | 103.6 |
| AMD layer4 | 0.757 | 0.723 | 0.838 | 95.7 |
| AMD full | 0.758 | 0.839 | 0.822 | 113.7 |
| 本机 head | 0.811 | 0.832 | 0.945 | 103.6 |
| 本机 layer4 | 0.757 | 0.745 | 1.043 | 90.8 |
| 本机 full | 0.760 | 0.732 | 0.915 | 109.5 |

A10 6D 没有对应的 R(v) 产物。

### 15.5 Support-Hessian top-k

k=16：

| 范围 | gradient energy in top-k | Adam Δθ energy in top-k | Δf energy in J(top-k) | R(v) median / max |
|---|---:|---:|---:|---:|
| head | 1.000 | 0.944 | 1.000 | 99.350 / 673.565 |
| layer4 | 1.000 | 0.007 | 0.998 | 0.824 / 1.065 |
| full | 0.999 | 0.019 | 0.994 | 0.829 / 1.197 |

top-16 集合含小特征值方向；因此“Δf energy 接近 1”不等价于只位于最大特征值方向。

### 15.6 MLP capacity×support

同一显式坐标回归控制；box：

| 架构/解冻 | pretrain | corners4 | G9 | G16 | G64 |
|---|---:|---:|---:|---:|---:|
| linear full | 0.1112 | 0.0013 | 0.0016 | 0.0013 | 0.0004 |
| small MLP last | 0.1381 | 0.2001 | 0.1702 | 0.0818 | 0.0670 |
| small MLP full | 0.1381 | 0.2658 | 0.1508 | 0.0801 | 0.0666 |
| medium MLP last | 0.0948 | 0.0811 | 0.0670 | 0.0319 | 0.0266 |
| medium MLP full | 0.0948 | 0.3448 | 0.0424 | 0.0233 | 0.0239 |
| wide MLP last | 0.0841 | 0.0352 | 0.0335 | 0.0309 | 0.0179 |
| wide MLP full | 0.0841 | 1.6472 | 0.3079 | 0.0194 | 0.0085 |

“last”与“full”改变可训练参数集合；support 列改变训练点数。

## 16. Constraint Removal（2026-08-19–20）

### 16.1 协议与完成范围

- optimizer=SGD，无 momentum；lr=1e−3，cosine 到 1e−5；weight decay=1e−4；batch=64；fp32。
- 2D run 3000 steps，6D run 4000 steps；每项从同一原始 checkpoint 独立初始化。
- 支持集为 dense/G64/G32/G16/G9/corners4；seed=20260816。
- 落盘完整性：本机 MLP 32 个训练单元与 5 个 MLP NTK 单元；AMD CNN 19 个、A10 CNN 16 个。P3 maintenance 未启动。
- 预设 P0 使用 corners4、SGD、layer4；只有 support anchor 回到或低于 step0，才继续把 off-support 变化解释为已拟合支持点条件下的函数变化。

### 16.2 P0 前置条件读数

| 平台 | lr | step0 box/u/anchor | endpoint box/u/anchor |
|---|---:|---|---|
| AMD 2D | 1e−3 | 4.5507 / 4.3023 / 0.2397 | 7.0255 / 6.1801 / 0.7788 |
| AMD 2D | 1e−2 | 同一 θ0 | 15.0702 / 11.9985 / 1.9128 |
| A10 6D | 1e−3 | 1.2128 / 1.2003 / 0.7028 | 22.1321 / 11.1768 / 9.3760 |
| A10 6D | 1e−2 | 同一 θ0 | 15.4956 / 8.9044 / 5.2327 |

两机 lr=1e−3 的 endpoint anchor 均高于各自 step0 anchor；lr=1e−2 补跑也未满足该前置条件。因此后续相图仅作为已执行的 optimizer/support 测量，不能当作“support 已拟合”条件下的读数。

### 16.3 AMD 2D SGD 相图

| support | head box/u/anchor | layer4 box/u/anchor | full box/u/anchor |
|---|---|---|---|
| dense | 3.741/3.772/3.744 | 4.026/4.032/4.136 | 3.878/3.879/3.950 |
| G64 | 4.647/4.310/0.136 | 4.621/4.298/0.187 | 4.630/4.273/0.187 |
| G32 | 4.611/4.292/0.118 | 4.772/4.371/0.211 | 4.752/4.355/0.211 |
| G16 | 4.646/4.295/0.136 | 5.225/4.980/0.523 | 5.109/4.878/0.529 |
| G9 | 4.635/4.316/0.121 | 4.946/4.726/0.478 | 4.823/4.627/0.452 |
| corners4 | 4.641/4.320/0.105 | 7.026/6.180/0.779 | 6.638/5.937/0.545 |

### 16.4 A10 6D SGD 相图

表内为 box/u；相应 anchor 需在各 run train_summary 中读取。

| support | head | layer4 | full |
|---|---|---|---|
| G64_continue | 1.2125/1.2004 | 18.1394/2.0677 | 17.0905/2.8589 |
| G32 | 1.2120/1.2006 | 18.1080/2.0639 | 16.8089/2.8320 |
| G16 | 1.2135/1.1990 | 17.7773/2.0400 | 17.4478/2.8406 |
| G9 | 1.2268/1.2029 | 11.8339/9.6652 | 15.2249/12.7964 |
| corners4 | 1.2496/1.2059 | 22.1321/11.1768 | 28.7901/17.4986 |

### 16.5 本机 MLP SGD

wide MLP pretrain box/u=0.0841/0.02018。SGD 后：

| 范围 | dense u | G64 u | G32 u | G16 u | G9 u | corners4 u |
|---|---:|---:|---:|---:|---:|---:|
| last | 0.0198 | 0.0197 | 0.0197 | 0.0201 | 0.0199 | 0.0200 |
| full | 0.0197 | 0.0197 | 0.0197 | 0.0201 | 0.0199 | 0.0200 |

linear full 的同列 u 为 7.20e−6、6.53e−6、7.51e−6、7.31e−6、7.07e−6、6.96e−6。

### 16.6 NTK-null 线性化 cosine

| 平台/模型 | 范围 | support | cos_lin | u0→u1 |
|---|---|---|---:|---|
| MLP wide | last | G64 | 1.0000 | 0.0202→0.0197 |
| MLP wide | last | corners4 | 1.0000 | 0.0202→0.0200 |
| MLP wide | full | G64 | 0.9998 | 0.0202→0.0197 |
| MLP wide | full | corners4 | 1.0000 | 0.0202→0.0200 |
| AMD CNN | head | corners4 | 1.0000 | 4.3023→4.3204 |
| AMD CNN | layer4 | corners4 | 0.3009 | 4.3023→6.1801 |
| AMD CNN | layer4 | G64 | −0.0771 | 4.3023→4.2982 |
| AMD CNN | full | corners4 | −0.2022 | 4.3023→5.9365 |
| A10 CNN | head | corners4 | 1.0000 | 1.3921→1.3982 |
| A10 CNN | layer4 | corners4 | −0.5890 | 1.3921→10.0457 |
| A10 CNN | head | G64_continue | 1.0000 | 1.3921→1.3912 |
| A10 CNN | layer4 | G64_continue | −0.8694 | 1.3921→2.0526 |

AMD CNN 与 A10 CNN 的 NTK 评测起点 box/u（4.5507/4.3023 与 1.4024/1.3921）不同；表内 cosine 不跨任务求平均。

## 17. SGD Anchor Rescue（2026-08-20）

### 17.1 预注册协议

- 单一实验单元：从合法 θ0 开始，corners4 支持、layer4 可训练；seed=20260816；batch=64；损失 MSE+0.25×L1；weight decay=1e−4；fp32。
- Candidate A：SGD momentum=0；candidate B：SGD momentum=0.9。共同使用 peak lr=1e−3、500-step warmup、cosine 到 20k、eta_min=1e−5。
- 5k 仅按 anchor 选择一个 candidate。若 20k anchor 仍下降，协议允许延长到 40k。
- strict anchor threshold：AMD 2D 为 0.10 px；A10 6D 为 0.17 px。只有达到阈值后才生成 dense field。
- AMD 的预设场保持读数为 endpoint u/u0≤1.5。该阈值与 strict anchor threshold 是两个独立条件。

### 17.2 5k candidate 选择

| 平台 | candidate | momentum | anchor@1k | anchor@5k | 5k 后处理 |
|---|---|---:|---:|---:|---|
| AMD 2D | A | 0 | 4.6995 | 3.9456 | 继续 |
| AMD 2D | B | 0.9 | 7.4267 | 8.5925 | 不再延长 |
| A10 6D | A | 0 | 15.3231 | 7.5436 | 继续 |
| A10 6D | B | 0.9 | 7.1608 | 7.2801 | 不再延长 |

### 17.3 A10 6D

Candidate A 延长到 40k；endpoint anchor=5.5440 px，高于 strict threshold 0.17 px。瘦包没有 dense field、完整训练轨迹或 20k/40k 场，因此没有 A10 dense/u 读数。本机仅对输入、样本数和阶段 0 做 preflight：full-train anchor=0.7027 px，train/val=224/32 且无重合；该 GTX 1060 preflight 不替代 A10 训练。

### 17.4 AMD 2D 轨迹与 dense reveal

20k anchor=0.063885，但 19k/20k 为 0.084859/0.063885，未满足 tail-close；随后延长到 40k。39k/40k=0.051461/0.051458，均低于 0.10。

| checkpoint | step | anchor | raw box | u | u/u0 |
|---|---:|---:|---:|---:|---:|
| step0 | 0 | 0.239660 | 4.550722 | 4.302342 | 1.000000 |
| step1 | 1 | 0.471944 | 4.684143 | 4.411065 | 1.025271 |
| step10 | 10 | 3.702343 | 6.068537 | 5.268018 | 1.224453 |
| step100 | 100 | 3.965729 | 6.701604 | 5.882858 | 1.367362 |
| step500 | 500 | 4.834218 | 7.178350 | 6.522573 | 1.516052 |
| step1k | 1,000 | 4.699486 | 7.139832 | 6.527004 | 1.517082 |
| step5k | 5,000 | 3.945578 | 6.981064 | 6.499537 | 1.510697 |
| step10k | 10,000 | 2.273547 | 6.874175 | 6.420884 | 1.492416 |
| step15k | 15,000 | 0.663286 | 7.289127 | 6.307062 | 1.465960 |
| step20k | 20,000 | 0.063885 | 6.944679 | 6.274052 | 1.458288 |
| step39k | 39,000 | 0.051461 | 6.933378 | 6.273728 | 1.458212 |
| endpoint | 40,000 | 0.051458 | 6.933424 | 6.273713 | 1.458209 |
| first threshold match | 18,600 | 0.094573 | 6.954491 | 6.276513 | 1.458860 |
| minimum anchor | 21,400 | 0.047095 | 6.934838 | 6.273389 | 1.458133 |

20 个 field 与 20 个 metrics 一一对应；每个 field 含 pred/true/err/u，形状 [41,41,2]，均为有限值。first threshold match、minimum anchor 与 endpoint 是三个不同 checkpoint。endpoint u/u0=1.458209，预设阈值为 1.5；历史 AdamW layer4 的 u=32.00699，但 optimizer、训练时长和轨迹协议不同。

### 17.5 跨平台边界

A10 与 AMD 使用不同输出维度、不同内容/数据、不同 θ0、不同 strict threshold，且只有 AMD 产生 dense field。因此只能分别记录两台设备的阈值与读数，不能把它们合并为一个 SGD 比例或一个平台无关标签。

## 18. 协议异常、缺失项与不可用字段

| 实验 | 事实记录 | 对本文的处理 |
|---|---|---|
| Phase 1 split | held-out shape–translation 配对，不是 held-out identity | 使用“重组” |
| Phase 1.7 erasure | 报告值约 30.030；从当前 Z/B_t/labels 复算约 0.27；源码含常数 30.030 | 并列冲突，不用于比较 |
| Phase 1.7c / 1.8 OLS | val=512，含 bias 参数=513 | 标作高维跨 split probe |
| Phase 1.8 metadata | resume=true，但 resume_from=null，train_summary 无 resume 记录 | 不由单一字段推断实际恢复 |
| energy-null | seed11 matched_ok=false；seed12 无有效样本 | 不把缺失样本记为零 |
| Bézier OOB | 二次 667/1000 OOB；三次 0/1000 OOB | 不作纯阶数控制 |
| Phase 2 budget | equal 5600 steps，非 equal epochs/images；G9/C9 仅 504 unique | 只在协议内比较 |
| shape generalization | cache 只有 7/16 个预设 unseen shapes | 报告实际覆盖 |
| EfficientNet recovery | fp32+clip；Atlas 为 AMP | 分开列 |
| content init_probe | train t 未填导致 t/constant=0 | 不作为位置结果 |
| Track E/F seed | 模型/新 head 在 seed_everything 前创建 | seed 不证明初始化相同 |
| A10 all6 / AMD cross-res | 缩放网格与固定网格不同；AMD L160 部分 OOB | 分表 |
| ols_G4 | frozen G64+四角 OLS；不是 G4 CNN | 使用实际模型名 |
| Task 5 G4 | checkpoint 缺失 | 记录为跳过 |
| torus | safe box 内 wrap=identity；卷积 padding=zeros；val=1.7505、dense=51.2879 | val/dense 分列 |
| Spatial Field remote slim | 部分缺权重或 dense field | 只按现有证据层级记录 |
| error-extension impulse | 反对称前置条件未满足 | seed17/superposition 记为未执行 |
| functional drift | 新建 AdamW，无原 optimizer state | step1 标作 reset optimizer |
| mechanism keep-state | 30-step G64 热身代理；full 热身自身改变场 | 不视作原状态恢复 |
| constraint removal P0 | AMD/A10 endpoint anchor 均高于 step0 | 相图不作“已拟合 support”解释 |
| constraint P3 | 未启动 | 不写 planned 数值 |
| SGD rescue A10 | 40k anchor=5.544>0.17；无 dense/完整轨迹 | 不补写 dense |
| SGD rescue AMD | 40k anchor=0.05146；有 20 个 dense checkpoints | 与 A10 分表 |

## 19. 主要证据入口

| 内容 | 路径 |
|---|---|
| 平台与资产边界 | <code>MACHINE_ASSETS.md</code> |
| Spatial Field 协议 | <code>SPATIAL_FIELD_DYNAMICS_PROTOCOL.md</code> |
| shortcycle 协议 | <code>TODAY_SHORTCYCLE_PROTOCOL.md</code> |
| error extension 协议 | <code>ERROR_EXTENSION_PROTOCOL.md</code> |
| functional drift 协议 | <code>FUNCTIONAL_DRIFT_PROTOCOL.md</code> |
| constraint removal 协议 | <code>CONSTRAINT_REMOVAL_PROTOCOL.md</code> |
| SGD rescue 协议 | <code>SGD_ANCHOR_RESCUE_PROTOCOL.md</code> |
| stride | <code>results/main/tables</code>、<code>results/baseline_60ep/tables</code> |
| Bézier | <code>results/bezier_inverse</code>、<code>results/quadratic_bezier_minimal</code>、<code>results/bezier_oob</code> |
| Phase 1 | <code>results/phase1_gap_rep</code>（含 <code>phase1_8_unseen_t</code>） |
| Phase 2 | <code>results/phase2_overnight_discovery</code>（含 <code>recovery</code>） |
| Phase 3 / spectroscopy | <code>results/phase3_high_upside_discovery</code>（含 <code>spectroscopy</code>） |
| AMD Batch 1/2 | <code>results/amd_batch1_discovery</code>、<code>results/amd_batch2_discovery</code> |
| Spatial Field | <code>results/spatial_field_dynamics</code> |
| shortcycle | <code>results/today_shortcycle</code> |
| error extension | <code>results/error_extension</code> |
| functional drift | <code>results/functional_drift</code> |
| mechanism probe | <code>results/mechanism_probe</code> |
| constraint removal | <code>results/constraint_removal</code> |
| SGD rescue | <code>results/sgd_anchor_rescue</code> |

第 1–19 节保留截至 2026-08-20 的原档案。以下各节只追加该截止点之后的新训练、零训练科学分析与正式审计；不把设备性能测试、环境准备、上传下载、结果导出、存储清理或关机状态计为实验。

## 20. S0 clean-room 证据收尾（2026-08-20–21）

### 20.1 范围与数据审计

本阶段是对已有 S0 产物的独立物化、复算和门禁审计，不是新训练。S0 综合阶段没有启动训练、远程任务或 S1；导入的 A10 materialization 是冻结 checkpoint 的前向物化，记录为 <code>training_enabled=false</code>、<code>optimizer_constructed=false</code>、<code>backward_called=false</code>。

A10 数据审计覆盖 64 个 shape × 64 个 translation，共 4096 个组合；每个 translation 按 56/8 划分，train/held-out=3584/512。独立 CPU renderer 对 4096 张 <code>uint8</code> 图逐字节比较，<code>mismatch_count=0</code>、<code>max_abs_diff=0</code>；split 无重合，geometry boundary violation=0。物化的 OLS 输入为 train [3584,512]、eval [53792,512]；kernel gap grid 为 [1681,16384]。

### 20.2 已保存场 headline 与 partial-out

headline 只从保存的 <code>field.npz</code> 重新读取 pred/true 并复算，没有重训：

| seed | raw MAE | affine-removed MAE |
|---:|---:|---:|
| 20260816 | 43.116354 | 30.937463 |
| 20260817 | 43.754964 | 23.786073 |
| 20260818 | 61.861726 | 26.309609 |

Frozen G64 historical field 的 raw/affine-removed MAE 为 1.192122/1.180124。与同源历史 sidecar 的 4/4 对齐在 5% 门内，但这是保存场的独立复评，不是 clean-room retraining replication。

partial-out 固定 nuisance removal 后的正式聚合为 <code>mixed</code>：

| group | original corr | partial corr | retained fraction | 记录状态 |
|---|---:|---:|---:|---|
| neural_affine_2d | 0.671296 | 0.206079 | 0.306987 | inconclusive |
| amd_2d | 0.735285 | 0.322827 | 0.439049 | inconclusive |
| a10_6d | 0.651997 | 0.617881 | 0.947675 | survived |

<code>a10_6d=survived</code> 只表示该固定残差化口径下相关性仍保留；它不是 held-out predictor，也不构成因果测量。

### 20.3 BN、OLS 与 kernel 审计

| 项目 | 关键读数 | 审计状态与边界 |
|---|---|---|
| BN-only control | max/final u ratio=1.277251/1.277239；门槛=1.320746 | control valid；仅 BN buffers 改变；<code>small / passed=false</code> |
| formal OLS | float64/float32 eval box MAE=1.19219144/1.19219112；历史场=1.19212214 | rank deficient；perturbation 最大变化=8.463646；<code>unstable</code> |
| formal kernel | kernel/baseline log-MAE=1.153060/0.414855；improvement fraction=−1.779431，要求≥0.2 | ranking correct，但 held-out 改善门失败；<code>closed / passed=false</code> |

OLS 的 <code>status=complete</code> 表示审计已执行，不表示数值稳定性通过。S0 综合决策随后把 Neural Affine clean minimal reproduction 作为下一条 S1 实验；综合包生成时其执行状态仍为 <code>paused_by_user</code>，后续实际 S1 运行单列于第 21 节。

### 20.4 复核边界

S0 收尾具有与历史训练代码分离的独立 dataset audit、headline 重算、OLS/kernel/BN/partial evaluator 和 protocol/hash 锁。它提高了输入、指标与 provenance 的可追溯性，但产物中没有单独记录 reviewer-agent 身份或第二模型签名，因此这里称“独立审计与复算链”，不声称已由另一实验室重训复现。

## 21. Neural Affine S1 clean reproduction（2026-08-21）

### 21.1 固定协议

- 任务：224×224 blob，sigma=6，<code>coord_scale=223</code>，safe box [59,164]；四角 tids=[0,7,56,63]；dense=41×41。
- seeds=20260816/20260817/20260818；vanilla ResNet18；3000 steps；batch=64；AdamW lr=1e−3、weight decay=1e−4；MSE+0.25×L1；cosine 到 1e−5；fp32、amp=false。
- 正式门：anchor≤0.25 px、raw full-box≥20 px、affine-removed≥10 px、独立 prediction replay≤0.001 px、metric diff≤1e−6。
- A10 正式路径在运行前验证 package、cache、初始权重和协议；独立 evaluator 从协议重建 support/dense images，并从 checkpoint 重新前向。

### 21.2 corrected vanilla 正式结果

| seed | anchor MAE | raw full-box MAE | affine-removed MAE | 数值门 | formal replay 最大差 | formal audit |
|---:|---:|---:|---:|---|---:|---|
| 20260816 | 0.108438 | 53.403000 | 28.040853 | valid/positive | 0.014236 | failed |
| 20260817 | 0.090804 | 43.361880 | 19.890558 | valid/positive | 0.008361 | failed |
| 20260818 | 0.102926 | 45.794585 | 26.903877 | valid/positive | 0.006992 | failed |

三 seed 的数值门均命中，但三项 checkpoint 独立重前向与 saved predictions 的最大差均超过 0.001 px。正式聚合因此保留 <code>verdict=inconclusive</code>、<code>reason=independent_audit_failed</code>；不能因 headline 数值满足预设范围而写成 formal pass。

### 21.3 补充 parity 与 controls

补充审计定位到 float32 预处理顺序差异：保存预测路径先在 CPU 执行 <code>uint8→float32→/255</code> 再传 CUDA；formal evaluator 原路径先传 CUDA 再转换。仅加入 deterministic flags 不消除差异；改成 CPU 预处理顺序后，corrected 与 legacy vanilla 共 6 项的最大 prediction diff 均为 0。补充文件明确 <code>does_not_replace_formal_aggregate=true</code>，所以正式状态不变。

| predictor | seed/范围 | anchor MAE | raw full-box MAE | affine-removed/residual MAE | 审计边界 |
|---|---|---:|---:|---:|---|
| closed-form identity | analytical | — | 6.84e−14 | 1.13e−13 | 解析 control |
| explicit-xy MLP | 20260816 | 0.002670 | 1.377552 | 0.600517 | 单 seed |
| CoordConv | 20260816 | 0.106561 | 31.812789 | 22.616700 | independent audit passed |
| CoordConv | 20260817 | 0.129414 | 29.517438 | 18.141536 | independent audit passed |
| CoordConv | 20260818 | 0.077996 | 37.542947 | 22.938406 | independent audit passed |

legacy exact-init vanilla 的三 seed raw/residual 未进入相应历史值的 5% tolerance，只作初始化与代码路径诊断。clean package 的 26/26 文件存在性、字节数和 SHA 均通过；formal return 的数值与审计状态已在本机回传产物中落盘。

## 22. 低成本并行实验与 S1 snapshot（2026-08-21–22）

### 22.1 AA/P32 三 seed 新训练

本机 GTX 1060 上运行 seeds=20260810/11/12；baseline 与 antialias 各 60 epochs，batch=64，AdamW lr=1e−3、weight decay=1e−4，AMP=true；dense set 为 x=32…191 的 160 点。六个 run 均 <code>SEALED</code>，独立 aggregate verification 为 <code>VERIFIED</code>。

| seed | baseline dense MAE | antialias dense MAE | AA−baseline | baseline P32 | antialias P32 | P32 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260810 | 0.261211 | 0.673045 | +0.411833 | 0.040831 | 0.161927 | +0.121096 |
| 20260811 | 0.306219 | 0.278616 | −0.027603 | 0.064436 | 0.070642 | +0.006206 |
| 20260812 | 0.228421 | 0.550377 | +0.321956 | 0.042003 | 0.069033 | +0.027030 |
| 三 seed 均值 | 0.265284 | 0.500679 | +0.235395 | 0.049090 | 0.100534 | +0.051444 |

三 seed 的固定 epoch 曲线都存在非单调波动；上述 n=3 只作描述，不作显著性或单因素因果推断。

### 22.2 A：anchor-implied extension

这里的 affine、bilinear 与 RBF 是用网络在四个角点的 predictions 构造的新解析 predictor，不是 CNN 原始 dense output。历史三场的 raw CNN MAE=43.1164/43.7550/61.8617，anchor-affine MAE=0.0695/0.0956/0.0696。

后写入 clean S1 snapshot 的最终独立复算值为：

| seed | raw CNN | anchor affine | anchor bilinear | thin-plate RBF | anchor constant | anchor nearest |
|---:|---:|---:|---:|---:|---:|---:|
| 20260816 | 53.403000 | 0.101045 | 0.101679 | 0.101843 | 41.167471 | 39.506051 |
| 20260817 | 43.361880 | 0.082981 | 0.083052 | 0.083066 | 41.167429 | 39.453146 |
| 20260818 | 45.794585 | 0.102154 | 0.102177 | 0.102181 | 41.167475 | 39.498572 |

### 22.3 B：endpoint feature ridge

B 封存 360 个 ridge 结果行与 72 个 fold manifest 行；四个 interlaced parity folds 每次以一 fold calibration、其余三 fold test。下表是在已测试 layer/dimension 中按 input 取最低 mean test MAE 的描述性索引，不是新的预注册 headline：

| 输入 | seed/run | 最低 mean test MAE | layer / dimension |
|---|---|---:|---|
| historical Neural Affine 2D | 20260816 | 8.59 | layer2 / raw |
| historical Neural Affine 2D | 20260817 | 3.17 | layer4 / raw |
| historical Neural Affine 2D | 20260818 | 6.09 | layer4 / raw |
| cached 6D | phase2_G64 | 0.65 | GAP / raw |
| cached 6D | unfreeze_g64_corners4_full | 1.44 | GAP / raw |
| cached 6D | random_init_s20260816 | 5.68 | GAP / raw |

cached 6D features 为 13448×512，且每个 translation 重复 8 次；2D endpoint 与 6D translation-only readout 不作为同一总体比较。

### 22.4 C、D 与最终 audit ledger

| seed | 历史 C：低频占比 / degree-2 residual MAE | clean snapshot C：低频占比 / degree-2 residual MAE |
|---:|---:|---:|
| 20260816 | 0.158320 / 23.487612 | 0.203772 / 23.550801 |
| 20260817 | 0.436182 / 14.510293 | 0.637059 / 9.919246 |
| 20260818 | 0.234460 / 26.077975 | 0.241341 / 19.048704 |

clean snapshot 的 <code>one_pixel_slices</code> 为 <code>UNAVAILABLE_FROM_FROZEN_FIELD_ONLY</code>，没有数值。D 是历史单 seed AA/P32 法医审计，状态 <code>AUDIT_COMPLETE</code>，但 ledger 明确标为 <code>SELF_AUDIT_NO_SECOND_REVIEW</code>，不能写成独立复核通过。

S1 snapshot 审计先后保留两次失败：第一次为三项 <code>snapshot_field_shape</code>，第二次为 36 项 A mismatch 与 3 项 C degree-0 missing。analyzer/auditor 更新并重新冻结后，最终 <code>analysis_verification=VERIFIED</code>、<code>aggregate_verification=VERIFIED</code>（6/6 runs）、<code>evidence_ledger=VERIFIED</code>、<code>failures=[]</code>、A/C <code>differences=[]</code>、protected comparison=<code>UNCHANGED</code>。最终成功只适用于更新后的冻结审计代码；两份 <code>AUDIT_HALT</code> 仍是有效历史。

## 23. Neural Affine NTK、frozen feature 与场差分（2026-08-22–23）

本节不含同批设备 benchmark 或 GPU endurance，只记录直接回答 Neural Affine 科学问题的结果。输入沿用三份 corrected exact-init 与同一 41×41 blob 场；比较中的 <code>trained</code> 是第 21 节已经保存的 S1 场，不在本节重新训练。

### 23.1 零训练轴向分析

三个 corrected vanilla 保存场的聚合分类为 <code>genuine2d</code>；最低 correct-axis R²=0.751117，低于 0.90 的 seeds 为 20260816、20260818；最大 interaction fraction=0.137570。该项只分析保存场，没有重新训练。

### 23.2 exact empirical NTK v2

NTK 使用完整 output-coupled 8×8 <code>K_SS</code>；BN 为 eval；只对 model parameters 求导；拟合时只可读取四个 support labels，dense labels 在预测冻结后才用于事后指标。三 seed 的 rank 均为 8，<code>K_SS</code> 最小特征值为 0.182606/0.199281/0.233128，最大 support anchor error 均小于 1e−12 px；有限性、对称性、PSD、support/query consistency 门均通过。

| seed | NTK raw | trained raw | NTK affine-removed | trained affine-removed | error-field Pearson（flat / norm） |
|---:|---:|---:|---:|---:|---:|
| 20260816 | 28.88 | 53.40 | 28.06 | 28.04 | 0.514 / 0.060 |
| 20260817 | 26.91 | 43.36 | 23.95 | 19.89 | 0.532 / −0.382 |
| 20260818 | 29.35 | 45.79 | 28.15 | 26.90 | 0.662 / −0.087 |

raw MAE 上 NTK 三 seed 均低于 saved trained 场；affine-removed 后三 seed 并不一致更低。相关系数是场结构的描述量，不作为因果验证。

### 23.3 frozen-feature 与 head-only

frozen-feature 使用 exact-init backbone 的 support features 做闭式插值，不训练；head-only 从同一 exact init 训练线性 head 3000 steps、batch=64，并保持 BN eval。三项 head-only 的 support/query backbone features 在训练前后 bitwise equal，backbone 可训练参数数为 0。下表的 aggregate 先对三个 seed 的 candidate/trained 场逐点取均值，再在均值场上计算指标；它不是三个 per-seed MAE 的算术平均。

| 三 seed mean-field aggregate | frozen-feature | head-only | saved trained |
|---|---:|---:|---:|
| raw MAE | 41.6406 | 15.8813 | 45.1354 |
| affine-removed MAE | 16.4341 | 14.8659 | 22.8976 |
| 与 trained error field 的 flattened Pearson | 0.7773 | 0.6888 | — |
| 与 trained error norm 的 Pearson | 0.7571 | −0.0585 | — |
| degree-2 explained fraction | 0.9540 | 0.4893 | 0.8839 |

degree-2 explained fraction 定义为未中心化的 <code>1−SSE_residual/SSE_raw</code>，不是经典 centered R²。预注册 renderer 触发门随后给出 <code>NO_GO</code>，第二 renderer 未上传、未启动，因此没有 renderer 科学结果。

### 23.4 保存场 delta 分解

<code>field_delta_20260823</code> 对历史 NTK/head/full 场做零训练差分分解。full−head 的 degree-1/degree-2 系数跨 seed Pearson 范围为 0.7484–0.9173 / 0.8254–0.9595，主非零频率三对中精确一致 1 对；head−NTK 对应范围为 0.9198–0.9872 / 0.8336–0.9744，主频精确一致 0/3。该产物明确 <code>classification=supplementary</code>、<code>mixed_backend_allowed=true</code>、<code>training_reexecuted=false</code>、<code>formal_s1_verdict_unchanged=true</code>，只作描述性函数场证据。

### 23.5 独立复核轨

独立复核智能体在代码、标签边界、pilot、三 seed probe、full-grid、post-hoc 和 renderer 门逐级给出只读审计。NTK v1 因预测冻结前可访问 dense points 而被中止，日志未产生 receipt 或科学结果；修订的 v2 只有在每级 <code>next_stage_allowed=true</code> 后继续。NTK v2、NTK-vs-trained、frozen/head-only 和 frozen-vs-trained 四项最终均记录复核 <code>GO</code>。

## 24. support-density × plasticity 正式矩阵（2026-08-23）

### 24.1 预注册协议

- 同一 224 px、sigma=6 blob 任务；safe box [59,164]；41×41 dense；seeds=20260816/17/18。
- supports=<code>corners4/G9/G16/G64</code>；regimes=<code>head_only/full/frozen_feature</code>；共 24 个 trained arms 与 12 个 frozen-feature evaluations，36 conditions、12 paired triplets。
- trained arms：3000 steps、batch=64、AdamW lr=1e−3、weight decay=1e−4、MSE+0.25×L1、cosine 到 1e−5、fp32、amp=false。
- 每个 support/seed 的 head/full 使用相同 exact init 与相同 deterministic batch stream；训练仅可读取 support labels；dense labels 在 prediction archive 原子冻结后才读取。
- primary statistic 为 <code>raw MAE(full)−raw MAE(head)</code>。有效 pair 要求 head/full support MAE≤0.25 px；稳定 crossover 还要求 corners4 median≥+2、G64 median≤−2、各端至少 2/3 seeds 符号满足且 Spearman≤−0.8。

### 24.2 support gate 与正式结果

矩阵 36/36 conditions、12/12 triplets 均完成；没有缺项、重复项、OOM 或非有限值。support MAE 按 seeds 20260816/17/18 排列：

| support | head-only support MAE | full support MAE | 有效 pair | 门禁事实 |
|---|---|---|---:|---|
| corners4 | 0.00417 / 0.01046 / 0.01012 | 0.13539 / 0.06084 / 0.06774 | 3/3 | head/full 全过门 |
| G9 | 2.38652 / 2.00742 / 1.65756 | 0.80831 / 0.93001 / 0.63905 | 0/3 | head/full 均 3/3 underfit |
| G16 | 3.76249 / 3.69214 / 2.41513 | 0.05800 / 0.14981 / 0.17230 | 0/3 | head 3/3 underfit；full 3/3 过门 |
| G64 | 4.40234 / 3.61809 / 3.34429 | 0.19123 / 0.08387 / 0.17185 | 0/3 | head 3/3 underfit；full 3/3 过门 |

全部 12 个 frozen-feature conditions 均通过其 0.001 px support gate。只有 corners4 的三对具有 primary 统计资格：

| 读数 | 20260816 | 20260817 | 20260818 | 中位数 / 符号 |
|---|---:|---:|---:|---:|
| raw full−head | +33.1942 | +21.9822 | +25.0512 | +25.0512 / +++ |
| affine-removed full−head | +8.8260 | +10.2273 | +5.7792 | +8.8260 / +++ |

G9/G16/G64 没有有效的 head/full paired comparison，故其 dense gap、median 与 Spearman 只属于 aggregate 中的诊断字段，不具 primary 统计资格。正式聚合为 <code>decision=incomplete</code>；该状态来自预注册 support-fit gate，而不是缺文件、身份漂移或 evaluator 错误。第二 renderer 未启动。

### 24.3 零训练 post-hoc

post-hoc 不改写 formal <code>aggregate.json</code>，也未启动新训练、ridge、LBFGS、延步或 rescue。探索性 full−frozen-feature 只纳入两格都 valid、evaluator passed 且各自 support gate 通过的 pair：

| support | 有效 pair | raw 中位数 / 符号 | affine-removed 中位数 / 符号 |
|---|---:|---:|---:|
| corners4 | 3/3 | −1.509409 / +-- | +7.598977 / +++ |
| G9 | 0/3 | 不可用 | 不可用 |
| G16 | 3/3 | +12.839164 / +++ | +13.534749 / +++ |
| G64 | 3/3 | +21.642027 / +++ | +19.878560 / +++ |

post-hoc 状态为 <code>no_observed_flip_stop</code>，formal 状态仍为 <code>incomplete</code>。G9 的 dense 值没有进入趋势、median 或机制判断。所有纳入格的 raw/affine-removed MAE 均从原 prediction archives 独立重算，并在 1e−8 px 内匹配 evaluator 与 aggregate。

### 24.4 独立 evaluator 与复核智能体

36/36 independent evaluators 为 <code>passed</code>；12/12 head-only backbone bitwise unchanged；12/12 full backbone parameter-only hash 发生变化；36/36 training summary 记录 <code>dense_labels_read=false</code>；12/12 triplets 的 init、source-init、batch stream、support、asset、package 和 protocol identity 一致，pair consistency error=0。独立复核智能体覆盖 package、preflight、smoke、首格、各 triplet、聚合与回传，最终意见为 <code>GO，无 RE-RUN</code>。这里的 GO 表示产物与预注册执行链通过复核，不把科学聚合的 <code>incomplete</code> 改写成 pass。

## 25. 2026-08-20 后的审计失败、缺失项与解释边界

| 项目 | 事实记录 | 本文处理 |
|---|---|---|
| S0 headline | 保存场独立复评，无重训 | 不称 clean-room replication |
| S0 manifest self-check | <code>verify_hashes=false</code> | 只作存在性检查；完整性以其他 asset/audit 链为准 |
| S1 corrected vanilla | 数值门 3/3；formal replay 0/3 | 保留 <code>inconclusive / independent_audit_failed</code> |
| S1 preprocess parity | CPU 预处理路径 6/6 exact | 只解释 replay 差异，不覆盖 formal verdict |
| S1 snapshot | 两次 AUDIT_HALT 后更新并重冻审计代码 | 失败保留为历史；最终 VERIFIED 只适用于新冻结版本 |
| 低成本 D | <code>SELF_AUDIT_NO_SECOND_REVIEW</code> | 不列为独立复核结果 |
| clean snapshot one-pixel | frozen fields 不足以恢复 | 记 <code>UNAVAILABLE_FROM_FROZEN_FIELD_ONLY</code> |
| NTK v1 | dense points 在预测冻结前可达；进程中止 | 无科学结果；只记录审计中止 |
| NTK/head/full 场差分 | 历史保存场、混合 backend | 仅 supplementary description |
| crossover G9/G16/G64 | head 或 full support gate 未满足 | dense gap 不进入 primary 判据 |
| crossover post-hoc | 预注册之后的 full−frozen 分析 | 不替代 full−head primary 或 formal aggregate |
| 独立复核 | 执行/审计轨分离，但均为 AI 生成 | 可信度相对提高；不等于人工或跨实验室复现 |

## 26. 2026-08-20 后的主要证据入口

| 内容 | 路径 |
|---|---|
| S0 clean-room protocol 与结果 | <code>closeout_sprint_clean/protocols</code>、<code>closeout_sprint_clean/results</code> |
| S0 综合事实档案 | <code>closeout_sprint_clean/results/s0_integrated_analysis_v1/S0_EXISTING_EVIDENCE_INTEGRATED_ANALYSIS_20260821.md</code> |
| S1 clean 协议与代码 | <code>neural_affine_s1_clean/protocol.json</code>、<code>neural_affine_s1_clean/s1clean</code> |
| S1 formal 回传与 snapshot | <code>s1clean_returns/a10_20260821</code>、<code>s1_parallel_lowcost_20260821/snapshots/s1</code> |
| S1 总结与审计账本 | <code>S1实验结果阶段性汇总.md</code>、<code>s1_parallel_lowcost_20260821/auditor</code> |
| AA/P32 与 A/B/C | <code>s1_parallel_lowcost_20260821/sealed</code> |
| axis、NTK、frozen/head-only | <code>remote_results/20260822/axis_analysis.json</code>、<code>remote_results/20260822/ntk_v2_grid_all3_01</code>、<code>remote_results/20260822/frozen_feature_head_only_all3_01</code> |
| NTK/frozen 与 trained post-hoc | <code>remote_results/20260822/ntk_vs_trained_compare_01</code>、<code>remote_results/20260822/frozen_vs_trained_compare_01</code> |
| 保存场 delta 分解 | <code>results/neural_affine_crossover_clean/field_delta_20260823</code> |
| crossover formal | <code>handoff/20260823_crossover_results/RESULTS_SUMMARY.md</code> 及其 <code>extracted</code> 结果树 |
| crossover post-hoc | <code>handoff/20260823_crossover_posthoc_v1</code> |

## 27. Track A MiniCNN 已知机制验证（2026-08-25）

### 27.1 实验角色与协议

按用户给定的研究定位，本实验用于验证个人已知的边界机制，不作为探索性发现、正式 Track B promotion 或普遍 CNN 规律。前序 <code>results/track_a_toy_v1</code> 保留为教学型验证记录；这里列最终的 valid-boundary MiniCNN 结果。

任务使用 64×64 固定三角形与 256 个位置；<code>visible=182</code>（train/test=133/49）、<code>safe=90</code>（65/25），另有 74 个 clipped 位置只作边界说明、不训练。模型为四层 3×3、stride=1、bias=false 的 MiniCNN，无 BN、残差、池化或 Dropout；三条件共享初始化与 batch 序列：Z=finite+zero，V=finite+true-valid padding0，C=torus+circular。六个 run 均固定 500 steps，只使用最终 checkpoint。

### 27.2 结果

主指标为 held-out circular64 MAE，单位 px：

| scope | 未训练 GAP probe：Z / V / C | probe 常数 baseline | step500：Z / V / C |
|---|---:|---:|---:|
| visible | 8.654 / 7.511 / 13.858 | 13.858 | 13.207 / 11.486 / 14.336 |
| safe | 9.803 / 9.803 / 9.803 | 9.803 | 9.808 / 9.808 / 9.808 |

在这一单 seed、固定形状与固定切分协议内，visible 的 Z/V 未训练特征可读出位置而 C 回到常数 baseline；safe 中三条件都与 baseline 数值上不可区分。该记录验证“有限边界及 valid-domain survival/cropping 可提供位置参照，严格周期边界不提供同类绝对参照”在此协议中的表现；它不证明 zero padding 是唯一来源，也不外推到其他架构或数据。

执行任务中的独立复核智能体检查了 checkpoint/history、mask、共享初始化与 schedule、pooling、周期不变性、报告和图，给出 <code>FINAL PASS</code>。但 <code>results/track_a_minicnn_v1</code> 内没有单独 reviewer receipt，因此这里记为“经过独立智能体复核的验证性实验”，不写成带正式回执链的 promotion。

## 28. Track B B1–B8 function-selection 正式矩阵（2026-08-25）

### 28.1 问题、指标与闭合状态

Track B 在稀疏监督下分别改变 normalization（B1）、appearance 数量（B2）、support/appearance 布局（B3）、optimizer（B4）、renderer（B5）、upstream/readout 与解冻层级（B6）、absolute/relative dense objective（B7）及解析/MLP/CNN function class（B8）。固定 seeds=20260816/17/18；主指标是 41×41、1681 个 query 上的 <code>full_box_raw_mae_px</code>，越低越好。只有 reviewer 重算 support MAE≤0.25 px 的 cell 才能进入对应 off-support contrast；差值统一为 condition−reference。

共同训练路径以 ResNet18、GAP、Linear(512→2) 为主体，使用 fp32；普通路径为 3000 steps、batch=64、AdamW，B1 按预注册条件替换 normalization，B4 的 causal 路径另有 20000-step 端点，B8 degree2 为解析 reference 而非随机训练。

正式 aggregate 为 <code>PASS_COMPLETE_84_CELL_CLOSURE</code>：84/84 logical cells 完整、84/84 review PASS，无 missing、extra、duplicate 或 non-pass review；68 个 cell 通过 support gate，16 个未通过。这里的 closure/PASS 表示执行和证据链闭合，不表示全部科学对比成立。

### 28.2 预注册 contrast

| 家族 | 预注册对比 | 可纳入 seed | 均值差 px | 正式状态 |
|---|---|---:|---:|---|
| B1 | bn−gn；frozen_bn−gn | 3；3 | −2.524；−6.937 | <code>INCONCLUSIVE</code>；<code>CONSISTENT_CONDITION_BETTER</code> |
| B2 | A8−A1；A64−A1 | 2；3 | −1.481；+0.240 | missing/excluded seed；<code>INCONCLUSIVE</code> |
| B3 | 16×4−4×16；64×1−4×16 | 3；0 | +6.250；— | <code>INCONCLUSIVE</code>；missing/excluded seed |
| B4 | SGD−AdamW，matched gate；step 20000 | 3；3 | +1.464；−4.144 | <code>INCONCLUSIVE</code>；<code>CONSISTENT_CONDITION_BETTER</code> |
| B5 | line−blob；triangle−blob | 3；3 | −1.579；+0.397 | 均 <code>INCONCLUSIVE</code> |
| B6 | head/lp_ft/layer4/full−ridge | 0 | — | reference support gate 失败；均不可纳入 |
| B7 | 两个 relative−dense_absolute | 0 | — | reference support gate 失败；均不可纳入 |
| B8 | explicit-XY MLP−degree2；CNN−degree2 | 0；3 | —；+14.276 | missing/excluded seed；<code>CONSISTENT_CONDITION_WORSE</code> |

三个满足 promotion 规则的逐 seed 差值为：

| 对比（condition−reference） | 20260816 | 20260817 | 20260818 | 协议内记录 |
|---|---:|---:|---:|---|
| B1 frozen_bn−gn | −7.645 | −5.304 | −7.862 | condition better |
| B4 SGD−AdamW，step 20000 | −6.038 | −2.520 | −3.874 | condition better；仅限该端点 |
| B8 CNN−degree2 | +14.907 | +10.694 | +17.227 | condition worse |

B4 的 matched-gate 端点三 seed 均为正（+1.514/+1.850/+1.029），与 step 20000 的方向不同，因此不能写成无端点限定的“SGD 更好”。B3 64×1、B6/B7 reference 与 B8 explicit-XY MLP 的低原始误差不能越过 support gate 后重新纳入。

B8 degree2 只有 seed 20260816 对应一次实际解析计算，seed 17/18 是同一确定性结果的逻辑引用；表中的三项 B8 差值是三个 CNN seed 对同一解析 reference 的比较，不是三次独立 degree2 训练。

### 28.3 独立复核

逐 cell reviewer 的 84 个 <code>review.json</code> 均为 PASS，且主指标与各自 <code>recomputed_metrics.json</code> 一致。本机只读脚本重新计算全部 18 个 contrast，保存值与复算值最大绝对差为 0；完整数值、过程边界和 B0 专项三份分离 reviewer receipt 均为 PASS。该复核提高了数字、纳入规则和证据路径的可信度，但不是独立人类或跨实现复现。

## 29. Track B B6 后验机制诊断（2026-08-25–26）

原正式 B6 以 ridge 为 reference；ridge 三 seed 均未通过 support gate，因此四项预注册 contrast 继续保持 <code>INCONCLUSIVE_MISSING_OR_EXCLUDED_SEED</code>。以下三项使用原 B6 的 <code>head_only/lp_ft/layer4/full</code> × 3 seeds 保存产物，属于 formal 之后的机制诊断，不改写 aggregate。

除 dense re-probe 的五折 held-out 值外，下表在去掉四角后的 1677 个非角位置上计分；它们不能与正式 1681 点 full-box 指标直接混为同一口径：

| 条件 | 当前输出 | dense re-probe | 二维 affine 后 | zero-centered minimum-norm | prior-centered minimum-change |
|---|---:|---:|---:|---:|---:|
| head-only | 1.089 | 0.175 | 1.097 | 25.324 | 0.966 |
| LP-FT | 10.013 | 0.302 | 10.017 | 22.258 | 10.439 |
| layer4 | 13.507 | 0.354 | 13.497 | 26.380 | 13.740 |
| full | 12.903 | 0.421 | 12.900 | 23.665 | 13.351 |

1. <code>dense re-probe</code>：冻结终点 backbone 后，位置仍能由新的 dense linear head 以 0.175–0.421 px held-out MAE 读出；feature adaptation 伴随小幅读出变差，但没有把线性位置信息消除。
2. <code>sparse readout decomposition</code>：二维 affine 校准几乎不改变当前 1–14 px 坏场；只用四角拟合的 minimum-norm/ridge 虽可拟合 support，却在非角达到约 20–28 px。四角中心化 feature 在 12/12 endpoint 均 rank=3；固定 dense-reference head 的权重平方范数至少 99.9195% 落在四角无法约束的 null 部分。该比例只属于当前 feature 参数化。
3. <code>prior/path M0</code>：head-only 在同一 upstream feature 上，zero-centered minimum-norm 为 25.324 px，而保留 upstream head 为中心的 exact minimum-change 为 0.966 px，接近实际 1.089 px。LP-FT/layer4/full 的 endpoint head×upstream feature 为 31.376/24.670/23.924 px，upstream head×endpoint feature 为 27.736/25.314/25.343 px，均高于实际 endpoint 的 10.013/13.507/12.903 px。

机器报告据此把最窄描述限定为：四角监督对自由 512 维读出严重欠定；upstream function prior 在 head-only 固定 feature 条件下足以产生与观察相符的延拓；feature adaptation 后 head 与 feature 的坐标兼容关系同时改变，联合终点只部分补偿。该结果不证明 AdamW 沿解析 minimum-change 路径运动，也不证明 optimizer 是唯一因果。M0 达到预先停止条件，因此没有启动 M1。

三项诊断没有新增 CNN 训练、renderer、appearance 或 seed。各报告记录独立复核智能体从原 feature/checkpoint 重建计算；本次汇总的独立复核再次核对 condition mean，并抽查一份保存场，未发现数值差异。但三个实验根没有单列 machine-readable reviewer receipt，故复核固化程度与正式 84-cell review 链不同。

## 30. 2026-08-24–26 证据入口与边界

| 内容 | 证据路径 |
|---|---|
| Track A MiniCNN 验证 | <code>results/track_a_minicnn_v1/report.md</code>、<code>probe.json</code>、<code>training/summary.json</code> |
| Track B 总报告与数值复算 | <code>track_b/docs/results/TRACK_B_EXPERIMENT_RESULTS_REPORT_20260825.md</code>、<code>REPORT_NUMERICAL_REVIEW_20260825.json</code> |
| Track B 正式 aggregate | <code>track_b/formal_archive/20260825_complete/extracted/function_selection_v1_trackb_unified_r1_20260824_deploy_20260824T2014/formal_execution_20260824/FORMAL_AGGREGATE.json</code> |
| Track B 报告层复核 | <code>track_b/docs/results/reviews</code> |
| B6 dense re-probe | <code>track_b/experiments/b6_dense_reprobe_r1/results/results.json</code> |
| B6 sparse decomposition | <code>track_b/experiments/b6_sparse_readout_decomposition_r1/results/results.json</code> |
| B6 prior/path M0 | <code>track_b/experiments/b6_prior_path_decomposition_r1/results/results.json</code> |

本文截至 2026-08-26；未落盘的 planned 项、旧 Track A 的准备材料和编排器修复记录不计为已完成实验。Track A 按验证性实验记录；Track B 的 formal closure、scientific promotion 与 post-hoc diagnosis 分层保存。设备性能、endurance、存储清理、上传下载与结果导出不属于本文实验内容。
