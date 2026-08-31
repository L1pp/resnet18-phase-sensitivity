# Track A-Toy v1

这是一个与旧 Track A 运行时完全独立的本机教学实验，用来观察绝对位置信息在一个固定三角形上的何处消失。它不复现旧的 1024 画布、长训练或云端工程。

四个主条件共用同一个精确复制的随机初始化和同形状的 `fc(512 -> 2)` head：

| 条件 | 输入边界 | 网络边界 | 总 stride | 用途 |
|---|---|---|---:|---|
| A0 | finite | zero padding | 32 | 主 finite 对照 |
| A1 | finite | zero padding | 1 | 保留空间分辨率 |
| A2 | torus | explicit circular conv/maxpool | 32 | 周期输入对照 |
| A3 | torus | explicit circular conv/maxpool | 1 | 周期输入、无下采样 |

另外保留 A0-R（reflect/stride32）和 A0-C（finite input + circular network/stride32）作为随机特征 probe 条件，不启动短训练。

## 默认无训练流程

在工作区根目录执行：

```powershell
python -m track_a_toy.cli check --device cuda
python -m track_a_toy.cli prepare
python -m track_a_toy.cli random-features --device cuda
python -m track_a_toy.cli probe
python -m track_a_toy.cli report
```

本轮已经完成上述无训练流程，并额外生成了中文主报告和图。若只想查看已有结果，直接打开
`results/track_a_toy_v1/report.md`；若要在本机重新记录全过程墙钟（包含非持久速度 profile，但不启动正式训练），执行：

```powershell
python -m track_a_toy.cli wallclock --device cuda
```

墙钟明细在 `results/track_a_toy_v1/墙钟估算.md`，机器可读收据在
`results/track_a_toy_v1/wallclock_receipt.json`。报告主图是
`figures/实验概念图.png`、`figures/有限画布_固定输入_10张.png`、
`figures/环面画布_固定输入_10张.png` 和 `figures/结果总览.png`；每个输入图严格为同一组 10 个固定中心位置。

默认检测到 CUDA 时会使用 CUDA；`--device cpu` 可用于单测或没有显卡的复核。本轮已经按授权执行 A0–A3 正式短训练；若要重复该训练，执行：

```powershell
python -m track_a_toy.cli train --device cuda
```

配置固定为 64x64、scalene triangle、位置网格 `0,4,...,60`。finite 和 torus 数据集都保留完整 256 个位置；finite 另存 `fully_visible`/`interior` mask（本三角形为 182 个 true、74 个 clipped），不会因裁切而丢掉图像。主随机特征/probe 口径是 all-256，finite fully-visible/interior 只作为补充口径；A0-C 也保留 finite renderer 的 crop 边界。torus 按 64 周期渲染。

随机特征是 `eval()` 下的 fc 前 GAP，随后使用固定 `(ix + 2*iy) mod 4 == 0` 测试划分做 ridge probe。每个 raw-xy、phase-sincos、coarse-cell quotient 任务都只用 train split 拟合 StandardScaler，并独立 CV 选择 alpha；同一 scaler 变换 train/test/all。coarse-cell quotient 的 target 明确定义为 `floor(x/32), floor(y/32)`，本数据标签为 0/1，采用回归输出阈值 0.5 的每轴及平均 accuracy，同时可报告 direct cell MAE/R²；它不是 phase，也不使用 circular MAE。`random-features` 还会对每个 `p` 和即时渲染的 `p+Δ` 计算 Δ=`1,2,4,8,16,32` 的分层差异；这些诊断不从 4 像素稀疏网格查找目标图像。

## 输出

所有新输出仅写入 `results/track_a_toy_v1/`：

- `check.json`：运行时、CUDA、模型 stage/head shape、网格和可见性计数。
- `data/finite.npz`、`data/torus.npz`、`data/manifest.json`：物化输入、finite 全可见/裁切 mask 及哈希。
- `random_features/*.npz`、`random_features/manifest.json`、`random_features/shift_metrics.json`：共享初始化下的冻结特征和即时 shift 诊断。
- `probe/*.json`、`probe/summary.json`：固定划分 ridge 结果。
- `figures/inputs_finite.png`、`figures/inputs_torus.png`：每个数据集固定十张输入图。
- `figures/predictions_*.png`：A0/A1 明确使用 finite，A2/A3 明确使用 torus。
- `training/*_best.pt`、`training/*_history.json`、`training/summary.json`：A0–A3 的最佳 checkpoint、全域评估历史和实际训练墙钟。
- `figures/training_curve_*.png`、`figures/trained_predictions_*.png`：中文学习曲线和同一组十个固定位置的训练后预测图。
- `report.md`、`report.json`：带边界解释和限制的汇总。

torus 的 raw `x,y` 回归只作诊断，因为周期 seam 使全局坐标解释不稳；phase 的 modulo-32 circular MAE 仍只表示 phase，coarse-cell quotient 则使用 direct cell accuracy，不是 circular 指标。StandardScaler 在 train split 内按 `rtol=1e-5, atol=1e-6` 识别 near-constant 列，并在所有 split 中精确置零；若 active feature dim 为 0，三个任务分别使用 train target 常数 baseline，不调用 ridge。这个 toy 也不是 publication-grade 证据，旧 Track A 资产未被导入、移动或删除。
