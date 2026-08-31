# 神经几何定位与空间表征实验公开归档

[English](README.md)

本仓库是一个已经结束的几何定位、空间表征与支持域外行为研究项目的公开快照。

它主要用于备份、展示研究过程并保留证据线索，**不是**原始本地实验目录的完整镜像，也不保证开箱即用。

## 研究范围

项目研究卷积网络如何从渲染图像中推断位置、平移和几何参数，重点包括：

- stride、抗混叠、全局平均池化与空间表征；
- 图像到坐标、图像到 Bézier 曲线参数的反问题；
- 稀疏支持点和支持域外泛化；
- 冻结特征、线性探针、Neural Affine 变体与 NTK 分析；
- 有限边界、边界条件和函数选择机制；
- 优化动力学、恢复训练与结构控制实验。

## 研究时间线

| 阶段 | 主要内容 |
|---|---|
| 早期实验 | 位置回归、stride/downsampling 相位敏感性、抗混叠、GAP 与 Bézier 反演 |
| Phase 1–1.8 | 冻结 GAP 表征、空间因子、子空间探针和未见平移 |
| Phase 2 | 位置覆盖、结构、padding、内容、对称性和恢复训练 |
| Phase 3 | 算子、群作用/等变性、内容、分辨率、初始化和表征诊断 |
| S0 / S1 | clean-room 物化、Neural Affine 控制实验、crossover 分析与独立审计 |
| Track A | 对指定有限边界机制的小型控制实验 |
| Track B | function-selection 实验矩阵及后验机制诊断 |

更详细的阶段关系见[研究时间线](docs/RESEARCH_TIMELINE.md)，实验事实与结论边界见[实验汇总](docs/EXPERIMENT_SUMMARY_CN.md)。

## 仓库结构

```text
experiments/
  legacy/                 早期 phase-sensitivity 与 Bézier 入口
  phase_and_mechanism/    精选 Phase 1/2/3 和机制分析代码
  neural_affine/          精选 S0、S1 与 crossover 代码
  tracks/                 精选 Track A/B 代码与测试
docs/                     实验总结、时间线和归档说明
showcase/                 精选图表和小型结果表
```

## 公开内容

- 经过筛选的实验源码与测试；
- 研究协议和小型配置；
- 事实型实验总结与证据说明；
- 精选图表和汇总表；
- 原仓库中已公开轻量结果的精选子集。

## 有意省略的内容

- 权重、checkpoint、dense field 和其他大文件；
- 生成数据集和完整预测明细；
- 云端、设备、部署、传输和存储操作材料；
- 设备清单、环境快照、网络记录和凭据；
- 大型压缩包、日志、缓存、临时文件和完整正式运行回传。

这些省略是公开归档设计的一部分。仓库用于展示研究问题、实验路线和证据边界，而不是复制完整本地运行环境。

## 可运行性与历史路径

历史实验代码来自多个执行环境，部分保留脚本仍含原始绝对路径或环境相关默认值。归档时没有对这些路径进行全局改写。

如需使用代码：

1. 请在独立工作副本中操作；
2. 自行检查并适配路径、依赖和目标环境；
3. 不要假设脚本能直接从仓库根目录运行；
4. 将旧 manifest、报告和结果摘要视为历史记录。

详见[归档与路径说明](docs/ARCHIVE_AND_PATH_NOTE.md)。

## 精选结果预览

[展示目录](showcase/README.md)收录了少量关键图和汇总表。下面列出三个示例。

![ResNet18 等变性分析摘要](showcase/legacy_phase_sensitivity/main/figures/equivariance_summary.png)

![二次 Bézier 预测叠加图](showcase/figures/quadratic_overlay.png)

![Track A 结果总览](showcase/figures/track_a_toy_overview.png)

## 证据边界

本项目区分 formal result、supplementary analysis、diagnostic 和 incomplete run，也区分对已有产物的独立审计与完整实验复现。

部分实验只使用固定合成协议或少量随机种子。某一协议内的数值一致，不代表该现象适用于所有网络结构、数据集或训练方式。

## 许可证

见 [LICENSE](LICENSE)。
