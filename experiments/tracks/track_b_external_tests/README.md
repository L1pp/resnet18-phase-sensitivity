# Track B 外部科学 oracle（2026-08-24）

这是一个独立的、只关注实验逻辑的 oracle 测试目录。它不修改
`function_selection_v1_trackb_unified_r1_20260824`，也不测试 SHA、打包、门禁、
收据、代码风格或重复逐文件校验。

测试内容仅限于可能直接改变结论的纯函数约定：

- B1--B8 的 84 logical / 82 physical / 78 optimizer 矩阵与条件名；
- 41×41 坐标顺序（x 外层、y 内层）和 `xy_px / 223`；
- B8 固定二次目标公式；
- B2 train/held-out appearance identity 不交；
- 所有固定 sampler 的 batch=64（有放回）；
- B7 112 条 canonical 轴向边、每步先抽 64 images 再抽 64 pairs、128 个有序端点、三 dense 条件共享 exposure、ordered displacement 与 equal pair/anchor loss；
- B6 centered float64 SVD ridge，包括 alpha、截断和 rank-zero；
- common cosine 与 B4 warmup/cosine 的关键学习率点；
- raw MAE、normalized MAE 和 support gate。

## 当前运行

目录包含本地参考实现，因此无需统一 runtime 即可运行：

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

测试只依赖 Python 标准库 `unittest` 作为测试框架，以及项目已有的 NumPy 数值运行时。

## 以后指向实现根

实现准备好后，提供一个很薄的 adapter 模块（可以位于实现根，也可以是实现根
下的测试 adapter），并将环境变量指向它：

```powershell
$env:TRACK_B_ORACLE_TARGET = "fsx.scientific_oracle_adapter"
python -m unittest discover -s . -p "test_*.py" -v
```

adapter 需要暴露与 `scientific_oracles.py` 同名的下列函数：

`matrix_contract`, `canonical_grid41`, `normalized_xy`, `b8_quadratic`,
`b2_identity_sets`, `fixed_batch_stream`, `b7_canonical_edges`,
`b7_interleaved_stream`, `b7_matched_exposure`, `b7_relative_loss`,
`b7_dense_absolute_loss`, `b6_ridge_reference`, `common_lr`, `b4_causal_lr`,
`metric_summary`, `support_gate`。

adapter 的职责只是把实现中的函数参数/返回值映射到这个小接口；不要把
runtime、打包、SHA 或 reviewer 门控带进 oracle。这样之后可以在同一套测试中
直接检查实现的科学语义，同时保持测试本身与实现根隔离。

## 证据边界

本目录中的参考函数是对 V3、V7、V8 的数值化表达，不是新的实验方案，也不替代
全程复核者或具体过程复核者。B7 的 oracle 特别固定了“完整 relative+anchor
objective 对 dense absolute labels、在 matched endpoint exposure 下”的解释；它
不声称单独识别 anchor 的贡献。

