# Clean-room S0 audit tools

这是收尾 sprint 的本机 `local_s0` 基础设施。它只依赖 Python 标准库、NumPy 和可选的 PyTorch；不导入旧实验包，也不覆盖旧 `results/`、`ckpts/` 或 A10 资产。

## 锁与执行边界

默认协议是 `protocols/s0_v1.json`。每次正式 runner 都会读取并计算协议文件 SHA-256，校验 `asset_manifest` 中每个文件的 SHA-256、NPZ schema、checkpoint tensor/head 维度，然后在计算前和写结果前再次重算资产 hash。协议当前 `current_stage=local_s0`，允许本机 `cpu`/`cuda:0`；训练命令不存在，CPU/local_s0 阶段无条件拒绝训练。

每个成功命令都会写入新的结果目录：`protocol_lock.json`、`asset_manifest.json`、`asset_lock.json`、`environment.json`、`manifest.json`、`metrics.json` 和 `decision.json`。输入资产只读。完整 `run-s0` 会先静态校验所有阶段资产和 schema，之后才创建输出目录，并固定按 `check → headlines → OLS → kernel → BN → partial` 执行；缺资产、错角色、shape/hash 不符会在任何阶段计算前失败。

## CPU 准备与单阶段审计

在 `closeout_sprint_clean` 目录执行（测试只使用标准库 `unittest`）：

```powershell
python -m unittest discover -s tests -v
python -m closeout_sprint.cli check
python -m closeout_sprint.cli audit-headlines `
  --input ..\results\today_shortcycle\a10\frozen_g64\ols_G64\field.npz `
  --output ..\results\closeout_sprint_clean\headlines
```

新增命令：

```powershell
python -m closeout_sprint.cli audit-kernel --input path\kernel_input.npz
python -m closeout_sprint.cli audit-bn --config path\bn_adapters.json
python -m closeout_sprint.cli audit-partial `
  --input neural_affine_2d=path\neural_affine.npz `
  --input amd_2d=path\amd.npz
python -m closeout_sprint.cli run-s0 --protocol protocols\s0_v1.json
```

OLS NPZ 必须使用正式 `frozen_g64_gap_to_p6_v1` schema：`train/eval_gap_features`、`train/eval_p6_norm`、`train_q_px[N,3,2]`、`train_t_px[N,2]`、`eval_q_px[N,3,2]`、`eval_t_px[N,2]`，以及 `train_shape_id/train_translation_id/eval_shape_id/eval_dense_index`；固定 `target_semantics=p6_normalized`、`coord_scale=223`、`head_dim=6`。正式可比路径强制 train=3584（每 translation 56 个互斥 pair）、eval=32×41×41，并从语义 ID 校验/构造 row IDs 与 JSON provenance。历史 `train_features/train_targets` 或仅有 `W/b` 的 `ols_head.npz` 会被明确拒绝。所有稳定性指标使用 `t_hat=mean(P_hat*223-Q)` 与 `eval_t_px` 的欧氏误差 `eval_box_mae_px`，另报 normalized/elementwise P6 MAE；居中的内部 shape residual 仅作补充诊断。kernel NPZ 必须分离 fit 与 held-out rows、显式 `feature_names` 和 held-out Euclidean baseline；held-out target 永远不会进入拟合。BN 配置必须显式声明 `model_adapter`、`data_adapter`、`evaluator_adapter`（`module:function`），并显式声明顶层 `device`；若有 `model_kwargs.device`，它必须与顶层值一致，runner 会用已验证值覆盖后再调用 model adapter。BN 的 evaluator 记录需要在 exposure 0 和后续 exposure 提供 `affine_removed_mae_px`，结果会报告 `u_ratio`、最大/最终比值和协议冻结阈值；缺指标只会得到 `inconclusive`，不会伪报 passed。partial 输入每个任务组单独拟合、单独输出，不混合 2D/6D 或设备组；必须提供显式 `anchors`，不会回退到 corners。除传统 `coords`+`signals` 外，runner 兼容 `signals_vector[appearance,y,x,seed,2]`，并要求显式 `x`、`y` 网格，输出中保存 x/y、pooled correlations 和 gate。

BN、kernel、partial 的完整资产尚未具备时，不要用历史重资产替代；运行对应单阶段命令或等待新的 hash-pinned manifest。A10/GPU 训练属于后续单独审阅的阶段，不在本 CPU runner 内启动。
