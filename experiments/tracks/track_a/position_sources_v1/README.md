# `position_sources_v1`

这是 Track A（绝对位置信息来源分解）的独立、本机可审查实验包。它只依赖
Python 标准库、NumPy 和当前环境已有的 PyTorch；不会导入历史
`neural_affine_*` 包，也不会修改其代码、checkpoint 或结果。

## 冻结范围

本版本把 Track A 的三组模型/数据协议冻结为：

- A1 padding/boundary：`zero_s32`、`reflection_s32`、`circular_s32`、
  `valid_core_s32`、`true_valid_s32`；主规范使用 GroupNorm32，避免 BatchNorm
  running-statistics 混淆。
- A2 stride/phase：`valid_core_s32`、`valid_core_aa32`、`valid_core_s1`。
- A3 strict torus：`torus_s32`、`torus_s1`；renderer 和卷积均使用周期边界。
- zero-padding BatchNorm 是单独的 `reference_only` 参考臂（不进入五臂主比较、解释门槛或聚合），同样固定三 seed、4000 steps、fp32/effective batch 16；它不能与主 `zero_s32` 混名。

主画布是 `1024x1024`，完整整数位置域为 `[448,575]^2`。位置域不使用 9x9
稀疏网格：按 quotient cell 和 residue 做确定性 SHA-256 80/20 split，train/test
均覆盖 16 个 quotient cells 以及两个轴的全部 32 个 residues。renderer 按需
生成图像，formal prepare 不物化整个 1024 域的图像。

`valid_core` 使用标准 zero-padded ResNet18 的中央 feature cells，并由代码
根据精确感受野计算安全 cell；preflight 必须通过 layer4 `19x19` 的 symbolic
RF certificate。`true_valid` 的所有 spatial convolution 和
pooling 都是 `padding=0`，projection shortcut 使用 3x3 valid projection
后做对称中心裁剪。后者是独立复核实现，不宣称与标准 ResNet18 逐参数等价。

renderer 是 compact Gaussian blob 与固定 scalene triangle，不得用偏移
Gaussian 代替三角形。主训练协议是 AdamW `lr=1e-3`、`wd=1e-4`、
`MSE+0.25L1`、cosine `eta_min=1e-5`、4000 steps、effective batch 16、
fp32、AMP 关闭；strict torus 为 `256x256`，目标用每轴 sin/cos 周期编码。

解释门槛也冻结在 protocol 中：global improvement 至少 2 px 且 quotient
R² 至少 0.10；phase-only 需要 phase R² 至少 0.50、quotient R² 不超过
0.05 且 32/64 px collision 不超过 1 px；null 需要 improvement 不超过
1 px 且两个 R² 都不超过 0.05；其它情况必须标记 `inconclusive`。

A2 的 GAP 可读性是独立证据：evaluate 会在 fc 前物化 512 维
`model.forward_features`，分别保存 train/eval feature fields 并记录 hash；ridge
probe 只用 train 拟合、只在 heldout eval 计算 phase/quotient R²，同时记录 1/32/64 px
feature L2/cosine distance。reviewer 不能把二维坐标输出场冒充 GAP field。

本包的训练 CLI 默认拒绝正式训练。必须显式传递 `--allow-formal` 才会进入
`run-family`；本次准备只运行 `prepare/preflight/smoke/profile` 和 CPU 单测。

## CLI

在本目录执行：

```powershell
python -m position_sources.cli prepare --out work\prepared --quick
python -m position_sources.cli preflight --root work\prepared
python -m position_sources.cli smoke --out work\smoke
python -m position_sources.cli profile --out work\profile --variants zero_s32,true_valid_s32

# 正式训练（本次不执行；必须显式授权；每条命令先注册 block/run id）
python -m position_sources.cli run-family --root work\prepared --family padding `
  --primitive blob --comparison-block-id A1-padding-blob `
  --run-id A1-padding-blob-full --microbatch 16 --allow-formal

# 仅供参考的 zero-padding BatchNorm 臂（不属于 primary 五臂）
python -m position_sources.cli run-reference --root work\prepared `
  --primitive blob --comparison-block-id A1-padding-bn-reference `
  --run-id A1-padding-bn-reference-full --microbatch 16 --allow-formal

python -m position_sources.cli evaluate --root work\prepared --checkpoint <checkpoint>
python -m position_sources.cli pack --root work\prepared --out dist\position_sources_v1.zip
python -m position_sources.cli verify --path dist\position_sources_v1.zip

python -m position_sources.cli review-block `
  --root <executor-root> --out <independent-review-root> `
  --allowed-reviewer-root <independent_review_v1>/track_a/<block-id> `
  --comparison-block-id <block-id> --family padding `
  --input inputs.npy --predictions predictions.npy --coordinates coordinates.npy `
  --gap-train-features gap_features_train.npy --gap-eval-features gap_features_eval.npy `
  --gap-train-coordinates train_points.npy --gap-eval-coordinates eval_points.npy `
  --gap-feature-metadata GAP_FEATURE_METADATA.json `
  --support-coordinates support_coordinates.npy `
  --support-predictions support_predictions.npy --support-targets support_targets.npy
```

`--microbatch` 只能是冻结 effective batch 16 的正整数因子（`1/2/4/8/16`）。
它按同一组预采样索引做梯度累积，每个逻辑 batch 只执行一次 optimizer 和
scheduler step；checkpoint/receipt 会同时记录 `microbatch`、
`effective_batch_size` 和 `optimizer_steps`。variants/seeds 可以作为并行 shard
提交，但 receipt 会标记 `status=shard`，不会冒充完整 family。

训练会在每 500 个 optimizer steps 写入带完整 identity、model/optimizer/scheduler、
RNG、sampler 和最后一批预采样 indices 的 `in_progress` checkpoint；终点 checkpoint
同样保存这些状态。`--resume-from` 只能指向同一 primitive/family/variant/seed/
comparison block/run/cache/protocol/effective-batch/microbatch/model-stack、device backend
和 torch runtime stack 的周期 checkpoint，所有字段逐项不一致都会拒绝。GPU/ROCm
checkpoint 必须含并恢复 `torch_cuda` RNG；CPU 与 GPU/ROCm checkpoint 互相拒绝，且不会
覆盖已完成的 formal 证据。单 primitive 全矩阵 receipt 只可标记
`family_primitive_complete`；只有显式聚合两个 primitive 后才可另行声明 block complete。

`reviewer/CONTRACT.md` 规定了独立只读复核边界。复核结果只能写入包外的
`independent_review_v1/track_a/<review_id>/`，并且 CLI/函数必须同时收到完全相同的
显式 `--allowed-reviewer-root`；不能写入代码、cache、checkpoint 或正式结果目录。
`verify` 会检查 manifest 中声明文件的 byte 数和 SHA-256。包根的
`package_manifest.json` 供独立 freeze-audit 使用，列出全部正式源文件并记录
按 path+sha256 字节序列计算的 `aggregate_sha256`；它本身不列入 files。

## 资产边界

本目录只生成新的协议、代码、测试、轻量 smoke/profile 输出和本目录下的
manifest。它不连接远端、不上传、不依赖远端路径、不读取历史 checkpoint，
也不执行正式训练。正式实验仍需在用户审阅 protocol、manifest、机器分配和
评测门槛后单独启动。
