# SGD Anchor Rescue Protocol

状态：方案草案，供用户和执行 AI 审阅

本协议只覆盖一次受控的生死判定实验：在已经训练好的同一函数上，使用 SGD 在 `corners4 + l4` 稀疏约束上继续训练，先确认能否达到与 AdamW 相近的 anchor 拟合，再揭示完整连续场。协议写成并审阅前，不得启动远端训练。

## 1. 研究问题与边界

唯一主问题是：

> 当 SGD 从同一个合法 `theta0` 出发，并且把 `corners4` anchors 拟合到与 AdamW 相近的水平时，是否也会产生大幅的 affine-removed residual `u`？

这不是一次新的全面参数扫描。以下内容明确不在本轮范围内：

- 其他 support 梯子（`G64/G32/G16/G9` 等）；
- MLP 或 linear model；
- NTK/null-space 分析；
- P3 maintenance complexity；
- 多 architecture、多 width、多 seed、replay 或额外正则化；
- 为得到漂亮的 dense field 而临时修改 optimizer 或挑选 checkpoint。

如果本轮没有得到清晰结论，fine-tuning 主线应停止扩展，回到 sparse-from-scratch function selection 问题；本协议不预授权后续补实验。

## 2. 固定对象与可追溯性

AMD 的 2D 和 A10 的 6D 是两个独立验证，不把两个维度拼成一个平均分数，也不把 AMD 选出的 schedule 当成 A10 已验证 schedule。两台机器必须分别从各自原实验使用的、可核验的原始 `theta0` 开始：

- AMD：`deploy/constraint_removal_amd_pack/blob_G64/best_slim.pt`，SHA 前缀 `dc7022ab98b28cc6`，head=2；
- A10：`ckpts/phase2/resnet18__G64__20260820/checkpoints/best_slim.pt`，SHA 前缀 `f304f50f2759a534`，head=6。

合法原始资产的文件名本身可以是 `best_slim.pt`。禁止的是 `results/constraint_removal/**/cnn_sgd/**`、`results/error_extension/**`、`results/functional_drift/**` 等已经解冻或微调后的 checkpoint。执行 AI 在运行前必须记录：

- `theta0` 的来源、文件路径、SHA 或等价内容哈希；
- 明确确认不是任何已解冻后的 checkpoint，也不是上一次候选的继续训练；
- 固定实验 seed（本轮沿用已确认的 seed `20260816`，除非用户在审阅时明确修改）；
- `corners4` 的 anchor 定义、坐标顺序和 target；
- batch size、损失定义、单位和现有 canonical anchor/objective 指标；
- 冻结/解冻范围；
- BatchNorm 或其他状态层的 train/eval 语义及其运行时状态处理；
- 代码版本、依赖/设备信息和输出目录。

除 momentum 外，同一机器 A/B 候选之间的模型、数据顺序、seed、batch、loss、weight decay、freeze/BN 语义和所有未列出的 optimizer flags 均必须相同，并写入 run manifest。冻结值为 batch `64`、weight decay `1e-4`、`MSE + 0.25 * L1`、amp=false；冻结层 BN 保持 eval，可训练层 BN 保持 train。AMD 与 A10 的数据规模、采样器和输出维数本来不同，不要求跨机器强行统一，只要求各机内部公平。

两候选必须使用同一预生成 batch-index 流或显式 `torch.Generator` seed，并写出 data-order fingerprint；不得只依赖全局 seed 推测 DataLoader 顺序一致。

新结果根冻结为 `results/sgd_anchor_rescue/{local,amd,a10}/`，每个候选再使用独立子目录；不得覆盖已有 `results/constraint_removal/` 或历史 AdamW 结果。运行结果必须同时保存配置、日志、固定 checkpoint、指标表和模型/数据哈希，避免把训练中最小值误当作 endpoint。

## 3. 阶段 0：只读准备与门禁

这一阶段不训练、不选择 dense checkpoint。执行 AI 先完成以下检查，并在 manifest 中逐项标记通过/失败：

1. 能读取并校验合法 `theta0`、`corners4`、seed 和目标定义。
2. 能在目标设备上加载模型，并以 eval 语义计算 step 0 的 full-train anchor/objective；A10 必须覆盖 corners4 support 的全部 224 个训练样本，禁止用 `train[:256]` 或其他前缀子集作为闸门指标。
3. 能保存 step 0 和训练过程中的预先规定 checkpoint，能在训练结束后按 checkpoint 重新计算指标。
4. 能区分训练 endpoint、首次达到门槛的 checkpoint 和训练期间 anchor 最小的 checkpoint。
5. 能确认输出目录为空或为本次新建目录，且没有从禁止的旧 checkpoint 或旧 optimizer state 恢复。
6. 新实现具备真正的 `anchor_only` 模式：阶段 1 禁止调用 `dense_official` / `official_eval_and_save`，禁止生成 `field.npz`，禁止在 stdout、日志或图中输出 raw box/`u`。step 0 的 dense 场也留到阶段 2 统一揭示。

阶段命令只在执行环境中由执行 AI 根据现有仓库入口填写：

```text
[PREPARE_COMMAND_PLACEHOLDER]
[SMOKE_COMMAND_PLACEHOLDER]
```

这里故意不写未经核验的 shell 命令。远端尚未连接或远端 Codex CLI 尚未完成登录时，只允许整理 manifest、核对输入和等待连接；不得上传、启动训练、申请实例、修改远端环境或把本地 smoke 的结果冒充远端验证。

## 4. 阶段 1：anchor-only schedule rescue

### 4.1 候选配置

最多注册两个主候选，且只能根据 on-support 指标进行比较：

| 候选 | optimizer 差异 | 其他安排 |
|---|---|---|
| A | SGD，`momentum=0` | 峰值学习率 `1e-3`，500 step linear warmup，随后 cosine 到 20k step |
| B | SGD，`momentum=0.9` | 与 A 完全相同 |

warmup 包含在 20k 总步数中；warmup 后对剩余步数执行 cosine，末端学习率固定为 `1e-5`。两候选 `dampening=0`、`nesterov=false`，候选的唯一区别是 momentum。不得因为任何中途结果改动这些值。

### 4.2 successive-halving 与观察规则

1. A、B 各从同一合法 `theta0` 独立启动，使用相同 batch-order fingerprint，先运行至 5k。
2. 5k 阶段只读取 anchor/objective、数值稳定性和下降趋势；不得读取或用 raw box、`u`、dense field、图像或其他 off-support 量挑选候选。
3. 5k 选择使用确定的词典序：先排除非有限值；再要求 `anchor_5000 < anchor_1000`；剩余候选按 full-train `anchor_5000` 较小者优先，再按同一 full-train objective 较小者优先，仍相同固定选择 A。日志不得引用其他字段。
4. 选中的候选沿同一合法轨迹继续到 20k。20k 结束时，固定的 19k 与 20k 检查点都必须达到相应机器的 anchor 门槛，不能只凭一个偶然 best checkpoint 过闸门。

full-train anchor 每 200 step 计算一次。固定保存状态点为 `0,1,5,10,50,100,200,500,1k,2k,5k,10k,15k,18k,19k,20k`；另保存首次过门状态，并用可追溯的单文件滚动保存当前 min-anchor 状态，避免每 200 step 保存一份大模型。anchor-only 阶段在候选冻结前不得生成、查看或汇报 dense-field 结论。

### 4.3 一次且仅一次的预定义补救

只允许以下两种补救分支中的一个，并且不能组合、重复或再开网格：

- 若 A/B 在 5k 至少有一个稳定候选，但选中候选到 20k 未过门且仍下降：沿同一轨迹从 20k 延长到 40k，学习率固定保持 `1e-5`；固定尾部检查点为 39k/40k；
- 若 A/B 在 5k 都不满足稳定规则：从同一合法 `theta0` 只重跑一次 20k，峰值学习率降为 `3e-4`、warmup 延长为 1k，其余相同；该分支不得再延长到 40k。

“20k 仍下降”固定定义为 full-train `anchor_20000 < anchor_15000`。若 20k 未过门且不满足该条件，不触发补救，直接归为 optimization pathology。40k 分支只允许从本次已选中的 20k endpoint 及其 optimizer state 继续；低学习率分支必须从原始 `theta0` 重启。两分支都不得使用 min-anchor/first-match 或历史微调状态。每机 optimizer step 总上限为 45k（A/B 各 5k，赢家补到 20k，再最多延长 20k）；补救结束即停。

## 5. Anchor 闸门

门槛使用已有 canonical endpoint anchor metric，单位为 px，不新造替代指标：

| 设备/维度 | 严格门槛 | AdamW 参考（仅作量级参照） |
|---|---:|---:|
| AMD / 2D | endpoint anchor `<= 0.10 px` | 约 `0.0740 px` |
| A10 / 6D | endpoint anchor `<= 0.17 px` | 约 `0.1359 px` |

这是预注册的 practical match gate，不表示两个 optimizer 的损失在数学上完全相等。A10 的 canonical gate 是全部 224 个 corners4 train 样本上的 MAE；历史 AdamW checkpoint 也必须用相同 evaluator 复核。这里的“过闸门”同时要求：

- endpoint 达到门槛；
- 最后两个固定 checkpoint 均达到门槛，且两者 anchor 绝对差不超过该机器 strict threshold 的 20%；
- 指标来自本次合法 theta0 的同一训练轨迹，而不是训练中最小 anchor 的另存模型。

AMD 和 A10 任意一台未过门，都不能写成“SGD 已复制 AdamW 现象”；应分别报告 `AMD pass/fail` 与 `A10 pass/fail`。

## 6. 阶段 2：冻结 schedule 后揭示 dense field

只有阶段 1 选出稳定候选并通过对应 anchor 闸门后，才允许进行完整场评估。schedule、seed、theta0 和训练轨迹在此阶段冻结；不得因看到 dense 结果再回头改配置。

对阶段 1 预先规定的固定状态点，以及自动保存的 first-match/min-anchor，统一计算并报告：

- canonical anchor/objective；
- raw box；
- affine-removed residual `u`；
- step 1 shock；
- endpoint；
- 首次达到 matched-anchor 门槛的 checkpoint（`first-match`）；
- 训练期间 anchor 最小的 checkpoint（`min-anchor`）。

定义固定如下：`endpoint` 是最后一步模型；`first-match` 是每 200 step full-train 检查序列中首个达到 strict gate 的模型；`min-anchor` 是全部每 200 step 的 full-train anchor 检查点及固定点中 anchor 最低的模型。三者必须分栏呈现，不能用 `min-anchor` 的 dense field 代替 endpoint，也不能把首次过门之后的恶化藏起来。step 1 shock 只作为正式结果，不参与 schedule 选择。

## 7. 预注册结论规则

结果只能按以下四类之一归档，先写分类再写解释：

1. **Anchor matched + 强 drift**：endpoint 与 first-match 都满足 `u >= 2 * u0`、raw box 至少比 step 0 增加 1 px，且 endpoint `u >= 0.5 * u_AdamW`。冻结参考为 AMD `u_AdamW=32.00699 px`（`results/functional_drift/amd/l4/train_summary.json`）和 A10 `u_AdamW=6.42162 px`（`results/functional_drift/a10/unfreeze_g64_corners4/l4/train_summary.json`）。这说明预注册且经 anchor-only 选择的 SGD schedule 也能出现强 functional drift；结论仍是 optimizer-conditional。
2. **Anchor matched + 保持原场**：endpoint 与 first-match 都满足 `u <= 1.5 * u0`。这说明同一 sparse objective 下 optimizer/implicit bias 或 parameterization 影响很大；不能再宣称 constraint removal 单独足以导致 harmful drift。
3. **40k/一次补救内仍不能稳定 matched anchors**：判为 optimization pathology；fine-tuning 主线到此停止，不继续补 support 梯子或调参。
4. **中间态、只在单 checkpoint 过门、first-match 与 endpoint 分类不同、或 AMD/A10 方向冲突**：判为 inconclusive；step 1 shock 单独报告，不用它替代终点分类，也不自动开启后续实验。

如果一台机器属于第 1/2 类而另一台属于第 3/4 类，应保留机器级结论并归档为跨架构不一致，不强行合并成普遍规律。

## 8. 机器职责与时间预算

本机 GTX 1060：协议、manifest、输入/哈希核对、smoke、回传后的统一分析和 closeout；不承担 ResNet 长训。

AMD：只负责 `corners4 + l4` 的 2D anchor-rescue 与阶段 2 的 2D dense reveal。

A10：只负责 `corners4 + l4` 的 6D anchor-rescue 与阶段 2 的 6D dense reveal。

在远端连接可用的前提下，AMD 与 A10 可并行执行，以缩短墙钟时间；本机不作为第三个训练节点。若只连通一台远端，另一台保持未执行，不能用单机结果代替双机验证。

时间必须把实测和估算分开记录：

- 已有实测参考：AMD 单条 l4/3000 step 约 3 分钟；A10 单条 l4/4000 step 约 7 分钟；上一轮 A10 的 P0+P1 总耗时约 2.3 小时。
- 本轮估算：协议/实现审阅和 smoke 约 60–90 分钟；两机 anchor-rescue 并行约 45–90 分钟；dense reveal、回传和 closeout 约 30–45 分钟；从批准到首轮生死判定约 2.5–4 小时。每台 GPU 内 A/B 串行，两台 GPU 之间才并行；无补救时每机累计 25k optimizer steps。
- 若触发一次 40k/低学习率补救，再增加约 30–60 分钟；只有首轮严格过闸门且用户另行批准，才估算额外 `full` 单 seed 或其他扩展，当前不纳入本协议。

网络、实例排队、人工上传和远端登录耗时另记，不与 GPU 训练时长混淆。未获得远端 Codex CLI 登录和实际连接前，状态只能是“prepared / not executed”。

## 9. 交付与停止条件

执行 AI 最终应交付一份每机独立、可复核的结果包，至少包括：运行 manifest、theta0/代码/数据哈希、A/B 5k 比较、最终 schedule、训练日志、固定 checkpoint、endpoint/first-match/min-anchor 对照、anchor 闸门状态、dense 指标表和四类结论分类。

以下任一情况即停止本轮，不自行扩大范围：远端未登录或不可连接；theta0/目标/冻结语义无法核验；两个候选均不稳定或无法区分；一次补救失败；40k 仍未过门；或结果落入 inconclusive。后续是否写代码、分配机器、启动训练和追加 `full`，均等待用户审阅本协议并明确批准。
