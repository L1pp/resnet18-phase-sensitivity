# Phase 3 科学协议（给人看，不是给云上 AI 改的）

本轮四个互不依赖命题：

1. **H1** vanilla CNN 是否自发形成近似 translation-group action
2. **H2** 绝对坐标能力是否被 symmetry-breaking 因果控制
3. **H3** 同一 position head 能否跨内容 / 跨分辨率复用
4. **H4** 连续坐标有多少来自 random architecture prior

结果只写 `results/phase3_high_upside_discovery/`。禁止写入 Phase 1 / Phase 2 目录。禁止默认 `resume=true`。禁止宣称严格数学群表示。

## Track A

复用云上已有 Phase 1.8 三 seed ResNet18 与 Phase 2 DenseNet121 G64。不重训 Phase 1.8。

- 32 个冻结 `eval_shape_ids`，安全盒 `[59, 164] px`
- 33×33 连续网格；算子 origin 为每 3 个格点（11×11）
- Δ：x/y 的 ±0.5/1/2/4/8 与对角 (1,1)(2,2)(4,4)(4,-4)
- 16/8/8 shape split；origin 70/15/15
- 拟合 `z' ≈ M z + b`，ridge + reduced-rank，超参只在 val
- 群律用 functional error，不用 512×512 Frobenius 当主指标
- frozen head：`t̂(Mz) ≈ t̂(z)+Δ`，`q̂` 尽量不变
- 对照：random-init、shuffled Δ、同秩随机算子

## Track B

新建约 1.1M 受控 CNN。**禁止**把 Phase 2 circular-ResNet 当成 S0（它仍有 stride-2）。

- 96×96 Gaussian blob，目标生成中心 `(x,y)`
- S0 circular+stride1+GAP（负对照，MAE 高是成功）
- S1 circular+stride；S2 zero+stride1；S3 zero+stride
- S4 circular+AA；S5 zero+AA；S6 = S0+CoordConv
- Dense / Sparse 3×3；同时测 `D_eq` 与 `D_GAP`

## Track C / D

中心 = 生成参数，不是 ink centroid。只预测 `(x,y)`。

- 六族：line / circle / arc / quadratic / polygon / blob
- LOFO seed `20260830`，禁止在 unseen family 上重训 head
- D：224 训练，测 160/192/224/256/320；relative vs absolute 物体尺寸

## Track E / F

- E：ResNet18 / DenseNet121 / EfficientNet-B0 × R0–R3 × G64/G9
- EfficientNet 强制 fp32+clip=1.0
- F：仅时间充足时跑 1 seed 的 `(tx,ty,θ,log s)`

## HIGH_UPSIDE 旗

只有同时满足协议里写死的阈值才打旗。负结果同样有信息量，尤其是 H1/H2。
