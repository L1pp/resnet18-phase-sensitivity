# Track A MiniCNN Valid-Boundary v1

独立的小型边界条件实验，不修改 `track_a_toy_v1`。三条件为：

- Z：finite + 每层显式 zero padding + S1；
- V：finite + 每层真正 `padding=0` valid 卷积 + S1；
- C：torus + 每层显式 circular padding + S1。

四层 3×3 卷积的通道为 `3→16→32→32→32`，均为 `bias=False`，ReLU，无 BN、残差、池化和 Dropout。GAP 按固定协议计算 `sum/4096`，head 为 `32→4`，输出 x/y 的 64 周期 sin/cos。

阶段1只生成未训练 probe、中文图和报告：

```powershell
python -m track_a_minicnn.cli report
```

正式六 run 由唯一入口直接启动，阶段1不会调用：

```powershell
python -m track_a_minicnn.cli run --device cuda
```

训练固定 visible=182、safe=90 两个 scope，每个 scope 的 Z/V/C 各500步；clipped=74 完全不训练、不排名。结果只写入 `results/track_a_minicnn_v1/`。
