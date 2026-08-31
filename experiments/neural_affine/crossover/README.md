# Neural Affine support-density × plasticity clean protocol

This directory is independent from `neural_affine_s1_clean`.  The formal
runner uses the legacy-compatible vanilla torchvision ResNet18 state layout,
the verified three corrected exact-init files, uint8 renderer bytes and a
materialized paired batch stream.

```powershell
python -m crossover.cli prepare --out <root> --init-dir <corrected_init_dir>
python -m crossover.cli smoke --root <root> --steps 2
python -m crossover.cli run-matrix --root <root> --device cpu
python -m crossover.cli aggregate --root <root>
```

`prepare` creates `cache/`, `exact_inits/`, and `batch_streams/`.  Formal
conditions are written only under `formal_runs/`; smoke conditions are under
`smoke_runs/` and are never consumed by `aggregate`.  Each training condition
receives support arrays only.  `independent_evaluator` atomically writes
`predictions_dense.npz` before opening dense labels.

The official matrix contains 36 cells: 4 supports × 3 seeds × 3 regimes.
There are 24 trained cells (`head_only`, `full`) and 12 closed-form
`frozen_feature` baselines.
