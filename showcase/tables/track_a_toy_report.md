# Track A toy boundary experiment — public summary

This experiment compared finite and toroidal boundaries, with total stride 32 or stride 1. It first measured position information in frozen random features and then ran a short fixed protocol.

## Frozen-feature probe

The table below uses the all-256-position scope. Coordinate and phase errors are lower-is-better; coarse-grid accuracy is higher-is-better.

| Condition | Boundary / stride | Coordinate MAE | 32-pixel phase MAE | Coarse-grid accuracy |
|---|---|---:|---:|---:|
| A0 | finite / zero / S32 | 3.874 px | 0.854 px | 0.992 |
| A1 | finite / zero / S1 | 5.867 px | 4.338 px | 0.812 |
| A2 | torus / circular / S32 | 16.000 px | 0.469 px | 0.500 |
| A3 | torus / circular / S1 | 16.000 px | 8.000 px | 0.500 |
| A0-reflect | finite / reflect / S32 | 4.127 px | 1.114 px | 0.992 |
| A0-circular | finite input / circular network / S32 | 15.737 px | 0.896 px | 0.523 |

![Track A toy overview](../figures/track_a_toy_overview.png)

## Short-training result

| Model | Actual steps | Best full-grid MAE | Best step |
|---|---:|---:|---:|
| A0 | 200 | 1.599 px | 150 |
| A1 | 1000 | 2.816 px | 500 |
| A2 | 1000 | 16.000 px | 900 |
| A3 | 300 | 16.000 px | 200 |

## Interpretation limits

- The random-feature probe and trained-model result are separate stages and should not be conflated.
- Absolute coordinate error on a torus is seam-sensitive; the phase and coarse-grid metrics answer different questions.
- This is a single-seed synthetic mechanism experiment, not evidence of universal architectural behavior.
