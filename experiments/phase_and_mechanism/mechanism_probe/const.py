"""Local short-cycle mechanism probe. Do not write into functional_drift or error_extension trees."""

from __future__ import annotations

from pathlib import Path

from error_extension.const import CORNERS4_TIDS
from phase2_overnight.protocol import regime_tids
from phase3_high_upside.protocol import L1_WEIGHT, LR, LR_MIN, WEIGHT_DECAY

PACK_ROOT = Path(__file__).resolve().parents[1]
CODE_REV = "mechanism_probe_20260819"
MACHINE = "local"
SEED = 20260816
GEOMETRY_FP = "195f37870e2cdb8685a7"

RESULTS_ROOT = PACK_ROOT / "results" / "mechanism_probe" / "local"
DRIFT_ROOT = PACK_ROOT / "results" / "functional_drift"

BLOB_CKPT = PACK_ROOT / "results" / "spatial_field_dynamics" / "content" / "blob_G64" / "best_slim.pt"
BLOB_SHA256_PREFIX = "dc7022ab98b28cc6"
HEAD_2 = 2
IMAGE_SIZE = 224
COORD_SCALE = 223.0
N_DENSE = 41
EVAL_BATCH = 16
TRAIN_BATCH = 16
ADAMW_BETAS = (0.9, 0.999)
WD = float(WEIGHT_DECAY)
LR_USED = float(LR)
LR_MIN_USED = float(LR_MIN)
L1_W = float(L1_WEIGHT)
STAGES = ("head", "l4", "full")
FFT_STEPS_2D = (0, 1, 10, 3000)
FFT_STEPS_6D = (0, 1, 10, 4000)
OPT_STEPS = 10
WARMUP_DENSE_STEPS = 30
HESS_K = 16
HESS_ITERS = 12
FD_REL_RMS = 1e-3

MLP_PRETRAIN_STEPS = 5000
MLP_PRETRAIN_BOX = 0.15
MLP_SPARSE_STEPS = 3000
MLP_DUMP = (0, 1, 10, 3000)

FORBIDDEN_INIT_SUBSTR = (
    "error_extension",
    "functional_drift",
    "gate_resnet18_g64",
)

SUPPORT_REGIMES = {
    "corners4": tuple(int(t) for t in CORNERS4_TIDS),
    "G9": tuple(int(t) for t in regime_tids()["G9"]),
    "G16": tuple(int(t) for t in regime_tids()["G16"]),
    "G64": tuple(int(t) for t in regime_tids()["G64"]),
}
