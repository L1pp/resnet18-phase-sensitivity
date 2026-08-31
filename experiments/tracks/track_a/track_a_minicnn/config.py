from __future__ import annotations

from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_ROOT = WORKSPACE_ROOT / "results" / "track_a_toy_v1" / "data"
OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "track_a_minicnn_v1"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
TRAIN_ROOT = OUTPUT_ROOT / "training"
REPORT_PATH = OUTPUT_ROOT / "report.md"
REPORT_JSON_PATH = OUTPUT_ROOT / "report.json"
PROBE_JSON_PATH = OUTPUT_ROOT / "probe.json"
ANCHOR_JSON_PATH = OUTPUT_ROOT / "anchor_shift_diagnostics.json"
TRAINING_SUMMARY_PATH = TRAIN_ROOT / "summary.json"

SEED = 20260825
IMAGE_SIZE = 64
GRID_VALUES = tuple(range(0, IMAGE_SIZE, 4))
TRIANGLE_VERTICES_XY = ((-6.0, -4.0), (6.0, -3.0), (0.0, 7.0))
BATCH_SIZE = 32
MAX_STEPS = 500
EVAL_INTERVAL = 25
POOL_DENOMINATOR = 4096.0
OPTIMIZER = {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4}
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
SPLIT_RULE = "(ix + 2*iy) mod 4 == 0 is held-out test"

CONDITIONS = {
    "Z": {"label": "Z：finite + zero + S1", "mode": "zero", "dataset": "finite"},
    "V": {"label": "V：finite + true-valid(padding0) + S1", "mode": "valid", "dataset": "finite"},
    "C": {"label": "C：torus + circular + S1", "mode": "circular", "dataset": "torus"},
}

# Every finite pair is on-canvas for delta=1/4/16 along +x.
SHIFT_ANCHORS_XY = (
    (8, 4),
    (16, 4),
    (32, 4),
    (40, 4),
    (8, 20),
    (32, 20),
    (40, 20),
    (8, 40),
    (32, 40),
    (40, 56),
)

VISIBLE_INPUTS_XY = (
    (8, 4),
    (16, 4),
    (32, 4),
    (48, 4),
    (8, 28),
    (32, 28),
    (56, 28),
    (8, 56),
    (32, 56),
    (56, 56),
)

SAFE_INPUTS_XY = (
    (16, 12),
    (32, 12),
    (48, 12),
    (16, 28),
    (32, 28),
    (48, 28),
    (16, 48),
    (32, 48),
    (40, 32),
    (48, 48),
)


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ANCHOR_JSON_PATH",
    "BATCH_SIZE",
    "CONDITIONS",
    "EVAL_INTERVAL",
    "FIGURE_ROOT",
    "GRID_VALUES",
    "IMAGE_SIZE",
    "MAX_STEPS",
    "OPTIMIZER",
    "OUTPUT_ROOT",
    "POOL_DENOMINATOR",
    "PROBE_JSON_PATH",
    "REPORT_JSON_PATH",
    "REPORT_PATH",
    "RIDGE_ALPHAS",
    "SAFE_INPUTS_XY",
    "SEED",
    "SHIFT_ANCHORS_XY",
    "SOURCE_DATA_ROOT",
    "SPLIT_RULE",
    "TRAINING_SUMMARY_PATH",
    "TRAIN_ROOT",
    "TRIANGLE_VERTICES_XY",
    "VISIBLE_INPUTS_XY",
    "ensure_output_dirs",
]
