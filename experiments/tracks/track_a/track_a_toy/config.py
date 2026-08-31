from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "track_a_toy_v1"
DATA_ROOT = OUTPUT_ROOT / "data"
FEATURE_ROOT = OUTPUT_ROOT / "random_features"
PROBE_ROOT = OUTPUT_ROOT / "probe"
TRAIN_ROOT = OUTPUT_ROOT / "training"
FIGURE_ROOT = OUTPUT_ROOT / "figures"


CONFIG: dict[str, object] = {
    "schema_version": 1,
    "experiment_id": "track_a_toy_v1",
    "purpose": "personal mechanism reproduction and teaching experiment",
    "seed": 20260825,
    "image_size": 64,
    "grid_step": 4,
    "grid_values": list(range(0, 64, 4)),
    "triangle_vertices_xy": [[-6.0, -4.0], [6.0, -3.0], [0.0, 7.0]],
    "shifts_px": [1, 2, 4, 8, 16, 32],
    "feature_batch_size": 16,
    "train_batch_size": 16,
    "train_max_steps": {"a0_standard": 1000, "a1_zero_s1": 1000, "a2_torus_s32": 1000, "a3_torus_s1": 300},
    "train_eval_interval": 50,
    "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
    "dtype": "float32",
    "amp": False,
    "ridge_alphas": [1e-6, 1e-4, 1e-2, 1.0, 100.0],
    "probe_test_rule": "(ix + 2*iy) mod 4 == 0",
    "visual_positions_xy": [
        [0, 0],
        [32, 0],
        [60, 0],
        [0, 32],
        [16, 16],
        [32, 32],
        [48, 16],
        [60, 32],
        [32, 60],
        [60, 60],
    ],
}


MODEL_SPECS: dict[str, dict[str, object]] = {
    "a0_standard": {"padding_mode": "zeros", "total_stride": 32, "renderer": "finite", "train": True},
    "a1_zero_s1": {"padding_mode": "zeros", "total_stride": 1, "renderer": "finite", "train": True},
    "a2_torus_s32": {"padding_mode": "circular", "total_stride": 32, "renderer": "torus", "train": True},
    "a3_torus_s1": {"padding_mode": "circular", "total_stride": 1, "renderer": "torus", "train": True},
    "a0_reflect_s32": {"padding_mode": "reflect", "total_stride": 32, "renderer": "finite", "train": False},
    "a0_finite_circular_s32": {
        "padding_mode": "circular",
        "total_stride": 32,
        "renderer": "finite",
        "train": False,
    },
}


def config_fingerprint() -> str:
    payload = {"config": CONFIG, "model_specs": MODEL_SPECS}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_ROOT, FEATURE_ROOT, PROBE_ROOT, TRAIN_ROOT, FIGURE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
