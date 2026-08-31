"""Frozen protocol helpers for the support-density experiment.

The protocol is deliberately JSON based so a remote runner can verify the
same values without importing the package.  No historical S1 package is
imported here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"
EXPECTED_SEEDS = (20260816, 20260817, 20260818)
SUPPORT_NAMES = ("corners4", "G9", "G16", "G64")
REGIME_NAMES = ("frozen_feature", "head_only", "full")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_supports(value: Mapping[str, Any]) -> None:
    expected = {
        "corners4": [0, 7, 56, 63],
        "G9": [0, 3, 7, 24, 27, 31, 56, 59, 63],
        "G16": [0, 2, 5, 7, 16, 18, 21, 23, 40, 42, 45, 47, 56, 58, 61, 63],
        "G64": list(range(64)),
    }
    if set(value) != set(expected):
        raise ValueError(f"support names drifted: {sorted(value)}")
    for name, tids in expected.items():
        actual = [int(item) for item in value[name]]
        if actual != tids:
            raise ValueError(f"{name} tids drifted: {actual} != {tids}")


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol_id",
        "task",
        "seeds",
        "image_size",
        "coord_scale",
        "safe_box_px",
        "grid",
        "supports",
        "dense_grid",
        "renderer",
        "model",
        "regimes",
        "training",
        "evaluation",
        "gates",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"protocol missing keys: {missing}")
    if int(payload["schema_version"]) != 1:
        raise ValueError("schema_version must be 1")
    if str(payload["protocol_id"]) != "neural_affine_support_plasticity_clean_v1":
        raise ValueError("protocol_id drifted")
    if str(payload["task"]) != "neural_affine_support_density_x_plasticity":
        raise ValueError("task drifted")
    if tuple(int(value) for value in payload["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("seeds must be 20260816/20260817/20260818")
    if int(payload["image_size"]) != 224 or float(payload["coord_scale"]) != 223.0:
        raise ValueError("image_size/coord_scale must be 224/223.0")
    box = payload["safe_box_px"]
    if (float(box["low"]), float(box["high"])) != (59.0, 164.0):
        raise ValueError("safe box must be [59,164] pixels")
    if int(payload["grid"]["levels"]) != 8 or payload["grid"].get("tid_order") != "tx_major_ty_minor":
        raise ValueError("grid must be an 8x8 tx-major/ty-minor grid")
    _validate_supports(payload["supports"])
    dense = payload["dense_grid"]
    if int(dense["n"]) != 41 or dense.get("order") != "tx_major_ty_minor":
        raise ValueError("dense grid must be 41x41 tx-major/ty-minor")
    renderer = payload["renderer"]
    if renderer.get("family") != "blob" or renderer.get("mode") != "L":
        raise ValueError("renderer family/mode must be blob/L")
    if float(renderer["sigma"]) != 6.0 or int(renderer["supersample"]) != 4:
        raise ValueError("renderer sigma/supersample must be 6/4")
    if renderer.get("downsample") != "PIL.Image.Resampling.LANCZOS" or renderer.get("dtype") != "uint8":
        raise ValueError("renderer must be uint8 PIL Lanczos")
    model = payload["model"]
    if model.get("family") != "vanilla_resnet18" or model.get("output_space") != "normalized" or int(model.get("head_dim", -1)) != 2:
        raise ValueError("model family/output_space/head_dim drifted")
    if tuple(payload["regimes"]) != REGIME_NAMES:
        raise ValueError("regimes must be frozen_feature/head_only/full")
    training = payload["training"]
    expected_training = {
        "steps": 3000,
        "batch_size": 64,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }
    for name, expected in expected_training.items():
        actual = training.get(name)
        if actual != expected:
            raise ValueError(f"training.{name} drifted: {actual!r} != {expected!r}")
    if training.get("dtype") != "fp32" or bool(training.get("amp")):
        raise ValueError("training must be fp32 with AMP disabled")
    if float(training["loss"]["mse_weight"]) != 1.0 or float(training["loss"]["l1_weight"]) != 0.25:
        raise ValueError("loss must be MSE + 0.25*L1")
    scheduler = training["scheduler"]
    if scheduler.get("name") != "CosineAnnealingLR" or float(scheduler["eta_min"]) != 1e-5:
        raise ValueError("scheduler name/eta_min drifted")
    if int(training.get("anchor_eval_every", -1)) != 100 or int(training.get("resume_every", -1)) != 200:
        raise ValueError("anchor_eval_every/resume_every drifted")
    evaluation = payload["evaluation"]
    if int(evaluation["dense_n"]) != 41 or int(evaluation.get("batch_size", -1)) != 64 or evaluation.get("preprocess") != "uint8_to_float32_then_divide_255":
        raise ValueError("evaluation preprocessing/dense grid drifted")
    gates = payload["gates"]
    expected_gates = {
        "support_mae_px": 0.25,
        "crossover_gap_px": 2.0,
        "crossover_spearman": -0.8,
        "minimum_positive_seeds": 2,
        "minimum_negative_seeds": 2,
    }
    for name, expected in expected_gates.items():
        actual = gates.get(name)
        if actual != expected:
            raise ValueError(f"gates.{name} drifted: {actual!r} != {expected!r}")
    return dict(payload)


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("protocol root must be an object")
    return validate_protocol(value)


def protocol_hash(path: str | Path | None = None, payload: Mapping[str, Any] | None = None) -> str:
    value = load_protocol(path) if payload is None else validate_protocol(payload)
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "EXPECTED_SEEDS",
    "REGIME_NAMES",
    "SUPPORT_NAMES",
    "canonical_json_bytes",
    "load_protocol",
    "protocol_hash",
    "validate_protocol",
    "write_json",
]
