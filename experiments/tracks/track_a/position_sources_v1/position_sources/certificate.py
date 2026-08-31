"""Symbolic receptive-field certificate for the valid-core model."""

from __future__ import annotations

from typing import Any

from .models import build_model


def _out_size(size: int, *, kernel: int, stride: int, padding: int) -> int:
    return (int(size) + 2 * int(padding) - int(kernel)) // int(stride) + 1


def _feature_shape(model: Any, input_size: int) -> tuple[int, int]:
    size = int(input_size)
    for kernel, stride, padding in model._rf_ops:
        size = _out_size(size, kernel=kernel, stride=stride, padding=padding)
        if size <= 0:
            raise ValueError(f"symbolic feature map collapsed at {kernel, stride, padding}")
    return size, size


def valid_core_certificate(*, input_size: int = 1024) -> dict[str, Any]:
    model = build_model("valid_core_s32", input_size=input_size)
    feature_height, feature_width = _feature_shape(model, input_size)
    info = model.valid_core_info(input_size=input_size, feature_shape=(feature_height, feature_width))
    height_indices = list(info["height_indices"])
    width_indices = list(info["width_indices"])
    certificate = {
        "schema_version": 1,
        "kind": "symbolic_receptive_field_certificate",
        "variant": "valid_core_s32",
        "input_size": int(input_size),
        "layer4_feature_shape": [feature_height, feature_width],
        "rf_ops": [list(map(int, op)) for op in model._rf_ops],
        "receptive_field_size": float(info["receptive_field"]),
        "jump": float(info["jump"]),
        "start": float(info["start"]),
        "core_indices": {"height": height_indices, "width": width_indices},
        "core_shape": [len(height_indices), len(width_indices)],
        "complete_rf_inside_input": bool(height_indices and width_indices),
        "target_core_shape": [19, 19],
    }
    expected_feature_shape = [32, 32]
    expected_rf_size = 435.0
    expected_jump = 32.0
    expected_core_shape = [19, 19]
    certificate["checks"] = {
        "feature_shape_expected": expected_feature_shape,
        "rf_size_expected": expected_rf_size,
        "jump_expected": expected_jump,
        "core_shape_expected": expected_core_shape,
        "passed": (
            certificate["layer4_feature_shape"] == expected_feature_shape
            and certificate["receptive_field_size"] == expected_rf_size
            and certificate["jump"] == expected_jump
            and certificate["core_shape"] == expected_core_shape
            and certificate["complete_rf_inside_input"]
        ),
    }
    if not certificate["checks"]["passed"]:
        raise AssertionError(f"valid-core certificate failed: {certificate}")
    return certificate


__all__ = ["valid_core_certificate"]
