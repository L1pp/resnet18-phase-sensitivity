"""Small, serializable metric records used by the protocol logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CheckpointMetrics:
    """Metrics belonging to one saved model/checkpoint.

    ``anchor`` and ``objective`` are the only fields allowed in stage 1.  The
    dense fields are optional because they are only populated during the
    stage-2 reveal.
    """

    step: int
    anchor: float
    objective: float | None = None
    raw_box: float | None = None
    u: float | None = None

    def as_dict(self, *, include_dense: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "step": self.step,
            "anchor": self.anchor,
        }
        if self.objective is not None:
            result["objective"] = self.objective
        if include_dense:
            if self.raw_box is not None:
                result["raw_box"] = self.raw_box
            if self.u is not None:
                result["u"] = self.u
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_step: int | None = None) -> "CheckpointMetrics":
        if "step" in value:
            step = int(value["step"])
        elif default_step is not None:
            step = int(default_step)
        else:
            raise ValueError("metric mapping is missing step")
        if "anchor" not in value:
            raise ValueError("metric mapping is missing anchor")
        return cls(
            step=step,
            anchor=float(value["anchor"]),
            objective=(None if value.get("objective") is None else float(value["objective"])),
            raw_box=(None if value.get("raw_box") is None else float(value["raw_box"])),
            u=(None if value.get("u") is None else float(value["u"])),
        )


def as_checkpoint_metrics(value: CheckpointMetrics | Mapping[str, Any], *, default_step: int | None = None) -> CheckpointMetrics:
    """Normalize a dataclass or JSON-like metric mapping."""

    if isinstance(value, CheckpointMetrics):
        return value
    if isinstance(value, Mapping):
        return CheckpointMetrics.from_mapping(value, default_step=default_step)
    raise TypeError("metrics must be CheckpointMetrics or a mapping")


def normalize_checkpoints(
    values: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | list[CheckpointMetrics | Mapping[str, Any]],
) -> dict[int, CheckpointMetrics]:
    """Normalize a step mapping/list and reject duplicate or invalid steps."""

    result: dict[int, CheckpointMetrics] = {}
    if isinstance(values, Mapping):
        iterable = values.items()
        for raw_step, value in iterable:
            step = int(raw_step)
            point = as_checkpoint_metrics(value, default_step=step)
            if point.step != step:
                raise ValueError(f"checkpoint key {step} disagrees with metric step {point.step}")
            if step < 0:
                raise ValueError("checkpoint steps must be non-negative")
            if step in result:
                raise ValueError(f"duplicate checkpoint step {step}")
            result[step] = point
        return dict(sorted(result.items()))
    for value in values:
        point = as_checkpoint_metrics(value)
        if point.step < 0:
            raise ValueError("checkpoint steps must be non-negative")
        if point.step in result:
            raise ValueError(f"duplicate checkpoint step {point.step}")
        result[point.step] = point
    return dict(sorted(result.items()))


def is_finite_number(value: object) -> bool:
    """Return true only for real finite numeric values (bool is rejected)."""

    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


__all__ = [
    "CheckpointMetrics",
    "as_checkpoint_metrics",
    "is_finite_number",
    "normalize_checkpoints",
]
