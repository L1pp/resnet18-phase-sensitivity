"""Frozen, cloud-agnostic constants for the SGD anchor-rescue protocol.

This module intentionally contains no model, CUDA, filesystem, or remote
machine assumptions.  The execution AI supplies those details when it builds
an experiment manifest on the target machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


PROTOCOL_NAME = "sgd_anchor_rescue"
PROTOCOL_VERSION = "1.0"
PROTOCOL_STATUS = "prepared_not_executed"

# The two candidates differ only in momentum.  Keeping these values in one
# place prevents a remote runner from silently changing one candidate only.
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
PEAK_LR = 1e-3
ETA_MIN = 1e-5
WARMUP_STEPS = 500
TOTAL_STEPS = 20_000
EXTENDED_TOTAL_STEPS = 40_000
LOW_LR_PEAK_LR = 3e-4
LOW_LR_WARMUP_STEPS = 1_000
ANCHOR_CHECK_INTERVAL = 200
SELECTION_STEP = 5_000
PRECHECK_STEP = 1_000
TAIL_STEPS_20K = (19_000, 20_000)
TAIL_STEPS_40K = (39_000, 40_000)
# The final two fixed checkpoints must be close as well as below the anchor
# threshold.  This is an absolute difference expressed relative to the gate.
TAIL_RELATIVE_TOLERANCE = 0.20
MAX_OPTIMIZER_STEPS_PER_MACHINE = 45_000

# Fixed state points are part of the protocol, not a suggestion to save a
# dense model at every point.  The runner may use a rolling min-anchor model.
FIXED_STATE_STEPS = (
    0,
    1,
    5,
    10,
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    15_000,
    18_000,
    19_000,
    20_000,
)

# Stage 1 may only expose on-support values.  This is a deny-list rather than
# a complete allow-list so that adding a harmless scalar to a manifest does
# not accidentally weaken the guard.
FORBIDDEN_STAGE1_KEY_TOKENS = frozenset(
    {
        "dense",
        "field",
        "raw_box",
        "rawbox",
        "affine_removed",
        "affineremoved",
        "residual_u",
        "dense_box",
        "dense_field",
        "off_support",
        "offsupport",
        "residual",
        "residual_u",
        "u",
        "u_metric",
        "u0",
        "u_adamw",
        "image",
        "image_ids",
        "plot",
    }
)


@dataclass(frozen=True)
class CandidateConfig:
    """A complete optimizer schedule independent of a cloud environment."""

    name: str
    optimizer: str = "SGD"
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    peak_lr: float = PEAK_LR
    warmup_steps: int = WARMUP_STEPS
    total_steps: int = TOTAL_STEPS
    eta_min: float = ETA_MIN
    batch_size: int = BATCH_SIZE
    weight_decay: float = WEIGHT_DECAY
    amp: bool = False
    loss_name: str = "MSE + 0.25 * L1"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable copy of the frozen configuration."""

        return {
            "name": self.name,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "dampening": self.dampening,
            "nesterov": self.nesterov,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "eta_min": self.eta_min,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "amp": self.amp,
            "loss_name": self.loss_name,
        }


CANDIDATE_A = CandidateConfig(name="A", momentum=0.0)
CANDIDATE_B = CandidateConfig(name="B", momentum=0.9)
CANDIDATES: Mapping[str, CandidateConfig] = MappingProxyType(
    {"A": CANDIDATE_A, "B": CANDIDATE_B}
)


@dataclass(frozen=True)
class GateConfig:
    """Machine-specific canonical anchor gate and drift references.

    The protocol deliberately keeps the dimensions separate: values for AMD
    and A10 must not be averaged or used as a cross-machine threshold.
    """

    device: str
    dimensions: int
    anchor_threshold_px: float
    adamw_u_px: float

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "dimensions": self.dimensions,
            "anchor_threshold_px": self.anchor_threshold_px,
            "adamw_u_px": self.adamw_u_px,
        }


AMD_GATE = GateConfig(
    device="amd",
    dimensions=2,
    anchor_threshold_px=0.10,
    adamw_u_px=32.00699,
)
A10_GATE = GateConfig(
    device="a10",
    dimensions=6,
    anchor_threshold_px=0.17,
    adamw_u_px=6.42162,
)
GATES: Mapping[str, GateConfig] = MappingProxyType(
    {"amd": AMD_GATE, "a10": A10_GATE}
)


def validate_candidate_config(config: CandidateConfig) -> tuple[str, ...]:
    """Return protocol violations for a candidate, without touching a model."""

    errors: list[str] = []
    if config.optimizer != "SGD":
        errors.append("optimizer must be SGD")
    if config.dampening != 0:
        errors.append("dampening must be 0")
    if config.nesterov:
        errors.append("nesterov must be false")
    if config.batch_size != BATCH_SIZE:
        errors.append(f"batch_size must be {BATCH_SIZE}")
    if config.weight_decay != WEIGHT_DECAY:
        errors.append(f"weight_decay must be {WEIGHT_DECAY}")
    if config.eta_min != ETA_MIN:
        errors.append(f"eta_min must be {ETA_MIN}")
    if config.peak_lr != PEAK_LR:
        errors.append(f"peak_lr must be {PEAK_LR}")
    if config.warmup_steps != WARMUP_STEPS:
        errors.append(f"warmup_steps must be {WARMUP_STEPS}")
    if config.total_steps != TOTAL_STEPS:
        errors.append(f"total_steps must be {TOTAL_STEPS}")
    if config.amp is not False:
        errors.append("amp must be false")
    if config.loss_name != "MSE + 0.25 * L1":
        errors.append("loss_name must be 'MSE + 0.25 * L1'")
    if config.name not in CANDIDATES:
        errors.append("candidate name must be A or B")
    else:
        expected_momentum = CANDIDATES[config.name].momentum
        if config.momentum != expected_momentum:
            errors.append(
                f"candidate {config.name} momentum must be {expected_momentum}"
            )
    return tuple(errors)


def validate_protocol_constants() -> tuple[str, ...]:
    """Validate both frozen candidates and the fixed protocol constants."""

    errors: list[str] = []
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        errors.extend(f"{candidate.name}: {error}" for error in validate_candidate_config(candidate))
    if FIXED_STATE_STEPS != tuple(sorted(set(FIXED_STATE_STEPS))):
        errors.append("FIXED_STATE_STEPS must be sorted and unique")
    if FIXED_STATE_STEPS[0] != 0 or FIXED_STATE_STEPS[-1] != TOTAL_STEPS:
        errors.append("FIXED_STATE_STEPS must start at 0 and end at TOTAL_STEPS")
    if SELECTION_STEP != 5_000 or PRECHECK_STEP != 1_000:
        errors.append("selection and precheck steps are protocol-fixed")
    return tuple(errors)


__all__ = [
    "A10_GATE",
    "AMD_GATE",
    "ANCHOR_CHECK_INTERVAL",
    "BATCH_SIZE",
    "CANDIDATE_A",
    "CANDIDATE_B",
    "CANDIDATES",
    "CandidateConfig",
    "ETA_MIN",
    "EXTENDED_TOTAL_STEPS",
    "FIXED_STATE_STEPS",
    "FORBIDDEN_STAGE1_KEY_TOKENS",
    "GATES",
    "GateConfig",
    "LOW_LR_PEAK_LR",
    "LOW_LR_WARMUP_STEPS",
    "MAX_OPTIMIZER_STEPS_PER_MACHINE",
    "PEAK_LR",
    "PRECHECK_STEP",
    "PROTOCOL_NAME",
    "PROTOCOL_STATUS",
    "PROTOCOL_VERSION",
    "SELECTION_STEP",
    "TAIL_STEPS_20K",
    "TAIL_STEPS_40K",
    "TAIL_RELATIVE_TOLERANCE",
    "TOTAL_STEPS",
    "WARMUP_STEPS",
    "WEIGHT_DECAY",
    "validate_candidate_config",
    "validate_protocol_constants",
]
