"""Pure learning-rate and checkpoint scheduling helpers.

The functions here only calculate numbers.  They do not import torch and do
not create an optimizer, so an execution AI can use them in a remote runner
without coupling this package to a particular CUDA/PyTorch installation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from .constants import (
    ANCHOR_CHECK_INTERVAL,
    CANDIDATES,
    ETA_MIN,
    EXTENDED_TOTAL_STEPS,
    FIXED_STATE_STEPS,
    LOW_LR_PEAK_LR,
    LOW_LR_WARMUP_STEPS,
    PEAK_LR,
    TOTAL_STEPS,
)


@dataclass(frozen=True)
class ScheduleConfig:
    """Parameters for warmup + cosine decay, including extension behavior."""

    peak_lr: float = PEAK_LR
    warmup_steps: int = 500
    total_steps: int = TOTAL_STEPS
    eta_min: float = ETA_MIN
    extension_steps: int = EXTENDED_TOTAL_STEPS

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not math.isfinite(self.peak_lr) or self.peak_lr <= 0:
            errors.append("peak_lr must be finite and positive")
        if not math.isfinite(self.eta_min) or self.eta_min < 0:
            errors.append("eta_min must be finite and non-negative")
        if self.warmup_steps <= 0:
            errors.append("warmup_steps must be positive")
        if self.total_steps <= self.warmup_steps:
            errors.append("total_steps must be greater than warmup_steps")
        if self.extension_steps < self.total_steps:
            errors.append("extension_steps must be >= total_steps")
        return tuple(errors)


def _require_valid_config(config: ScheduleConfig) -> None:
    errors = config.validate()
    if errors:
        raise ValueError("invalid schedule: " + "; ".join(errors))


def linear_warmup_cosine_lr(
    step: int,
    *,
    peak_lr: float = PEAK_LR,
    warmup_steps: int = 500,
    total_steps: int = TOTAL_STEPS,
    eta_min: float = ETA_MIN,
) -> float:
    """Return the protocol LR at a zero-based optimizer step.

    Step ``0`` has LR zero.  The warmup reaches ``peak_lr`` at
    ``warmup_steps``.  Cosine decay then reaches ``eta_min`` at
    ``total_steps``.  Any step after ``total_steps`` is held at ``eta_min``;
    this is the prescribed 20k-to-40k extension behavior.
    """

    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
    config = ScheduleConfig(
        peak_lr=peak_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=eta_min,
        extension_steps=total_steps,
    )
    _require_valid_config(config)
    if step == 0:
        return 0.0
    if step <= warmup_steps:
        return peak_lr * (step / warmup_steps)
    if step >= total_steps:
        return float(eta_min)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min + (peak_lr - eta_min) * cosine


def schedule_lr(step: int, config: ScheduleConfig) -> float:
    """Apply :func:`linear_warmup_cosine_lr` and extension hold semantics."""

    _require_valid_config(config)
    # The base function already holds eta_min after total_steps.  Keeping the
    # explicit branch documents that 20k-to-40k extension does not decay again.
    if step > config.total_steps:
        if step > config.extension_steps:
            # The protocol never trains beyond extension_steps, but returning
            # eta_min remains a safe pure-function behavior for validation.
            return float(config.eta_min)
        return float(config.eta_min)
    return linear_warmup_cosine_lr(
        step,
        peak_lr=config.peak_lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        eta_min=config.eta_min,
    )


def candidate_schedule(name: str) -> ScheduleConfig:
    """Return the frozen 20k schedule for candidate ``A`` or ``B``."""

    try:
        candidate = CANDIDATES[name]
    except KeyError as exc:
        raise ValueError("candidate must be A or B") from exc
    return ScheduleConfig(
        peak_lr=candidate.peak_lr,
        warmup_steps=candidate.warmup_steps,
        total_steps=candidate.total_steps,
        eta_min=candidate.eta_min,
        extension_steps=EXTENDED_TOTAL_STEPS,
    )


def low_lr_rescue_schedule() -> ScheduleConfig:
    """Return the one allowed low-LR/longer-warmup restart schedule."""

    return ScheduleConfig(
        peak_lr=LOW_LR_PEAK_LR,
        warmup_steps=LOW_LR_WARMUP_STEPS,
        total_steps=TOTAL_STEPS,
        eta_min=ETA_MIN,
        extension_steps=TOTAL_STEPS,
    )


def full_train_anchor_steps(
    max_step: int = TOTAL_STEPS,
    *,
    interval: int = ANCHOR_CHECK_INTERVAL,
    include_fixed: bool = True,
) -> tuple[int, ...]:
    """Return the prescribed full-train anchor check sequence.

    Checks occur every ``interval`` steps, with protocol fixed points added
    even when they are not multiples of the interval (for example step 1).
    The result is sorted and unique.  The function is intentionally usable for
    a 40k extension as well as the regular 20k run.
    """

    if not isinstance(max_step, int) or isinstance(max_step, bool) or max_step < 0:
        raise ValueError("max_step must be a non-negative integer")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("interval must be a positive integer")
    steps = set(range(0, max_step + 1, interval))
    if include_fixed:
        steps.update(step for step in FIXED_STATE_STEPS if step <= max_step)
    return tuple(sorted(steps))


def fixed_state_steps(max_step: int = TOTAL_STEPS) -> tuple[int, ...]:
    """Return state-save points, including the 39k/40k extension tail."""

    if max_step < 0:
        raise ValueError("max_step must be non-negative")
    steps = {step for step in FIXED_STATE_STEPS if step <= max_step}
    if max_step >= 40_000:
        steps.update((39_000, 40_000))
    return tuple(sorted(steps))


def extension_uses_eta_min(config: ScheduleConfig, step: int) -> bool:
    """Whether ``step`` is in the fixed-eta-min extension portion."""

    _require_valid_config(config)
    return step > config.total_steps and step <= config.extension_steps


__all__ = [
    "ScheduleConfig",
    "candidate_schedule",
    "extension_uses_eta_min",
    "fixed_state_steps",
    "full_train_anchor_steps",
    "linear_warmup_cosine_lr",
    "low_lr_rescue_schedule",
    "schedule_lr",
]
