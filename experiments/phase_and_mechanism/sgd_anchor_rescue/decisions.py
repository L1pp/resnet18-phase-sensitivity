"""Deterministic selection, gate, checkpoint, and outcome decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .constants import (
    ANCHOR_CHECK_INTERVAL,
    EXTENDED_TOTAL_STEPS,
    TAIL_RELATIVE_TOLERANCE,
    TAIL_STEPS_20K,
    TAIL_STEPS_40K,
    TOTAL_STEPS,
)
from .metrics import CheckpointMetrics, as_checkpoint_metrics, is_finite_number, normalize_checkpoints


class RemediationAction(str, Enum):
    """The only allowed transition after the 5k/20k decision points."""

    NO_REMEDIATION = "no_remediation"
    EXTEND_TO_40K = "extend_to_40k"
    LOW_LR_RESTART = "low_lr_restart"
    OPTIMIZATION_PATHOLOGY = "optimization_pathology"


class OutcomeCategory(str, Enum):
    """Pre-registered machine-level conclusion categories."""

    STRONG_DRIFT = "strong_drift"
    PRESERVE_ORIGINAL_FIELD = "preserve_original_field"
    OPTIMIZATION_PATHOLOGY = "optimization_pathology"
    INCONCLUSIVE = "inconclusive"


class FinalAnchorGateCategory(str, Enum):
    """Disposition after the single permitted rescue run."""

    CONTINUE_DENSE_REVEAL = "continue_dense_reveal"
    OPTIMIZATION_PATHOLOGY = "optimization_pathology"


@dataclass(frozen=True)
class SelectionDecision:
    winner: str | None
    eligible: tuple[str, ...]
    excluded: Mapping[str, str]
    sort_keys: Mapping[str, tuple[float, float, int]]

    @property
    def stable_candidate_exists(self) -> bool:
        return bool(self.eligible)

    def as_dict(self) -> dict[str, object]:
        return {
            "winner": self.winner,
            "eligible": list(self.eligible),
            "excluded": dict(self.excluded),
            "sort_keys": {name: list(key) for name, key in self.sort_keys.items()},
        }


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _candidate_metrics(value: Any) -> Mapping[str, Any] | CheckpointMetrics:
    if isinstance(value, Mapping) or isinstance(value, CheckpointMetrics):
        return value
    raise TypeError("candidate 5k metrics must be mappings or CheckpointMetrics")


def select_5k_candidate(
    results: Mapping[str, Mapping[str, Any] | CheckpointMetrics],
) -> SelectionDecision:
    """Select A/B using the protocol's fixed deterministic lexicographic rule.

    The rule is exactly: remove non-finite rows; require
    ``anchor_5000 < anchor_1000``; sort by ``anchor_5000``, then
    ``objective_5000``; break an exact tie in favor of A.
    """

    eligible: list[tuple[str, float, float]] = []
    excluded: dict[str, str] = {}
    sort_keys: dict[str, tuple[float, float, int]] = {}
    for name in ("A", "B"):
        if name not in results:
            excluded[name] = "missing_candidate"
            continue
        value = _candidate_metrics(results[name])
        if isinstance(value, CheckpointMetrics):
            # A single CheckpointMetrics is useful for callers that already
            # computed the 5k point but has no 1k comparison; reject it
            # explicitly rather than silently treating a missing value as zero.
            excluded[name] = "missing_anchor_1000_or_objective_5000"
            continue
        anchor_5k = _mapping_value(value, "anchor_5000", "anchor5k", "anchor_5k")
        anchor_1k = _mapping_value(value, "anchor_1000", "anchor1k", "anchor_1k")
        objective_5k = _mapping_value(
            value, "objective_5000", "objective5k", "objective_5k"
        )
        if not all(is_finite_number(item) for item in (anchor_5k, anchor_1k, objective_5k)):
            excluded[name] = "non_finite_or_missing_metric"
            continue
        anchor_5k = float(anchor_5k)
        anchor_1k = float(anchor_1k)
        objective_5k = float(objective_5k)
        if not anchor_5k < anchor_1k:
            excluded[name] = "anchor_5000_not_below_anchor_1000"
            continue
        rank = 0 if name == "A" else 1
        sort_keys[name] = (anchor_5k, objective_5k, rank)
        eligible.append((name, anchor_5k, objective_5k))
    eligible.sort(key=lambda item: (item[1], item[2], 0 if item[0] == "A" else 1))
    return SelectionDecision(
        winner=eligible[0][0] if eligible else None,
        eligible=tuple(item[0] for item in eligible),
        excluded=excluded,
        sort_keys=sort_keys,
    )


def _require_complete_5k_results(
    results: Mapping[str, Mapping[str, Any] | CheckpointMetrics],
) -> None:
    """Reject incomplete A/B 5k inputs before the low-LR decision branch."""

    required = ("anchor_1000", "anchor_5000", "objective_5000")
    errors: list[str] = []
    for name in ("A", "B"):
        if name not in results:
            errors.append(f"missing candidate {name}")
            continue
        value = results[name]
        if not isinstance(value, Mapping):
            errors.append(f"candidate {name} must provide a metric mapping")
            continue
        for key in required:
            if key not in value or not is_finite_number(value[key]):
                errors.append(f"candidate {name} missing/non-finite {key}")
    if errors:
        raise ValueError("incomplete 5k candidate inputs: " + "; ".join(errors))


@dataclass(frozen=True)
class GateEvaluation:
    threshold_px: float
    endpoint_step: int | None
    endpoint_anchor: float | None
    tail_steps: tuple[int, ...]
    tail_anchors: tuple[float | None, ...]
    endpoint_pass: bool
    tail_pass: bool
    tail_close: bool
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold_px": self.threshold_px,
            "endpoint_step": self.endpoint_step,
            "endpoint_anchor": self.endpoint_anchor,
            "tail_steps": list(self.tail_steps),
            "tail_anchors": list(self.tail_anchors),
            "endpoint_pass": self.endpoint_pass,
            "tail_pass": self.tail_pass,
            "tail_close": self.tail_close,
            "passed": self.passed,
        }


def strict_gate(
    checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    threshold_px: float,
    *,
    endpoint_step: int | None = None,
    tail_steps: Sequence[int] | None = None,
    tail_relative_tolerance: float = TAIL_RELATIVE_TOLERANCE,
) -> GateEvaluation:
    """Evaluate endpoint and the final fixed checkpoints.

    The protocol requires both tail values to be finite and below the
    threshold, with an absolute tail difference no larger than
    ``0.20 * threshold`` by default.  A caller may provide a different
    non-negative relative tolerance for an explicitly documented diagnostic.
    """

    if not is_finite_number(threshold_px) or float(threshold_px) < 0:
        raise ValueError("threshold_px must be finite and non-negative")
    normalized = normalize_checkpoints(checkpoints)
    if not normalized:
        return GateEvaluation(
            threshold_px=float(threshold_px),
            endpoint_step=None,
            endpoint_anchor=None,
            tail_steps=tuple(tail_steps or ()),
            tail_anchors=tuple(None for _ in (tail_steps or ())),
            endpoint_pass=False,
            tail_pass=False,
            tail_close=False,
            passed=False,
        )
    resolved_endpoint = max(normalized) if endpoint_step is None else endpoint_step
    endpoint_point = normalized.get(resolved_endpoint)
    endpoint_anchor = endpoint_point.anchor if endpoint_point is not None else None
    endpoint_pass = endpoint_point is not None and is_finite_number(endpoint_anchor) and endpoint_anchor <= float(threshold_px)
    resolved_tail = tuple(tail_steps or ((TAIL_STEPS_40K if resolved_endpoint >= EXTENDED_TOTAL_STEPS else TAIL_STEPS_20K)))
    tail_anchors = tuple(
        normalized[step].anchor if step in normalized else None for step in resolved_tail
    )
    tail_pass = bool(tail_anchors) and all(
        is_finite_number(anchor) and float(anchor) <= float(threshold_px)
        for anchor in tail_anchors
    )
    finite_tail = [float(anchor) for anchor in tail_anchors if is_finite_number(anchor)]
    if not is_finite_number(tail_relative_tolerance) or float(tail_relative_tolerance) < 0:
        raise ValueError("tail_relative_tolerance must be finite and non-negative")
    if len(finite_tail) != len(tail_anchors) or not finite_tail:
        tail_close = False
    else:
        tail_close = max(finite_tail) - min(finite_tail) <= float(tail_relative_tolerance) * float(threshold_px)
    return GateEvaluation(
        threshold_px=float(threshold_px),
        endpoint_step=resolved_endpoint,
        endpoint_anchor=(None if endpoint_anchor is None else float(endpoint_anchor)),
        tail_steps=resolved_tail,
        tail_anchors=tuple(None if value is None else float(value) for value in tail_anchors),
        endpoint_pass=bool(endpoint_pass),
        tail_pass=bool(tail_pass),
        tail_close=bool(tail_close),
        passed=bool(endpoint_pass and tail_pass and tail_close),
    )


def anchor_gate_pass(
    checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    threshold_px: float,
    **kwargs: Any,
) -> bool:
    """Boolean convenience wrapper around :func:`strict_gate`."""

    return strict_gate(checkpoints, threshold_px, **kwargs).passed


def classify_final_anchor_gate(
    checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    threshold_px: float,
    *,
    endpoint_step: int | None = None,
    tail_steps: Sequence[int] | None = None,
    tail_relative_tolerance: float = TAIL_RELATIVE_TOLERANCE,
) -> FinalAnchorGateCategory:
    """Classify the final checkpoint after the one allowed rescue run.

    This function intentionally has only two dispositions: a strict final
    gate pass authorizes the already-frozen dense reveal, while every failed
    final gate is optimization pathology.  In particular, a failed low-LR
    restart must not fall through to the generic ``inconclusive`` category.
    """

    gate = strict_gate(
        checkpoints,
        threshold_px,
        endpoint_step=endpoint_step,
        tail_steps=tail_steps,
        tail_relative_tolerance=tail_relative_tolerance,
    )
    if gate.passed:
        return FinalAnchorGateCategory.CONTINUE_DENSE_REVEAL
    return FinalAnchorGateCategory.OPTIMIZATION_PATHOLOGY


# Explicit name for callers handling the one-shot rescue branch.  It keeps
# rescue failure semantics separate from the later four-way dense-field
# classification.
classify_final_rescue_result = classify_final_anchor_gate


@dataclass(frozen=True)
class RemediationDecision:
    action: RemediationAction
    selected_candidate: str | None
    reason: str
    gate: GateEvaluation | None
    allowed_endpoint_step: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "selected_candidate": self.selected_candidate,
            "reason": self.reason,
            "gate": None if self.gate is None else self.gate.as_dict(),
            "allowed_endpoint_step": self.allowed_endpoint_step,
        }


def decide_20k_remediation(
    five_k_results: Mapping[str, Mapping[str, Any] | CheckpointMetrics],
    selected_checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    threshold_px: float,
    *,
    selected_candidate: str | None = None,
) -> RemediationDecision:
    """Apply the one-shot 5k/20k remediation decision tree.

    If neither candidate is stable at 5k, the only allowed branch is a fresh
    low-LR restart.  Otherwise a failed 20k gate may extend only when
    ``anchor_20000 < anchor_15000``; all other failures stop as optimization
    pathology.
    """

    _require_complete_5k_results(five_k_results)
    selection = select_5k_candidate(five_k_results)
    candidate = selection.winner if selected_candidate is None else selected_candidate
    if candidate is None:
        return RemediationDecision(
            action=RemediationAction.LOW_LR_RESTART,
            selected_candidate=None,
            reason="no_candidate_satisfied_5k_stability_rule",
            gate=None,
            allowed_endpoint_step=TOTAL_STEPS,
        )
    if selected_candidate is not None and selected_candidate != selection.winner:
        raise ValueError(
            "selected_candidate must equal the deterministic 5k selection winner"
        )
    if candidate not in ("A", "B") or candidate != selection.winner:
        raise ValueError("selected_candidate must be the eligible deterministic winner")
    gate = strict_gate(selected_checkpoints, threshold_px, endpoint_step=TOTAL_STEPS, tail_steps=TAIL_STEPS_20K)
    if gate.passed:
        return RemediationDecision(
            action=RemediationAction.NO_REMEDIATION,
            selected_candidate=candidate,
            reason="strict_20k_anchor_gate_passed",
            gate=gate,
            allowed_endpoint_step=TOTAL_STEPS,
        )
    normalized = normalize_checkpoints(selected_checkpoints)
    anchor_15k = normalized.get(15_000)
    anchor_20k = normalized.get(20_000)
    if (
        anchor_15k is not None
        and anchor_20k is not None
        and is_finite_number(anchor_15k.anchor)
        and is_finite_number(anchor_20k.anchor)
        and anchor_20k.anchor < anchor_15k.anchor
    ):
        return RemediationDecision(
            action=RemediationAction.EXTEND_TO_40K,
            selected_candidate=candidate,
            reason="20k_gate_failed_but_anchor_is_still_decreasing",
            gate=gate,
            allowed_endpoint_step=EXTENDED_TOTAL_STEPS,
        )
    return RemediationDecision(
        action=RemediationAction.OPTIMIZATION_PATHOLOGY,
        selected_candidate=candidate,
        reason="20k_gate_failed_and_anchor_is_not_still_decreasing",
        gate=gate,
        allowed_endpoint_step=None,
    )


def select_endpoint(
    checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    *,
    endpoint_step: int | None = None,
) -> CheckpointMetrics:
    """Select the final training endpoint, never the min-anchor checkpoint."""

    normalized = normalize_checkpoints(checkpoints)
    if not normalized:
        raise ValueError("cannot select endpoint from empty checkpoints")
    step = max(normalized) if endpoint_step is None else endpoint_step
    if step not in normalized:
        raise KeyError(f"endpoint step {step} is not present")
    return normalized[step]


def select_first_match(
    checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    threshold_px: float,
    *,
    check_steps: Sequence[int] | None = None,
) -> CheckpointMetrics | None:
    """Return the earliest checked model whose anchor reaches the gate."""

    normalized = normalize_checkpoints(checkpoints)
    # The protocol's first-match sequence is the full-train check stream, not
    # the irregular fixed state-save points (1, 5, 10, 50, ...).  A caller
    # wanting a different explicitly pre-registered sequence must pass it.
    steps = (
        tuple(sorted(check_steps))
        if check_steps is not None
        else tuple(step for step in normalized if step % ANCHOR_CHECK_INTERVAL == 0)
    )
    for step in steps:
        point = normalized.get(step)
        if point is not None and is_finite_number(point.anchor) and point.anchor <= threshold_px:
            return point
    return None


def select_min_anchor(
    checkpoints: Mapping[int | str, CheckpointMetrics | Mapping[str, Any]]
    | Sequence[CheckpointMetrics | Mapping[str, Any]],
    *,
    check_steps: Sequence[int] | None = None,
) -> CheckpointMetrics:
    """Return the lowest-anchor checked point, breaking ties by earliest step."""

    normalized = normalize_checkpoints(checkpoints)
    if check_steps is None:
        points = tuple(normalized.values())
    else:
        points = tuple(normalized[step] for step in sorted(check_steps) if step in normalized)
    finite_points = tuple(point for point in points if is_finite_number(point.anchor))
    if not finite_points:
        raise ValueError("no finite anchor checkpoint available")
    return min(finite_points, key=lambda point: (float(point.anchor), point.step))


def _dense_point_is_finite(point: CheckpointMetrics) -> bool:
    return is_finite_number(point.u) and is_finite_number(point.raw_box)


def classify_outcome(
    *,
    endpoint: CheckpointMetrics | Mapping[str, Any] | None,
    first_match: CheckpointMetrics | Mapping[str, Any] | None,
    u0: float,
    raw_box0: float,
    adamw_u: float,
    threshold_px: float,
    final_gate: GateEvaluation | Mapping[str, Any] | None = None,
    anchor_matched: bool | None = None,
    remediation_action: RemediationAction | str | None = None,
) -> OutcomeCategory:
    """Apply the four pre-registered outcome classifications.

    Strong drift requires both endpoint and first-match to show the twofold
    residual and +1px raw-box criteria, plus the endpoint half-AdamW criterion.
    Preserve-original-field requires both points to stay within 1.5x ``u0``.
    Any failed one-shot rescue is classified as optimization pathology;
    missing/non-finite dense evidence is inconclusive.  ``anchor_matched`` is
    retained as a compatibility field but is deliberately ignored: the
    endpoint and first-match anchor values, plus an optional final gate, are
    the source of truth.
    """

    if remediation_action is not None:
        action = RemediationAction(remediation_action)
        if action == RemediationAction.OPTIMIZATION_PATHOLOGY:
            return OutcomeCategory.OPTIMIZATION_PATHOLOGY
    if endpoint is None or first_match is None:
        return OutcomeCategory.INCONCLUSIVE
    endpoint_point = as_checkpoint_metrics(endpoint)
    first_point = as_checkpoint_metrics(first_match)
    if not all(is_finite_number(value) for value in (threshold_px, u0, raw_box0, adamw_u)):
        return OutcomeCategory.INCONCLUSIVE
    threshold_px = float(threshold_px)
    if threshold_px < 0:
        return OutcomeCategory.INCONCLUSIVE
    anchor_values = (endpoint_point.anchor, first_point.anchor)
    if not all(
        is_finite_number(value) and float(value) <= threshold_px for value in anchor_values
    ):
        return OutcomeCategory.INCONCLUSIVE
    if final_gate is not None:
        if isinstance(final_gate, GateEvaluation):
            final_gate_passed = final_gate.passed
        elif isinstance(final_gate, Mapping):
            final_gate_passed = final_gate.get("passed") is True
        else:
            raise TypeError("final_gate must be GateEvaluation, mapping, or None")
        if not final_gate_passed:
            return OutcomeCategory.INCONCLUSIVE
    if not _dense_point_is_finite(endpoint_point) or not _dense_point_is_finite(first_point):
        return OutcomeCategory.INCONCLUSIVE
    u0 = float(u0)
    raw_box0 = float(raw_box0)
    adamw_u = float(adamw_u)
    endpoint_u = float(endpoint_point.u)
    first_u = float(first_point.u)
    strong = (
        endpoint_u >= 2.0 * u0
        and first_u >= 2.0 * u0
        and float(endpoint_point.raw_box) >= raw_box0 + 1.0
        and float(first_point.raw_box) >= raw_box0 + 1.0
        and endpoint_u >= 0.5 * adamw_u
    )
    if strong:
        return OutcomeCategory.STRONG_DRIFT
    keep = endpoint_u <= 1.5 * u0 and first_u <= 1.5 * u0
    if keep:
        return OutcomeCategory.PRESERVE_ORIGINAL_FIELD
    return OutcomeCategory.INCONCLUSIVE


__all__ = [
    "FinalAnchorGateCategory",
    "GateEvaluation",
    "OutcomeCategory",
    "RemediationAction",
    "RemediationDecision",
    "SelectionDecision",
    "anchor_gate_pass",
    "classify_outcome",
    "classify_final_anchor_gate",
    "classify_final_rescue_result",
    "decide_20k_remediation",
    "select_5k_candidate",
    "select_endpoint",
    "select_first_match",
    "select_min_anchor",
    "strict_gate",
]
