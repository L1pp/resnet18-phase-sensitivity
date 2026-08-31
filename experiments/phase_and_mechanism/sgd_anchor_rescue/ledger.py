"""Pure run-ledger and optimizer-step budget validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .constants import MAX_OPTIMIZER_STEPS_PER_MACHINE


@dataclass(frozen=True)
class LedgerEntry:
    """One bounded optimizer run segment, independent of machine details."""

    run_id: str
    candidate: str
    phase: str
    optimizer_steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "candidate": self.candidate,
            "phase": self.phase,
            "optimizer_steps": self.optimizer_steps,
        }


def validate_run_budget(
    entries: Sequence[LedgerEntry],
    *,
    max_steps: int = MAX_OPTIMIZER_STEPS_PER_MACHINE,
) -> tuple[str, ...]:
    """Return deterministic errors for duplicate/invalid/over-budget runs."""

    errors: list[str] = []
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        return ("max_steps must be a non-negative integer",)
    seen: set[str] = set()
    total = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, LedgerEntry):
            errors.append(f"entry[{index}] must be LedgerEntry")
            continue
        if not entry.run_id:
            errors.append(f"entry[{index}] run_id must be non-empty")
        elif entry.run_id in seen:
            errors.append(f"duplicate run_id: {entry.run_id}")
        seen.add(entry.run_id)
        if isinstance(entry.optimizer_steps, bool) or not isinstance(entry.optimizer_steps, int):
            errors.append(f"entry[{index}] optimizer_steps must be integer")
        elif entry.optimizer_steps < 0:
            errors.append(f"entry[{index}] optimizer_steps must be non-negative")
        else:
            total += entry.optimizer_steps
    if total > max_steps:
        errors.append(f"optimizer step budget exceeded: {total} > {max_steps}")
    return tuple(errors)


@dataclass(frozen=True)
class RunLedger:
    """Immutable append-only ledger capped at the protocol's 45k steps."""

    entries: tuple[LedgerEntry, ...] = ()
    max_steps: int = MAX_OPTIMIZER_STEPS_PER_MACHINE

    @property
    def total_optimizer_steps(self) -> int:
        return sum(entry.optimizer_steps for entry in self.entries)

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - self.total_optimizer_steps

    def validate(self) -> tuple[str, ...]:
        return validate_run_budget(self.entries, max_steps=self.max_steps)

    def append(
        self,
        *,
        run_id: str,
        candidate: str,
        phase: str,
        optimizer_steps: int,
    ) -> "RunLedger":
        """Return a new ledger, rejecting any append over the fixed budget."""

        entry = LedgerEntry(
            run_id=run_id,
            candidate=candidate,
            phase=phase,
            optimizer_steps=optimizer_steps,
        )
        proposed = replace(self, entries=self.entries + (entry,))
        errors = proposed.validate()
        if errors:
            raise ValueError("invalid run ledger: " + "; ".join(errors))
        return proposed

    add = append

    def as_dict(self) -> dict[str, object]:
        return {
            "max_steps": self.max_steps,
            "total_optimizer_steps": self.total_optimizer_steps,
            "remaining_steps": self.remaining_steps,
            "entries": [entry.as_dict() for entry in self.entries],
        }


__all__ = ["LedgerEntry", "RunLedger", "validate_run_budget"]
