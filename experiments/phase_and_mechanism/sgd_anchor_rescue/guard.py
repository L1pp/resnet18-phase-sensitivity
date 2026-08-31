"""Guards that seal stage-1 records against dense/off-support leakage."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .constants import FORBIDDEN_STAGE1_KEY_TOKENS


class DenseMetricLeakError(ValueError):
    """Raised when stage 1 attempts to store a dense/off-support field."""


def _normalise_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _key_is_forbidden(key: object) -> bool:
    normalized = _normalise_key(key)
    if normalized in FORBIDDEN_STAGE1_KEY_TOKENS:
        return True
    return any(
        token in normalized
        for token in (
            "dense",
            "field",
            "raw_box",
            "rawbox",
            "affine_removed",
            "affineremoved",
            "off_support",
            "offsupport",
            "residual",
        )
    )


def find_forbidden_stage1_keys(value: Any, *, path: str = "") -> tuple[str, ...]:
    """Find dense/field keys recursively, returning deterministic paths."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _key_is_forbidden(key):
                found.append(child_path)
            found.extend(find_forbidden_stage1_keys(child, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(find_forbidden_stage1_keys(child, path=f"{path}[{index}]"))
    return tuple(found)


def validate_anchor_only_record(record: Mapping[str, Any]) -> None:
    """Raise if a stage-1 record contains any dense/off-support key."""

    if not isinstance(record, Mapping):
        raise TypeError("anchor-only record must be a mapping")
    forbidden = find_forbidden_stage1_keys(record)
    if forbidden:
        joined = ", ".join(forbidden)
        raise DenseMetricLeakError(f"stage-1 record contains forbidden key(s): {joined}")


def seal_anchor_only_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and deep-copy one immutable stage-1 record."""

    validate_anchor_only_record(record)
    return MappingProxyType(copy.deepcopy(dict(record)))


class AnchorOnlyGuard:
    """Append-only stage-1 record collector with a dense-metric deny-list."""

    def __init__(self) -> None:
        self._records: list[Mapping[str, Any]] = []

    def append(self, record: Mapping[str, Any]) -> None:
        self._records.append(seal_anchor_only_record(record))

    # Clear aliases make integration with common logging loops explicit while
    # preserving one guarded implementation.
    add = append
    record = append

    def extend(self, records: Sequence[Mapping[str, Any]]) -> None:
        for record in records:
            self.append(record)

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._records)

    def as_list(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(dict(record)) for record in self._records]

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize only the already-validated stage-1 records."""

        return json.dumps(self.as_list(), ensure_ascii=False, sort_keys=True, indent=indent)


__all__ = [
    "AnchorOnlyGuard",
    "DenseMetricLeakError",
    "find_forbidden_stage1_keys",
    "seal_anchor_only_record",
    "validate_anchor_only_record",
]
