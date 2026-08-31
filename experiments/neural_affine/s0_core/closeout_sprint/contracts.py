"""Small, dependency-light contracts shared by the clean-room S0 tools.

The package deliberately has no imports from the historical experiment trees.
Arrays returned by the numerical code stay as NumPy arrays until the final
JSON boundary, where non-finite values are converted to ``null``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FieldBundle:
    """The only values an evaluator is allowed to use for a field audit."""

    pred: np.ndarray
    true: np.ndarray
    source: str = "unknown"


class SchemaError(ValueError):
    """Raised when an input artifact does not satisfy a clean-room schema."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/path/non-finite values to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
