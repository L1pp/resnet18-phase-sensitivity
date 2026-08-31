from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("protocol root must be a JSON object")
    return value


def protocol_hash(path: str | Path | None = None) -> str:
    return hashlib.sha256(canonical_json_bytes(load_protocol(path))).hexdigest()

