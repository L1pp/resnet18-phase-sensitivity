"""JSON / hash / path helpers. Never write bare NaN into JSON."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(json_sanitize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return json_sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, Path):
        return str(obj).replace("\\", "/")
    if isinstance(obj, Mapping):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return str(obj)


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_sanitize(payload), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_dict_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        value = state[key]
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def disk_free_gb(path: Path) -> float:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    usage = os.statvfs(path) if hasattr(os, "statvfs") else None
    if usage is not None:
        return float(usage.f_bavail * usage.f_frsize) / (1024**3)
    import shutil

    return float(shutil.disk_usage(str(path)).free) / (1024**3)


def dir_size_gb(path: Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return float(total) / (1024**3)
