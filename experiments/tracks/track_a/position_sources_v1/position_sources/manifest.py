"""SHA-256 manifests and read-only preflight checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .protocol import canonical_json_bytes, protocol_hash, write_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_safe(path: str | Path) -> str:
    value = Path(path).as_posix()
    if value.startswith("/") or value.startswith("../") or "/../" in value or value == "..":
        raise ValueError(f"unsafe manifest path: {value}")
    return value


def file_record(root: Path, path: Path) -> dict[str, Any]:
    relative = relative_safe(path.relative_to(root))
    return {
        "path": relative,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def records_for_paths(root: str | Path, paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root_path / path
        path = path.resolve()
        if root_path not in path.parents and path != root_path:
            raise ValueError(f"manifest path escapes root: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(file_record(root_path, path))
    return sorted(records, key=lambda row: row["path"])


def manifest_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_manifest(
    root: str | Path,
    *,
    protocol_path: str | Path,
    files: Iterable[str | Path],
    kind: str = "position_sources_v1_manifest",
    mode: str = "formal",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    protocol = Path(protocol_path).resolve()
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_sha = protocol_hash(path=protocol)
    all_files = list(files)
    if protocol not in [Path(p).resolve() if Path(p).is_absolute() else (root_path / p).resolve() for p in all_files]:
        all_files.append(protocol)
    records = records_for_paths(root_path, all_files)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "mode": str(mode),
        "protocol_id": str(protocol_payload["protocol_id"]),
        "protocol_hash": protocol_sha,
        "files": records,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    payload["manifest_sha256"] = manifest_digest(payload)
    return payload


def write_manifest(root: str | Path, payload: Mapping[str, Any], name: str = "MANIFEST.json") -> Path:
    target = Path(root) / name
    write_json(target, payload)
    return target


def verify_manifest(root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = Path(manifest_path) if manifest_path is not None else root_path / "MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("manifest schema_version must be 1")
    declared_digest = payload.get("manifest_sha256")
    digest_input = dict(payload)
    digest_input.pop("manifest_sha256", None)
    actual_digest = manifest_digest(digest_input)
    if declared_digest != actual_digest:
        raise ValueError("manifest_sha256 mismatch")
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest files must be a non-empty list")
    errors: list[str] = []
    for row in records:
        try:
            relative = relative_safe(row["path"])
            target = (root_path / relative).resolve()
            if root_path not in target.parents:
                errors.append(f"path escapes root: {relative}")
                continue
            if not target.is_file():
                errors.append(f"missing: {relative}")
                continue
            if int(row["bytes"]) != int(target.stat().st_size):
                errors.append(f"bytes mismatch: {relative}")
            if str(row["sha256"]) != sha256_file(target):
                errors.append(f"sha256 mismatch: {relative}")
        except Exception as exc:  # pragma: no cover - defensive malformed manifest path
            errors.append(f"invalid record: {exc}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "passed",
        "manifest_path": str(path),
        "manifest_sha256": actual_digest,
        "file_count": len(records),
        "protocol_hash": payload.get("protocol_hash"),
    }


__all__ = [
    "build_manifest",
    "file_record",
    "manifest_digest",
    "records_for_paths",
    "relative_safe",
    "sha256_file",
    "verify_manifest",
    "write_manifest",
]
