from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", "outputs", "cache", "cache_gpu"}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_package_files(root: str | Path) -> Iterable[Path]:
    base = Path(root).resolve()
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if rel.name in {"MANIFEST.sha256", "package_manifest.json"}:
            continue
        yield path


def build_manifest(root: str | Path) -> dict[str, object]:
    base = Path(root).resolve()
    entries = []
    for path in iter_package_files(base):
        rel = path.relative_to(base).as_posix()
        entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(entry["path"].encode("utf-8"))
        aggregate.update(entry["sha256"].encode("ascii"))
    # Keep the manifest portable between the local staging directory and DSW.
    return {"root": ".", "files": entries, "aggregate_sha256": aggregate.hexdigest()}


def write_manifest(root: str | Path) -> dict[str, object]:
    base = Path(root).resolve()
    payload = build_manifest(base)
    (base / "package_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [f"{item['sha256']}  {item['path']}" for item in payload["files"]]
    (base / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def verify_manifest(root: str | Path) -> tuple[bool, list[str]]:
    base = Path(root).resolve()
    manifest_path = base / "package_manifest.json"
    if not manifest_path.is_file():
        return False, ["missing:package_manifest.json"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected_paths = [str(item["path"]) for item in payload.get("files", [])]
    actual_paths = [path.relative_to(base).as_posix() for path in iter_package_files(base)]
    if expected_paths != actual_paths:
        missing = sorted(set(expected_paths).difference(actual_paths))
        unexpected = sorted(set(actual_paths).difference(expected_paths))
        failures.extend(f"unlisted_missing:{path}" for path in missing)
        failures.extend(f"unexpected:{path}" for path in unexpected)
    aggregate = hashlib.sha256()
    for item in payload["files"]:
        aggregate.update(str(item["path"]).encode("utf-8"))
        aggregate.update(str(item["sha256"]).encode("ascii"))
        path = base / item["path"]
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
        elif path.stat().st_size != item["bytes"]:
            failures.append(f"size:{item['path']}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"sha256:{item['path']}")
    if aggregate.hexdigest() != payload.get("aggregate_sha256"):
        failures.append("aggregate_sha256")
    checksum_path = base / "MANIFEST.sha256"
    expected_checksum_text = "".join(f"{item['sha256']}  {item['path']}\n" for item in payload["files"])
    if not checksum_path.is_file():
        failures.append("missing:MANIFEST.sha256")
    elif checksum_path.read_text(encoding="utf-8") != expected_checksum_text:
        failures.append("MANIFEST.sha256")
    return not failures, failures
