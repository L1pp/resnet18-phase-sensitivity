from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .config import PACKAGE_ROOT

_SAFE_PART = re.compile(r"^[A-Za-z0-9_+.-]+$")
RELEASE_ID = "function_selection_v1_trackb_unified_r1_20260824"
FORMAL_ROOT_MARKER_NAME = "FORMAL_EXECUTION_ROOT.json"
FORMAL_ROOT_PURPOSE = "amd48_track_b_b1_b8_formal_execution"


class FormalRootError(ValueError):
    pass


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold one short cross-process byte/file lock around catalog mutation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_part(value: str, name: str) -> str:
    if not _SAFE_PART.fullmatch(value):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def attempt_parent(
    family: str,
    condition: str,
    seed: int,
    *,
    project_root: str | Path = PACKAGE_ROOT,
    review: bool = False,
) -> Path:
    root = Path(project_root)
    branch = "reviews" if review else "runs"
    return root / branch / _safe_part(family, "family") / _safe_part(condition, "condition") / f"seed_{int(seed)}"


def next_attempt_number(parent: str | Path) -> int:
    path = Path(parent)
    observed: list[int] = []
    if path.is_dir():
        for child in path.iterdir():
            match = re.fullmatch(r"attempt_(\d{3})", child.name)
            if match:
                observed.append(int(match.group(1)))
    return max(observed, default=0) + 1


def create_attempt(
    family: str,
    condition: str,
    seed: int,
    *,
    project_root: str | Path = PACKAGE_ROOT,
) -> dict[str, str]:
    run_parent = attempt_parent(family, condition, seed, project_root=project_root)
    review_parent = attempt_parent(family, condition, seed, project_root=project_root, review=True)
    number = next_attempt_number(run_parent)
    run_path = run_parent / f"attempt_{number:03d}"
    review_path = review_parent / f"attempt_{number:03d}"
    if run_path.exists() or review_path.exists():
        raise FileExistsError("attempt or matching review path already exists")
    run_path.mkdir(parents=True, exist_ok=False)
    return {
        "attempt_id": f"attempt_{number:03d}",
        "run_path": str(run_path.resolve()),
        "review_path": str(review_path.resolve()),
    }


def write_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # On Windows a scanner can briefly hold the destination between rapid
        # state snapshots.  Keep the same atomic replace, with a small bounded
        # retry for that transient sharing violation.
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def _formal_marker_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": FORMAL_ROOT_PURPOSE,
    }


def require_formal_execution_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    marker = root / FORMAL_ROOT_MARKER_NAME
    if not marker.is_file():
        raise FormalRootError(f"formal execution root marker is missing: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise FormalRootError(f"formal execution root marker is unreadable: {marker}") from exc
    if payload != _formal_marker_payload():
        raise FormalRootError("formal execution root marker does not match this release/purpose")
    return root


def initialize_formal_execution_root(project_root: str | Path) -> Path:
    """Initialize only an absent/empty root, or reopen an already marked root."""
    root = Path(project_root).resolve()
    marker = root / FORMAL_ROOT_MARKER_NAME
    if root.exists():
        if not root.is_dir():
            raise FormalRootError("formal execution root is not a directory")
        if marker.is_file():
            require_formal_execution_root(root)
            initialize_project_records(root)
            return root
        if any(root.iterdir()):
            raise FormalRootError("refusing to initialize a nonempty directory without a formal root marker")
    else:
        root.mkdir(parents=True, exist_ok=False)
    write_json(marker, _formal_marker_payload())
    initialize_project_records(root)
    return root


def path_within_root(path: str | Path, root: str | Path, *, label: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FormalRootError(f"{label} escapes the formal execution root") from exc
    if resolved == resolved_root:
        raise FormalRootError(f"{label} must be below the formal execution root")
    return resolved


def initialize_project_records(project_root: str | Path = PACKAGE_ROOT) -> None:
    """Controller-only initialization before any formal child process starts."""
    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=True)
    release_id = RELEASE_ID
    defaults = {
        "RUN_CATALOG.json": {"schema_version": 1, "release_id": release_id, "attempts": []},
        "RESULTS_INDEX.json": {
            "schema_version": 1,
            "release_id": release_id,
            "reviewed_terminal_results": [],
            "family_summaries": [],
        },
    }
    for filename, payload in defaults.items():
        path = root / filename
        if not path.exists():
            write_json(path, payload)


def append_run_catalog(record: Mapping[str, Any], *, project_root: str | Path = PACKAGE_ROOT) -> None:
    path = Path(project_root) / "RUN_CATALOG.json"
    required = {"cell_id", "family", "condition", "seed", "run_path", "review_path", "status"}
    if not required.issubset(record):
        raise ValueError(f"run catalog record missing {sorted(required.difference(record))}")
    lock_path = Path(project_root) / ".RUN_CATALOG.lock"
    with _exclusive_file_lock(lock_path):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        attempts = catalog.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("RUN_CATALOG attempts must be a list")
        if any(item.get("run_path") == record["run_path"] for item in attempts):
            raise ValueError("run path is already catalogued")
        payload = dict(record)
        payload.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
        attempts.append(payload)
        write_json(path, catalog, overwrite=True)


def register_reviewed_result(record: Mapping[str, Any], *, project_root: str | Path = PACKAGE_ROOT) -> None:
    raise RuntimeError("RESULTS_INDEX is reducer-owned; per-attempt writes are forbidden")


def write_reduced_results_index(
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    project_root: str | Path = PACKAGE_ROOT,
) -> Path:
    """The one final reducer write after exact 84-PASS closure."""
    if aggregate.get("closure") is not True or aggregate.get("aggregate_status") != "PASS_COMPLETE_84_CELL_CLOSURE":
        raise ValueError("cannot write RESULTS_INDEX without a complete PASS aggregate")
    if len(records) != 84 or len({str(record.get("cell_id")) for record in records}) != 84:
        raise ValueError("RESULTS_INDEX reducer requires exactly 84 unique logical records")
    path = Path(project_root) / "RESULTS_INDEX.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("reviewed_terminal_results"):
        raise FileExistsError("RESULTS_INDEX already contains terminal results")
    payload = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "reviewed_terminal_results": [dict(record) for record in records],
        "family_summaries": aggregate["contrasts"],
        "aggregate_status": aggregate["aggregate_status"],
    }
    write_json(path, payload, overwrite=True)
    return path
