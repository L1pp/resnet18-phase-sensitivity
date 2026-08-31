from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .config import resolve_cell
from .records import attempt_parent, path_within_root, require_formal_execution_root


class FormalProvenanceError(ValueError):
    pass


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise FormalProvenanceError(f"unreadable formal evidence: {path}") from exc
    if not isinstance(value, dict):
        raise FormalProvenanceError(f"formal evidence is not an object: {path}")
    return value


def validate_formal_attempt(
    run_path: str | Path,
    *,
    execution_root: str | Path | None = None,
    expected_cell_id: str | None = None,
    require_complete: bool,
) -> dict[str, Any]:
    run = Path(run_path).resolve()
    config = _json_object(run / "run_config.json")
    status = _json_object(run / "status.json")
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise FormalProvenanceError("run_config.execution is missing")
    declared_root = Path(str(execution.get("project_root", ""))).resolve()
    root = require_formal_execution_root(execution_root or declared_root)
    if declared_root != root:
        raise FormalProvenanceError("run_config project_root does not match the formal execution root")
    path_within_root(run, root / "runs", label="run path")
    if Path(str(execution.get("run_path", ""))).resolve() != run:
        raise FormalProvenanceError("run_config run_path does not match the inspected attempt")
    if execution.get("mode") != "formal" or execution.get("formal_eligible") is not True:
        raise FormalProvenanceError("run_config is not formal/formal_eligible")

    cell_id = str(expected_cell_id or status.get("cell_id") or config.get("cell", {}).get("cell_id", ""))
    cell = resolve_cell(cell_id)
    if config.get("cell") != cell or status.get("cell_id") != cell_id:
        raise FormalProvenanceError("formal run cell does not exactly match the frozen matrix")
    if not re.fullmatch(r"attempt_\d{3}", run.name):
        raise FormalProvenanceError("formal run attempt name is invalid")
    expected_run_parent = attempt_parent(
        str(cell["family"]),
        str(cell["condition_id"]),
        int(cell["run_seed"]),
        project_root=root,
    ).resolve()
    if run.parent != expected_run_parent:
        raise FormalProvenanceError("formal run path is not the frozen cell/seed attempt path")
    review = Path(str(execution.get("review_path", ""))).resolve()
    expected_review = (
        attempt_parent(
            str(cell["family"]),
            str(cell["condition_id"]),
            int(cell["run_seed"]),
            project_root=root,
            review=True,
        ).resolve()
        / run.name
    )
    path_within_root(review, root / "reviews", label="review path")
    if review != expected_review:
        raise FormalProvenanceError("formal review path is not the matching attempt path")

    state = status.get("state")
    if require_complete:
        if state != "COMPLETE" or status.get("formal_eligible") is not True:
            raise FormalProvenanceError("formal run is not COMPLETE/formal_eligible")
    elif state not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise FormalProvenanceError("formal attempt status is invalid")
    elif state in {"RUNNING", "COMPLETE"} and status.get("formal_eligible") is not True:
        raise FormalProvenanceError("active/complete formal status is not formal_eligible")
    return {"root": root, "run_path": run, "review_path": review, "cell": cell, "config": config, "status": status}


def validate_formal_review(
    review_path: str | Path,
    *,
    execution_root: str | Path,
    expected_cell_id: str | None = None,
    expected_status: str | None = None,
) -> dict[str, Any]:
    root = require_formal_execution_root(execution_root)
    directory = Path(review_path).resolve()
    path_within_root(directory, root / "reviews", label="review directory")
    review = _json_object(directory / "review.json")
    cell_id = str(expected_cell_id or review.get("cell_id", ""))
    attempt = validate_formal_attempt(
        str(review.get("run_path", "")),
        execution_root=root,
        expected_cell_id=cell_id,
        require_complete=True,
    )
    if directory != attempt["review_path"] or Path(str(review.get("review_path", ""))).resolve() != directory:
        raise FormalProvenanceError("review path does not match the producer-declared attempt")
    if review.get("cell_id") != cell_id or review.get("formal_eligible") is not True:
        raise FormalProvenanceError("review cell/formal_eligible evidence is invalid")
    if review.get("review_status") not in {"PASS", "FAIL"}:
        raise FormalProvenanceError("review status is not terminal")
    if expected_status is not None and review.get("review_status") != expected_status:
        raise FormalProvenanceError("review status differs from the required formal status")
    if Path(str(review.get("run_path", ""))).resolve() != attempt["run_path"]:
        raise FormalProvenanceError("review run_path does not match its formal attempt")
    return {**attempt, "review": review, "review_directory": directory}


def validate_formal_catalog_record(record: Mapping[str, Any], *, execution_root: str | Path) -> dict[str, Any]:
    root = require_formal_execution_root(execution_root)
    cell_id = str(record.get("cell_id", ""))
    cell = resolve_cell(cell_id)
    if record.get("mode") != "formal":
        raise FormalProvenanceError("RUN_CATALOG record is not formal")
    if (
        record.get("family") != cell["family"]
        or record.get("condition") != cell["condition_id"]
        or int(record.get("seed", -1)) != int(cell["run_seed"])
    ):
        raise FormalProvenanceError("RUN_CATALOG cell fields do not match the frozen matrix")
    run = Path(str(record.get("run_path", ""))).resolve()
    attempt = validate_formal_attempt(
        run,
        execution_root=root,
        expected_cell_id=cell_id,
        require_complete=record.get("status") == "COMPLETE",
    )
    if Path(str(record.get("review_path", ""))).resolve() != attempt["review_path"]:
        raise FormalProvenanceError("RUN_CATALOG review path does not match the attempt")
    if record.get("status") == "COMPLETE" and record.get("formal_eligible") is not True:
        raise FormalProvenanceError("complete RUN_CATALOG record is not formal_eligible")
    if record.get("status") not in {"COMPLETE", "FAILED"}:
        raise FormalProvenanceError("RUN_CATALOG record has an unsupported status")
    return attempt
