"""Unattended progress.json. Honest failures, no silent rewrite of old phases."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .io_util import dir_size_gb, disk_free_gb, dump_json, load_json
from .protocol import PHASE3_ROOT, PROGRESS_PATH, summarize_failures


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_progress() -> Dict[str, Any]:
    return {
        "current_track": None,
        "current_run": None,
        "finished_runs": [],
        "failed_runs": [],
        "gpu_hours": 0.0,
        "elapsed_sec": 0.0,
        "disk_usage_gb": 0.0,
        "disk_free_gb": None,
        "started_at": _now(),
        "updated_at": _now(),
        "resume": False,
    }


def load_progress() -> Dict[str, Any]:
    if PROGRESS_PATH.exists():
        blob = load_json(PROGRESS_PATH)
        blob.setdefault("finished_runs", [])
        blob.setdefault("failed_runs", [])
        return blob
    return empty_progress()


def save_progress(progress: Dict[str, Any]) -> None:
    progress["updated_at"] = _now()
    progress["disk_usage_gb"] = dir_size_gb(PHASE3_ROOT)
    try:
        progress["disk_free_gb"] = disk_free_gb(PHASE3_ROOT)
    except Exception:
        progress["disk_free_gb"] = None
    progress["failures_mirror"] = summarize_failures()
    dump_json(PROGRESS_PATH, progress)


def mark_current(progress: Dict[str, Any], track: str, run: str) -> None:
    progress["current_track"] = track
    progress["current_run"] = run
    save_progress(progress)


def mark_finished(progress: Dict[str, Any], run: str, elapsed_sec: float) -> None:
    if run not in progress["finished_runs"]:
        progress["finished_runs"].append(run)
    progress["elapsed_sec"] = float(progress.get("elapsed_sec") or 0.0) + float(elapsed_sec)
    progress["gpu_hours"] = float(progress["elapsed_sec"]) / 3600.0
    progress["current_run"] = None
    save_progress(progress)


def mark_failed(progress: Dict[str, Any], run: str, elapsed_sec: float) -> None:
    if run not in progress["failed_runs"]:
        progress["failed_runs"].append(run)
    progress["elapsed_sec"] = float(progress.get("elapsed_sec") or 0.0) + float(elapsed_sec)
    progress["gpu_hours"] = float(progress["elapsed_sec"]) / 3600.0
    progress["current_run"] = None
    save_progress(progress)


def already_done(progress: Dict[str, Any], run: str) -> bool:
    return run in progress.get("finished_runs", [])


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
