"""Phase 3 unattended entry. Order: A→B→C→D→E→F. Fail one track, continue."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3_high_upside.io_util import disk_free_gb, dump_json
from phase3_high_upside.progress import Timer, already_done, load_progress, mark_current, mark_failed, mark_finished, save_progress
from phase3_high_upside.protocol import MIN_DISK_GB, PHASE3_ROOT, TIME_BUDGET_SEC, freeze_protocol, record_failure, record_preflight
from phase3_high_upside.ranking import write_handoff, write_ranking
from phase3_high_upside.track_a import run_track_a
from phase3_high_upside.track_b import run_track_b
from phase3_high_upside.track_c import run_track_c
from phase3_high_upside.track_d import run_track_d
from phase3_high_upside.track_e import run_track_e
from phase3_high_upside.track_f import run_track_f

TRACKS = (
    ("A", run_track_a),
    ("B", run_track_b),
    ("C", run_track_c),
    ("D", run_track_d),
    ("E", run_track_e),
)


def _run_named(name: str, fn, progress: dict, resume: bool) -> None:
    run = f"track_{name}"
    if resume and already_done(progress, run):
        print(f"[phase3] skip completed {run}")
        return
    mark_current(progress, name, run)
    timer = Timer()
    try:
        result = fn() if name != "F" else fn(float(progress.get("elapsed_sec") or 0.0))
        dump_json(PHASE3_ROOT / f"track_{name.lower() if name != 'F' else 'f'}_status.json", {"ok": True, "run": run})
        mark_finished(progress, run, timer.elapsed())
        print(f"[phase3] done {run} in {timer.elapsed():.1f}s")
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        record_failure(run, str(exc), {"tb": tb[-4000:]})
        mark_failed(progress, run, timer.elapsed())
        print(f"[phase3] FAIL {run}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="skip finished_runs only; does not resume mid-training")
    parser.add_argument("--skip-f", action="store_true")
    args = parser.parse_args()
    PHASE3_ROOT.mkdir(parents=True, exist_ok=True)
    freeze_protocol(force=False)
    pre = record_preflight()
    progress = load_progress()
    if args.resume:
        progress["resume"] = True
        progress["resume_note"] = "skip completed tracks only; trainers still start from random init"
    else:
        progress["resume"] = False
    save_progress(progress)
    try:
        free = disk_free_gb(PHASE3_ROOT)
        if free < MIN_DISK_GB:
            record_failure("disk", f"free={free:.1f}GB < {MIN_DISK_GB}")
            print(f"[phase3] disk {free:.1f}GB < {MIN_DISK_GB}GB; continue anyway unless later write fails")
    except Exception:
        pass
    dump_json(PHASE3_ROOT / "config" / "preflight_pointer.json", {"protocol_hash": pre.get("protocol_hash")})

    for name, fn in TRACKS:
        _run_named(name, fn, progress, args.resume)
        if float(progress.get("elapsed_sec") or 0.0) > TIME_BUDGET_SEC:
            print("[phase3] time budget reached after", name)
            break
    if not args.skip_f:
        _run_named("F", run_track_f, progress, args.resume)
    write_ranking()
    write_handoff()
    from phase3_high_upside.pack_return import pack_return

    try:
        pack_return()
    except Exception as exc:
        record_failure("pack_return", str(exc))
    progress["current_track"] = None
    progress["current_run"] = None
    save_progress(progress)
    print("[phase3] all requested tracks finished or failed-open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
