"""Cloud overnight entry: C → D/E/H → branches → atlas → extras. Skip A/B. Supports --resume."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# allow `python run_overnight.py` from package dir or repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1_gap_rep.common import device, dump_json, environment_versions, fingerprint, profile_spec
from phase2_overnight.atlas import run_atlas
from phase2_overnight.branches import run_branches
from phase2_overnight.content import run_content_panel
from phase2_overnight.layers import run_layers
from phase2_overnight.part_c import run_part_c
from phase2_overnight.parts_deh import run_parts_deh
from phase2_overnight.protocol import GEOMETRY_FP, PHASE2_ROOT, freeze_protocol, record_failure, summarize_failures
from phase2_overnight.ranking import write_cloud_handoff
from phase2_overnight.render_dense import build_dense_cache
from phase2_overnight.symmetry import run_symmetry_and_extras
from phase2_overnight.train_regime import n_steps_ref

PROGRESS = PHASE2_ROOT / "progress.json"


def _load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"completed": [], "failures": [], "started_at": datetime.now(timezone.utc).isoformat()}


def _save_progress(p: dict) -> None:
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["failures"] = summarize_failures()
    dump_json(PROGRESS, p)


def _run_stage(name: str, fn, progress: dict, resume: bool):
    if resume and name in progress.get("completed", []):
        print(f"[overnight] skip completed {name}")
        return progress.get("results", {}).setdefault(name, {})
    progress["current"] = name
    _save_progress(progress)
    t0 = time.time()
    try:
        result = fn()
        progress.setdefault("completed", []).append(name)
        progress.setdefault("results", {})[name] = "ok"
        progress["elapsed_" + name] = time.time() - t0
        _save_progress(progress)
        print(f"[overnight] done {name} in {progress['elapsed_' + name]:.1f}s")
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        record_failure(name, str(exc), {"tb": tb[-2000:]})
        _save_progress(progress)
        print(f"[overnight] FAIL {name}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ab-only", action="store_true", help="local Part A/B only")
    args = parser.parse_args()
    PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
    proto = freeze_protocol()
    fp = fingerprint(profile_spec("factorial"))
    if fp != GEOMETRY_FP:
        raise SystemExit(f"fingerprint {fp} != {GEOMETRY_FP}")
    env = environment_versions()
    env["device"] = str(device())
    dump_json(PHASE2_ROOT / "config" / "cloud_environment.json", env)
    print(f"[overnight] protocol={proto['protocol_hash']} env={env}")

    if args.ab_only:
        from phase2_overnight.parts_ab import run_part_ab

        run_part_ab()
        return

    progress = _load_progress() if args.resume else {"completed": [], "failures": [], "started_at": datetime.now(timezone.utc).isoformat()}
    progress["n_steps_ref"] = n_steps_ref()
    _save_progress(progress)

    _run_stage("dense_cache", build_dense_cache, progress, args.resume)
    c = _run_stage("part_c", run_part_c, progress, args.resume)
    deh = _run_stage("parts_deh", run_parts_deh, progress, args.resume) or {}
    _run_stage("branches", lambda: run_branches(deh if isinstance(deh, dict) else {}), progress, args.resume)
    atlas = _run_stage("atlas", run_atlas, progress, args.resume) or {}
    _run_stage("symmetry", lambda: run_symmetry_and_extras(atlas if isinstance(atlas, dict) else {}, deh if isinstance(deh, dict) else {}), progress, args.resume)
    _run_stage("layers", run_layers, progress, args.resume)
    _run_stage("content", run_content_panel, progress, args.resume)
    write_cloud_handoff(progress)
    print("[overnight] finished writing ranking/handoff")


if __name__ == "__main__":
    main()
