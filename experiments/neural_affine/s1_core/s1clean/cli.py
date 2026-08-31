from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_protocol, protocol_hash
from .data import render_points, support_points_px
from .environment import environment_snapshot, gpu_gate
from .manifest import verify_manifest, write_manifest
from .models import build_model, seed_all
from .preflight import cpu_preflight
from .runner import run_all, run_one
from .status import StatusStore
from .train import train_run


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


def _local_gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "reason": "nvidia_smi_unavailable"}
    name, free, total, utilization = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    python_processes = []
    for row in process_result.stdout.splitlines():
        if "python" in row.lower() and not row.lstrip().startswith(str(os.getpid())):
            python_processes.append(row.strip())
    available = int(float(free)) >= 4096 and int(float(utilization)) <= 10 and not python_processes
    return {
        "available": available,
        "name": name,
        "free_mib": int(float(free)),
        "total_mib": int(float(total)),
        "utilization_pct": int(float(utilization)),
        "other_python_compute_processes": python_processes,
    }


def command_local_smoke(args: argparse.Namespace) -> int:
    package = Path(args.package_root).resolve()
    output = Path(args.output).resolve()
    protocol = load_protocol(package / "protocol.json")
    if args.device == "cuda":
        gate = _local_gpu_snapshot()
        if not gate["available"]:
            _json({"status": "skipped", "reason": "local_gpu_busy_or_too_small", "gpu": gate})
            return 0
    seed = int(args.seed)
    support = support_points_px(protocol)
    images = render_points(support, workers=1, protocol=protocol)
    targets = support.astype(np.float32) / float(protocol["coord_scale"])
    seed_all(seed)
    model = build_model(args.variant)
    manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    config = {
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "anchor_every": max(1, min(int(args.steps), 10)),
        "resume_every": max(1, min(int(args.steps), 10)),
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "lr_min": 1e-5,
        "l1_weight": 0.25,
        "variant": args.variant,
        "protocol_hash": protocol_hash(package / "protocol.json"),
        "code_hash": manifest["aggregate_sha256"],
    }
    summary = train_run(
        model,
        images,
        targets,
        images,
        targets,
        output,
        config,
        seed,
        device=args.device,
        resume=False,
    )
    _json({"status": "passed", "summary": summary, "environment": environment_snapshot()})
    return 0


def command_pack(run_root: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_dirs = {"cache_cpu", "cache_gpu", "__pycache__"}
    with tarfile.open(output, mode="w") as archive:
        for path in sorted(run_root.rglob("*")):
            if not path.is_file() or path.resolve() == output.resolve():
                continue
            relative = path.relative_to(run_root)
            if any(part in excluded_dirs for part in relative.parts):
                continue
            if path.name == "resume.pt":
                continue
            archive.add(path, arcname=(Path("run") / relative).as_posix(), recursive=False)
    from .manifest import sha256_file

    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s1clean")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--package-root", required=True)

    verify = sub.add_parser("verify-manifest")
    verify.add_argument("--package-root", required=True)

    env = sub.add_parser("environment")
    env.add_argument("--output")

    cpu = sub.add_parser("cpu-preflight")
    cpu.add_argument("--package-root", required=True)
    cpu.add_argument("--run-root", required=True)
    cpu.add_argument("--historical-root", required=True)
    cpu.add_argument("--workers", type=int, default=None)

    gpu = sub.add_parser("gpu-gate")
    gpu.add_argument("--run-root", required=True)
    gpu.add_argument("--require-a10", action="store_true", default=False)

    smoke = sub.add_parser("local-smoke")
    smoke.add_argument("--package-root", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--variant", choices=("vanilla", "coordconv"), default="vanilla")
    smoke.add_argument("--steps", type=int, default=20)
    smoke.add_argument("--batch-size", type=int, default=8)
    smoke.add_argument("--seed", type=int, default=20260816)

    one = sub.add_parser("run-one")
    one.add_argument("--package-root", required=True)
    one.add_argument("--run-root", required=True)
    one.add_argument("--cache-dir", required=True)
    one.add_argument("--run-id", required=True)
    one.add_argument("--variant", choices=("vanilla", "coordconv"), required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--init-source")
    one.add_argument("--device", default="cuda")

    all_runs = sub.add_parser("run-all")
    all_runs.add_argument("--package-root", required=True)
    all_runs.add_argument("--run-root", required=True)
    all_runs.add_argument("--hard-new-work-sec", type=int, default=6300)

    audit = sub.add_parser("audit")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--protocol", required=True)
    audit.add_argument("--device", default="cpu")

    status = sub.add_parser("status")
    status.add_argument("--run-root", required=True)

    pack = sub.add_parser("pack")
    pack.add_argument("--run-root", required=True)
    pack.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            _json(write_manifest(args.package_root))
        elif args.command == "verify-manifest":
            ok, failures = verify_manifest(args.package_root)
            _json({"passed": ok, "failures": failures})
            return 0 if ok else 2
        elif args.command == "environment":
            payload = environment_snapshot()
            if args.output:
                Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _json(payload)
        elif args.command == "cpu-preflight":
            _json(cpu_preflight(args.package_root, args.run_root, args.historical_root, workers=args.workers))
        elif args.command == "gpu-gate":
            payload = gpu_gate(require_a10=args.require_a10)
            StatusStore(Path(args.run_root) / "status").update("GPU_GATE_PASSED", environment=payload)
            _json(payload)
        elif args.command == "local-smoke":
            return command_local_smoke(args)
        elif args.command == "run-one":
            _json(
                run_one(
                    args.package_root,
                    args.run_root,
                    args.cache_dir,
                    run_id=args.run_id,
                    variant=args.variant,
                    seed=args.seed,
                    init_source=args.init_source,
                    device=args.device,
                )
            )
        elif args.command == "run-all":
            _json(run_all(args.package_root, args.run_root, hard_new_work_sec=args.hard_new_work_sec))
        elif args.command == "audit":
            from .independent_eval import audit_run

            _json(audit_run(args.run_dir, args.protocol, device=args.device))
        elif args.command == "status":
            _json(StatusStore(Path(args.run_root) / "status").read())
        elif args.command == "pack":
            _json(command_pack(Path(args.run_root), Path(args.output)))
        return 0
    except Exception as exc:
        if getattr(args, "run_root", None):
            try:
                StatusStore(Path(args.run_root) / "status").update(
                    "FAILED", error_type=type(exc).__name__, error=str(exc), command=args.command
                )
            except Exception:
                pass
        print(f"S1ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

