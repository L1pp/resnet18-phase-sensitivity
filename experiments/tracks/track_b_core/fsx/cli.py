from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import PACKAGE_ROOT, iter_cells, load_matrix, resolve_cell, validate_matrix
from .data import prepare_inputs, validate_prepared_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track B unified exploratory local preparation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-inputs", help="materialize and semantically validate frozen inputs")
    prepare.add_argument("--asset-root", type=Path, default=PACKAGE_ROOT / "assets")
    validate = commands.add_parser("validate-inputs", help="semantically validate prepared inputs")
    validate.add_argument("--asset-root", type=Path, default=PACKAGE_ROOT / "assets")
    commands.add_parser("matrix-summary", help="show exact logical/physical/optimizer counts")
    show = commands.add_parser("show-cell", help="resolve one frozen logical cell")
    show.add_argument("cell_id")
    smoke = commands.add_parser("smoke-distinct-paths", help="run isolated non-formal smoke coverage")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--device", choices=["cpu", "cuda"], default=None)
    review = commands.add_parser("review-attempt", help="independently recompute one attempt from raw fields")
    review.add_argument("run_path", type=Path)
    aggregate = commands.add_parser("aggregate-reviews", help="aggregate exactly 84 PASS review files")
    aggregate.add_argument("output_path", type=Path)
    aggregate.add_argument("review_files", nargs="+", type=Path)
    formal_cell = commands.add_parser("formal-run-cell", help="run one frozen logical cell in formal mode")
    formal_cell.add_argument("cell_id")
    formal_cell.add_argument("--project-root", type=Path, required=True)
    formal_cell.add_argument("--asset-root", type=Path, required=True)
    formal_cell.add_argument("--device", choices=["cpu", "cuda"], required=True)
    formal_cell.add_argument("--upstream-run", type=Path)
    formal_cell.add_argument("--paired-initialization", type=Path)
    formal_cell.add_argument("--resume-from", type=Path)
    formal_b8 = commands.add_parser(
        "formal-b8-degree2-shared",
        help="run the one shared analytic B8 degree-2 physical job",
    )
    formal_b8.add_argument("--project-root", type=Path, required=True)
    formal_b8.add_argument("--asset-root", type=Path, required=True)
    formal_b4 = commands.add_parser("formal-b4-paired-init", help="atomically prepare one B4 paired initialization")
    formal_b4.add_argument("seed", type=int)
    formal_b4.add_argument("--project-root", type=Path, required=True)
    formal_review = commands.add_parser("formal-review", help="independently review one completed formal attempt")
    formal_review.add_argument("run_path", type=Path)
    formal_review.add_argument("--review-path", type=Path)
    orchestrate = commands.add_parser("formal-orchestrate", help="run or restart the bounded AMD48 formal DAG")
    orchestrate.add_argument("--execution-root", type=Path, default=PACKAGE_ROOT / "formal_execution_20260824")
    orchestrate.add_argument("--asset-root", type=Path, default=PACKAGE_ROOT / "assets")
    orchestrate.add_argument("--gpu-workers", type=int, default=2)
    orchestrate.add_argument("--cpu-workers", type=int, default=2)
    orchestrate.add_argument("--reviewer-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-inputs":
        result = prepare_inputs(args.asset_root)
    elif args.command == "validate-inputs":
        result = validate_prepared_inputs(args.asset_root)
    elif args.command == "matrix-summary":
        matrix = load_matrix()
        result = {**validate_matrix(matrix), "families": {family: len(items) * len(matrix["seeds"]) for family, items in matrix["families"].items()}}
    elif args.command == "show-cell":
        result = resolve_cell(args.cell_id)
    elif args.command == "smoke-distinct-paths":
        from .smoke import run_distinct_path_smoke

        result = run_distinct_path_smoke(args.output_root, device=args.device)
    elif args.command == "review-attempt":
        from .evaluator import review_attempt

        result = review_attempt(args.run_path)
    elif args.command == "aggregate-reviews":
        from .aggregate import aggregate_review_files

        result = aggregate_review_files(args.review_files, args.output_path)
    elif args.command == "formal-run-cell":
        from .formal_worker import run_formal_cell

        result = run_formal_cell(
            args.cell_id,
            project_root=args.project_root,
            asset_root=args.asset_root,
            device=args.device,
            upstream_run=args.upstream_run,
            paired_initialization=args.paired_initialization,
            resume_from=args.resume_from,
        )
    elif args.command == "formal-b8-degree2-shared":
        from .formal_worker import run_formal_b8_degree2_shared

        result = run_formal_b8_degree2_shared(project_root=args.project_root, asset_root=args.asset_root)
    elif args.command == "formal-b4-paired-init":
        from .formal_worker import prepare_formal_b4_pair

        result = prepare_formal_b4_pair(args.seed, project_root=args.project_root)
    elif args.command == "formal-review":
        from .formal_worker import review_formal_attempt

        result = review_formal_attempt(args.run_path, review_path=args.review_path)
    elif args.command == "formal-orchestrate":
        from .formal_orchestrator import ResourceLimits, run_formal_orchestration

        limits = ResourceLimits(
            gpu_workers=args.gpu_workers,
            cpu_workers=args.cpu_workers,
            reviewer_workers=args.reviewer_workers,
        )
        result = run_formal_orchestration(
            execution_root=args.execution_root,
            asset_root=args.asset_root,
            limits=limits,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0
