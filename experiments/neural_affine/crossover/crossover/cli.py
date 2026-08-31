"""Minimal CLI for the clean crossover protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import aggregate_runs
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol
from .runner import prepare_assets, run_matrix, run_smoke


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-affine-crossover")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="render cache and exact initializations")
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--init-dir", type=Path, default=None, help="verified corrected_vanilla exact-init directory")
    prepare.add_argument("--allow-generated-inits", action="store_true", help="development-only fallback; not formal")
    smoke = sub.add_parser("smoke", help="short local CPU smoke")
    smoke.add_argument("--root", required=True, type=Path)
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--tiny", action=argparse.BooleanOptionalAction, default=True)
    matrix = sub.add_parser("run-matrix", help="run support x seed x regime matrix")
    matrix.add_argument("--root", required=True, type=Path)
    matrix.add_argument("--protocol", type=Path, default=None)
    matrix.add_argument("--device", default="cpu")
    matrix.add_argument("--steps", type=int, default=None)
    matrix.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    matrix.add_argument("--tiny", action="store_true")
    aggregate = sub.add_parser("aggregate", help="aggregate independent evaluator outputs")
    aggregate.add_argument("--root", required=True, type=Path)
    aggregate.add_argument("--protocol", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_assets(args.out, protocol_path=args.protocol, workers=args.workers, init_dir=args.init_dir, allow_generated_inits=args.allow_generated_inits)
    elif args.command == "smoke":
        result = run_smoke(args.root, steps=args.steps, device=args.device, tiny=args.tiny)
    elif args.command == "run-matrix":
        result = run_matrix(args.root, protocol_path=args.protocol, device=args.device, steps=args.steps, resume=args.resume, tiny=args.tiny)
    else:
        result = aggregate_runs(args.root, protocol=None if args.protocol is None else load_protocol(args.protocol))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
