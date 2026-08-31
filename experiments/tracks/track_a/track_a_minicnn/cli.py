"""Direct CLI for the independent MiniCNN experiment."""

from __future__ import annotations

import argparse

from .pipeline import run_all, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m track_a_minicnn.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the six fixed MiniCNN runs and write the Chinese report")
    run_parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    subparsers.add_parser("report", help="write the probe/report figures without starting formal training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_all(args.device)
    else:
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
