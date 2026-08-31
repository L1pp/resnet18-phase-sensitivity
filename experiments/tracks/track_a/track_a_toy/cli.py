"""Command line entry points for the independent Track A-Toy experiment."""

from __future__ import annotations

import argparse

from .pipeline import (
    check_environment,
    prepare_datasets,
    probe_features,
    random_features,
    train_models,
    write_report,
)
from .wallclock import run_wallclock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m track_a_toy.cli",
        description="Run the small, independent Track A-Toy mechanism experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function, help_text in (
        ("check", check_environment, "check the local runtime, shapes and renderer counts"),
        ("prepare", prepare_datasets, "materialize finite/torus inputs and input montages"),
        ("random-features", random_features, "extract frozen random GAP features and on-demand shift diagnostics"),
        ("probe", probe_features, "fit the fixed-split ridge probes"),
        ("report", write_report, "write the JSON and Markdown report"),
        ("wallclock", run_wallclock, "measure the complete wall-clock workflow and isolated speed profile"),
        ("train", train_models, "optional short training; not part of the default no-train workflow"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        if name in {"check", "random-features", "wallclock", "train"}:
            command.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu (default: CUDA when available)")
        command.set_defaults(function=function)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    kwargs = {}
    if hasattr(args, "device"):
        kwargs["device_name"] = args.device
    args.function(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
