"""Local pure-logic CLI for protocol validation and result classification.

The CLI intentionally has no train/run command.  It only consumes JSON and
prints deterministic decisions; cloud paths, devices, and model execution
remain the responsibility of the execution AI on each server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .constants import CANDIDATES, GATES, validate_protocol_constants
from .decisions import classify_outcome, select_5k_candidate


def _load_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sgd_anchor_rescue",
        description="Cloud-agnostic SGD anchor-rescue protocol utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate-config",
        help="validate frozen constants and print candidates/gates",
    )

    select = subparsers.add_parser(
        "select",
        help="apply deterministic 5k A/B selection to a JSON mapping",
    )
    select.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="JSON file, or - for stdin; contains A/B 1k/5k metrics",
    )

    classify = subparsers.add_parser(
        "classify",
        help="classify a frozen stage-2 JSON outcome",
    )
    classify.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="JSON file, or - for stdin",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        errors = validate_protocol_constants()
        _dump(
            {
                "valid": not errors,
                "errors": list(errors),
                "candidates": {name: config.as_dict() for name, config in CANDIDATES.items()},
                "gates": {name: gate.as_dict() for name, gate in GATES.items()},
            }
        )
        return 0 if not errors else 1
    if args.command == "select":
        payload = _load_json(args.input)
        if not isinstance(payload, dict):
            raise SystemExit("select input must be a JSON object keyed by A/B")
        decision = select_5k_candidate(payload)
        _dump(decision.as_dict())
        return 0 if decision.winner is not None else 2
    if args.command == "classify":
        payload = _load_json(args.input)
        if not isinstance(payload, dict):
            raise SystemExit("classify input must be a JSON object")
        required = (
            "endpoint",
            "first_match",
            "u0",
            "raw_box0",
            "adamw_u",
            "threshold_px",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise SystemExit("classify input missing: " + ", ".join(missing))
        category = classify_outcome(
            endpoint=payload["endpoint"],
            first_match=payload["first_match"],
            u0=payload["u0"],
            raw_box0=payload["raw_box0"],
            adamw_u=payload["adamw_u"],
            threshold_px=payload["threshold_px"],
            final_gate=payload.get("final_gate"),
            anchor_matched=payload.get("anchor_matched"),
            remediation_action=payload.get("remediation_action"),
        )
        _dump({"category": category.value})
        return 0
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
