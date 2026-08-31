"""Command line entry points for protocol-bound local_s0 runners."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocols import DEFAULT_PROTOCOL_PATH
from .runners import (
    run_bn,
    run_check,
    run_headlines,
    run_kernel,
    run_ols,
    run_partial,
    run_s0,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = [
    REPO_ROOT / "MACHINE_ASSETS.md",
    REPO_ROOT / "ckpts/phase2/resnet18__G64__20260820/checkpoints/best_slim.pt",
    REPO_ROOT / "results/today_shortcycle/a10/frozen_g64/ols_G64/field.npz",
]
DEFAULT_ROLES = ["machine_assets", "checkpoint_6d", "field_rendered_xy"]


def _default_output(name: str) -> Path:
    return Path("results") / "closeout_sprint_clean" / name


def _lock_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    command.add_argument("--asset-manifest", type=Path, default=None)
    command.add_argument("--source-machine", default="local")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protocol-bound clean-room local S0 audit tools")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="read-only asset and environment preflight")
    check.add_argument("--asset", action="append", dest="assets", help="asset path; repeatable")
    check.add_argument("--role", action="append", dest="roles", help="role matching each --asset (repeatable)")
    check.add_argument("--output", type=Path, default=_default_output("check"))
    _lock_options(check)

    headline = sub.add_parser("audit-headlines", help="recompute pred/true headline metrics")
    headline.add_argument("--input", action="append", dest="inputs", required=True, help="field.npz; repeatable")
    headline.add_argument("--output", type=Path, default=_default_output("headlines"))
    headline.add_argument("--support-index", action="append", type=int, default=None)
    headline.add_argument("--run-id", action="append", dest="run_ids", default=None, help="retained for CLI compatibility")
    _lock_options(headline)

    ols = sub.add_parser("audit-ols", help="run float64 SVD/ridge/whitening audit")
    ols.add_argument("--input", type=Path, required=True, help="NPZ containing separated train/eval arrays")
    ols.add_argument("--output", type=Path, default=_default_output("ols"))
    ols.add_argument("--feature-key")
    ols.add_argument("--target-key")
    ols.add_argument("--svd-cutoff", action="append", type=float, dest="svd_cutoffs")
    ols.add_argument("--ridge-lambda", action="append", type=float, dest="ridge_lambdas")
    ols.add_argument("--perturbation-scale", action="append", type=float, dest="perturbation_scales")
    ols.add_argument("--no-standardize", action="store_true")
    ols.add_argument("--no-whiten", action="store_true")
    ols.add_argument("--seed", type=int, default=20260821)
    _lock_options(ols)

    kernel = sub.add_parser("audit-kernel", help="fit kernel predictor on fit rows and gate held-out rows")
    kernel.add_argument("--input", type=Path, required=True, help="NPZ with fit/heldout features and targets")
    kernel.add_argument("--output", type=Path, default=_default_output("kernel"))
    kernel.add_argument("--ridge-lambda", action="append", type=float, dest="ridge_lambdas")
    _lock_options(kernel)

    bn = sub.add_parser("audit-bn", help="run explicit model/data/evaluator BN-only control")
    bn.add_argument("--config", type=Path, required=True, help="JSON adapter config")
    bn.add_argument("--output", type=Path, default=_default_output("bn"))
    _lock_options(bn)

    partial = sub.add_parser("audit-partial", help="partial-out audit with independent task groups")
    partial.add_argument("--input", action="append", dest="inputs", required=False, help="named group input as group=path (repeatable)")
    partial.add_argument("--group", action="append", dest="groups", help="alias for --input group=path")
    partial.add_argument("--output", type=Path, default=_default_output("partial"))
    partial.add_argument("--block-size", type=float, default=None)
    _lock_options(partial)

    s0 = sub.add_parser("run-s0", help="execute check→headlines→OLS→kernel→BN→partial on a protocol-allowed local_s0 device")
    s0.add_argument("--output", type=Path, default=_default_output("s0"))
    _lock_options(s0)

    # There is intentionally no train subcommand.  argparse rejects it before
    # any model or GPU code can be loaded; a future GPU protocol must add a
    # separately reviewed command and stage.
    return parser


def _parse_group_inputs(values: list[str] | None, aliases: list[str] | None = None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in [*(values or []), *(aliases or [])]:
        if "=" not in raw:
            raise SystemExit(f"partial input must be group=path: {raw!r}")
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"partial input must be group=path: {raw!r}")
        if name in result:
            raise SystemExit(f"duplicate partial group: {name}")
        result[name] = Path(path)
    if not result:
        raise SystemExit("audit-partial requires at least one group=path input")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        paths = [Path(p) for p in args.assets] if args.assets else DEFAULT_ASSETS
        roles = args.roles or DEFAULT_ROLES
        if len(roles) != len(paths):
            raise SystemExit(f"--role count {len(roles)} must match asset count {len(paths)}")
        result = run_check(
            output_dir=args.output,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            input_paths=None if args.asset_manifest else paths,
            input_roles=None if args.asset_manifest else roles,
            source_machine=args.source_machine,
        )
        print(f"[check] status={result['decision']['status']} output={result['output_dir']}")
        return 0

    if args.command == "audit-headlines":
        result = run_headlines(
            args.inputs,
            output_dir=args.output,
            support_indices=args.support_index,
            run_ids=args.run_ids,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            source_machine=args.source_machine,
        )
        print(f"[audit-headlines] runs={result['metrics']['run_count']} output={result['output_dir']}")
        return 0

    if args.command == "audit-ols":
        result = run_ols(
            args.input,
            output_dir=args.output,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            feature_key=args.feature_key,
            target_key=args.target_key,
            svd_cutoffs=args.svd_cutoffs or (1e-14, 1e-12, 1e-10, 1e-8),
            ridge_lambdas=args.ridge_lambdas or (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
            perturbation_scales=args.perturbation_scales or (0.0, 1e-8, 1e-6),
            standardize=not args.no_standardize,
            whiten=not args.no_whiten,
            seed=args.seed,
            source_machine=args.source_machine,
        )
        print(f"[audit-ols] verdict={result['metrics']['stability']['verdict']} output={result['output_dir']}")
        return 0

    if args.command == "audit-kernel":
        result = run_kernel(
            args.input,
            output_dir=args.output,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            ridge_grid=args.ridge_lambdas,
            source_machine=args.source_machine,
        )
        print(f"[audit-kernel] status={result['decision']['status']} output={result['output_dir']}")
        return 0

    if args.command == "audit-bn":
        result = run_bn(
            args.config,
            output_dir=args.output,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            source_machine=args.source_machine,
        )
        print(f"[audit-bn] status={result['decision']['status']} output={result['output_dir']}")
        return 0 if result["decision"]["passed"] else 2

    if args.command == "audit-partial":
        result = run_partial(
            _parse_group_inputs(args.inputs, args.groups),
            output_dir=args.output,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            block_size=args.block_size,
            source_machine=args.source_machine,
        )
        print(f"[audit-partial] groups={result['metrics']['group_count']} output={result['output_dir']}")
        return 0

    if args.command == "run-s0":
        result = run_s0(
            output_dir=args.output,
            protocol_path=args.protocol,
            asset_manifest_path=args.asset_manifest,
            source_machine=args.source_machine,
        )
        print(f"[run-s0] status={result['decision']['status']} output={result['output_dir']}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
