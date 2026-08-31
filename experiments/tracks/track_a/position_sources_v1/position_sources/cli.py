"""Command line boundary for local Track A preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import DEFAULT_PROTOCOL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="position-sources")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="write protocol, dense split metadata and lazy family caches")
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    prepare.add_argument("--family", action="append", choices=("padding", "stride", "torus"))
    prepare.add_argument("--primitive", action="append", choices=("blob", "triangle"))
    prepare.add_argument("--quick", action="store_true", help="small non-formal cache subset with images")

    preflight = sub.add_parser("preflight", help="verify a prepared root read-only")
    preflight.add_argument("--root", required=True, type=Path)

    smoke_cmd = sub.add_parser("smoke", help="run lightweight CPU model smoke")
    smoke_cmd.add_argument("--out", required=True, type=Path)
    smoke_cmd.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    smoke_cmd.add_argument("--variant", action="append", dest="variants")
    smoke_cmd.add_argument("--input-size", type=int, default=None)

    profile_cmd = sub.add_parser("profile", help="run lightweight CPU timing profile")
    profile_cmd.add_argument("--out", required=True, type=Path)
    profile_cmd.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    profile_cmd.add_argument("--variant", action="append", dest="variants")
    profile_cmd.add_argument("--input-size", type=int, default=None)
    profile_cmd.add_argument("--backward", action="store_true")

    run_cmd = sub.add_parser("run-family", help="run a formal family only after explicit authorization")
    run_cmd.add_argument("--root", required=True, type=Path)
    run_cmd.add_argument("--family", required=True, choices=("padding", "stride", "torus"))
    run_cmd.add_argument("--primitive", choices=("blob", "triangle"), default="blob")
    run_cmd.add_argument("--variant", action="append", dest="variants")
    run_cmd.add_argument("--seed", action="append", dest="seeds", type=int)
    run_cmd.add_argument("--device", default="cpu")
    run_cmd.add_argument("--steps", type=int, default=None)
    run_cmd.add_argument(
        "--microbatch",
        type=int,
        default=None,
        help="physical microbatch divisor of frozen effective batch 16; one optimizer step remains one logical batch",
    )
    run_cmd.add_argument("--comparison-block-id", required=True)
    run_cmd.add_argument("--run-id", required=True)
    run_cmd.add_argument("--resume-from", type=Path, default=None, help="in-progress periodic checkpoint for the same arm identity")
    run_cmd.add_argument("--allow-formal", action="store_true")

    reference_cmd = sub.add_parser("run-reference", help="run the reference-only zero-padding BatchNorm arm")
    reference_cmd.add_argument("--root", required=True, type=Path)
    reference_cmd.add_argument("--primitive", choices=("blob", "triangle"), default="blob")
    reference_cmd.add_argument("--seed", action="append", dest="seeds", type=int)
    reference_cmd.add_argument("--device", default="cpu")
    reference_cmd.add_argument("--steps", type=int, default=None)
    reference_cmd.add_argument("--microbatch", type=int, default=None)
    reference_cmd.add_argument("--comparison-block-id", required=True)
    reference_cmd.add_argument("--run-id", required=True)
    reference_cmd.add_argument("--resume-from", type=Path, default=None)
    reference_cmd.add_argument("--allow-formal", action="store_true")

    evaluate = sub.add_parser("evaluate", help="evaluate one checkpoint with committed predictions")
    evaluate.add_argument("--root", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--device", default="cpu")

    pack = sub.add_parser("pack", help="create a local zip package")
    pack.add_argument("--root", required=True, type=Path)
    pack.add_argument("--out", required=True, type=Path)

    verify = sub.add_parser("verify", help="verify a prepared root or zip")
    verify.add_argument("--path", required=True, type=Path)

    review = sub.add_parser("review", help="run independent static review to a separate output root")
    review.add_argument("--root", required=True, type=Path)
    review.add_argument("--out", required=True, type=Path)
    review.add_argument("--allowed-reviewer-root", required=True, type=Path)
    review.add_argument("--review-id", required=True)

    block_review = sub.add_parser("review-block", help="review one comparison block from read-only artifacts")
    block_review.add_argument("--root", required=True, type=Path, help="read-only executor root")
    block_review.add_argument("--out", required=True, type=Path, help="separate reviewer writable root")
    block_review.add_argument("--allowed-reviewer-root", required=True, type=Path)
    block_review.add_argument("--comparison-block-id", required=True)
    block_review.add_argument("--config-id", default=None)
    block_review.add_argument("--family", required=True, choices=("padding", "stride", "torus"))
    block_review.add_argument("--variant", default=None)
    block_review.add_argument("--input", required=True, type=Path, dest="input_path")
    block_review.add_argument("--predictions", required=True, type=Path, dest="prediction_path")
    block_review.add_argument("--coordinates", required=True, type=Path, dest="coordinate_path")
    block_review.add_argument("--gap-train-features", required=True, type=Path, dest="gap_train_feature_path")
    block_review.add_argument("--gap-eval-features", required=True, type=Path, dest="gap_eval_feature_path")
    block_review.add_argument("--gap-train-coordinates", required=True, type=Path, dest="gap_train_coordinate_path")
    block_review.add_argument("--gap-eval-coordinates", required=True, type=Path, dest="gap_eval_coordinate_path")
    block_review.add_argument("--gap-feature-metadata", required=True, type=Path, dest="gap_feature_metadata_path")
    block_review.add_argument("--support-coordinates", required=True, type=Path, dest="support_coordinate_path")
    block_review.add_argument("--support-predictions", required=True, type=Path, dest="support_prediction_path")
    block_review.add_argument("--support-targets", required=True, type=Path, dest="support_target_path")
    block_review.add_argument(
        "--prediction-space",
        choices=("pixels_xy", "normalized_xy", "torus_sincos_xy"),
        default="pixels_xy",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from .experiment import prepare_root

        result = prepare_root(
            args.out,
            protocol_path=args.protocol,
            families=args.family,
            primitives=args.primitive,
            quick=args.quick,
        )
    elif args.command == "preflight":
        from .experiment import preflight_root

        result = preflight_root(args.root)
    elif args.command == "smoke":
        from .experiment import smoke

        result = smoke(args.out, protocol_path=args.protocol, variants=args.variants, input_size_override=args.input_size)
    elif args.command == "profile":
        from .experiment import profile

        result = profile(
            args.out,
            protocol_path=args.protocol,
            variants=args.variants,
            input_size_override=args.input_size,
            backward=args.backward,
        )
    elif args.command == "run-family":
        from .experiment import run_family

        result = run_family(
            args.root,
            family=args.family,
            primitive=args.primitive,
            variants=args.variants,
            seeds=args.seeds,
            device=args.device,
            steps=args.steps,
            microbatch=args.microbatch,
            allow_formal=args.allow_formal,
            comparison_block_id=args.comparison_block_id,
            run_id=args.run_id,
            resume_from=args.resume_from,
        )
    elif args.command == "run-reference":
        from .experiment import run_reference

        result = run_reference(
            args.root,
            primitive=args.primitive,
            seeds=args.seeds,
            device=args.device,
            steps=args.steps,
            microbatch=args.microbatch,
            allow_formal=args.allow_formal,
            comparison_block_id=args.comparison_block_id,
            run_id=args.run_id,
            resume_from=args.resume_from,
        )
    elif args.command == "evaluate":
        from .experiment import evaluate_checkpoint

        result = evaluate_checkpoint(args.root, args.checkpoint, device=args.device)
    elif args.command == "pack":
        from .packaging import pack_root

        result = pack_root(args.root, args.out)
    elif args.command == "verify":
        from .packaging import verify_package

        result = verify_package(args.path)
    elif args.command == "review-block":
        from .reviewer import review_block

        result = review_block(
            args.root,
            args.out,
            comparison_block_id=args.comparison_block_id,
            allowed_reviewer_root=args.allowed_reviewer_root,
            config_id=args.config_id,
            family=args.family,
            variant=args.variant,
            input_path=args.input_path,
            prediction_path=args.prediction_path,
            coordinate_path=args.coordinate_path,
            gap_train_feature_path=args.gap_train_feature_path,
            gap_eval_feature_path=args.gap_eval_feature_path,
            gap_train_coordinate_path=args.gap_train_coordinate_path,
            gap_eval_coordinate_path=args.gap_eval_coordinate_path,
            gap_feature_metadata_path=args.gap_feature_metadata_path,
            support_coordinate_path=args.support_coordinate_path,
            support_prediction_path=args.support_prediction_path,
            support_target_path=args.support_target_path,
            prediction_space=args.prediction_space,
        )
    else:
        from .reviewer import run_review

        result = run_review(
            args.root,
            args.out,
            allowed_reviewer_root=args.allowed_reviewer_root,
            review_id=args.review_id,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
