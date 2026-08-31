"""CLI for the Phase 1 GAP representation experiment.

Default stage is check. Training is a separate explicit command and must wait
until the frozen dataset has been reviewed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "phase1_gap_rep"

from .common import PROFILES, TRAIN_RUNS, build_model, device, environment_versions, profile_spec


def check() -> None:
    versions = environment_versions()
    for key, value in versions.items():
        print(f"[check] {key}={value}")
    missing = [key for key, value in versions.items() if str(value).startswith("unavailable")]
    if missing:
        raise RuntimeError(f"missing dependencies: {missing}; ask before installing")
    import torch

    print(f"[check] cuda={torch.cuda.is_available()} device={device()}")
    model = build_model()
    n_params = sum(item.numel() for item in model.parameters())
    print(f"[check] resnet18 Linear(512,6) params={n_params} avgpool={model.avgpool}")
    print("[check] profiles=" + ", ".join(sorted(PROFILES)))
    print("[check] train_runs=" + ", ".join(sorted(TRAIN_RUNS)))
    print("[check] ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 ResNet18+GAP Bézier representation experiment")
    parser.add_argument(
        "stage",
        nargs="?",
        default="check",
        choices=(
            "check",
            "prepare",
            "train",
            "eval",
            "extract",
            "analyze",
            "audit",
            "cuts",
            "bt_robust",
            "explore",
            "phase17",
            "phase17c",
            "phase18",
            "all",
        ),
    )
    parser.add_argument("--profile", default="factorial", choices=sorted(PROFILES))
    parser.add_argument(
        "--run",
        default=None,
        choices=(*sorted(TRAIN_RUNS), "all_l1"),
        help="new L1 training runs; writes to results/phase1_gap_rep/<run>/ and does not touch factorial wreckage",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue a run from last.pt")
    parser.add_argument("--seed", type=int, default=None, help="Phase 1.8: run a single training seed")
    args = parser.parse_args()
    spec = profile_spec(args.profile)
    if args.stage == "all":
        raise SystemExit("all 已禁用，避免误开训。请先 prepare，人工看预览图后再显式运行 train。")
    if args.stage == "check":
        check()
        return
    if args.stage == "prepare":
        from .generate_data import prepare

        prepare(spec.name, force=args.force)
        return
    if args.stage == "train":
        from .train import train_all_l1_runs, train_run

        if args.run is None:
            raise SystemExit(
                "train 需要 --run adamw_l1_resume|adamw_l1_scratch|sgd_l1|all_l1，"
                "避免覆盖 results/phase1_gap_rep/factorial 残骸。"
            )
        if args.run == "all_l1":
            train_all_l1_runs(force=args.force, resume=args.resume)
            return
        train_run(args.run, force=args.force, resume=args.resume)
        return
    if args.stage == "eval":
        from .train import evaluate, evaluate_run

        if args.run:
            if args.run == "all_l1":
                for name in sorted(TRAIN_RUNS):
                    evaluate_run(name)
                return
            evaluate_run(args.run)
            return
        evaluate(spec.name)
        return
    if args.stage == "extract":
        from .extract import extract_run

        if args.run is None or args.run == "all_l1":
            raise SystemExit(
                "extract 需要 --run <train_run>（例如 adamw_l1_scratch），"
                "避免覆盖 results/phase1_gap_rep/factorial 残骸。"
            )
        extract_run(args.run)
        return
    if args.stage == "analyze":
        from .analyze import analyze_run

        if args.run is None or args.run == "all_l1":
            raise SystemExit(
                "analyze 需要 --run <train_run>（例如 adamw_l1_scratch），"
                "避免覆盖 results/phase1_gap_rep/factorial 残骸。"
            )
        analyze_run(args.run)
        return
    if args.stage == "audit":
        from .audit import audit_run

        if args.run is None or args.run == "all_l1":
            raise SystemExit(
                "audit 需要 --run <train_run>（例如 adamw_l1_scratch），"
                "避免覆盖 results/phase1_gap_rep/factorial 残骸。"
            )
        audit_run(args.run)
        return
    if args.stage == "cuts":
        from .cuts import cuts_run

        if args.run is None or args.run == "all_l1":
            raise SystemExit(
                "cuts 需要 --run <train_run>（例如 adamw_l1_scratch），"
                "避免覆盖 results/phase1_gap_rep/factorial 残骸。"
            )
        cuts_run(args.run)
        return
    if args.stage == "bt_robust":
        from .bt_robustness import bt_robust

        if args.run is None or args.run == "all_l1":
            raise SystemExit("bt_robust 需要 --run adamw_l1_scratch（只读其特征，写入 phase1_6_bt_robustness/）")
        bt_robust(args.run)
        return
    if args.stage == "explore":
        from .bt_explore import run_explore

        if args.run is None or args.run == "all_l1":
            raise SystemExit("explore 需要 --run adamw_l1_scratch（只读其特征，写入 phase1_6_bt_robustness/explore/）")
        run_explore(args.run)
        return
    if args.stage == "phase17":
        from .phase17 import run_phase17

        run_phase17()
        return
    if args.stage == "phase17c":
        from .phase17c import run_phase17c

        run_phase17c()
        return
    if args.stage == "phase18":
        from .phase18 import SEEDS, run_phase18

        if args.seed is not None and args.seed not in SEEDS:
            raise SystemExit(f"phase18 --seed 必须是 {SEEDS} 之一")
        run_phase18(seed=args.seed)
        return
    raise ValueError(args.stage)


if __name__ == "__main__":
    main()
