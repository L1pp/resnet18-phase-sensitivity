"""Local mechanism probe. CPU first (S1/S4a), then 1060 (S2/S3/L3)."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from .blob_io import assert_blob_gate, device, ensure_results, write_json
from .const import BLOB_CKPT, CODE_REV, MACHINE, RESULTS_ROOT


def _env() -> dict:
    import torch

    return {
        "code_rev": CODE_REV,
        "machine": MACHINE,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "blob_ckpt": str(BLOB_CKPT),
        "blob_sha": assert_blob_gate(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="subset: fft rv opt tangent mlp report",
    )
    args = parser.parse_args(argv)
    wanted = set(args.only) if args.only else {"fft", "rv", "opt", "tangent", "mlp", "report"}
    ensure_results()
    env = _env()
    write_json(RESULTS_ROOT / "env.json", env)
    print(env, flush=True)
    t0 = time.time()
    if "fft" in wanted:
        from .fft_traj import run as run_fft

        run_fft()
    if "rv" in wanted:
        from .rv_table import run as run_rv

        run_rv()
    if "opt" in wanted:
        from .opt_audit import run as run_opt

        run_opt()
    if "tangent" in wanted:
        from .tangent_proj import run as run_tan

        run_tan()
    if "mlp" in wanted:
        from .mlp_phase import run as run_mlp

        run_mlp()
    if "report" in wanted:
        from .report import run as run_report

        run_report()
    print(f"[done] elapsed_h={(time.time() - t0) / 3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
