"""Asset preparation and matrix orchestration."""

from __future__ import annotations

import time
import shutil
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .aggregate import aggregate_runs
from .assets import _asset_records, asset_bundle_sha256, validate_assets
from .data import load_batch_stream, load_support_bank, materialize_batch_stream, materialize_cache, sha256_file
from .evaluator import evaluate_run
from .models import build_model, load_exact_init, save_exact_init, seed_all
from .protocol import EXPECTED_SEEDS, REGIME_NAMES, SUPPORT_NAMES, load_protocol, protocol_hash, write_json
from .training import _stream_seed, run_condition


KNOWN_EXACT_INIT_SHA256 = {
    20260816: "e2edf8c57b8201ed5c973c4e57f9620e129a5cc62c033d93953c44a0ce0f0b2d",
    20260817: "1695d625c07b2232bfa53d98008342759a2253c595fcf42c4c1868ca13fe24ef",
    20260818: "01bae741789811bcb2e31bea270784187e144f323fc27540ce03e42dcc18b794",
}
EXECUTION_SUPPORT_ORDER = ("corners4", "G64", "G9", "G16")
EXECUTION_REGIME_ORDER = ("head_only", "full", "frozen_feature")


def _discover_exact_init_dir() -> Path | None:
    candidate = Path(__file__).resolve().parents[2] / "handoff" / "20260823_crossover_inputs" / "neural_affine_autorun_20260822" / "assets" / "corrected_init"
    return candidate if candidate.exists() else None


def prepare_assets(
    out_root: str | Path,
    *,
    protocol_path: str | Path | None = None,
    workers: int = 1,
    init_dir: str | Path | None = None,
    allow_generated_inits: bool = False,
) -> dict[str, Any]:
    """Generate fresh renderer cache and exact init assets."""

    cfg = load_protocol(protocol_path)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "cache"
    source_init_dir = Path(init_dir) if init_dir is not None else _discover_exact_init_dir()
    target_init_dir = root / "exact_inits"
    target_init_dir.mkdir(parents=True, exist_ok=True)
    manifest = materialize_cache(cache_dir, workers=int(workers), protocol=cfg)
    p_hash = protocol_hash(payload=cfg)
    init_rows = []
    for seed in EXPECTED_SEEDS:
        source_path = None if source_init_dir is None else next((candidate for candidate in (source_init_dir / f"s{seed}_init.pt", source_init_dir / f"seed_{seed}.pt") if candidate.exists()), None)
        target_path = target_init_dir / f"seed_{seed}.pt"
        if source_path is None:
            if not allow_generated_inits:
                raise FileNotFoundError(f"missing verified exact init for {seed}; pass --init-dir or allow_generated_inits for non-formal smoke")
            seed_all(seed)
            model = build_model()
            payload = save_exact_init(model, target_path, seed=seed, protocol_hash=p_hash)
            generated_sha = sha256_file(target_path)
            init_rows.append({**payload, "source": "generated", "source_protocol_hash": p_hash, "sha256": generated_sha, "target_sha256": generated_sha})
            continue
        source_sha = sha256_file(source_path)
        expected_sha = KNOWN_EXACT_INIT_SHA256.get(seed)
        if expected_sha is not None and source_sha != expected_sha:
            raise ValueError(f"verified exact init SHA mismatch for seed {seed}: {source_sha} != {expected_sha}")
        shutil.copyfile(source_path, target_path)
        seed_all(seed)
        model = build_model()
        payload = load_exact_init(target_path, model, expected_seed=seed)
        init_rows.append({
            "seed": seed,
            "source": str(source_path.resolve()),
            "source_sha256": source_sha,
            "target": str(target_path.resolve()),
            "target_sha256": sha256_file(target_path),
            "source_protocol_hash": payload.get("protocol_hash"),
            "source_code_hash": payload.get("code_hash"),
            "init_hash": payload.get("init_hash"),
            "variant": payload.get("variant"),
        })
    stream_rows = []
    for support_name in SUPPORT_NAMES:
        support_count = len(cfg["supports"][support_name])
        for seed in EXPECTED_SEEDS:
            stream_path = root / "batch_streams" / support_name / f"seed_{seed}.npy"
            row = materialize_batch_stream(stream_path, count=support_count, batch_size=int(cfg["training"]["batch_size"]), batches=int(cfg["training"]["steps"]), seed=_stream_seed(seed, support_name))
            sidecar_path = stream_path.with_suffix(".json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar.update({"support": support_name, "experiment_seed": int(seed), "stream_seed": int(_stream_seed(seed, support_name))})
            write_json(sidecar_path, sidecar)
            row.update({
                "support": support_name,
                "seed": int(seed),
                "path": str(stream_path.relative_to(root)).replace("\\", "/"),
                "sidecar_path": str(sidecar_path.relative_to(root)).replace("\\", "/"),
                "sidecar_sha256": sha256_file(sidecar_path),
            })
            stream_rows.append(row)
    cache_manifest_path = cache_dir / "manifest.json"
    write_json(root / "protocol.json", cfg)
    receipt = {
        "schema_version": 1,
        "kind": "prepare_receipt",
        "protocol_hash": p_hash,
        "cache_dir": str(cache_dir.resolve()),
        "init_dir": str(target_init_dir.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "cache_manifest": manifest,
        "exact_inits": init_rows,
        "batch_streams": stream_rows,
    }
    # Bind every generated asset to one canonical content hash before any
    # matrix arm can start.  Validation also writes the read-only audit JSON.
    records, prepare_errors = _asset_records(root, cfg)
    if prepare_errors:
        raise ValueError("freshly prepared assets failed validation: " + "; ".join(prepare_errors[:8]))
    receipt["asset_bundle_sha256"] = asset_bundle_sha256(records)
    write_json(root / "prepare_receipt.json", receipt)
    validate_assets(root, cfg)
    return receipt


def _condition_inputs(cache_dir: Path, support_name: str, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bank_images, bank_points = load_support_bank(cache_dir)
    tids = np.asarray(cfg["supports"][support_name], dtype=np.int64)
    return np.asarray(bank_images[tids]), np.asarray(bank_points[tids])


def run_matrix(
    root_dir: str | Path,
    *,
    protocol_path: str | Path | None = None,
    device: str = "cpu",
    steps: int | None = None,
    resume: bool = True,
    supports: tuple[str, ...] = EXECUTION_SUPPORT_ORDER,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    regimes: tuple[str, ...] = EXECUTION_REGIME_ORDER,
    tiny: bool = False,
    runs_subdir: str = "formal_runs",
) -> dict[str, Any]:
    """Run all requested matrix cells and independent dense evaluations."""

    root = Path(root_dir)
    cfg = load_protocol(protocol_path or (root / "protocol.json" if (root / "protocol.json").exists() else None))
    cache_dir = root / "cache"
    init_dir = root / ("smoke_exact_inits" if tiny else "exact_inits")
    if not cache_dir.exists() or not init_dir.exists():
        raise FileNotFoundError("prepare assets first: cache/ and exact_inits/ are required")
    asset_validation = None
    asset_bundle = None
    package_id = None
    package_manifest_sha256 = None
    if not tiny and runs_subdir == "formal_runs":
        # This call is deliberately before the first support load/model build.
        asset_validation = validate_assets(root, cfg, require_package_identity=True)
        asset_bundle = str(asset_validation["asset_bundle_sha256"])
        identity = asset_validation.get("package_identity") or {}
        package_id = identity.get("package_id")
        package_manifest_sha256 = identity.get("package_manifest_sha256")
    matrix_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for support_name in supports:
        if support_name not in SUPPORT_NAMES:
            raise ValueError(f"unknown support {support_name}")
        support_images, support_points = _condition_inputs(cache_dir, support_name, cfg)
        for seed in seeds:
            init_path = init_dir / f"seed_{seed}.pt"
            if not init_path.exists():
                raise FileNotFoundError(init_path)
            for regime in regimes:
                seed_all(seed)
                model = build_model(tiny=tiny)
                condition_dir = root / runs_subdir / support_name / str(seed) / regime
                stream_path = root / "batch_streams" / support_name / f"seed_{seed}.npy"
                stream_array = load_batch_stream(stream_path, count=len(support_images), batch_size=int(cfg["training"]["batch_size"]), batches=int(cfg["training"]["steps"]))
                cell_started = time.perf_counter()
                cell_started_utc = datetime.now(timezone.utc).isoformat()
                summary = run_condition(
                    model,
                    support_images,
                    support_points,
                    condition_dir,
                    protocol=cfg,
                    seed=seed,
                    support_name=support_name,
                    regime=regime,
                    init_checkpoint=init_path,
                    device=device,
                    steps=steps,
                    resume=resume,
                    batch_stream=stream_array,
                    batch_stream_sha256=sha256_file(stream_path),
                    asset_bundle_sha256=asset_bundle,
                    package_id=package_id,
                    package_manifest_sha256=package_manifest_sha256,
                )
                evaluator = evaluate_run(
                    condition_dir,
                    cache_dir,
                    protocol=cfg,
                    device=device,
                    model_factory=lambda tiny=tiny: build_model(tiny=tiny),
                    asset_bundle_sha256=asset_bundle,
                    package_id=package_id,
                    package_manifest_sha256=package_manifest_sha256,
                )
                cell_elapsed = float(time.perf_counter() - cell_started)
                row = {
                    "support": support_name,
                    "seed": seed,
                    "regime": regime,
                    "summary": summary,
                    "evaluator": evaluator,
                    "started_utc": cell_started_utc,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "train_elapsed_sec": float(summary.get("elapsed_sec", 0.0)),
                    "eval_elapsed_sec": float(evaluator.get("elapsed_sec", 0.0)),
                    "cell_elapsed_sec": cell_elapsed,
                }
                rows.append(row)
                progress_name = "formal_matrix_progress.json" if runs_subdir == "formal_runs" else "smoke_matrix_progress.json"
                expected_cells = len(supports) * len(seeds) * len(regimes)
                observed_by_regime = {}
                for observed_regime in sorted({str(item["regime"]) for item in rows}):
                    samples = [float(item["cell_elapsed_sec"]) for item in rows if str(item["regime"]) == observed_regime]
                    observed_by_regime[observed_regime] = {"n": len(samples), "mean_cell_elapsed_sec": float(np.mean(samples))}
                remaining = max(0, expected_cells - len(rows))
                mean_elapsed = float(np.mean([float(item["cell_elapsed_sec"]) for item in rows]))
                write_json(root / progress_name, {
                    "completed": len(rows),
                    "expected": expected_cells,
                    "remaining": remaining,
                    "last": row,
                    "asset_bundle_sha256": asset_bundle,
                    "package_id": package_id,
                    "package_manifest_sha256": package_manifest_sha256,
                    "observed_by_regime": observed_by_regime,
                    "eta_sec": float(remaining * mean_elapsed),
                    "elapsed_sec": time.perf_counter() - matrix_started,
                })
    elapsed_sec = float(time.perf_counter() - matrix_started)
    result = {"schema_version": 1, "kind": "matrix_receipt", "protocol_hash": protocol_hash(payload=cfg), "asset_bundle_sha256": asset_bundle, "package_id": package_id, "package_manifest_sha256": package_manifest_sha256, "asset_validation": asset_validation, "completed": len(rows), "rows": rows, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": elapsed_sec}
    receipt_name = "formal_matrix_receipt.json" if runs_subdir == "formal_runs" else "smoke_matrix_receipt.json"
    write_json(root / receipt_name, result)
    return result


def run_smoke(
    root_dir: str | Path,
    *,
    steps: int = 2,
    device: str = "cpu",
    tiny: bool = True,
) -> dict[str, Any]:
    """Run a short local CPU smoke on all regimes and four supports.

    ``tiny=True`` is the default so smoke validates I/O, exact-init loading,
    resume and evaluator semantics without spending the formal ResNet budget.
    """

    root = Path(root_dir)
    cfg = load_protocol(root / "protocol.json" if (root / "protocol.json").exists() else None)
    if not (root / "cache").exists():
        prepare_assets(root, workers=1)
    if tiny:
        smoke_init = root / "smoke_exact_inits"
        smoke_init.mkdir(parents=True, exist_ok=True)
        p_hash = protocol_hash(payload=cfg)
        for seed in EXPECTED_SEEDS[:1]:
            seed_all(seed)
            save_exact_init(build_model(tiny=True), smoke_init / f"seed_{seed}.pt", seed=seed, protocol_hash=p_hash)
        return run_matrix(root, device=device, steps=int(steps), resume=False, supports=SUPPORT_NAMES, seeds=EXPECTED_SEEDS[:1], regimes=REGIME_NAMES, tiny=True, runs_subdir="smoke_runs")
    return run_matrix(root, device=device, steps=max(20, int(steps)), resume=False, supports=("corners4", "G64"), seeds=EXPECTED_SEEDS[:1], regimes=("head_only", "full"), tiny=False, runs_subdir="smoke_runs")


__all__ = ["prepare_assets", "run_matrix", "run_smoke"]
