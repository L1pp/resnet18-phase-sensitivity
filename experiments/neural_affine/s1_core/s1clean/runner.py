from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .baselines import ExplicitXYMLP, closed_form_identity, run_explicit_xy_mlp
from .config import load_protocol, protocol_hash
from .data import load_cache, materialize_cache, render_points, sample_hashes
from .environment import environment_snapshot, gpu_gate
from .manifest import sha256_file, verify_manifest
from .metrics import affine_residual_mae_px, fit_affine, mae_px
from .models import build_model, load_model_state, seed_all, state_dict_hash
from .status import StatusStore
from .train import predict_images, train_run


CORRECTED_SEEDS = (20260816, 20260817, 20260818)
LEGACY_REFERENCE = {
    20260816: {"anchor_mae_px": 0.10829, "raw_full_box_mae_px": 43.11635, "affine_removed_mae_px": 30.93746},
    20260817: {"anchor_mae_px": 0.09841, "raw_full_box_mae_px": 43.75496, "affine_removed_mae_px": 23.78607},
    20260818: {"anchor_mae_px": 0.07558, "raw_full_box_mae_px": 61.86173, "affine_removed_mae_px": 26.30961},
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _gate(gates: Mapping[str, Any], *names: str, default: float) -> float:
    for name in names:
        if name in gates:
            return float(gates[name])
    return float(default)


def _legacy_reference_comparison(seed: int, metric: Mapping[str, Any]) -> dict[str, Any]:
    reference = LEGACY_REFERENCE[int(seed)]
    rows: dict[str, Any] = {}
    passed = True
    for name in ("raw_full_box_mae_px", "affine_removed_mae_px"):
        actual = float(metric[name])
        expected = float(reference[name])
        relative = abs(actual - expected) / max(abs(expected), 1e-12)
        rows[name] = {"actual": actual, "historical": expected, "relative_diff": relative, "tolerance": 0.05, "passed": relative <= 0.05}
        passed = passed and relative <= 0.05
    anchor_actual = float(metric["anchor_mae_px"])
    anchor_expected = float(reference["anchor_mae_px"])
    anchor_diff = abs(anchor_actual - anchor_expected)
    rows["anchor_mae_px"] = {
        "actual": anchor_actual,
        "historical": anchor_expected,
        "abs_diff": anchor_diff,
        "tolerance": 0.05,
        "passed": anchor_diff <= 0.05,
    }
    passed = passed and anchor_diff <= 0.05
    return {"passed": passed, "diagnostic_only": True, "fields": rows}


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _training_config(protocol: Mapping[str, Any], variant: str, package_hash: str, p_hash: str) -> dict[str, Any]:
    training = dict(protocol.get("training", {}))
    loss = dict(training.get("loss", {})) if isinstance(training.get("loss"), Mapping) else {}
    scheduler = dict(training.get("scheduler", {})) if isinstance(training.get("scheduler"), Mapping) else {}
    return {
        "steps": int(training.get("steps", 3000)),
        "batch_size": int(training.get("batch_size", 64)),
        "anchor_every": int(training.get("anchor_every", 100)),
        "resume_every": int(training.get("resume_every", 200)),
        "lr": float(training.get("lr", 1e-3)),
        "weight_decay": float(training.get("weight_decay", 1e-4)),
        "lr_min": float(training.get("lr_min", scheduler.get("eta_min", 1e-5))),
        "l1_weight": float(training.get("l1_weight", loss.get("l1_weight", 0.25))),
        "variant": variant,
        "protocol_hash": p_hash,
        "code_hash": package_hash,
    }


def cross_environment_cache(run_root: str | Path, protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(run_root)
    cpu_dir = root / "cache_cpu"
    cpu = load_cache(cpu_dir, verify=True)
    current_probe_images = render_points(cpu["probe_points"], workers=min(4, os.cpu_count() or 1), protocol=protocol)
    current_hashes = sample_hashes(current_probe_images, cpu["probe_points"])
    expected_hashes = list(cpu["manifest"]["probes"]["sample_sha256"])
    comparison = {
        "cpu_cache": str(cpu_dir),
        "probe_count": len(expected_hashes),
        "expected": expected_hashes,
        "current": current_hashes,
        "exact_match": current_hashes == expected_hashes,
        "environment": environment_snapshot(),
    }
    if comparison["exact_match"]:
        chosen = cpu_dir
        comparison["chosen_cache"] = "cpu"
    else:
        chosen = root / "cache_gpu"
        materialize_cache(chosen, workers=min(8, os.cpu_count() or 1), protocol=protocol)
        comparison["chosen_cache"] = "gpu_regenerated"
    _write_json(root / "cache_environment_comparison.json", comparison)
    return chosen, comparison


def evaluate_trained_run(
    run_dir: str | Path,
    cache_dir: str | Path,
    *,
    variant: str,
    seed: int,
    coord_scale: float,
    device: str,
) -> dict[str, Any]:
    output = Path(run_dir)
    cache = load_cache(cache_dir, verify=True)
    seed_all(seed)
    model = build_model(variant)
    checkpoint = output / "checkpoints" / "best.pt"
    payload = load_model_state(checkpoint, model, variant=variant, device=device)
    return evaluate_model(
        model,
        output,
        cache,
        variant=variant,
        seed=seed,
        coord_scale=coord_scale,
        device=device,
        checkpoint=checkpoint,
        checkpoint_payload=payload,
    )


def evaluate_model(
    model: torch.nn.Module,
    output: Path,
    cache: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
    coord_scale: float,
    device: str,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model.to(device).eval()
    pred_norm = predict_images(model, cache["dense_images"], device=device, batch_size=64)
    anchor_pred_norm = predict_images(model, cache["support_images"], device=device, batch_size=64)
    pred_px = pred_norm.astype(np.float64) * coord_scale
    anchor_pred_px = anchor_pred_norm.astype(np.float64) * coord_scale
    true_px = np.asarray(cache["dense_points"], dtype=np.float64)
    anchor_true_px = np.asarray(cache["support_points"], dtype=np.float64)
    raw = mae_px(pred_px, true_px, coord_scale=1.0)
    residual = affine_residual_mae_px(pred_px, true_px, coord_scale=1.0)
    anchor = mae_px(anchor_pred_px, anchor_true_px, coord_scale=1.0)
    matrix, bias = fit_affine(pred_px, true_px)
    np.savez_compressed(
        output / "predictions.npz",
        pred_px=pred_px,
        true_px=true_px,
        anchor_pred_px=anchor_pred_px,
        anchor_true_px=anchor_true_px,
    )
    metrics = {
        "run_id": output.name,
        "variant": variant,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "checkpoint_init_hash": checkpoint_payload.get("init_hash") if isinstance(checkpoint_payload, Mapping) else None,
        "anchor_mae_px": anchor,
        "raw_full_box_mae_px": raw,
        "affine_removed_mae_px": residual,
        "affine_matrix": matrix.tolist(),
        "affine_bias": bias.tolist(),
        "n_full_box": int(len(true_px)),
        "n_anchor": int(len(anchor_true_px)),
    }
    _write_json(output / "metrics.json", metrics)
    return metrics


def run_legacy_simulated_sequence(
    package: Path,
    root: Path,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Diagnostic fallback preserving the historical build-before-seed order.

    This intentionally stays in one process.  The model for seed N+1 is built
    from the RNG state left by training seed N, then the training seed is set.
    Formal corrected runs never use this path.
    """

    protocol = load_protocol(package / "protocol.json")
    p_hash = protocol_hash(package / "protocol.json")
    package_manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    package_hash = str(package_manifest["aggregate_sha256"])
    cache = load_cache(cache_dir, verify=True)
    coord_scale = float(protocol.get("coord_scale", 223.0))
    targets_norm = np.asarray(cache["support_points"], dtype=np.float32) / coord_scale
    # Historical prelude: a 2->32->2 MLP was built before the seed was reset
    # for its training.  Even though this path is diagnostic-only, reproduce
    # the full RNG/action order before constructing vanilla seed 16.
    seed_all(20260816)
    prelude_model = ExplicitXYMLP().to("cuda")
    prelude_init_hash = state_dict_hash(prelude_model.state_dict())
    seed_all(20260816)
    support_norm = torch.from_numpy(targets_norm).to("cuda")
    batch_size = int(dict(protocol.get("training", {})).get("batch_size", 64))
    repeats = (batch_size + len(support_norm) - 1) // len(support_norm)
    prelude_batch = support_norm.repeat((repeats, 1))[:batch_size]
    prelude_optimizer = torch.optim.AdamW(prelude_model.parameters(), lr=1e-3, weight_decay=1e-4)
    prelude_steps = int(dict(protocol.get("training", {})).get("steps", 3000))
    prelude_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        prelude_optimizer, T_max=prelude_steps, eta_min=1e-5
    )
    prelude_last_loss = math.nan
    for _ in range(prelude_steps):
        prelude_optimizer.zero_grad(set_to_none=True)
        prediction = prelude_model(prelude_batch)
        loss = torch.mean((prediction - prelude_batch) ** 2) + 0.25 * torch.mean(torch.abs(prediction - prelude_batch))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite legacy MLP prelude")
        loss.backward()
        prelude_optimizer.step()
        prelude_scheduler.step()
        prelude_last_loss = float(loss.detach().cpu())
    _write_json(
        root / "outputs" / "legacy_order_simulation_mlp_prelude" / "metrics.json",
        {
            "diagnostic_mode": "legacy_order_simulation_rng_prelude",
            "seed": 20260816,
            "steps": prelude_steps,
            "init_hash": prelude_init_hash,
            "final_loss": prelude_last_loss,
        },
    )
    rows: list[dict[str, Any]] = []
    for seed in CORRECTED_SEEDS:
        model = build_model("vanilla")
        simulated_init_hash = state_dict_hash(model.state_dict())
        seed_all(seed)
        run_id = f"legacy_order_simulation_s{seed}"
        output = root / "outputs" / run_id
        train_run(
            model,
            cache["support_images"],
            targets_norm,
            cache["support_images"],
            targets_norm,
            output,
            _training_config(protocol, "vanilla", package_hash, p_hash),
            seed,
            device="cuda",
            resume=True,
        )
        best_path = output / "checkpoints" / "best.pt"
        best_payload = _torch_load(best_path)
        model.load_state_dict(best_payload["model_state"], strict=True)
        metric = evaluate_model(
            model,
            output,
            cache,
            variant="vanilla",
            seed=seed,
            coord_scale=coord_scale,
            device="cuda",
            checkpoint=best_path,
            checkpoint_payload=best_payload,
        )
        metric.update(
            {
                "diagnostic_mode": "legacy_order_simulation",
                "simulated_init_hash": simulated_init_hash,
                "equivalence_claim": False,
            }
        )
        _write_json(output / "metrics.json", metric)
        rows.append(metric)
    return rows


def run_one(
    package_root: str | Path,
    run_root: str | Path,
    cache_dir: str | Path,
    *,
    run_id: str,
    variant: str,
    seed: int,
    init_source: str | Path | None = None,
    device: str = "cuda",
    seed_before_build: bool = True,
) -> dict[str, Any]:
    package = Path(package_root)
    root = Path(run_root)
    output = root / "outputs" / run_id
    protocol = load_protocol(package / "protocol.json")
    p_hash = protocol_hash(package / "protocol.json")
    manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    package_hash = str(manifest["aggregate_sha256"])
    cache = load_cache(cache_dir, verify=True)
    coord_scale = float(protocol.get("coord_scale", 223.0))
    targets_norm = np.asarray(cache["support_points"], dtype=np.float32) / coord_scale
    # Corrected runs freeze seed -> construct.  An exact historical init is a
    # diagnostic replay: construct first, reset the historical training seed,
    # then overwrite weights from the sealed init.  This keeps the data/order
    # RNG at the historical seed while exact weights make the throwaway fresh
    # construction irrelevant.
    effective_seed_before_build = bool(seed_before_build and init_source is None)
    if effective_seed_before_build:
        seed_all(seed)
        model = build_model(variant)
    else:
        model = build_model(variant)
        seed_all(seed)
    config = _training_config(protocol, variant, package_hash, p_hash)
    summary = train_run(
        model,
        cache["support_images"],
        targets_norm,
        cache["support_images"],
        targets_norm,
        output,
        config,
        seed,
        init_source=init_source,
        device=device,
        resume=True,
    )
    metrics = evaluate_trained_run(
        output,
        cache_dir,
        variant=variant,
        seed=seed,
        coord_scale=coord_scale,
        device=device,
    )
    return {"summary": summary, "metrics": metrics}


def _subprocess_run_one(
    package: Path,
    run_root: Path,
    cache_dir: Path,
    *,
    run_id: str,
    variant: str,
    seed: int,
    init_source: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "s1clean.cli",
        "run-one",
        "--package-root",
        str(package),
        "--run-root",
        str(run_root),
        "--cache-dir",
        str(cache_dir),
        "--run-id",
        run_id,
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if init_source is not None:
        command.extend(["--init-source", str(init_source)])
    log_path = run_root / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=package, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"run {run_id} failed with exit code {process.returncode}; see {log_path}")
    return json.loads((run_root / "outputs" / run_id / "metrics.json").read_text(encoding="utf-8"))


def _verdict(corrected: list[dict[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    anchor_gate = _gate(gates, "anchor_valid_px", "anchor_valid_mae_px", default=0.25)
    raw_gate = _gate(gates, "raw_positive_px", "raw_positive_mae_px", default=20.0)
    residual_gate = _gate(
        gates,
        "residual_positive_px",
        "affine_residual_positive_mae_px",
        default=10.0,
    )
    rows = []
    for metric in corrected:
        valid = float(metric["anchor_mae_px"]) <= anchor_gate
        positive = valid and float(metric["raw_full_box_mae_px"]) >= raw_gate and float(metric["affine_removed_mae_px"]) >= residual_gate
        rows.append({"seed": metric["seed"], "valid": valid, "positive": positive})
    if len(rows) != 3 or not all(row["valid"] for row in rows):
        label = "inconclusive"
    else:
        positives = sum(bool(row["positive"]) for row in rows)
        label = "strong_replication" if positives == 3 else "replicated" if positives >= 2 else "not_replicated"
    return {"verdict": label, "per_seed": rows, "gates": dict(gates)}


def _audit_completed_runs(
    runs: list[dict[str, Any]], root: Path, protocol_path: Path, *, device: str
) -> tuple[list[dict[str, Any]], bool]:
    from .independent_eval import audit_run

    rows: list[dict[str, Any]] = []
    failed = False
    for metric in runs:
        run_dir = root / "outputs" / str(metric["run_id"])
        audit = audit_run(run_dir, protocol_path, device=device)
        passed = bool(audit.get("passed")) or str(audit.get("status", "")).lower() == "passed"
        rows.append({"run_id": metric["run_id"], "passed": passed, "report": str(run_dir / "independent_audit.json")})
        failed = failed or not passed
    return rows, failed


def run_all(package_root: str | Path, run_root: str | Path, *, hard_new_work_sec: int = 6300) -> dict[str, Any]:
    package = Path(package_root).resolve()
    root = Path(run_root).resolve()
    status = StatusStore(root / "status")
    manifest_ok, manifest_failures = verify_manifest(package)
    if not manifest_ok:
        status.update("FAILED", reason="gpu_package_manifest", failures=manifest_failures)
        raise RuntimeError(f"GPU package manifest verification failed: {manifest_failures}")
    cpu_preflight_path = root / "cpu_preflight.json"
    if not cpu_preflight_path.is_file():
        status.update("FAILED", reason="missing_cpu_preflight")
        raise RuntimeError(f"missing CPU preflight record: {cpu_preflight_path}")
    cpu_preflight = json.loads(cpu_preflight_path.read_text(encoding="utf-8"))
    current_manifest_sha256 = sha256_file(package / "package_manifest.json")
    if current_manifest_sha256 != cpu_preflight.get("package_manifest_sha256"):
        status.update(
            "FAILED",
            reason="cpu_gpu_package_drift",
            expected=cpu_preflight.get("package_manifest_sha256"),
            actual=current_manifest_sha256,
        )
        raise RuntimeError("package manifest changed after CPU preflight")
    live_gpu = gpu_gate(require_a10=True)
    status.update("GPU_RUNNING", package_root=str(package), run_root=str(root), gpu_environment=live_gpu)
    protocol = load_protocol(package / "protocol.json")
    gates = dict(protocol.get("gates", {}))
    coord_scale = float(protocol.get("coord_scale", 223.0))
    started = time.time()
    cache_dir, cache_comparison = cross_environment_cache(root, protocol)
    cache = load_cache(cache_dir, verify=True)

    baseline_root = root / "outputs" / "baselines"
    closed = closed_form_identity(cache["support_points"], cache["dense_points"])
    _write_json(baseline_root / "closed_form" / "metrics.json", closed)
    mlp = run_explicit_xy_mlp(
        cache["support_points"],
        cache["dense_points"],
        baseline_root / "explicit_xy_mlp",
        seed=20260816,
        coord_scale=coord_scale,
        steps=int(dict(protocol.get("training", {})).get("steps", 3000)),
        batch_size=int(dict(protocol.get("training", {})).get("batch_size", 64)),
        device="cuda",
    )
    if closed["raw_mae_px"] > _gate(gates, "closed_form_mae_px", default=1e-6):
        status.update("INCONCLUSIVE", reason="closed_form_baseline", metrics=closed)
        return {"verdict": "inconclusive", "reason": "closed_form_baseline"}
    if mlp["anchor_mae_px"] > _gate(gates, "mlp_anchor_mae_px", default=0.25) or mlp["raw_mae_px"] >= _gate(gates, "mlp_raw_mae_px", default=5.0):
        status.update("INCONCLUSIVE", reason="explicit_xy_mlp_baseline", metrics=mlp)
        return {"verdict": "inconclusive", "reason": "explicit_xy_mlp_baseline"}

    corrected: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    corrected_early_reason: str | None = None
    anchor_gate = _gate(gates, "anchor_valid_px", "anchor_valid_mae_px", default=0.25)
    for seed in CORRECTED_SEEDS:
        if time.time() - started >= hard_new_work_sec:
            corrected_early_reason = "hard_new_work_deadline"
            break
        metric = _subprocess_run_one(
            package, root, cache_dir, run_id=f"corrected_vanilla_s{seed}", variant="vanilla", seed=seed
        )
        corrected.append(metric)
        all_runs.append(metric)
        if float(metric["anchor_mae_px"]) > anchor_gate:
            corrected_early_reason = "invalid_corrected_seed"
            break

    if corrected_early_reason is not None:
        audit_rows, audit_failed = _audit_completed_runs(
            all_runs, root, package / "protocol.json", device="cuda"
        )
        verdict = _verdict(corrected, gates)
        verdict["verdict"] = "inconclusive"
        verdict["reason"] = corrected_early_reason
        aggregate = {
            **verdict,
            "cache_comparison": cache_comparison,
            "baselines": {"closed_form": closed, "explicit_xy_mlp": mlp},
            "runs": all_runs,
            "audits": audit_rows,
            "audit_failed": audit_failed,
            "diagnostic_complete": False,
            "elapsed_sec": time.time() - started,
            "environment": environment_snapshot(),
        }
        _write_json(root / "aggregate_decision.json", aggregate)
        status.update("INCONCLUSIVE", reason=corrected_early_reason, verdict="inconclusive")
        return aggregate

    init_manifest = json.loads((root / "inputs" / "legacy_init" / "manifest.json").read_text(encoding="utf-8"))
    legacy_mode = str(init_manifest.get("mode", "legacy_order_simulation"))
    if legacy_mode == "exact_init_replay":
        for seed in CORRECTED_SEEDS:
            if time.time() - started >= hard_new_work_sec:
                break
            init_path = root / "inputs" / "legacy_init" / f"s{seed}_init.pt"
            state = _torch_load(init_path)
            source_state = state["model_state"] if isinstance(state, Mapping) and "model_state" in state else state
            expected = next(row["state_dict_sha256"] for row in init_manifest["runs"] if int(row["seed"]) == seed)
            if state_dict_hash(source_state) != expected:
                legacy_mode = "legacy_order_simulation"
                break
            metric = _subprocess_run_one(
                package,
                root,
                cache_dir,
                run_id=f"legacy_exact_vanilla_s{seed}",
                variant="vanilla",
                seed=seed,
                init_source=init_path,
            )
            metric["diagnostic_mode"] = "exact_init_replay"
            metric["historical_comparison"] = _legacy_reference_comparison(seed, metric)
            _write_json(root / "outputs" / f"legacy_exact_vanilla_s{seed}" / "metrics.json", metric)
            all_runs.append(metric)
    if legacy_mode != "exact_init_replay":
        # Do not mix a partial exact branch with simulation.  Exact assets are
        # currently preflighted all-or-nothing, so this is only a defensive gate.
        if any(str(row.get("diagnostic_mode")) == "exact_init_replay" for row in all_runs):
            raise RuntimeError("exact init changed after a partial legacy replay; refusing a mixed diagnostic branch")
        all_runs.extend(run_legacy_simulated_sequence(package, root, cache_dir))

    for seed in CORRECTED_SEEDS:
        if time.time() - started >= hard_new_work_sec:
            break
        metric = _subprocess_run_one(
            package, root, cache_dir, run_id=f"corrected_coordconv_s{seed}", variant="coordconv", seed=seed
        )
        metric["diagnostic_mode"] = "coordconv_control"
        _write_json(root / "outputs" / f"corrected_coordconv_s{seed}" / "metrics.json", metric)
        all_runs.append(metric)

    diagnostic_complete = len(all_runs) == 9
    audit_rows, audit_failed = _audit_completed_runs(
        all_runs, root, package / "protocol.json", device="cuda"
    )
    verdict = _verdict(corrected, gates)
    if not diagnostic_complete:
        verdict["verdict"] = "inconclusive"
        verdict["reason"] = "diagnostic_runs_incomplete"
    elif audit_failed:
        verdict["verdict"] = "inconclusive"
        verdict["reason"] = "independent_audit_failed"
    aggregate = {
        **verdict,
        "cache_comparison": cache_comparison,
        "baselines": {"closed_form": closed, "explicit_xy_mlp": mlp},
        "runs": all_runs,
        "diagnostic_complete": diagnostic_complete,
        "audits": audit_rows,
        "elapsed_sec": time.time() - started,
        "environment": environment_snapshot(),
    }
    _write_json(root / "aggregate_decision.json", aggregate)
    status.update("INCONCLUSIVE" if aggregate["verdict"] == "inconclusive" else "DONE", verdict=aggregate["verdict"])
    return aggregate


__all__ = ["cross_environment_cache", "evaluate_trained_run", "run_all", "run_one"]
