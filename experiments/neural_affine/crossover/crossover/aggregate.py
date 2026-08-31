"""Aggregate independent condition evaluators into the frozen crossover gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .assets import validate_assets
from .protocol import SUPPORT_NAMES, load_protocol, protocol_hash, write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[int(order[j])] == values[int(order[i])]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xr, yr = _rank(x), _rank(y)
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _condition_dirs(root: Path) -> list[Path]:
    # ``smoke_runs`` is intentionally not part of formal aggregation.
    return sorted(path for path in root.joinpath("formal_runs").glob("*/*/*") if path.is_dir())


def aggregate_runs(root_dir: str | Path, *, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_protocol() if protocol is None else dict(protocol)
    root = Path(root_dir)
    p_hash = protocol_hash(payload=cfg)
    conditions: list[dict[str, Any]] = []
    expected_steps = int(cfg["training"]["steps"])
    validation_path = root / "formal_asset_validation.json"
    formal_validation = None
    # Re-read the immutable cache/stream/init files when a formal asset tree
    # is present.  The persisted JSON remains useful for synthetic fixtures,
    # but a writable validation report alone must never authorize aggregation.
    if (root / "cache").exists() and (root / "exact_inits").exists():
        try:
            formal_validation = validate_assets(root, cfg, require_package_identity=True)
        except Exception as exc:
            # Never fall back to a previously-written passed report after a
            # live asset check fails: that would make a mutable report an
            # authorization token for altered cache/streams/inits.
            formal_validation = {
                "schema_version": 1,
                "kind": "formal_asset_validation",
                "status": "failed",
                "protocol_hash": p_hash,
                "asset_bundle_sha256": None,
                "package_identity": None,
                "errors": [f"live asset validation exception: {exc}"],
            }
    elif validation_path.exists():
        formal_validation = _read(validation_path)
    validation_bundle = None if formal_validation is None else formal_validation.get("asset_bundle_sha256")
    validation_identity = None if formal_validation is None else formal_validation.get("package_identity")
    if not isinstance(validation_identity, Mapping):
        validation_identity = None
    validation_package_id = None if validation_identity is None else validation_identity.get("package_id")
    validation_package_manifest_sha = None if validation_identity is None else validation_identity.get("package_manifest_sha256")
    expected_init_rows = {}
    has_init_records = False
    if isinstance(formal_validation, Mapping):
        records = formal_validation.get("records")
        if isinstance(records, Mapping):
            has_init_records = True
            expected_init_rows = {
                int(row.get("seed")): row
                for row in records.get("exact_inits", [])
                if isinstance(row, Mapping) and row.get("seed") is not None
            }
    for path in _condition_dirs(root):
        summary_path, evaluator_path = path / "summary.json", path / "independent_evaluator.json"
        if not summary_path.exists():
            continue
        summary = _read(summary_path)
        evaluator = _read(evaluator_path) if evaluator_path.exists() else None
        receipt_path = path / "receipt.json"
        receipt = _read(receipt_path) if receipt_path.exists() else None
        support = str(summary.get("support"))
        regime = str(summary.get("regime"))
        step_target = 0 if regime == "frozen_feature" else expected_steps
        threshold = 0.001 if regime == "frozen_feature" else float(cfg["gates"]["support_mae_px"])
        invalid: list[str] = []
        if str(summary.get("protocol_hash")) != p_hash:
            invalid.append("summary_protocol_hash")
        if str(summary.get("protocol_id")) != str(cfg["protocol_id"]):
            invalid.append("summary_protocol_id")
        expected_package_id = "neural_affine_crossover_clean" if validation_package_id is None else str(validation_package_id)
        if str(summary.get("package_id")) != expected_package_id:
            invalid.append("summary_package_id")
        if validation_package_id is not None and str(summary.get("package_manifest_sha256")) != str(validation_package_manifest_sha):
            invalid.append("summary_package_identity_mismatch")
        if evaluator is None:
            invalid.append("missing_independent_evaluator")
        elif str(evaluator.get("protocol_hash")) != p_hash:
            invalid.append("evaluator_protocol_hash")
        asset_hash = summary.get("asset_bundle_sha256")
        if not asset_hash:
            invalid.append("missing_asset_bundle_sha256")
        if validation_bundle is not None and str(asset_hash) != str(validation_bundle):
            invalid.append("summary_asset_bundle_mismatch")
        if evaluator is not None and str(evaluator.get("asset_bundle_sha256")) != str(asset_hash):
            invalid.append("evaluator_asset_bundle_mismatch")
        if evaluator is not None and str(evaluator.get("package_id")) != str(summary.get("package_id")):
            invalid.append("evaluator_package_id_mismatch")
        if evaluator is not None and str(evaluator.get("package_manifest_sha256")) != str(summary.get("package_manifest_sha256")):
            invalid.append("evaluator_package_manifest_mismatch")
        if receipt is None:
            invalid.append("missing_training_receipt")
        if regime not in ("frozen_feature", "head_only", "full") or support not in SUPPORT_NAMES:
            invalid.append("unexpected_condition_key")
        if int(summary.get("steps_completed", -1)) != step_target:
            invalid.append("wrong_steps")
        if not str(summary.get("init_hash", "")):
            invalid.append("missing_init_hash")
        expected_init = expected_init_rows.get(int(summary.get("seed", -1)))
        if not has_init_records or expected_init is None:
            invalid.append("missing_validation_init_record")
        if expected_init is not None:
            if str(summary.get("init_hash")) != str(expected_init.get("init_hash")) or str(summary.get("init_state_hash")) != str(expected_init.get("init_hash")):
                invalid.append("init_identity_mismatch")
            if str(summary.get("init_source_file_sha256")) != str(expected_init.get("sha256")):
                invalid.append("init_file_identity_mismatch")
            if str(summary.get("init_source_protocol_hash")) != str(expected_init.get("source_protocol_hash")):
                invalid.append("init_source_protocol_identity_mismatch")
        if not str(summary.get("stream_digest", "")):
            invalid.append("missing_stream_sha")
        if receipt is not None and not str(receipt.get("support_input_hash", "")):
            invalid.append("missing_support_hash")
        if receipt is not None and str(receipt.get("asset_bundle_sha256")) != str(asset_hash):
            invalid.append("receipt_asset_bundle_mismatch")
        if receipt is not None and str(receipt.get("package_id")) != str(summary.get("package_id")):
            invalid.append("receipt_package_id_mismatch")
        if receipt is not None and str(receipt.get("package_manifest_sha256")) != str(summary.get("package_manifest_sha256")):
            invalid.append("receipt_package_manifest_mismatch")
        if not str(summary.get("init_state_hash", "")):
            invalid.append("missing_init_state_hash")
        if not str(summary.get("init_source_protocol_hash", "")):
            invalid.append("missing_init_source_protocol_hash")
        if not str(summary.get("init_source_file_sha256", "")):
            invalid.append("missing_init_source_file_sha256")
        if receipt is not None:
            for field in ("init_state_hash", "init_source_protocol_hash", "init_source_file_sha256"):
                if str(receipt.get(field)) != str(summary.get(field)):
                    invalid.append(f"receipt_{field}_mismatch")
        if evaluator is not None and evaluator.get("status") != "passed":
            invalid.append("evaluator_failed")
        if evaluator is not None and regime == "head_only" and evaluator.get("backbone", {}).get("head_only_bitwise_unchanged") is not True:
            invalid.append("head_backbone_changed")
        if evaluator is not None and regime == "full" and evaluator.get("backbone", {}).get("full_backbone_changed") is not True:
            invalid.append("full_backbone_unchanged")
        if evaluator is not None and regime == "full" and evaluator.get("backbone", {}).get("full_backbone_parameter_changed") is not True:
            invalid.append("full_backbone_parameter_unchanged")
        row = {
            "support": support,
            "support_count": int(summary.get("support_count", 0)),
            "seed": int(summary.get("seed", -1)),
            "regime": regime,
            "path": str(path.resolve()),
            "status": str(summary.get("status")),
            "steps_completed": int(summary.get("steps_completed", 0)),
            "support_mae_px": float(summary.get("support_mae_px", math.inf)),
            "support_gate_threshold_px": threshold,
            "support_gate": bool(float(summary.get("support_mae_px", math.inf)) <= threshold),
            "init_hash": str(summary.get("init_hash", "")),
            "init_state_hash": str(summary.get("init_state_hash", "")),
            "init_source_protocol_hash": str(summary.get("init_source_protocol_hash", "")),
            "init_source_file_sha256": str(summary.get("init_source_file_sha256", "")),
            "stream_sha256": str(summary.get("stream_digest", "")),
            "support_input_hash": None if receipt is None else str(receipt.get("support_input_hash", "")),
            "asset_bundle_sha256": None if asset_hash is None else str(asset_hash),
            "package_id": None if summary.get("package_id") is None else str(summary.get("package_id")),
            "package_manifest_sha256": None if summary.get("package_manifest_sha256") is None else str(summary.get("package_manifest_sha256")),
            "raw_mae_px": None if evaluator is None else float(evaluator.get("metrics", {}).get("raw_mae_px", math.inf)),
            "affine_removed_mae_px": None if evaluator is None else float(evaluator.get("metrics", {}).get("affine_removed_mae_px", math.inf)),
            "evaluator_status": None if evaluator is None else str(evaluator.get("status")),
            "head_backbone_bitwise_unchanged": None if evaluator is None else evaluator.get("backbone", {}).get("head_only_bitwise_unchanged"),
            "full_backbone_changed": None if evaluator is None else evaluator.get("backbone", {}).get("full_backbone_changed"),
            "full_backbone_parameter_changed": None if evaluator is None else evaluator.get("backbone", {}).get("full_backbone_parameter_changed"),
            "invalid_reasons": invalid,
            "valid": not invalid and bool(summary.get("status") == "completed"),
        }
        conditions.append(row)

    expected_keys = {(support, seed, regime) for support in SUPPORT_NAMES for seed in (20260816, 20260817, 20260818) for regime in ("frozen_feature", "head_only", "full")}
    key_list = [(row["support"], row["seed"], row["regime"]) for row in conditions]
    duplicate_keys = sorted({key for key in key_list if key_list.count(key) > 1})
    actual_keys = set(key_list)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    by_key = {(row["support"], row["seed"], row["regime"]): row for row in conditions}
    pair_rows: list[dict[str, Any]] = []
    for support in SUPPORT_NAMES:
        for seed in (20260816, 20260817, 20260818):
            head = by_key.get((support, seed, "head_only"))
            full = by_key.get((support, seed, "full"))
            frozen = by_key.get((support, seed, "frozen_feature"))
            gap = None
            secondary_gap = None
            if head is not None and full is not None and head["raw_mae_px"] is not None and full["raw_mae_px"] is not None:
                gap = float(full["raw_mae_px"] - head["raw_mae_px"])
            if head is not None and full is not None and head["affine_removed_mae_px"] is not None and full["affine_removed_mae_px"] is not None:
                secondary_gap = float(full["affine_removed_mae_px"] - head["affine_removed_mae_px"])
            consistency: list[str] = []
            triplet = [row for row in (frozen, head, full) if row is not None]
            if len(triplet) == 3:
                if len({row["init_hash"] for row in triplet}) != 1:
                    consistency.append("init_hash_mismatch")
                if len({row["stream_sha256"] for row in triplet}) != 1:
                    consistency.append("stream_sha_mismatch")
                if len({row["support_input_hash"] for row in triplet}) != 1:
                    consistency.append("support_hash_mismatch")
                if len({row["asset_bundle_sha256"] for row in triplet}) != 1:
                    consistency.append("asset_bundle_mismatch")
                if len({row["package_id"] for row in triplet}) != 1:
                    consistency.append("package_id_mismatch")
                if len({row["package_manifest_sha256"] for row in triplet}) != 1:
                    consistency.append("package_manifest_mismatch")
                if len({row["init_state_hash"] for row in triplet}) != 1:
                    consistency.append("init_state_hash_mismatch")
                if len({row["init_source_protocol_hash"] for row in triplet}) != 1:
                    consistency.append("init_source_protocol_mismatch")
                if len({row["init_source_file_sha256"] for row in triplet}) != 1:
                    consistency.append("init_source_file_mismatch")
            pair_rows.append({
                "support": support,
                "seed": seed,
                "head": head,
                "full": full,
                "frozen_feature": frozen,
                "primary_gap_full_minus_head_px": gap,
                "secondary_gap_full_minus_head_px": secondary_gap,
                "consistency_errors": consistency,
                "pair_gate": bool(frozen and head and full and all(row["valid"] and row["support_gate"] for row in (frozen, head, full)) and not consistency),
            })

    medians: dict[str, dict[str, Any]] = {}
    for support in SUPPORT_NAMES:
        rows = [row for row in pair_rows if row["support"] == support and row["primary_gap_full_minus_head_px"] is not None]
        gaps = [float(row["primary_gap_full_minus_head_px"]) for row in rows]
        medians[support] = {
            "support_count": int(len(cfg["supports"][support])),
            "n_pairs": len(rows),
            "n_pair_gate_pass": sum(bool(row["pair_gate"]) for row in rows),
            "median_gap_full_minus_head_px": None if not gaps else float(np.median(gaps)),
            "positive_seed_count": sum(value > 0 for value in gaps),
            "negative_seed_count": sum(value < 0 for value in gaps),
            "gates": {
                "all_pairs_present": len(rows) == 3,
                "all_pair_gates": bool(rows) and all(bool(row["pair_gate"]) for row in rows),
            },
        }
    ordered = [medians[name] for name in SUPPORT_NAMES]
    support_counts = [float(item["support_count"]) for item in ordered if item["median_gap_full_minus_head_px"] is not None]
    median_gaps = [float(item["median_gap_full_minus_head_px"]) for item in ordered if item["median_gap_full_minus_head_px"] is not None]
    rho = _spearman([math.log2(value) for value in support_counts], median_gaps) if len(support_counts) == len(SUPPORT_NAMES) else None
    gap_gate = float(cfg["gates"]["crossover_gap_px"])
    corners = medians["corners4"]
    dense = medians["G64"]
    stable = bool(
        len(support_counts) == 4
        and all(item["gates"]["all_pair_gates"] for item in medians.values())
        and corners["median_gap_full_minus_head_px"] is not None
        and dense["median_gap_full_minus_head_px"] is not None
        and corners["median_gap_full_minus_head_px"] >= gap_gate
        and dense["median_gap_full_minus_head_px"] <= -gap_gate
        and corners["positive_seed_count"] >= int(cfg["gates"]["minimum_positive_seeds"])
        and dense["negative_seed_count"] >= int(cfg["gates"]["minimum_negative_seeds"])
        and rho is not None
        and rho <= float(cfg["gates"]["crossover_spearman"])
    )
    catch_up = bool(
        not stable
        and corners["median_gap_full_minus_head_px"] is not None
        and dense["median_gap_full_minus_head_px"] is not None
        and corners["median_gap_full_minus_head_px"] >= gap_gate
        and abs(dense["median_gap_full_minus_head_px"]) < gap_gate
    )
    pair_invalid = any(bool(row["consistency_errors"]) or not bool(row["pair_gate"]) for row in pair_rows)
    protocol_invalid = bool(
        formal_validation is None
        or formal_validation.get("status") != "passed"
        or formal_validation.get("protocol_hash") != p_hash
        or any(("protocol" in reason or "package" in reason or "init_source" in reason) for row in conditions for reason in row["invalid_reasons"])
    )
    global_invalid = bool(
        missing_keys
        or unexpected_keys
        or duplicate_keys
        or any(not row["valid"] for row in conditions)
        or pair_invalid
        or formal_validation is None
        or formal_validation.get("status") != "passed"
        or formal_validation.get("protocol_hash") != p_hash
        or protocol_invalid
    )
    if len(actual_keys) != len(expected_keys) or global_invalid:
        decision = "invalid_protocol" if protocol_invalid else "incomplete"
    elif stable:
        decision = "stable_crossover"
    elif catch_up:
        decision = "catch_up"
    else:
        decision = "no_stable_crossover"
    result = {
        "schema_version": 1,
        "kind": "support_density_plasticity_aggregate",
        "protocol_hash": p_hash,
        "decision": decision,
        "conditions_present": len(conditions),
        "expected_conditions": 36,
        "expected_trained_conditions": 24,
        "missing_keys": [list(key) for key in missing_keys],
        "unexpected_keys": [list(key) for key in unexpected_keys],
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "global_invalid": global_invalid,
        "formal_asset_validation": formal_validation,
        "asset_bundle_sha256": validation_bundle,
        "package_id": validation_package_id,
        "package_manifest_sha256": validation_package_manifest_sha,
        "conditions": conditions,
        "pairs": pair_rows,
        "support_medians": medians,
        "spearman_log2_support_vs_median_gap": rho,
        "gates": cfg["gates"],
    }
    write_json(root / "aggregate.json", result)
    return result


__all__ = ["aggregate_runs"]
