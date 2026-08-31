from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import PACKAGE_ROOT, iter_cells, load_protocol
from .records import write_json


def _primary_error(record: Mapping[str, Any]) -> float:
    value = float(record["primary_metrics"]["full_box_raw_mae_px"])
    if not np.isfinite(value):
        raise ValueError("reviewed primary metric must be finite")
    return value


def _promotion(values: list[float], expected_seed_count: int = 3) -> str:
    if len(values) != expected_seed_count:
        return "INCONCLUSIVE_MISSING_OR_EXCLUDED_SEED"
    if all(abs(value) <= 1.0 for value in values):
        return "PRACTICALLY_TIED_ALL_THREE_WITHIN_1PX"
    same_positive = all(value > 0.0 for value in values)
    same_negative = all(value < 0.0 for value in values)
    if (same_positive or same_negative) and all(abs(value) >= 2.0 for value in values):
        return "CONSISTENT_CONDITION_WORSE" if same_positive else "CONSISTENT_CONDITION_BETTER"
    return "INCONCLUSIVE"


def _indexed(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for source in records:
        record = dict(source)
        cell_id = str(record.get("cell_id", ""))
        if cell_id in indexed:
            duplicates.append(cell_id)
        else:
            indexed[cell_id] = record
    return indexed, sorted(set(duplicates))


def _standard_contrast(
    family: str,
    reference: str,
    conditions: list[str],
    by_id: Mapping[str, Mapping[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {"reference": reference, "direction": "condition_minus_reference", "conditions": {}}
    for condition in conditions:
        rows: list[dict[str, Any]] = []
        included_values: list[float] = []
        for seed in seeds:
            reference_record = by_id[f"{family}.{reference}.s{seed}"]
            condition_record = by_id[f"{family}.{condition}.s{seed}"]
            eligible = bool(reference_record["support_eligible"] and condition_record["support_eligible"])
            reference_value = _primary_error(reference_record)
            condition_value = _primary_error(condition_record)
            difference = condition_value - reference_value
            row = {
                "seed": seed,
                "reference_value": reference_value,
                "condition_value": condition_value,
                "condition_minus_reference": difference,
                "included": eligible,
                "exclusion_reason": None if eligible else "support_gate_failed_in_reference_or_condition",
            }
            rows.append(row)
            if eligible:
                included_values.append(difference)
        result["conditions"][condition] = {
            "per_seed": rows,
            "included_seed_count": len(included_values),
            "mean_condition_minus_reference": float(np.mean(included_values)) if included_values else None,
            "promotion": _promotion(included_values, len(seeds)),
        }
    return result


def _b4_contrast(by_id: Mapping[str, Mapping[str, Any]], seeds: list[int]) -> dict[str, Any]:
    endpoints = [
        ("matched_gate_when_available", "first_gate"),
        ("step_20000", "step_20000"),
    ]
    result: dict[str, Any] = {
        "paired_arms": ["causal_adamw", "causal_sgd"],
        "direction": "causal_sgd_minus_causal_adamw",
        "reference_adamw_role": "context_only",
        "endpoints": {},
    }
    for aggregate_endpoint, record_endpoint in endpoints:
        rows: list[dict[str, Any]] = []
        values: list[float] = []
        for seed in seeds:
            adamw = by_id[f"B4.causal_adamw.s{seed}"]
            sgd = by_id[f"B4.causal_sgd.s{seed}"]
            adamw_metric = (adamw.get("endpoint_metrics") or {}).get(record_endpoint)
            sgd_metric = (sgd.get("endpoint_metrics") or {}).get(record_endpoint)
            eligible = bool(
                adamw_metric
                and sgd_metric
                and adamw_metric.get("support_eligible") is True
                and sgd_metric.get("support_eligible") is True
            )
            difference = None
            if adamw_metric and sgd_metric:
                difference = float(sgd_metric["full_box_raw_mae_px"]) - float(adamw_metric["full_box_raw_mae_px"])
            rows.append(
                {
                    "seed": seed,
                    "adamw_value": None if not adamw_metric else float(adamw_metric["full_box_raw_mae_px"]),
                    "sgd_value": None if not sgd_metric else float(sgd_metric["full_box_raw_mae_px"]),
                    "sgd_minus_adamw": difference,
                    "adamw_first_gate_step": (adamw.get("endpoint_metrics") or {}).get("first_gate_selection_step") if record_endpoint == "first_gate" else None,
                    "sgd_first_gate_step": (sgd.get("endpoint_metrics") or {}).get("first_gate_selection_step") if record_endpoint == "first_gate" else None,
                    "included": eligible,
                    "exclusion_reason": None if eligible else "missing_endpoint_or_support_gate_failed",
                }
            )
            if eligible and difference is not None:
                values.append(difference)
        result["endpoints"][aggregate_endpoint] = {
            "per_seed": rows,
            "included_seed_count": len(values),
            "mean_sgd_minus_adamw": float(np.mean(values)) if values else None,
            "promotion": _promotion(values, len(seeds)),
        }
    return result


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only complete PASS reviews; never infer or fill a cell."""
    expected_cells = list(iter_cells())
    expected_ids = {cell["cell_id"] for cell in expected_cells}
    seeds = [int(seed) for seed in load_protocol()["discovery_seeds"]]
    by_id, duplicates = _indexed(records)
    observed_ids = set(by_id)
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    failed = sorted(cell_id for cell_id, record in by_id.items() if record.get("review_status") != "PASS")
    closure = not missing and not extra and not duplicates and not failed
    inventory = {
        "expected_cells": len(expected_ids),
        "observed_unique_cells": len(observed_ids),
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "non_pass_reviews": failed,
    }
    if not closure:
        return {"aggregate_status": "INCOMPLETE_OR_FAILED", "closure": False, "inventory": inventory, "contrasts": None}
    protocol_contrasts = load_protocol()["preregistered_contrasts"]
    contrasts: dict[str, Any] = {}
    for family in ("B1", "B2", "B3", "B5", "B6", "B7", "B8"):
        spec = protocol_contrasts[family]
        contrasts[family] = _standard_contrast(
            family,
            str(spec["reference"]),
            [str(value) for value in spec["conditions"]],
            by_id,
            seeds,
        )
    contrasts["B4"] = _b4_contrast(by_id, seeds)
    per_cell = [
        {
            "cell_id": cell["cell_id"],
            "review_status": by_id[cell["cell_id"]]["review_status"],
            "support_eligible": bool(by_id[cell["cell_id"]]["support_eligible"]),
            "primary_full_box_raw_mae_px": _primary_error(by_id[cell["cell_id"]]),
            "run_path": by_id[cell["cell_id"]].get("run_path"),
            "review_path": by_id[cell["cell_id"]].get("review_path"),
        }
        for cell in expected_cells
    ]
    return {
        "aggregate_status": "PASS_COMPLETE_84_CELL_CLOSURE",
        "closure": True,
        "inventory": inventory,
        "per_cell": per_cell,
        "contrasts": contrasts,
    }


def aggregate_review_files(
    review_files: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in review_files]
    result = aggregate_records(records)
    target = Path(output_path)
    write_json(target, result)
    return result
