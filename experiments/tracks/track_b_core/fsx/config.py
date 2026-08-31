from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocols" / "runtime_protocol.json"
MATRIX_PATH = PACKAGE_ROOT / "protocols" / "matrix.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _load_json(path)
    if protocol.get("release_id") != "function_selection_v1_trackb_unified_r1_20260824":
        raise ValueError("wrong Track B release id")
    if not isinstance(protocol.get("formal_training_authorized"), bool):
        raise ValueError("formal_training_authorized must be an explicit boolean")
    if protocol.get("formal_training_authorized") is True and protocol.get("status") != "MILESTONE_3_FORMAL_RELEASE_AWAITING_TWO_LAYER_REVIEW":
        raise ValueError("formal authorization requires the reviewed Milestone 3 release status")
    common = protocol.get("training", {}).get("common", {})
    if common.get("steps") != 3000 or common.get("batch_occurrences") != 64:
        raise ValueError("common training must be 3000 steps with fixed batch 64")
    if protocol.get("model", {}).get("groupnorm_groups") != 32:
        raise ValueError("GroupNorm must use 32 groups")
    return protocol


def load_matrix(path: str | Path = MATRIX_PATH) -> dict[str, Any]:
    matrix = _load_json(path)
    if matrix.get("release_id") != "function_selection_v1_trackb_unified_r1_20260824":
        raise ValueError("matrix release id does not match runtime")
    if matrix.get("status") != "MILESTONE_3_FORMAL_RELEASE_MATRIX_AWAITING_TWO_LAYER_REVIEW":
        raise ValueError("matrix is not the Milestone 3 formal release candidate")
    validate_matrix(matrix)
    return matrix


def _stream_id(kind: str | None, population: int | None, seed: int) -> str | None:
    if kind is None:
        return None
    if kind == "common":
        return f"S_COMMON_N{population}_T3000_s{seed}"
    if kind == "b4_causal":
        return f"S_B4_CAUSAL_N4_T20000_s{seed}"
    if kind == "b7_double_draw":
        return f"S_B7_DOUBLE_DRAW_T3000_s{seed}"
    raise ValueError(f"unknown stream kind {kind}")


def iter_cells(matrix: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    matrix = matrix or _load_json(MATRIX_PATH)
    seeds = [int(value) for value in matrix["seeds"]]
    for family, conditions in matrix["families"].items():
        for condition in conditions:
            for seed in seeds:
                condition_id = str(condition["condition_id"])
                shared = bool(condition.get("shared_physical_job", False))
                physical_job_id = (
                    f"job/{family}/{condition_id}/shared"
                    if shared
                    else f"job/{family}/{condition_id}/s{seed}"
                )
                upstream = condition.get("requires_upstream")
                cell = copy.deepcopy(condition)
                cell.update(
                    {
                        "cell_id": f"{family}.{condition_id}.s{seed}",
                        "family": family,
                        "run_seed": seed,
                        "physical_job_id": physical_job_id,
                        "sampler_id": _stream_id(
                            condition.get("stream_kind"), condition.get("population"), seed
                        ),
                        "upstream_physical_job_id": (
                            f"job/{family}/{upstream}/s{seed}" if upstream else None
                        ),
                        "logical_record_role": (
                            "seed_labelled_reference_to_shared_analytic_artifact"
                            if shared
                            else "physical_job_result"
                        ),
                    }
                )
                yield cell


def validate_matrix(matrix: Mapping[str, Any]) -> dict[str, int]:
    cells = list(iter_cells(matrix))
    expected = matrix["expected"]
    summary = {
        "logical_cells": len(cells),
        "physical_jobs": len({cell["physical_job_id"] for cell in cells}),
        "optimizer_jobs": sum(bool(cell["optimizer_job"]) for cell in cells),
    }
    if summary != {key: int(expected[key]) for key in summary}:
        raise ValueError(f"matrix count mismatch: {summary} != {expected}")
    ids = [cell["cell_id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("matrix contains duplicate cell ids")
    return summary


def resolve_cell(cell_id: str, matrix: Mapping[str, Any] | None = None) -> dict[str, Any]:
    matches = [cell for cell in iter_cells(matrix) if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate cell id {cell_id}")
    return matches[0]


def resolved_scientific_config(
    cell_id: str,
    protocol: Mapping[str, Any] | None = None,
    matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = protocol or load_protocol()
    cell = resolve_cell(cell_id, matrix)
    path = str(cell["training_path"])
    if path == "b4_causal":
        training = copy.deepcopy(protocol["training"]["b4_causal"])
    elif path == "b8_explicit_xy_mlp":
        training = copy.deepcopy(protocol["training"]["b8_mlp"])
    else:
        training = copy.deepcopy(protocol["training"]["common"])
    return {
        "release_id": protocol["release_id"],
        "cell": cell,
        "data": copy.deepcopy(protocol["data"]),
        "model": copy.deepcopy(protocol["model"]),
        "training": training,
        "evaluation": copy.deepcopy(protocol["evaluation"]),
        "family_specific": copy.deepcopy(protocol["family_specific"].get(cell["family"], {})),
    }
