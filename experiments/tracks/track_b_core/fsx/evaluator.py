from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .checkpoints import load_checkpoint_payload, resume_identity_from_cell, validate_resume_identity
from .config import PACKAGE_ROOT, load_protocol, resolve_cell
from .data import b7_endpoint_occurrences, canonical_b7_pair_graph, coordinate_target, coordinate_target_f64
from .formal_provenance import validate_formal_attempt
from .records import write_json
from .schedules import b4_causal_lr_for_update, common_lr_for_update
from .science import b6_ridge, metric_summary, support_gate

# Scientific independence boundary: this module intentionally never imports
# fsx.trainer and never consumes producer aggregates as evidence.


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _same_number(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-7, atol=atol))
    except (TypeError, ValueError):
        return False


def _nested_exact(left: Any, right: Any) -> bool:
    if hasattr(left, "shape") and hasattr(left, "dtype"):
        try:
            return bool(np.array_equal(left.detach().cpu().numpy(), right.detach().cpu().numpy()))
        except AttributeError:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_exact(a, b) for a, b in zip(left, right))
    return left == right


def _stacked_equal(values: Any, expected: np.ndarray) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        return False
    try:
        return bool(np.array_equal(np.stack([np.asarray(value) for value in values]), np.asarray(expected)))
    except (TypeError, ValueError):
        return False


def _checkpoint_evidence_path(config: Mapping[str, Any], filename: str) -> Path:
    """Resolve immutable checkpoint evidence from this attempt or its resume source."""
    local = Path(config["execution"]["run_path"]) / "checkpoints" / filename
    if local.is_file():
        return local
    resume_text = config["execution"].get("resume_from")
    if isinstance(resume_text, str) and resume_text:
        source_sibling = Path(resume_text).parent / filename
        if source_sibling.is_file():
            return source_sibling
    return local


def _recompute_metrics(
    predictions_path: Path,
    query_manifest: list[Mapping[str, Any]],
    primary_query_id: str,
    target_kind: str,
    analytic_float64: bool,
) -> dict[str, Any]:
    with np.load(predictions_path, allow_pickle=False) as values:
        queries: dict[str, Any] = {}
        appearance_breakdown: dict[str, Any] = {}
        for item in query_manifest:
            query_id = str(item["query_id"])
            prefix = str(item["prefix"])
            pred = np.asarray(values[f"{prefix}_pred_px"])
            xy = np.asarray(values[f"{prefix}_xy_px"])
            if analytic_float64 and any(
                np.asarray(values[name]).dtype != np.float64
                for name in (
                    f"{prefix}_pred_u",
                    f"{prefix}_truth_u",
                    f"{prefix}_pred_px",
                    f"{prefix}_truth_px",
                )
            ):
                raise ValueError(f"analytic query fields are not float64: {query_id}")
            expected_truth_u = coordinate_target_f64(xy, target_kind) if analytic_float64 else coordinate_target(xy, target_kind)
            truth = expected_truth_u * (223.0 if analytic_float64 else np.float32(223.0))
            truth_atol = 1e-14 if analytic_float64 else 1e-7
            if not np.allclose(np.asarray(values[f"{prefix}_truth_u"]), expected_truth_u, atol=truth_atol, rtol=0.0):
                raise ValueError(f"saved query truth differs from resolved cell target: {query_id}")
            if not np.allclose(np.asarray(values[f"{prefix}_truth_px"]), truth, atol=1e-12 if analytic_float64 else 1e-5, rtol=0.0):
                raise ValueError(f"saved query pixel truth differs from resolved cell target: {query_id}")
            if int(item["rows"]) != len(pred):
                raise ValueError(f"saved query row count mismatch: {query_id}")
            queries[query_id] = metric_summary(pred, truth)
            field_key = f"{prefix}_field_index"
            appearance_key = f"{prefix}_appearance_id"
            if field_key in values.files and appearance_key in values.files:
                field_index = np.asarray(values[field_key], dtype=np.int64)
                appearance_id = np.asarray(values[appearance_key], dtype=np.int64)
                per_field: list[dict[str, Any]] = []
                for field_id in np.unique(field_index):
                    mask = field_index == field_id
                    field_metrics = metric_summary(pred[mask], truth[mask])
                    per_field.append(
                        {
                            "field_index": int(field_id),
                            "appearance_id": int(appearance_id[mask][0]),
                            **field_metrics,
                        }
                    )
                appearance_breakdown[query_id] = {
                    "field_count": len(per_field),
                    "per_field": per_field,
                    "macro_full_box_raw_mae_px": float(np.mean([row["full_box_raw_mae_px"] for row in per_field])),
                    "macro_full_box_u_mae": float(np.mean([row["full_box_u_mae"] for row in per_field])),
                    "pooled": queries[query_id],
                }
        support_pred = np.asarray(values["support_pred_px"])
        support_xy = np.asarray(values["support_xy_px"])
        if analytic_float64 and any(
            np.asarray(values[name]).dtype != np.float64
            for name in ("support_pred_u", "support_truth_u", "support_pred_px", "support_truth_px")
        ):
            raise ValueError("analytic support fields are not float64")
        expected_support_u = coordinate_target_f64(support_xy, target_kind) if analytic_float64 else coordinate_target(support_xy, target_kind)
        support_truth = expected_support_u * (223.0 if analytic_float64 else np.float32(223.0))
        if not np.allclose(np.asarray(values["support_truth_u"]), expected_support_u, atol=1e-14 if analytic_float64 else 1e-7, rtol=0.0):
            raise ValueError("saved support truth differs from resolved cell target")
        if support_pred.shape != support_truth.shape or support_pred.ndim != 2 or support_pred.shape[1] != 2:
            raise ValueError("saved support fields are malformed")
        support_mae = float(np.mean(np.abs(support_pred.astype(np.float64) - support_truth.astype(np.float64))))
    if primary_query_id not in queries:
        raise ValueError("primary query is absent from raw fields")
    return {
        "primary_query_id": primary_query_id,
        "primary": queries[primary_query_id],
        "queries": queries,
        "appearance_breakdown": appearance_breakdown,
        "support_anchor_raw_mae_px": support_mae,
        "support_eligible": support_gate(support_mae),
    }


def _producer_matches(recomputed: Mapping[str, Any], producer: Mapping[str, Any]) -> bool:
    if recomputed["primary_query_id"] != producer.get("primary_query_id"):
        return False
    if recomputed["support_eligible"] is not producer.get("support_eligible"):
        return False
    if not _same_number(recomputed["support_anchor_raw_mae_px"], producer.get("support_anchor_raw_mae_px")):
        return False
    if set(recomputed["queries"]) != set(producer.get("queries", {})):
        return False
    for query_id, metrics in recomputed["queries"].items():
        candidate = producer["queries"][query_id]
        if set(metrics) != set(candidate) or any(not _same_number(metrics[key], candidate[key]) for key in metrics):
            return False
    if not _nested_metric_match(recomputed.get("appearance_breakdown", {}), producer.get("appearance_breakdown", {})):
        return False
    return True


def _nested_metric_match(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_metric_match(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nested_metric_match(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _same_number(left, right)
    return left == right


def _check_trace(config: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, bool]:
    cell = config["cell"]
    path = str(cell["training_path"])
    result: dict[str, bool] = {}
    if path in {"b8_degree2", "b6_ridge"}:
        result["analytic_has_no_optimizer_trace"] = len(trace) == 0
        return result
    result["trace_nonempty"] = len(trace) > 0
    result["steps_strictly_ordered"] = [int(row["actual_step"]) for row in trace] == list(range(1, len(trace) + 1))
    if path == "b8_explicit_xy_mlp":
        result["b8_mlp_fixed_lr_no_scheduler"] = all(_same_number(row["lr"], 1e-3) and int(row["full_batch_rows"]) == 9 for row in trace)
        return result
    if path == "b4_causal":
        result["b4_exact_lr_formula"] = all(
            _same_number(row["lr"], b4_causal_lr_for_update(int(row["protocol_step"])), atol=1e-12)
            for row in trace
        )
        result["b4_fixed_batch64"] = all(int(row["batch_occurrences"]) == 64 for row in trace)
        return result
    if path in {"common", "common_frozen_bn"} or path.startswith("b6_"):
        result["common_exact_lr_formula"] = all(
            _same_number(row["lr"], common_lr_for_update(int(row["protocol_step"])), atol=1e-12)
            for row in trace
        )
        result["common_fixed_batch64"] = all(int(row["batch_occurrences"]) == 64 for row in trace)
    elif path == "b7_context":
        result["b7_context_exact_lr_formula"] = all(_same_number(row["lr"], common_lr_for_update(int(row["protocol_step"])), atol=1e-12) for row in trace)
        result["b7_context_fixed_batch64"] = all(int(row["batch_occurrences"]) == 64 for row in trace)
    elif path in {"b7_dense_absolute", "b7_relative"}:
        result["b7_dense_exact_lr_formula"] = all(_same_number(row["lr"], common_lr_for_update(int(row["protocol_step"])), atol=1e-12) for row in trace)
        result["b7_64_pairs_128_endpoints"] = all(int(row["pair_occurrences"]) == 64 and int(row["endpoint_occurrences"]) == 128 for row in trace)
    return result


def _check_b4(config: Mapping[str, Any], facts: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, bool]:
    paired = Path(config["execution"]["paired_initialization"] or "")
    mode = str(config["execution"]["mode"])
    first_gate = facts.get("first_gate", {})
    selected = first_gate.get("selection_step")
    checks = {
        "b4_paired_initialization_exists": paired.is_file(),
        "b4_pure_mse_declared": facts.get("loss_semantics") == "pure_mean_mse",
        "b4_optimizer_matches_cell": facts.get("optimizer") == config["cell"].get("optimizer"),
        "b4_gate_selection_recorded": isinstance(first_gate, dict) and {"selection_step", "before", "at", "after"}.issubset(first_gate),
    }
    final_checkpoint = Path(config["execution"]["run_path"]) / "checkpoints" / "final.pt"
    checks["b4_final_checkpoint_exists"] = final_checkpoint.is_file()
    if final_checkpoint.is_file():
        final_payload = load_checkpoint_payload(final_checkpoint)
        runtime = dict(final_payload.get("runtime_state", {}))
        runtime_first_gate = runtime.get("first_gate")
        checks["b4_checkpoint_has_complete_trace"] = (
            int(final_payload["completed_step"]) == len(trace)
            and int(final_payload["stream_cursor"]) == len(trace)
            and _nested_exact(runtime.get("trace"), trace)
        )
        checks["b4_checkpoint_has_complete_gate_history"] = len(runtime.get("gate_history", [])) == len(trace)
        checks["b4_checkpoint_has_complete_support_history"] = np.asarray(runtime.get("gate_predictions", [])).shape[0] == len(trace)
        checks["b4_checkpoint_first_gate_state_matches"] = (
            runtime_first_gate is None and selected is None
        ) or _nested_exact(runtime_first_gate, first_gate)
    trajectory_path = Path(str(facts.get("support_trajectory_path", "")))
    checks["b4_raw_support_trajectory_exists"] = trajectory_path.is_file()
    if trajectory_path.is_file():
        with np.load(trajectory_path, allow_pickle=False) as raw:
            prediction = np.asarray(raw["support_pred_u"], dtype=np.float64)
            truth = np.asarray(raw["support_truth_u"], dtype=np.float64)
        raw_mae = np.mean(np.abs((prediction - truth[None, :, :]) * 223.0), axis=(1, 2))
        raw_pass = raw_mae <= float(load_protocol()["evaluation"]["support_gate_anchor_mae_px_lte"])
        first_raw = int(np.flatnonzero(raw_pass)[0] + 1) if np.any(raw_pass) else None
        checks["b4_gate_recomputed_from_raw_trajectory"] = (
            prediction.shape[0] == len(trace)
            and first_raw == selected
            and all(_same_number(raw_mae[index], trace[index]["support_anchor_raw_mae_px"]) for index in range(len(trace)))
        )
    if selected is not None:
        checks["b4_gate_at_row_is_first_pass"] = bool(first_gate["at"]["passes"]) and all(not bool(row["passes"]) for row in trace if int(row["actual_step"]) < int(selected))
        checks["b4_first_gate_checkpoint_exists"] = _checkpoint_evidence_path(config, "first_gate.pt").is_file()
    if mode == "formal":
        run_path = Path(config["execution"]["run_path"])
        checks["b4_formal_step3000_snapshot"] = _checkpoint_evidence_path(config, "step_3000.pt").is_file()
        checks["b4_formal_step20000_snapshot"] = _checkpoint_evidence_path(config, "step_20000.pt").is_file()
        checks["b4_formal_completed_20000"] = len(trace) == 20000
        snapshot_metrics = facts.get("snapshot_metrics", {})
        for label in ("step_3000", "step_20000"):
            snapshot_path = run_path / "snapshots" / f"{label}_predictions.npz"
            checks[f"b4_{label}_raw_snapshot"] = snapshot_path.is_file()
            if snapshot_path.is_file():
                with np.load(snapshot_path, allow_pickle=False) as raw:
                    recomputed = metric_summary(np.asarray(raw["pred_px"]), np.asarray(raw["truth_px"]))
                checks[f"b4_{label}_metric_recomputed"] = all(
                    _same_number(recomputed[key], snapshot_metrics.get(label, {}).get(key)) for key in recomputed
                )
        if selected is not None:
            checks["b4_first_gate_raw_snapshot"] = (run_path / "snapshots" / "first_gate_predictions.npz").is_file()
    else:
        checks["b4_smoke_not_misrepresented_as_endpoints"] = facts.get("formal_eligible") is False
    return checks


def _check_b6(config: Mapping[str, Any], facts: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, bool]:
    cell = config["cell"]
    upstream_text = config["execution"].get("upstream_run")
    checks: dict[str, bool] = {"b6_upstream_path_recorded": isinstance(upstream_text, str) and bool(upstream_text)}
    if upstream_text:
        upstream = Path(upstream_text)
        try:
            upstream_status = _load_json(upstream / "status.json")
            upstream_config = _load_json(upstream / "run_config.json")
            upstream_cell = upstream_config["cell"]
            checks["b6_same_seed_new_g64_upstream"] = (
                upstream_status.get("state") == "COMPLETE"
                and upstream_cell.get("family") == "B6"
                and upstream_cell.get("condition_id") == "g64_upstream"
                and int(upstream_cell.get("run_seed")) == int(cell["run_seed"])
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            checks["b6_same_seed_new_g64_upstream"] = False
    if cell["training_path"] == "b6_ridge":
        ridge_artifact = Path(config["execution"]["run_path"]) / "ridge_artifact.npz"
        checks.update(
            {
                "b6_ridge_four_rows": int(facts.get("input_rows", -1)) == 4,
                "b6_ridge_float64": facts.get("fit_dtype") == "float64",
                "b6_ridge_no_heldout_fit": facts.get("fit_heldout_data_used") is False,
                "b6_ridge_primary_multiplier": _same_number(facts.get("primary_multiplier"), 1e-4, atol=0.0),
                "b6_ridge_artifact_exists": ridge_artifact.is_file(),
            }
        )
        if ridge_artifact.is_file():
            with np.load(ridge_artifact, allow_pickle=False) as raw:
                train_features_raw = np.asarray(raw["train_features"])
                train_truth_raw = np.asarray(raw["train_truth_u"])
                query_features_raw = np.asarray(raw["query_features"])
                saved_beta_raw = np.asarray(raw["beta"])
                saved_singular_values_raw = np.asarray(raw["singular_values"])
            train_features = train_features_raw.astype(np.float64, copy=False)
            train_truth = train_truth_raw.astype(np.float64, copy=False)
            query_features = query_features_raw.astype(np.float64, copy=False)
            saved_beta = saved_beta_raw.astype(np.float64, copy=False)
            saved_singular_values = saved_singular_values_raw.astype(np.float64, copy=False)
            index = _load_json(Path(config["execution"]["asset_root"]) / "indices" / f"{cell['index_id']}.json")
            train_xy = np.asarray([row["position_px"] for row in index["rows"]], dtype=np.float64)
            expected_train_truth = coordinate_target_f64(train_xy, str(cell["target_kind"]))
            checks["b6_ridge_all_analytic_arrays_are_float64"] = all(
                value.dtype == np.float64
                for value in (train_features_raw, train_truth_raw, query_features_raw, saved_beta_raw, saved_singular_values_raw)
            )
            checks["b6_ridge_truth_direct_from_coordinates_float64"] = np.array_equal(train_truth_raw, expected_train_truth)
            recomputed = b6_ridge(train_features, train_truth, query_features, multiplier=1e-4, rcond=1e-6)
            with np.load(Path(config["execution"]["run_path"]) / "predictions.npz", allow_pickle=False) as predictions:
                raw_primary_prediction = np.asarray(predictions["q0_pred_u"])
                raw_support_prediction = np.asarray(predictions["support_pred_u"])
            checks["b6_ridge_saved_predictions_are_float64"] = raw_primary_prediction.dtype == np.float64 and raw_support_prediction.dtype == np.float64
            recomputed_support = b6_ridge(train_features, train_truth, train_features, multiplier=1e-4, rcond=1e-6)
            checks["b6_ridge_beta_recomputed"] = np.allclose(saved_beta, recomputed["beta"], atol=1e-12, rtol=1e-10)
            checks["b6_ridge_singular_values_recomputed"] = np.allclose(saved_singular_values, recomputed["singular_values"], atol=1e-12, rtol=1e-10)
            checks["b6_ridge_primary_prediction_recomputed"] = np.allclose(raw_primary_prediction, recomputed["prediction"], atol=1e-6, rtol=1e-6)
            checks["b6_ridge_support_prediction_recomputed"] = np.allclose(raw_support_prediction, recomputed_support["prediction"], atol=1e-6, rtol=1e-6)
            sensitivity = facts.get("sensitivity", {})
            checks["b6_ridge_all_frozen_multipliers_recomputed"] = all(
                np.allclose(
                    np.asarray(sensitivity[str(multiplier)]["prediction_u"], dtype=np.float64),
                    b6_ridge(train_features, train_truth, query_features, multiplier=multiplier, rcond=1e-6)["prediction"],
                    atol=1e-10,
                    rtol=1e-10,
                )
                for multiplier in (1e-6, 1e-4, 1e-2)
            )
    elif cell["training_path"] == "b6_lp_ft":
        transition = facts.get("lpft_transition")
        checks.update(
            {
                "lpft_transition_present": isinstance(transition, dict),
                "lpft_protocol_boundary_500_501": isinstance(transition, dict) and transition.get("selection_step") == 500 and transition.get("first_full_update_step") == 501,
                "lpft_unfreeze_does_not_change_weights": isinstance(transition, dict) and transition.get("weights_changed_at_unfreeze") is False,
                "lpft_fc_optimizer_state_preserved": isinstance(transition, dict) and transition.get("fc_optimizer_state_preserved") is True,
                "lpft_new_state_zero_before_step501": isinstance(transition, dict) and transition.get("new_parameter_optimizer_state_entries_before_first_full") == 0,
                "lpft_trace_has_500_then_501_semantics": len(trace) >= 2 and [int(trace[0]["protocol_step"]), int(trace[1]["protocol_step"])] == [500, 501] if config["execution"]["mode"] == "smoke" else len(trace) == 3000,
                "lpft_boundary_checkpoint_exists": _checkpoint_evidence_path(config, "lp_end.pt").is_file(),
                "lpft_pre_step501_checkpoint_exists": _checkpoint_evidence_path(config, "pre_step501.pt").is_file(),
                "lpft_post_step501_checkpoint_exists": _checkpoint_evidence_path(config, "post_step501.pt").is_file(),
            }
        )
        run_path = Path(config["execution"]["run_path"])
        lp_end_path = _checkpoint_evidence_path(config, "lp_end.pt")
        pre_path = _checkpoint_evidence_path(config, "pre_step501.pt")
        if lp_end_path.is_file() and pre_path.is_file():
            lp_end = load_checkpoint_payload(lp_end_path)
            pre = load_checkpoint_payload(pre_path)
            checks["lpft_weights_exact_across_unfreeze"] = _nested_exact(lp_end["model_state"], pre["model_state"])
            checks["lpft_fc_optimizer_state_exact_across_unfreeze"] = _nested_exact(lp_end["optimizer_state"]["state"], pre["optimizer_state"]["state"])
            known_state_ids = set(pre["optimizer_state"]["state"])
            new_parameter_ids = set(pre["optimizer_state"]["param_groups"][-1]["params"])
            checks["lpft_new_parameters_have_zero_state_before_step501"] = not bool(known_state_ids.intersection(new_parameter_ids))
            checks["lpft_stream_cursor_continues_at_boundary"] = int(lp_end["stream_cursor"]) == int(pre["stream_cursor"])
            checks["lpft_pre_checkpoint_declares_full_pre_update"] = pre.get("stage_state", {}).get("stage") == "full_pre_update"
        final_path = run_path / "checkpoints" / "final.pt"
        checks["lpft_final_checkpoint_exists"] = final_path.is_file()
        if final_path.is_file():
            final_payload = load_checkpoint_payload(final_path)
            runtime = dict(final_payload.get("runtime_state", {}))
            final_parameter_ids = {
                parameter_id
                for group in final_payload["optimizer_state"]["param_groups"]
                for parameter_id in group["params"]
            }
            checks["lpft_final_checkpoint_has_complete_trace"] = (
                int(final_payload["completed_step"]) == len(trace)
                and int(final_payload["stream_cursor"]) == len(trace)
                and _nested_exact(runtime.get("trace"), trace)
            )
            checks["lpft_step501_created_state_for_all_trainable_parameters"] = final_parameter_ids.issubset(
                set(final_payload["optimizer_state"]["state"])
            )
        boundary_metrics = facts.get("lpft_boundary_metrics", {})
        for label in ("lp_end_step500", "post_first_full_step501"):
            snapshot_path = run_path / "snapshots" / f"{label}_predictions.npz"
            checks[f"lpft_{label}_raw_snapshot_exists"] = snapshot_path.is_file()
            if snapshot_path.is_file():
                with np.load(snapshot_path, allow_pickle=False) as raw:
                    xy = np.asarray(raw["xy_px"])
                    prediction_px = np.asarray(raw["pred_px"])
                    expected_truth_px = coordinate_target(xy, str(cell["target_kind"])) * np.float32(223.0)
                recomputed_boundary = metric_summary(prediction_px, expected_truth_px)
                checks[f"lpft_{label}_metric_recomputed"] = all(
                    _same_number(recomputed_boundary[key], boundary_metrics.get(label, {}).get(key))
                    for key in recomputed_boundary
                )
    return checks


def _check_b7(config: Mapping[str, Any], facts: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> dict[str, bool]:
    path = str(config["cell"]["training_path"])
    asset_root = Path(config["execution"]["asset_root"])
    index = _load_json(asset_root / "indices" / f"{config['cell']['index_id']}.json")
    row_xy = np.asarray([row["position_px"] for row in index["rows"]], dtype=np.float64)
    row_truth = coordinate_target(row_xy, str(config["cell"]["target_kind"]))
    training_raw_path = Path(str(facts.get("training_raw_path", "")))
    final_checkpoint = Path(config["execution"]["run_path"]) / "checkpoints" / "final.pt"
    checkpoint_checks: dict[str, bool] = {"b7_final_checkpoint_exists": final_checkpoint.is_file()}
    final_runtime: dict[str, Any] = {}
    if final_checkpoint.is_file():
        final_payload = load_checkpoint_payload(final_checkpoint)
        final_runtime = dict(final_payload.get("runtime_state", {}))
        checkpoint_checks.update(
            {
                "b7_checkpoint_path_matches": final_payload.get("stage_state", {}).get("b7_path") == path,
                "b7_checkpoint_cursor_and_trace_complete": (
                    int(final_payload["completed_step"]) == len(trace)
                    and int(final_payload["stream_cursor"]) == len(trace)
                    and _nested_exact(final_runtime.get("trace"), trace)
                ),
            }
        )
    if path == "b7_context":
        checks = {
            **checkpoint_checks,
            "b7_context_pure_mse": facts.get("loss_semantics") == "pure_mean_mse_over_64_common_stream_occurrences",
            "b7_context_has_no_dense_exposure": facts.get("exposure_path") is None,
            "b7_context_training_raw_exists": training_raw_path.is_file(),
        }
        if training_raw_path.is_file():
            with np.load(training_raw_path, allow_pickle=False) as raw:
                batch_ids = np.asarray(raw["batch_row_ids"])
                prediction = np.asarray(raw["prediction_u"], dtype=np.float64)
                saved_truth = np.asarray(raw["truth_u"], dtype=np.float64)
            stream = np.load(asset_root / "streams" / f"{config['cell']['sampler_id']}.npy", allow_pickle=False, mmap_mode="r")
            expected_ids = np.asarray(stream[: len(trace)])
            expected_truth = row_truth[expected_ids]
            recomputed_loss = np.mean((prediction - expected_truth) ** 2, axis=(1, 2))
            checks["b7_context_batch_ids_exact"] = np.array_equal(batch_ids, expected_ids)
            checks["b7_context_truth_resolved_independently"] = np.allclose(saved_truth, expected_truth, atol=1e-7, rtol=0.0)
            checks["b7_context_loss_recomputed_from_raw"] = all(_same_number(recomputed_loss[i], trace[i]["loss"]) for i in range(len(trace)))
            checks["b7_context_checkpoint_raw_history_complete"] = (
                _stacked_equal(final_runtime.get("batch_row_ids"), batch_ids)
                and _stacked_equal(final_runtime.get("prediction_u"), prediction)
                and _stacked_equal(final_runtime.get("truth_u"), saved_truth)
            )
        return checks
    exposure_path = Path(str(facts.get("exposure_path", "")))
    checks = {
        **checkpoint_checks,
        "b7_exposure_exists": exposure_path.is_file(),
        "b7_endpoint_order_declared": facts.get("endpoint_order") == "left64_then_right64",
        "b7_image_draw_role_declared": facts.get("image_draw_role") == "consumed_first_to_preserve_frozen_double_draw_rng_but_not_used_for_matched_dense_loss",
        "b7_training_raw_exists": training_raw_path.is_file(),
    }
    if path == "b7_dense_absolute":
        checks["b7_absolute_pure_mse_128"] = facts.get("loss_semantics") == "pure_mean_mse_over_128_endpoint_occurrences"
    else:
        checks["b7_relative_ordered_displacement_plus_anchor"] = facts.get("loss_semantics") == "ordered_right_minus_left_displacement_mean_mse_plus_anchor_mean_mse_1_to_1"
    if exposure_path.is_file():
        sampler = str(config["cell"]["sampler_id"])
        with np.load(asset_root / "streams" / f"{sampler}.npz", allow_pickle=False) as stream, np.load(exposure_path, allow_pickle=False) as exposure:
            steps = len(trace)
            pair_ids = np.asarray(exposure["pair_ids"])
            endpoint_ids = np.asarray(exposure["endpoint_row_ids"])
            image_ids = np.asarray(exposure["image_draw_ids"])
            expected_pairs = np.asarray(stream["pair_ids"][:steps])
            expected_images = np.asarray(stream["image_ids"][:steps])
            expected_endpoints = b7_endpoint_occurrences(expected_pairs, canonical_b7_pair_graph())
            checks["b7_pair_ids_exact_frozen_rows"] = np.array_equal(pair_ids, expected_pairs)
            checks["b7_image_draw_ids_exact_frozen_rows"] = np.array_equal(image_ids, expected_images)
            checks["b7_ordered_endpoints_exact"] = np.array_equal(endpoint_ids, expected_endpoints)
            checks["b7_exposure_shapes"] = pair_ids.shape == (steps, 64) and endpoint_ids.shape == (steps, 128)
            if training_raw_path.is_file():
                with np.load(training_raw_path, allow_pickle=False) as raw_training:
                    prediction = np.asarray(raw_training["endpoint_prediction_u"], dtype=np.float64)
                    saved_truth = np.asarray(raw_training["endpoint_truth_u"], dtype=np.float64)
                    anchor_prediction = np.asarray(raw_training["anchor_prediction_u"], dtype=np.float64) if "anchor_prediction_u" in raw_training.files else None
                    anchor_saved_truth = np.asarray(raw_training["anchor_truth_u"], dtype=np.float64) if "anchor_truth_u" in raw_training.files else None
                expected_training_truth = row_truth[expected_endpoints]
                checks["b7_endpoint_truth_resolved_independently"] = np.allclose(saved_truth, expected_training_truth, atol=1e-7, rtol=0.0)
                checks["b7_dense_checkpoint_exposure_history_complete"] = (
                    _stacked_equal(final_runtime.get("pair_ids"), pair_ids)
                    and _stacked_equal(final_runtime.get("endpoint_row_ids"), endpoint_ids)
                    and _stacked_equal(final_runtime.get("image_draw_ids"), image_ids)
                    and _stacked_equal(final_runtime.get("endpoint_prediction_u"), prediction)
                    and _stacked_equal(final_runtime.get("endpoint_truth_u"), saved_truth)
                )
                if path == "b7_dense_absolute":
                    recomputed_loss = np.mean((prediction - expected_training_truth) ** 2, axis=(1, 2))
                else:
                    pair_loss = np.mean(
                        ((prediction[:, 64:] - prediction[:, :64]) - (expected_training_truth[:, 64:] - expected_training_truth[:, :64])) ** 2,
                        axis=(1, 2),
                    )
                    anchors = np.asarray(config["cell"]["anchors"], dtype=np.int64)
                    expected_anchor_truth = row_truth[anchors]
                    checks["b7_anchor_truth_resolved_independently"] = anchor_saved_truth is not None and np.allclose(anchor_saved_truth, expected_anchor_truth[None, :, :], atol=1e-7, rtol=0.0)
                    checks["b7_relative_checkpoint_anchor_history_complete"] = (
                        anchor_prediction is not None
                        and anchor_saved_truth is not None
                        and _stacked_equal(final_runtime.get("anchor_prediction_u"), anchor_prediction)
                        and _stacked_equal(final_runtime.get("anchor_truth_u"), anchor_saved_truth)
                    )
                    anchor_loss = np.mean((anchor_prediction - expected_anchor_truth[None, :, :]) ** 2, axis=(1, 2))
                    recomputed_loss = pair_loss + anchor_loss
                checks["b7_loss_recomputed_from_raw"] = all(_same_number(recomputed_loss[i], trace[i]["loss"]) for i in range(len(trace)))
    return checks


def _check_b8(config: Mapping[str, Any], facts: Mapping[str, Any], recomputed: Mapping[str, Any]) -> dict[str, bool]:
    path = str(config["cell"]["training_path"])
    checks: dict[str, bool] = {"b8_quadratic_target": config["cell"]["target_kind"] == "quadratic_b8"}
    if path in {"b8_degree2", "b8_explicit_xy_mlp"}:
        asset_root = Path(config["execution"]["asset_root"])
        index = _load_json(asset_root / "indices" / f"{config['cell']['index_id']}.json")
        run_files = [candidate.name.lower() for candidate in Path(config["execution"]["run_path"]).rglob("*") if candidate.is_file()]
        checks.update(
            {
                "b8_coordinate_only_modalities": facts.get("input_modalities") == ["coordinate", "target"] and config["execution"].get("input_modalities") == ["coordinate", "target"],
                "b8_no_image_fit": facts.get("image_data_read") is False,
                "b8_no_activation_fit": facts.get("activation_data_read") is False,
                "b8_no_heldout_fit": facts.get("heldout_fit_data_read") is False,
                "b8_exact_nine_fit_rows": facts.get("fit_rows") == 9,
                "b8_resolved_index_is_coordinate_shard": index.get("kind") == "coordinate_shard" and index.get("renderer") is None and int(index.get("row_count", -1)) == 9,
                "b8_no_saved_image_or_activation_fit_artifact": not any("image" in name or "activation" in name for name in run_files),
            }
        )
    if path == "b8_degree2":
        checks["b8_degree2_full_rank"] = facts.get("rank") == 6
        coefficients = np.asarray(facts.get("coefficients"), dtype=np.float64)
        target_coefficients = np.asarray(load_protocol()["targets"]["quadratic_b8"]["coefficients"], dtype=np.float64).T
        checks["b8_degree2_recovers_frozen_map_float64_precision"] = (
            coefficients.shape == target_coefficients.shape
            and np.allclose(coefficients, target_coefficients, atol=1e-12, rtol=1e-12)
        )
        shared_source = Path(str(facts.get("shared_physical_source", "")))
        checks["b8_degree2_shared_physical_source_exists"] = shared_source.is_dir() and (shared_source / "analytic_degree2.npz").is_file()
        if (shared_source / "analytic_degree2.npz").is_file():
            with np.load(shared_source / "analytic_degree2.npz", allow_pickle=False) as analytic:
                saved_coefficients = np.asarray(analytic["coefficients"])
            checks["b8_degree2_saved_coefficients_float64_exact"] = (
                saved_coefficients.dtype == np.float64 and np.array_equal(saved_coefficients, coefficients)
            )
        checks["b8_degree2_logical_reference_role"] = facts.get("logical_record_role") == "seed_labelled_reference_to_shared_analytic_artifact"
    elif path == "b8_explicit_xy_mlp":
        checks["b8_mlp_no_scheduler"] = facts.get("scheduler") == "none" and _same_number(facts.get("fixed_lr"), 1e-3, atol=0.0)
    else:
        manifest_ids = {item["query_id"] for item in facts.get("query_manifest", [])}
        checks["b8_cnn_has_canonical_and_heldout_fields"] = manifest_ids == {"Q_B8_CANONICAL_FULLBOX", "Q_B8_HELDOUT_FULLBOX"}
        heldout = recomputed.get("appearance_breakdown", {}).get("Q_B8_HELDOUT_FULLBOX", {})
        expected_fields = 4 if config["execution"]["mode"] == "formal" else 1
        checks["b8_cnn_heldout_per_appearance_and_macro_recomputed"] = (
            int(heldout.get("field_count", 0)) >= expected_fields
            and len(heldout.get("per_field", [])) == int(heldout.get("field_count", -1))
            and np.isfinite(float(heldout.get("macro_full_box_raw_mae_px", np.nan)))
            and isinstance(heldout.get("pooled"), Mapping)
        )
    return checks


def review_attempt(run_path: str | Path, review_path: str | Path | None = None) -> dict[str, Any]:
    """Independently review one immutable attempt from raw saved fields."""
    run = Path(run_path).resolve()
    config = _load_json(run / "run_config.json")
    status = _load_json(run / "status.json")
    if config.get("execution", {}).get("mode") == "formal":
        validate_formal_attempt(
            run,
            execution_root=config["execution"].get("project_root"),
            expected_cell_id=str(status.get("cell_id", "")),
            require_complete=True,
        )
    producer = _load_json(run / "producer_metrics.json")
    facts = _load_json(run / "run_facts.json")
    trace_payload = _load_json(run / "training_trace.json")
    trace = trace_payload.get("steps")
    if not isinstance(trace, list):
        raise ValueError("training trace must contain a steps list")
    target = Path(review_path or config["execution"]["review_path"]).resolve()
    if str(target) != str(Path(config["execution"]["review_path"]).resolve()):
        raise ValueError("review path is not the producer-declared matching attempt")
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True, exist_ok=False)
    recomputed = _recompute_metrics(
        run / "predictions.npz",
        list(facts["query_manifest"]),
        str(producer["primary_query_id"]),
        str(config["cell"]["target_kind"]),
        str(config["cell"]["training_path"]) in {"b6_ridge", "b8_degree2"},
    )
    expected_cell = resolve_cell(str(status["cell_id"]))
    checks: dict[str, bool] = {
        "producer_complete": status.get("state") == "COMPLETE",
        "resolved_cell_exact": config.get("cell") == expected_cell,
        "run_path_exact": str(run) == str(Path(config["execution"]["run_path"]).resolve()),
        "raw_predictions_exist": (run / "predictions.npz").is_file(),
        "producer_metrics_recomputed_from_raw": _producer_matches(recomputed, producer),
        "init_seed_is_run_seed": int(config["execution"].get("init_seed", -1)) == int(expected_cell["run_seed"]),
        "smoke_never_formal_eligible": config["execution"]["mode"] != "smoke" or (config["execution"].get("formal_eligible") is False and producer.get("formal_eligible") is False and facts.get("formal_eligible") is False),
    }
    if expected_cell["training_path"] not in {"b6_ridge", "b8_degree2"}:
        final_checkpoint = run / "checkpoints" / "final.pt"
        checks["final_checkpoint_exists_for_resumable_path"] = final_checkpoint.is_file()
        if final_checkpoint.is_file():
            final_payload = load_checkpoint_payload(final_checkpoint)
            try:
                validate_resume_identity(final_payload.get("resume_identity"), resume_identity_from_cell(expected_cell))
                checks["checkpoint_resume_identity_exact"] = True
            except ValueError:
                checks["checkpoint_resume_identity_exact"] = False
    checks.update(_check_trace(config, trace))
    family = str(expected_cell["family"])
    if expected_cell["training_path"] == "b4_causal":
        checks.update(_check_b4(config, facts, trace))
    if family == "B6" and expected_cell["condition_id"] != "g64_upstream":
        checks.update(_check_b6(config, facts, trace))
    if family == "B7":
        checks.update(_check_b7(config, facts, trace))
    if family == "B8":
        checks.update(_check_b8(config, facts, recomputed))
    passed = bool(checks) and all(checks.values())
    recomputed_payload = {
        **recomputed,
        "mode": config["execution"]["mode"],
        "formal_eligible": config["execution"]["formal_eligible"],
    }
    write_json(target / "recomputed_metrics.json", recomputed_payload)
    endpoint_metrics: dict[str, Any] | None = None
    if expected_cell["training_path"] == "b4_causal":
        endpoint_metrics = {}
        support_values: np.ndarray | None = None
        trajectory_path = Path(str(facts.get("support_trajectory_path", "")))
        if trajectory_path.is_file():
            with np.load(trajectory_path, allow_pickle=False) as raw_support:
                support_prediction = np.asarray(raw_support["support_pred_u"], dtype=np.float64)
                support_truth = np.asarray(raw_support["support_truth_u"], dtype=np.float64)
            support_values = np.mean(np.abs((support_prediction - support_truth[None, :, :]) * 223.0), axis=(1, 2))
        selection_step = facts.get("first_gate", {}).get("selection_step")
        for label in ("first_gate", "step_3000", "step_20000"):
            snapshot = run / "snapshots" / f"{label}_predictions.npz"
            if snapshot.is_file():
                with np.load(snapshot, allow_pickle=False) as raw:
                    endpoint = metric_summary(np.asarray(raw["pred_px"]), np.asarray(raw["truth_px"]))
                support_step = int(selection_step) if label == "first_gate" and selection_step is not None else int(label.split("_")[1])
                support_value = None if support_values is None or support_step > len(support_values) else float(support_values[support_step - 1])
                endpoint["support_anchor_raw_mae_px"] = support_value
                endpoint["support_eligible"] = False if support_value is None else support_gate(support_value)
                endpoint_metrics[label] = endpoint
            else:
                endpoint_metrics[label] = None
        endpoint_metrics["first_gate_selection_step"] = selection_step
    review = {
        "review_status": "PASS" if passed else "FAIL",
        "cell_id": expected_cell["cell_id"],
        "run_path": str(run),
        "review_path": str(target),
        "checks": checks,
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "support_eligible": recomputed["support_eligible"],
        "formal_eligible": config["execution"]["formal_eligible"],
        "primary_metrics": recomputed["primary"],
        "endpoint_metrics": endpoint_metrics,
    }
    write_json(target / "review.json", review)
    return review
