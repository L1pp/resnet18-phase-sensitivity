import io
import json
import math
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

from sgd_anchor_rescue.cli import main
from sgd_anchor_rescue.constants import (
    CANDIDATE_A,
    CANDIDATE_B,
    ETA_MIN,
    FIXED_STATE_STEPS,
    TOTAL_STEPS,
    TAIL_RELATIVE_TOLERANCE,
    CandidateConfig,
    validate_candidate_config,
    validate_protocol_constants,
)
from sgd_anchor_rescue.decisions import (
    FinalAnchorGateCategory,
    GateEvaluation,
    OutcomeCategory,
    RemediationAction,
    anchor_gate_pass,
    classify_outcome,
    classify_final_anchor_gate,
    decide_20k_remediation,
    select_5k_candidate,
    select_endpoint,
    select_first_match,
    select_min_anchor,
    strict_gate,
)
from sgd_anchor_rescue.guard import AnchorOnlyGuard, DenseMetricLeakError
from sgd_anchor_rescue.ledger import LedgerEntry, RunLedger, validate_run_budget
from sgd_anchor_rescue.manifest import (
    assert_shared_manifests,
    build_manifest,
    compare_shared_manifests,
    expected_optimizer_schedule,
    fingerprint_data_order,
    validate_manifest,
)
from sgd_anchor_rescue.metrics import CheckpointMetrics
from sgd_anchor_rescue.schedule import (
    ScheduleConfig,
    candidate_schedule,
    fixed_state_steps,
    full_train_anchor_steps,
    linear_warmup_cosine_lr,
    low_lr_rescue_schedule,
    schedule_lr,
)


class ProtocolConstantTests(unittest.TestCase):
    def test_candidates_only_differ_in_momentum(self):
        self.assertEqual(validate_protocol_constants(), ())
        a = CANDIDATE_A.as_dict()
        b = CANDIDATE_B.as_dict()
        self.assertEqual({key for key in a if a[key] != b[key]}, {"name", "momentum"})

    def test_candidate_protocol_fields_are_locked(self):
        altered = replace(CANDIDATE_A, peak_lr=2e-3)
        self.assertIn("peak_lr", " ".join(validate_candidate_config(altered)))
        altered = replace(CANDIDATE_A, warmup_steps=1_000, amp=True)
        errors = " ".join(validate_candidate_config(altered))
        self.assertIn("warmup_steps", errors)
        self.assertIn("amp", errors)
        altered = replace(CANDIDATE_A, loss_name="MSE")
        self.assertIn("loss_name", " ".join(validate_candidate_config(altered)))

    def test_schedule_warmup_cosine_and_extension_hold(self):
        self.assertEqual(linear_warmup_cosine_lr(0), 0.0)
        self.assertAlmostEqual(linear_warmup_cosine_lr(500), 1e-3)
        self.assertGreater(linear_warmup_cosine_lr(501), ETA_MIN)
        self.assertAlmostEqual(linear_warmup_cosine_lr(TOTAL_STEPS), ETA_MIN)
        config = candidate_schedule("A")
        self.assertAlmostEqual(schedule_lr(40_000, config), ETA_MIN)
        self.assertTrue(schedule_lr(20_001, config) == ETA_MIN)
        low = low_lr_rescue_schedule()
        self.assertAlmostEqual(low.peak_lr, 3e-4)
        self.assertEqual(low.warmup_steps, 1_000)

    def test_fixed_and_anchor_check_schedule(self):
        self.assertEqual(FIXED_STATE_STEPS[0], 0)
        self.assertEqual(FIXED_STATE_STEPS[-1], 20_000)
        checks = full_train_anchor_steps(20_000)
        self.assertTrue(set(FIXED_STATE_STEPS).issubset(checks))
        self.assertIn(1, checks)
        self.assertEqual(len(checks), len(set(checks)))
        self.assertIn(39_000, fixed_state_steps(40_000))
        self.assertIn(40_000, fixed_state_steps(40_000))


class SelectionAndGateTests(unittest.TestCase):
    def test_5k_selection_is_stable_and_tie_breaks_to_a(self):
        result = select_5k_candidate(
            {
                "A": {"anchor_1000": 2.0, "anchor_5000": 1.0, "objective_5000": 4.0},
                "B": {"anchor_1000": 2.0, "anchor_5000": 1.0, "objective_5000": 4.0},
            }
        )
        self.assertEqual(result.winner, "A")
        self.assertEqual(result.eligible, ("A", "B"))

    def test_5k_selection_rejects_nonfinite_and_non_decreasing_rows(self):
        result = select_5k_candidate(
            {
                "A": {"anchor_1000": 1.0, "anchor_5000": 1.0, "objective_5000": 1.0},
                "B": {"anchor_1000": 1.0, "anchor_5000": math.nan, "objective_5000": 1.0},
            }
        )
        self.assertIsNone(result.winner)
        self.assertIn("A", result.excluded)
        self.assertIn("B", result.excluded)

    def test_strict_gate_requires_endpoint_and_both_tail_points(self):
        passing = {
            15_000: {"step": 15_000, "anchor": 0.3},
            19_000: {"step": 19_000, "anchor": 0.09},
            20_000: {"step": 20_000, "anchor": 0.08},
        }
        evaluation = strict_gate(passing, 0.10)
        self.assertTrue(evaluation.passed)
        self.assertTrue(anchor_gate_pass(passing, 0.10))
        failing = dict(passing)
        failing[19_000] = {"step": 19_000, "anchor": 0.11}
        self.assertFalse(strict_gate(failing, 0.10).passed)
        too_far_apart = dict(passing)
        too_far_apart[19_000] = {"step": 19_000, "anchor": 0.09}
        too_far_apart[20_000] = {"step": 20_000, "anchor": 0.05}
        self.assertFalse(strict_gate(too_far_apart, 0.10).passed)
        close_enough = dict(passing)
        close_enough[19_000] = {"step": 19_000, "anchor": 0.09}
        close_enough[20_000] = {"step": 20_000, "anchor": 0.07}
        self.assertTrue(strict_gate(close_enough, 0.10).passed)
        self.assertEqual(TAIL_RELATIVE_TOLERANCE, 0.20)

    def test_final_rescue_gate_is_dense_reveal_or_pathology(self):
        passing = {
            19_000: {"step": 19_000, "anchor": 0.09},
            20_000: {"step": 20_000, "anchor": 0.08},
        }
        self.assertEqual(
            classify_final_anchor_gate(passing, 0.10),
            FinalAnchorGateCategory.CONTINUE_DENSE_REVEAL,
        )
        failing = dict(passing)
        failing[20_000] = {"step": 20_000, "anchor": 0.12}
        self.assertEqual(
            classify_final_anchor_gate(failing, 0.10),
            FinalAnchorGateCategory.OPTIMIZATION_PATHOLOGY,
        )

    def test_remediation_tree(self):
        five_k = {
            "A": {"anchor_1000": 1.0, "anchor_5000": 0.5, "objective_5000": 0.5},
            "B": {"anchor_1000": 1.0, "anchor_5000": 0.6, "objective_5000": 0.6},
        }
        points = {
            15_000: {"step": 15_000, "anchor": 0.30},
            19_000: {"step": 19_000, "anchor": 0.20},
            20_000: {"step": 20_000, "anchor": 0.19},
        }
        self.assertEqual(
            decide_20k_remediation(five_k, points, 0.10).action,
            RemediationAction.EXTEND_TO_40K,
        )
        no_stable = {
            "A": {"anchor_1000": 1.0, "anchor_5000": 1.1, "objective_5000": 1.0},
            "B": {"anchor_1000": 1.0, "anchor_5000": 1.2, "objective_5000": 1.0},
        }
        self.assertEqual(
            decide_20k_remediation(no_stable, points, 0.10).action,
            RemediationAction.LOW_LR_RESTART,
        )
        with self.assertRaises(ValueError):
            decide_20k_remediation({"A": no_stable["A"]}, points, 0.10)
        with self.assertRaises(ValueError):
            decide_20k_remediation(five_k, points, 0.10, selected_candidate="B")
        stopped = dict(points)
        stopped[15_000] = {"step": 15_000, "anchor": 0.19}
        stopped[20_000] = {"step": 20_000, "anchor": 0.20}
        self.assertEqual(
            decide_20k_remediation(five_k, stopped, 0.10).action,
            RemediationAction.OPTIMIZATION_PATHOLOGY,
        )

    def test_endpoint_first_match_and_min_anchor_are_distinct(self):
        points = [
            CheckpointMetrics(step=0, anchor=0.5),
            CheckpointMetrics(step=100, anchor=0.08),
            CheckpointMetrics(step=200, anchor=0.09),
            CheckpointMetrics(step=400, anchor=0.07),
        ]
        self.assertEqual(select_endpoint(points).step, 400)
        self.assertEqual(select_first_match(points, 0.10).step, 200)
        self.assertEqual(select_first_match(points, 0.10, check_steps=(100, 200)).step, 100)
        self.assertEqual(select_min_anchor(points).step, 400)


class ClassificationTests(unittest.TestCase):
    def test_four_outcome_categories(self):
        endpoint = {"step": 20_000, "anchor": 0.08, "raw_box": 5.0, "u": 5.0}
        first = {"step": 10_000, "anchor": 0.09, "raw_box": 4.5, "u": 4.0}
        self.assertEqual(
            classify_outcome(
                endpoint=endpoint,
                first_match=first,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                anchor_matched=True,
            ),
            OutcomeCategory.STRONG_DRIFT,
        )
        keep_endpoint = {"step": 20_000, "anchor": 0.08, "raw_box": 3.0, "u": 1.4}
        keep_first = {"step": 10_000, "anchor": 0.09, "raw_box": 3.1, "u": 1.2}
        self.assertEqual(
            classify_outcome(
                endpoint=keep_endpoint,
                first_match=keep_first,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                anchor_matched=True,
            ),
            OutcomeCategory.PRESERVE_ORIGINAL_FIELD,
        )
        self.assertEqual(
            classify_outcome(
                endpoint=endpoint,
                first_match=first,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                anchor_matched=False,
                remediation_action=RemediationAction.OPTIMIZATION_PATHOLOGY,
            ),
            OutcomeCategory.OPTIMIZATION_PATHOLOGY,
        )
        self.assertEqual(
            classify_outcome(
                endpoint=None,
                first_match=None,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                anchor_matched=False,
            ),
            OutcomeCategory.INCONCLUSIVE,
        )

    def test_classification_uses_actual_anchor_values_and_final_gate(self):
        endpoint = {"step": 20_000, "anchor": 0.08, "raw_box": 5.0, "u": 5.0}
        first = {"step": 10_000, "anchor": 0.09, "raw_box": 4.5, "u": 4.0}
        final_gate = GateEvaluation(
            threshold_px=0.10,
            endpoint_step=20_000,
            endpoint_anchor=0.08,
            tail_steps=(19_000, 20_000),
            tail_anchors=(0.09, 0.08),
            endpoint_pass=True,
            tail_pass=True,
            tail_close=True,
            passed=True,
        )
        # A stale false boolean must not override the actual anchor metrics.
        self.assertEqual(
            classify_outcome(
                endpoint=endpoint,
                first_match=first,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                final_gate=final_gate,
                anchor_matched=False,
            ),
            OutcomeCategory.STRONG_DRIFT,
        )
        # A stale true boolean must not allow an endpoint over the threshold.
        endpoint_bad = dict(endpoint)
        endpoint_bad["anchor"] = 0.20
        self.assertEqual(
            classify_outcome(
                endpoint=endpoint_bad,
                first_match=first,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                final_gate=final_gate,
                anchor_matched=True,
            ),
            OutcomeCategory.INCONCLUSIVE,
        )
        rejected_gate = replace(final_gate, passed=False)
        self.assertEqual(
            classify_outcome(
                endpoint=endpoint,
                first_match=first,
                u0=1.0,
                raw_box0=3.0,
                adamw_u=8.0,
                threshold_px=0.10,
                final_gate=rejected_gate,
            ),
            OutcomeCategory.INCONCLUSIVE,
        )


class ManifestAndGuardTests(unittest.TestCase):
    def test_data_order_fingerprint_preserves_batch_boundaries(self):
        flat = fingerprint_data_order([0, 1, 2, 3])
        batched = fingerprint_data_order([[0, 1], [2, 3]])
        self.assertNotEqual(flat, batched)
        self.assertEqual(flat, fingerprint_data_order([0, 1, 2, 3]))

    def test_manifest_schema_validation(self):
        manifest = build_manifest(
            stage="anchor_only",
            candidate="A",
            seed=20260816,
            theta0={"sha256": "abc", "source": "execution_ai"},
            anchors={"name": "corners4", "count": 4},
            data_order_fingerprint="deadbeef",
        )
        self.assertEqual(validate_manifest(manifest), ())
        self.assertEqual(manifest["optimizer"]["name"], "SGD")
        self.assertEqual(manifest["schedule"]["peak_lr"], 1e-3)
        manifest["batch_size"] = 32
        self.assertTrue(validate_manifest(manifest))

    def test_manifest_strict_candidate_schedule_and_stage1_guard(self):
        base = dict(
            stage="anchor_only",
            candidate="A",
            seed=20260816,
            theta0={"sha256": "abc"},
            anchors={"name": "corners4", "count": 4},
            data_order_fingerprint="deadbeef",
            stage1_records=[{"step": 0, "anchor": 1.0, "objective": 2.0}],
        )
        manifest = build_manifest(**base)
        self.assertEqual(validate_manifest(manifest), ())
        bad_schedule = dict(manifest)
        bad_schedule["schedule"] = dict(manifest["schedule"], peak_lr=2e-3)
        self.assertTrue(validate_manifest(bad_schedule))
        bad_records = dict(manifest)
        bad_records["stage1_records"] = [{"step": 1, "off_support_error": 2.0}]
        self.assertTrue(validate_manifest(bad_records))
        bad_guard = dict(manifest)
        bad_guard["anchor_only_guard"] = False
        self.assertTrue(validate_manifest(bad_guard))
        low_optimizer, low_schedule = expected_optimizer_schedule("low_lr_restart")
        self.assertEqual(low_optimizer["name"], "SGD")
        self.assertEqual(low_schedule["warmup_steps"], 1_000)

    def test_shared_ab_manifests_require_same_order_and_inputs(self):
        common = dict(
            stage="anchor_only",
            seed=20260816,
            theta0={"sha256": "abc"},
            anchors={"name": "corners4", "count": 4},
            data_order_fingerprint="same-order",
        )
        a = build_manifest(candidate="A", **common)
        b = build_manifest(candidate="B", **common)
        self.assertEqual(compare_shared_manifests(a, b), ())
        assert_shared_manifests(a, b)
        b_bad = dict(b, data_order_fingerprint="other-order")
        self.assertTrue(compare_shared_manifests(a, b_bad))

    def test_anchor_only_guard_rejects_dense_nested_keys_and_copies(self):
        guard = AnchorOnlyGuard()
        row = {"step": 0, "anchor": 1.0, "objective": 2.0, "meta": {"seed": 3}}
        guard.append(row)
        row["anchor"] = 99.0
        self.assertEqual(guard.records[0]["anchor"], 1.0)
        with self.assertRaises(DenseMetricLeakError):
            guard.append({"step": 1, "metrics": {"dense_box": 2.0}})
        with self.assertRaises(DenseMetricLeakError):
            guard.append({"step": 1, "u": 3.0})
        for key in ("u_metric", "u0", "u_adamw", "image_ids"):
            with self.assertRaises(DenseMetricLeakError):
                guard.append({"step": 1, key: 3.0})
        for key in ("off_support_error", "residual_norm"):
            with self.assertRaises(DenseMetricLeakError):
                guard.append({"step": 1, key: 3.0})
        guard.append({"step": 2, "duration": 1.0, "cuda": "available"})

    def test_cli_validate_config_is_local_pure_logic(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["validate-config"]), 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["valid"])

    def test_cli_classify_requires_threshold_and_final_gate(self):
        payload = {
            "endpoint": {"step": 20_000, "anchor": 0.08, "raw_box": 5.0, "u": 5.0},
            "first_match": {"step": 10_000, "anchor": 0.09, "raw_box": 4.5, "u": 4.0},
            "u0": 1.0,
            "raw_box0": 3.0,
            "adamw_u": 8.0,
            "threshold_px": 0.10,
            "final_gate": {"passed": True},
        }
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(payload))), redirect_stdout(output):
            self.assertEqual(main(["classify", "--input", "-"]), 0)
        self.assertEqual(json.loads(output.getvalue())["category"], "strong_drift")


class LedgerTests(unittest.TestCase):
    def test_protocol_budget_allows_exact_45k_and_rejects_overrun(self):
        ledger = RunLedger()
        ledger = ledger.append(run_id="A-5k", candidate="A", phase="halving", optimizer_steps=5_000)
        ledger = ledger.append(run_id="B-5k", candidate="B", phase="halving", optimizer_steps=5_000)
        ledger = ledger.append(run_id="A-20k", candidate="A", phase="main", optimizer_steps=15_000)
        ledger = ledger.append(run_id="A-40k", candidate="A", phase="extension", optimizer_steps=20_000)
        self.assertEqual(ledger.total_optimizer_steps, 45_000)
        self.assertEqual(ledger.remaining_steps, 0)
        self.assertEqual(ledger.validate(), ())
        with self.assertRaises(ValueError):
            ledger.append(run_id="extra", candidate="A", phase="extra", optimizer_steps=1)

    def test_budget_validator_reports_duplicate_and_negative_entries(self):
        entries = (
            LedgerEntry("same", "A", "halving", 5_000),
            LedgerEntry("same", "B", "halving", -1),
        )
        errors = validate_run_budget(entries)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("non-negative" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
