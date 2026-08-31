from __future__ import annotations

import numpy as np
import unittest
from unittest.mock import patch

from track_a_toy.config import CONFIG, OUTPUT_ROOT, PROBE_ROOT, TRAIN_ROOT
from track_a_toy.pipeline import (
    _coarse_cell_targets,
    _fit_probe_scope,
    _fit_standard_scaler,
    _load_dataset,
    _split_mask,
    _transform_standard_scaler,
    _visual_points,
)
from track_a_toy.renderer import position_grid


class PipelineTests(unittest.TestCase):
    def test_fixed_split_is_disjoint_and_deterministic(self) -> None:
        _points, ix, iy = position_grid()
        train, test = _split_mask(ix, iy)
        self.assertEqual(len(ix), 256)
        self.assertEqual(int(test.sum()), 64)
        self.assertEqual(int(train.sum()), 192)
        self.assertFalse(np.any(train & test))
        self.assertTrue(np.all(train | test))
        train_again, test_again = _split_mask(ix, iy)
        self.assertTrue(np.array_equal(train, train_again))
        self.assertTrue(np.array_equal(test, test_again))

    def test_finite_keeps_all_256_and_visuals_use_fixed_ten(self) -> None:
        manifest = __import__("json").loads((OUTPUT_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["grid_count"], 256)
        self.assertEqual(manifest["finite_fully_visible_count"], 182)
        self.assertEqual(manifest["finite_clipped_grid_count"], 74)
        finite = _load_dataset("finite")
        torus = _load_dataset("torus")
        self.assertEqual(len(finite["points"]), 256)
        self.assertEqual(len(torus["points"]), 256)
        self.assertEqual(finite["images"].shape, (256, 3, 64, 64))
        self.assertEqual(finite["fully_visible"].shape, (256,))
        self.assertEqual(finite["interior"].shape, (256,))
        self.assertEqual(int(finite["fully_visible"].sum()), 182)
        self.assertTrue(np.array_equal(finite["fully_visible"], finite["interior"]))
        expected = np.asarray(CONFIG["visual_positions_xy"], dtype=np.float32)
        self.assertTrue(np.array_equal(_visual_points("finite"), expected))
        self.assertTrue(np.array_equal(_visual_points("torus"), expected))

    def test_scaler_is_train_only_and_safe_for_constant_columns(self) -> None:
        values = np.asarray([[1.0, 10.0], [3.0, 10.0], [101.0, 30.0]], dtype=np.float64)
        scaler = _fit_standard_scaler(values[:2])
        train_scaled = _transform_standard_scaler(values[:2], scaler)
        all_scaled = _transform_standard_scaler(values, scaler)
        self.assertTrue(np.allclose(train_scaled.mean(axis=0), 0.0))
        self.assertAlmostEqual(float(train_scaled.std(axis=0)[0]), 1.0)
        self.assertAlmostEqual(float(train_scaled.std(axis=0)[1]), 0.0)
        self.assertEqual(int(scaler["zero_variance_count"]), 1)
        self.assertEqual(int(scaler["near_constant_count"]), 1)
        self.assertTrue(np.all(all_scaled[:, np.asarray(scaler["near_constant"], dtype=bool)] == 0.0))
        self.assertGreater(float(np.abs(all_scaled[2]).max()), 1.0)

    def test_near_constant_noise_uses_constant_baseline_without_ridge(self) -> None:
        points, ix, iy = position_grid()
        rng = np.random.default_rng(12)
        features = np.column_stack(
            [
                np.full(len(points), 0.25, dtype=np.float64),
                2.0 + rng.normal(0.0, 1e-8, size=len(points)),
            ]
        ).astype(np.float32)
        scaler = _fit_standard_scaler(features[((ix + 2 * iy) % 4) != 0])
        self.assertEqual(int(scaler["active_feature_count"]), 0)
        transformed = _transform_standard_scaler(features, scaler)
        self.assertTrue(np.all(transformed == 0.0))
        with patch("track_a_toy.pipeline._select_alpha", side_effect=AssertionError("ridge must not run")):
            scope = _fit_probe_scope(
                features=features,
                points=points,
                ix=ix,
                iy=iy,
                scope="near_constant_noise",
            )
        self.assertTrue(scope["constant_baseline"])
        self.assertEqual(scope["active_feature_count"], 0)
        for task in scope["tasks"].values():
            self.assertTrue(task["constant_baseline"])
            self.assertFalse(task["ridge_fit"])
            self.assertIsNone(task["alpha"])
            self.assertTrue(np.allclose(task["_test_prediction"], task["constant_prediction"]))

    def test_a3_stored_gap_is_reported_as_constant_baseline(self) -> None:
        feature_path = OUTPUT_ROOT / "random_features" / "a3_torus_s1.npz"
        self.assertTrue(feature_path.exists())
        with np.load(feature_path, allow_pickle=False) as payload:
            features = np.asarray(payload["gap"], dtype=np.float32)
            points = np.asarray(payload["points"], dtype=np.float64)
            ix = np.asarray(payload["ix"], dtype=np.int64)
            iy = np.asarray(payload["iy"], dtype=np.int64)
        scope = _fit_probe_scope(
            features=features,
            points=points,
            ix=ix,
            iy=iy,
            scope="a3_stored_gap",
        )
        self.assertEqual(scope["active_feature_count"], 0)
        self.assertTrue(scope["constant_baseline"])
        raw = scope["tasks"]["raw_xy"]
        self.assertTrue(raw["constant_baseline"])
        self.assertTrue(np.allclose(raw["_test_prediction"], raw["constant_prediction"]))
        self.assertAlmostEqual(raw["test_mae_px"], raw["constant_test_mae_px"], places=12)

    def test_probe_scope_selects_independent_alpha_for_three_targets(self) -> None:
        points, ix, iy = position_grid()
        rng = np.random.default_rng(7)
        features = rng.normal(size=(256, 8)).astype(np.float32)
        with patch("track_a_toy.pipeline._select_alpha", side_effect=[1e-6, 1e-2, 100.0]) as chooser:
            scope = _fit_probe_scope(
                features=features,
                points=points,
                ix=ix,
                iy=iy,
                scope="test_all_256",
            )
        self.assertEqual(chooser.call_count, 3)
        self.assertEqual([call.args[1].shape[1] for call in chooser.call_args_list], [2, 4, 2])
        self.assertEqual(set(scope["tasks"]), {"raw_xy", "phase_sincos", "quotient_cell"})
        self.assertEqual(scope["feature_scaler"]["fit_scope"], "train_split_only")
        self.assertEqual(scope["active_feature_count"], 8)
        self.assertFalse(scope["constant_baseline"])
        self.assertNotIn("test_circular_mae_px", scope["tasks"]["quotient_cell"])
        self.assertIn("test_average_accuracy", scope["tasks"]["quotient_cell"])
        self.assertEqual(
            {name: task["alpha"] for name, task in scope["tasks"].items()},
            {"raw_xy": 1e-6, "phase_sincos": 1e-2, "quotient_cell": 100.0},
        )

    def test_quotient_target_is_direct_coarse_cell_not_phase(self) -> None:
        points = np.asarray([[0.0, 0.0], [31.9, 32.0], [32.0, 63.9], [60.0, 60.0]])
        expected = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.array_equal(_coarse_cell_targets(points), expected))

    def test_chinese_report_reads_summary_numbers(self) -> None:
        report_text = (OUTPUT_ROOT / "report.md").read_text(encoding="utf-8")
        summary = __import__("json").loads((PROBE_ROOT / "summary.json").read_text(encoding="utf-8"))
        value = summary["models"]["a0_standard"]["global_xy"]["test_mae_px"]
        self.assertIn("# Track A-Toy v1：边界与位置信息小实验", report_text)
        self.assertIn(f"{value:.3f}", report_text)
        self.assertIn("粗网格准确率", report_text)
        if (TRAIN_ROOT / "summary.json").exists():
            self.assertIn("正式短训练结果", report_text)
            training = __import__("json").loads((TRAIN_ROOT / "summary.json").read_text(encoding="utf-8"))
            for row in training["models"].values():
                self.assertIn(f"{row['best_mae_px']:.3f}", report_text)
            receipt_path = OUTPUT_ROOT / "wallclock_receipt.json"
            if receipt_path.exists():
                receipt = __import__("json").loads(receipt_path.read_text(encoding="utf-8"))
                actual_total = float(training["training_elapsed_seconds"]) + float(
                    receipt["observed_nontraining_seconds"]
                )
                self.assertAlmostEqual(actual_total, 2640.3199360000035, places=6)
                self.assertIn(f"{actual_total:.3f} 秒", report_text)
                self.assertIn("50–60 分钟", report_text)
                self.assertIn("60–90 分钟", report_text)
                self.assertIn("被实跑推翻", report_text)
        else:
            self.assertIn("本轮没有正式短训练", report_text)
