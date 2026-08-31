from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from track_a_toy.renderer import render_one

from track_a_minicnn.config import FIGURE_ROOT, REPORT_JSON_PATH, REPORT_PATH
from track_a_minicnn.model import build_shared_models, model_shape_report
from track_a_minicnn.pipeline import (
    _fit_probe_task,
    _fit_standardizer,
    _transform_standardizer,
    anchor_shift_diagnostics,
    build_masks,
    load_source_data,
    scope_splits,
)


class MiniCNNTests(unittest.TestCase):
    def test_shapes_and_direct_shared_state_copy(self) -> None:
        models = build_shared_models(device="cpu")
        reference = models["Z"].state_dict()
        for condition, model in models.items():
            report = model_shape_report(model)
            expected_size = [64, 64, 64, 64] if condition in {"Z", "C"} else [62, 60, 58, 56]
            self.assertEqual([report["stage_shapes"][f"conv{i}"][-1] for i in range(1, 5)], expected_size)
            self.assertEqual(report["pool_shape"], [1, 32])
            self.assertEqual(report["head_shape"], [4, 32])
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), (condition, key))

    def test_valid_is_zero_core_of_same_finite_input(self) -> None:
        models = build_shared_models(device="cpu")
        value = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            z = models["Z"].forward_stages(value)["conv4"]
            v = models["V"].forward_stages(value)["conv4"]
        self.assertEqual(tuple(z.shape[-2:]), (64, 64))
        self.assertEqual(tuple(v.shape[-2:]), (56, 56))
        self.assertTrue(torch.allclose(z[:, :, 4:-4, 4:-4], v, atol=1e-6, rtol=1e-6))

    def test_circular_roll_and_pool_are_invariant(self) -> None:
        models = build_shared_models(device="cpu")
        value = torch.randn(1, 3, 64, 64)
        rolled = torch.roll(value, shifts=(7, 11), dims=(2, 3))
        with torch.no_grad():
            left = models["C"].forward_features(value)
            right = models["C"].forward_features(rolled)
        self.assertTrue(torch.allclose(left, right, atol=1e-5, rtol=1e-5))

    def test_masks_counts_and_finite_torus_agreement_on_visible(self) -> None:
        data = load_source_data()
        masks = build_masks(data["finite"])
        splits = scope_splits(masks)
        self.assertEqual(int(masks["visible"].sum()), 182)
        self.assertEqual(int(masks["safe"].sum()), 90)
        self.assertEqual(int(masks["visible_outer"].sum()), 92)
        self.assertEqual(int(masks["clipped"].sum()), 74)
        self.assertEqual((int(splits["visible"]["train"].sum()), int(splits["visible"]["test"].sum())), (133, 49))
        self.assertEqual((int(splits["safe"]["train"].sum()), int(splits["safe"]["test"].sum())), (65, 25))
        for index in np.flatnonzero(masks["visible"]):
            point = data["finite"]["points"][index]
            self.assertTrue(np.array_equal(render_one(point, torus=False), render_one(point, torus=True)))

    def test_near_constant_noise_is_zeroed_and_not_amplified(self) -> None:
        rng = np.random.default_rng(4)
        values = np.ones((12, 5), dtype=np.float64) + rng.normal(0.0, 1e-8, size=(12, 5))
        scaler = _fit_standardizer(values[:8])
        transformed = _transform_standardizer(values, scaler)
        self.assertEqual(scaler["active_feature_count"], 0)
        self.assertTrue(np.array_equal(transformed, np.zeros_like(transformed)))
        points = np.asarray([(0, 0), (4, 0), (8, 0), (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0), (36, 0), (40, 0), (44, 0)], dtype=np.float64)
        train = np.asarray([True] * 8 + [False] * 4)
        row = _fit_probe_task(values, points, train, ~train, target_kind="phase_sincos")
        self.assertTrue(row["constant_baseline"])
        self.assertEqual(row["active_feature_count"], 0)
        self.assertAlmostEqual(row["test_metric"], row["baseline_metric"], places=12)

    def test_anchor_diagnostic_has_three_deltas_and_ten_anchors(self) -> None:
        result = anchor_shift_diagnostics()
        self.assertEqual(result["anchor_count"], 10)
        self.assertEqual(result["deltas_px"], [1, 4, 16])
        for condition in ("Z", "V", "C"):
            rows = result["models"][condition]["rows"]
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row["valid"] and row["gap_mean_abs"] is not None for row in rows))

    def test_generated_report_numbers_and_figures_when_present(self) -> None:
        if not REPORT_PATH.exists() or not REPORT_JSON_PATH.exists():
            self.skipTest("phase-1 report has not been generated yet")
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        report = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
        self.assertIn("Track A MiniCNN Valid-Boundary v1", report_text)
        for scope in ("visible", "safe"):
            self.assertIn(scope, report["probe"]["scopes"])
            figure = Path(report["figures"][f"{scope}_probe"])
            self.assertTrue(figure.exists())
            for condition in ("Z", "V", "C"):
                value = report["probe"]["scopes"][scope]["conditions"][condition]["phase_sincos"]["test_metric"]
                self.assertIn(f"{value:.3f}", report_text)
        self.assertEqual(len(list(FIGURE_ROOT.glob("*固定输入10张.png"))), 2)


if __name__ == "__main__":
    unittest.main()
