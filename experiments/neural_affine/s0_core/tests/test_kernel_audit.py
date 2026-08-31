import math
import unittest

import numpy as np

from closeout_sprint.kernel_audit import (
    FIT_SUPPORT_NAMES,
    HELD_OUT_SUPPORT_NAMES,
    PREDICTOR_FEATURE_NAMES,
    build_geometry_feature_table,
    evaluate_kernel_gate,
    fit_fixed_log_error_predictor,
    geometry_features,
    kernel_posterior_variance,
    validate_support_partition,
)


class KernelAuditTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.asarray([[float(x), float(y)] for x in range(4) for y in range(4)])
        self.gap = np.column_stack(
            [
                self.grid[:, 0],
                self.grid[:, 1],
                self.grid[:, 0] * self.grid[:, 1],
            ]
        )
        self.manifest = {
            "appearance_aligned": True,
            "n_grid": len(self.grid),
            "feature_shape": [len(self.grid), self.gap.shape[1]],
            "grid_coords": self.grid,
            "fingerprint": "synthetic-gap-grid",
        }

    def test_fixed_fit_and_held_out_names_are_explicit(self):
        self.assertEqual(FIT_SUPPORT_NAMES, ("G64", "G32", "G16", "G9", "maximin9", "boundary8_center"))
        self.assertEqual(HELD_OUT_SUPPORT_NAMES, ("random9", "cross5", "diagonal8"))

    def test_coordinate_overlap_is_not_target_leakage(self):
        points = self.grid[:4]
        supports = {name: points for name in (*FIT_SUPPORT_NAMES, *HELD_OUT_SUPPORT_NAMES)}
        fit, held = validate_support_partition(supports)
        self.assertEqual(len(fit), 6)
        self.assertEqual(len(held), 3)
        with self.assertRaises(ValueError):
            validate_support_partition(
                supports,
                geometry_provenance={name: "same-run" for name in supports},
            )

    def test_gap_grid_and_manifest_are_required(self):
        with self.assertRaises(ValueError):
            geometry_features(self.grid[:3], self.grid, gap_feature_grid=None, gap_manifest=self.manifest)
        with self.assertRaises(ValueError):
            geometry_features(self.grid[:3], self.grid, gap_feature_grid=self.gap, gap_manifest=None)
        result = geometry_features(
            self.grid[:3],
            self.grid,
            gap_feature_grid=self.gap,
            gap_manifest=self.manifest,
            support_name="G9",
            support_indices={"G9": [0, 1, 2]},
        )
        self.assertEqual(tuple(result), PREDICTOR_FEATURE_NAMES)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_public_kernel_requires_manifest_backed_gap_grid(self):
        with self.assertRaises(ValueError):
            kernel_posterior_variance(self.grid, self.gap[:3])
        values = kernel_posterior_variance(
            self.gap,
            self.gap[:3],
            gap_manifest=self.manifest,
        )
        self.assertEqual(values.shape, (len(self.gap),))

    def test_feature_table_uses_gap_rows_and_only_two_predictor_features(self):
        supports = {"G9": self.grid[:3], "random9": self.grid[3:6]}
        names, feature_names, matrix = build_geometry_feature_table(
            supports,
            self.grid,
            gap_feature_grid=self.gap,
            gap_manifest=self.manifest,
            support_indices={"G9": [0, 1, 2], "random9": [3, 4, 5]},
        )
        self.assertEqual(names, ("G9", "random9"))
        self.assertEqual(feature_names, PREDICTOR_FEATURE_NAMES)
        self.assertEqual(matrix.shape, (2, 2))

    def test_low_degree_predictor_rejects_overfit_feature_count(self):
        with self.assertRaises(ValueError):
            fit_fixed_log_error_predictor(np.ones((2, 2)), [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_fixed_log_error_predictor(np.ones((6, 3)), np.arange(1.0, 7.0))
        predictor = fit_fixed_log_error_predictor(
            np.asarray([[1.0, 0.1], [2.0, 0.2], [3.0, 0.4], [4.0, 0.8], [5.0, 1.6], [6.0, 3.2]]),
            [1.0, 1.4, 2.0, 3.0, 4.5, 7.0],
            feature_names=PREDICTOR_FEATURE_NAMES,
        )
        self.assertIn(predictor.ridge, (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0))

    def test_gate_treats_euclidean_baseline_as_raw_positive_errors(self):
        observed = np.asarray([1.0, 2.1, 3.8])
        predicted_raw = np.asarray([1.0, 2.0, 4.0])
        baseline_raw = np.asarray([3.0, 5.0, 7.0])
        result = evaluate_kernel_gate(
            np.log(predicted_raw),
            observed,
            baseline_raw,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.within_50_percent_count, 3)
        self.assertGreaterEqual(result.improvement_fraction, 0.20)


if __name__ == "__main__":
    unittest.main()
