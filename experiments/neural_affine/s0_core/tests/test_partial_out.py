import unittest

import numpy as np

from closeout_sprint.partial_out import (
    nuisance_basis,
    partial_out_group,
    partial_out_groups,
    spatial_block_cv_ridge,
    partial_out_vector_group,
)


class PartialOutTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.asarray(
            [[float(x), float(y)] for x in range(8) for y in range(8)],
            dtype=np.float64,
        )

    def test_basis_contains_fixed_nuisance_families(self):
        basis, names = nuisance_basis(self.coords, domain=((0.0, 7.0), (0.0, 7.0)))
        self.assertEqual(basis.shape[0], len(self.coords))
        expected = {
            "nearest_anchor_distance",
            "nearest_anchor_distance_sq",
            "boundary_distance",
            "poly_x2",
            "poly_xy",
            "poly_y2",
            "sin_x_stride32",
            "cos_x_stride32",
            "sin_y_stride32",
            "cos_y_stride32",
        }
        self.assertTrue(expected.issubset(set(names)))

    def test_single_block_is_rejected_without_in_sample_gate(self):
        basis, _ = nuisance_basis(self.coords)
        with self.assertRaises(ValueError):
            spatial_block_cv_ridge(
                basis,
                np.arange(len(self.coords), dtype=np.float64),
                self.coords,
                block_size=1000.0,
            )

    def test_known_anchor_distance_nuisance_is_removed(self):
        basis, names = nuisance_basis(self.coords, domain=((0.0, 7.0), (0.0, 7.0)))
        nearest = basis[:, names.index("nearest_anchor_distance")]
        rng = np.random.default_rng(20260821)
        signals = np.column_stack(
            [nearest + 0.002 * rng.normal(size=len(nearest)) for _ in range(4)]
        )
        result = partial_out_group(
            signals,
            self.coords,
            domain=((0.0, 7.0), (0.0, 7.0)),
            block_size=2.0,
        )
        self.assertLess(result.gate.median_partial_correlation, 0.20)
        self.assertEqual(result.gate.status, "closed")

    def test_unmodelled_checkerboard_shared_signal_is_retained(self):
        checkerboard = np.asarray(
            [1.0 if int(x + y) % 2 == 0 else -1.0 for x, y in self.coords]
        )
        rng = np.random.default_rng(7)
        signals = np.column_stack(
            [checkerboard + 0.05 * rng.normal(size=len(checkerboard)) for _ in range(4)]
        )
        result = partial_out_group(
            signals,
            self.coords,
            domain=((0.0, 7.0), (0.0, 7.0)),
            block_size=2.0,
        )
        self.assertGreater(result.gate.median_partial_correlation, 0.50)
        self.assertGreaterEqual(result.gate.retained_fraction, 0.70)
        self.assertEqual(result.gate.status, "survived")

    def test_task_groups_are_audited_independently(self):
        rng = np.random.default_rng(12)
        payload = {
            "neural_affine_2d": {
                "coords": self.coords,
                "signals": rng.normal(size=(len(self.coords), 3)),
            },
            "amd_2d": {
                "coords": self.coords + 100.0,
                "signals": rng.normal(size=(len(self.coords), 3)),
            },
        }
        results = partial_out_groups(payload, block_size=2.0)
        self.assertEqual(set(results), {"neural_affine_2d", "amd_2d"})
        self.assertEqual(results["neural_affine_2d"].residual_signals.shape, (64, 3))
        self.assertEqual(results["amd_2d"].residual_signals.shape, (64, 3))
        # The groups use separate nuisance fits and do not emit a pooled 6-seed
        # correlation matrix.
        self.assertEqual(results["neural_affine_2d"].original_correlations.shape, (3, 3))
        self.assertEqual(results["amd_2d"].original_correlations.shape, (3, 3))

    def test_all_seed_rows_in_a_block_are_held_out_together(self):
        checkerboard = np.asarray(
            [1.0 if int(x + y) % 2 == 0 else -1.0 for x, y in self.coords]
        )
        signals = np.column_stack([checkerboard, checkerboard + 0.01])
        result = partial_out_group(signals, self.coords, block_size=2.0)
        # The pooled CV used to choose ridge repeats each coordinate's block for
        # all seeds, so the number of folds is the spatial block count, not 2x.
        self.assertEqual(result.ridge_cv.block_count, 16)

    def test_vector_contract_reports_pooled_x_y_and_uses_pooled_gate(self):
        # appearance x spatial x seed x component; nuisance is a smooth
        # spatial term and checkerboard is the shared signal that should stay.
        coords = np.asarray(
            [[float(x), float(y)] for x in range(5) for y in range(5)],
            dtype=np.float64,
        )
        n_app, n_seed = 3, 3
        checker = np.asarray([1.0 if int(x + y) % 2 == 0 else -1.0 for x, y in coords])
        smooth = coords[:, 0] + 0.5 * coords[:, 1]
        vectors = np.empty((n_app, len(coords), n_seed, 2), dtype=np.float64)
        for app in range(n_app):
            for seed in range(n_seed):
                noise = 0.01 * np.random.default_rng(100 + seed + app).normal(size=len(coords))
                vectors[app, :, seed, 0] = checker + smooth * 0.01 + noise
                vectors[app, :, seed, 1] = checker + smooth * 0.01 + noise
        result = partial_out_vector_group(vectors, coords, block_size=1.0)
        self.assertEqual(result.residual_vectors.shape, vectors.shape)
        self.assertEqual(result.pooled_original_correlations.shape, (n_seed, n_seed))
        self.assertEqual(result.x_original_correlations.shape, (n_seed, n_seed))
        self.assertEqual(result.y_partial_correlations.shape, (n_seed, n_seed))
        self.assertEqual(result.to_dict()["gate_signal"], "pooled_vector_flatten")
        self.assertTrue(result.to_dict()["u_norm_is_supplementary"])

    def test_vector_grid_contract_rejects_appearance_dependent_coords(self):
        coords = np.asarray([[float(x), float(y)] for x in range(4) for y in range(4)], dtype=np.float64).reshape(4, 4, 2)
        vectors = np.zeros((2, 4, 4, 2, 2), dtype=np.float64)
        bad_coords = np.stack([coords, coords + 1.0], axis=0)
        with self.assertRaises(ValueError):
            partial_out_vector_group(vectors, bad_coords, block_size=1.0)


if __name__ == "__main__":
    unittest.main()
