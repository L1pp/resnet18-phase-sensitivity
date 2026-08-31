"""Independent numerical oracles for the exploratory Track B implementation.

The tests use the local reference functions by default.  To exercise a future
implementation, set ``TRACK_B_ORACLE_TARGET`` to a module exposing the API
listed in README.md.  The target module is an adapter boundary; this directory
does not import or edit the unified implementation root.
"""

from __future__ import annotations

import importlib
import os
import unittest
from typing import Any, Callable

import numpy as np

import scientific_oracles as ref


class _Target:
    def __init__(self) -> None:
        spec = os.environ.get("TRACK_B_ORACLE_TARGET", "").strip()
        self.module = importlib.import_module(spec) if spec else None

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self.module is None:
            fn: Callable[..., Any] = getattr(ref, name)
        else:
            fn = getattr(self.module, name)
        return fn(*args, **kwargs)


class ScientificOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = _Target()

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.target.call(name, *args, **kwargs)

    def test_matrix_counts_and_conditions(self) -> None:
        contract = self.call("matrix_contract")
        self.assertEqual(contract["logical_cells"], 84)
        self.assertEqual(contract["physical_jobs"], 82)
        self.assertEqual(contract["optimizer_jobs"], 78)
        self.assertEqual(tuple(contract["seeds"]), ref.SEEDS)
        self.assertEqual(contract["family_conditions"], ref.MATRIX_CONDITIONS)

    def test_coordinate_order_and_normalization(self) -> None:
        grid = self.call("canonical_grid41")
        np.testing.assert_equal(np.asarray(grid).shape, (1681, 2))
        np.testing.assert_allclose(grid[0], [59.0, 59.0])
        np.testing.assert_allclose(grid[1], [59.0, 61.625])
        np.testing.assert_allclose(grid[40], [59.0, 164.0])
        np.testing.assert_allclose(grid[41], [61.625, 59.0])
        np.testing.assert_allclose(grid[-1], [164.0, 164.0])
        normalized = self.call("normalized_xy", np.asarray([[223.0, 0.0], [0.0, 223.0]]))
        np.testing.assert_allclose(normalized, [[1.0, 0.0], [0.0, 1.0]])

    def test_b8_quadratic_formula(self) -> None:
        xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, -0.25]])
        observed = self.call("b8_quadratic", xy)
        expected = np.asarray(
            [
                [0.0, 0.0],
                [0.76, -0.10],
                [0.12, 0.77],
                [0.70 * 0.5 + 0.08 * -0.25 + 0.06 * 0.25 - 0.05 * 0.5 * -0.25 + 0.04 * 0.0625,
                 -0.06 * 0.5 + 0.72 * -0.25 - 0.04 * 0.25 + 0.06 * 0.5 * -0.25 + 0.05 * 0.0625],
            ]
        )
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)

    def test_b2_training_and_heldout_identity_sets_are_disjoint(self) -> None:
        observed = self.call("b2_identity_sets")
        expected = ref.b2_identity_sets()
        self.assertEqual(observed, expected)
        for condition, sets in observed.items():
            self.assertTrue(set(sets["train"]).isdisjoint(sets["heldout"]), condition)
            self.assertTrue(set(sets["train"]).isdisjoint(set(sets["heldout"])))

    def test_fixed_stream_is_always_batch_64_with_replacement(self) -> None:
        for n_rows in (4, 32, 64, 72, 256):
            stream = np.asarray(self.call("fixed_batch_stream", n_rows, 20260816, 2))
            self.assertEqual(stream.shape, (2, 64))
            self.assertTrue(np.all((stream >= 0) & (stream < n_rows)))
        tiny = np.asarray(self.call("fixed_batch_stream", 4, 20260816, 1))[0]
        self.assertEqual(len(tiny), 64)
        self.assertLess(len(np.unique(tiny)), 64)

    def test_b7_canonical_edges_are_112_unique_axial_edges(self) -> None:
        edges = np.asarray(self.call("b7_canonical_edges"))
        self.assertEqual(edges.shape, (112, 2))
        self.assertEqual(len({tuple(row) for row in edges.tolist()}), 112)
        np.testing.assert_array_equal(edges[:4], [[0, 8], [0, 1], [1, 9], [1, 2]])
        for left, right in edges:
            lx, ly = divmod(int(left), 8)
            rx, ry = divmod(int(right), 8)
            self.assertIn((int(rx - lx), int(ry - ly)), ((1, 0), (0, 1)))

    def test_b7_pcg64_draws_images_before_pairs_on_each_step(self) -> None:
        stream = self.call("b7_interleaved_stream", 20260816, 64, 2)
        images = np.asarray(stream["image_ids"])
        pairs = np.asarray(stream["pair_ids"])
        self.assertEqual(images.shape, (2, 64))
        self.assertEqual(pairs.shape, (2, 64))
        np.testing.assert_array_equal(images[0, :12], [35, 21, 47, 59, 6, 22, 2, 26, 18, 30, 62, 0])
        np.testing.assert_array_equal(pairs[0, :12], [9, 30, 43, 105, 85, 19, 6, 16, 103, 109, 86, 108])
        # Drawing all image batches first would produce a different second
        # pair batch.  This assertion catches that otherwise subtle ordering bug.
        block_rng = np.random.Generator(np.random.PCG64(20260816 + 17))
        _ = block_rng.integers(0, 64, size=(2, 64))
        block_pairs = block_rng.integers(0, 112, size=(2, 64))
        self.assertFalse(np.array_equal(pairs[0], block_pairs[0]))

    def test_b7_three_dense_conditions_share_pair_and_endpoint_occurrences(self) -> None:
        exposure = self.call("b7_matched_exposure", 20260816, 0)
        images = np.asarray(exposure["image_ids"])
        pair_ids = np.asarray(exposure["pair_ids"])
        endpoint_ids = np.asarray(exposure["endpoint_ids"])
        self.assertEqual(images.shape, (64,))
        self.assertEqual(pair_ids.shape, (64,))
        self.assertEqual(endpoint_ids.shape, (128,))
        edges = ref.b7_canonical_edges()
        selected = edges[pair_ids]
        np.testing.assert_array_equal(endpoint_ids[:64], selected[:, 0])
        np.testing.assert_array_equal(endpoint_ids[64:], selected[:, 1])
        # Passing the same exposure to all three arms is the scientific
        # contract; the endpoint list intentionally retains duplicate draws.
        for _condition in ("dense_absolute", "dense_relative_4_neighbor_plus_1_anchor", "dense_relative_4_neighbor_plus_4_corners"):
            np.testing.assert_array_equal(endpoint_ids, np.concatenate((selected[:, 0], selected[:, 1])))

    def test_b7_ordered_displacement_and_equal_loss_weights(self) -> None:
        pred = np.stack((np.arange(64, dtype=np.float64), 2.0 * np.arange(64, dtype=np.float64)), axis=1)
        truth = np.zeros((64, 2), dtype=np.float64)
        pair_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
        anchors = np.asarray([0], dtype=np.int64)
        result = self.call("b7_relative_loss", pred, truth, pair_ids, anchors)
        selected = ref.b7_canonical_edges()[pair_ids]
        direct_pair = pred[selected[:, 1]] - pred[selected[:, 0]]
        expected_pair = float(np.mean(direct_pair * direct_pair))
        expected_anchor = float(np.mean(pred[anchors] * pred[anchors]))
        self.assertAlmostEqual(result["pair_mse"], expected_pair)
        self.assertAlmostEqual(result["anchor_mse"], expected_anchor)
        self.assertAlmostEqual(result["total"], expected_pair + expected_anchor)
        # Reversing an edge reverses both predicted and true displacement.
        reversed_pred = pred[selected[:, 0]] - pred[selected[:, 1]]
        reversed_true = truth[selected[:, 0]] - truth[selected[:, 1]]
        np.testing.assert_allclose(reversed_pred - reversed_true, -direct_pair)

    def test_b7_absolute_uses_128_endpoint_occurrences(self) -> None:
        pred = np.arange(128, dtype=np.float64).reshape(64, 2)
        truth = np.zeros_like(pred)
        pair_ids = np.arange(64, dtype=np.int64)
        endpoint_ids = ref.b7_endpoint_occurrences(pair_ids)
        expected = float(np.mean(pred[endpoint_ids] ** 2))
        observed = self.call("b7_dense_absolute_loss", pred, truth, pair_ids)
        self.assertAlmostEqual(observed, expected)

    def test_b6_ridge_matches_small_hand_computation(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = np.asarray([[0.0, 0.0], [2.0, -1.0], [4.0, -2.0], [6.0, -3.0]])
        q = np.asarray([[4.0]])
        result = self.call("b6_ridge_reference", x, y, q, 1e-4, 1e-6)
        alpha = 1e-4 * (14.0 / 4.0)
        slope = np.asarray([10.0, -5.0]) / (5.0 + alpha)
        expected_prediction = np.asarray([[3.0, -1.5]]) + 2.5 * slope
        np.testing.assert_allclose(result["prediction"], expected_prediction, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result["beta"], slope.reshape(1, 2), rtol=0, atol=1e-12)
        self.assertAlmostEqual(result["alpha_scale"], 3.5)
        self.assertAlmostEqual(result["alpha"], alpha)
        self.assertEqual(result["rank"], 1)

    def test_b6_ridge_rank_zero_returns_mean_target(self) -> None:
        x = np.ones((4, 2), dtype=np.float64)
        y = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        q = np.asarray([[1.0, 1.0]])
        result = self.call("b6_ridge_reference", x, y, q, 1e-4, 1e-6)
        np.testing.assert_allclose(result["beta"], np.zeros((2, 2)))
        np.testing.assert_allclose(result["prediction"], [[4.0, 5.0]])
        self.assertEqual(result["rank"], 0)

    def test_common_and_b4_learning_rate_formulas(self) -> None:
        self.assertAlmostEqual(self.call("common_lr", 0), 1e-3)
        self.assertAlmostEqual(self.call("common_lr", 1500), (1e-3 + 1e-5) / 2.0)
        self.assertAlmostEqual(self.call("common_lr", 3000), 1e-5)
        self.assertAlmostEqual(self.call("b4_causal_lr", 1), 1e-3 / 500.0)
        self.assertAlmostEqual(self.call("b4_causal_lr", 500), 1e-3)
        self.assertGreater(self.call("b4_causal_lr", 501), 1e-5)
        self.assertLess(self.call("b4_causal_lr", 501), 1e-3)
        self.assertAlmostEqual(self.call("b4_causal_lr", 20000), 1e-5)

    def test_metrics_recompute_raw_mae_and_inclusive_support_gate(self) -> None:
        pred = np.asarray([[1.0, 3.0], [5.0, 7.0]])
        truth = np.asarray([[0.0, 1.0], [1.0, 3.0]])
        anchor_pred = np.asarray([[1.0, 2.0], [3.0, 7.0]])
        anchor_truth = np.asarray([[1.0, 1.0], [1.0, 7.0]])
        metrics = self.call("metric_summary", pred, truth, anchor_pred, anchor_truth)
        self.assertAlmostEqual(metrics["full_box_raw_mae_px"], 2.75)
        self.assertAlmostEqual(metrics["full_box_u_mae"], 2.75 / 223.0)
        self.assertAlmostEqual(metrics["anchor_raw_mae_px"], 0.75)
        self.assertTrue(self.call("support_gate", 0.25))
        self.assertFalse(self.call("support_gate", 0.2500001))
        self.assertFalse(self.call("support_gate", float("nan")))


if __name__ == "__main__":
    unittest.main()
