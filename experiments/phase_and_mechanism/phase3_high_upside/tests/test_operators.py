from __future__ import annotations

import unittest

import numpy as np

from phase3_high_upside.operators import (
    apply_affine,
    compose_affine,
    fit_affine_ridge,
    fit_operator,
    normalized_error,
    random_same_rank_operator,
    shuffle_pairs,
)
from phase3_high_upside.tests.synthetic_group import make_latent_field, pair_features, true_operator


class OperatorTests(unittest.TestCase):
    def test_compose_affine_matches_definition(self):
        rng = np.random.default_rng(0)
        m1, b1 = rng.standard_normal((4, 4)), rng.standard_normal(4)
        m2, b2 = rng.standard_normal((4, 4)), rng.standard_normal(4)
        z = rng.standard_normal((7, 4))
        m, b = compose_affine(m2, b2, m1, b1)
        left = apply_affine(apply_affine(z, m1, b1), m2, b2)
        right = apply_affine(z, m, b)
        self.assertTrue(np.allclose(left, right))

    def test_ridge_recovers_true_group_action(self):
        origins = np.stack(np.meshgrid(np.arange(0.0, 9.0), np.arange(0.0, 9.0), indexing="ij"), axis=-1).reshape(-1, 2)
        field = make_latent_field(12, origins, dim=8, seed=1)
        z, zp = pair_features(field, 1.0, 0.0)
        n = len(z)
        z_fit, zp_fit = z[: n // 2], zp[: n // 2]
        z_val, zp_val = z[n // 2 : 3 * n // 4], zp[n // 2 : 3 * n // 4]
        z_te, zp_te = z[3 * n // 4 :], zp[3 * n // 4 :]
        result = fit_operator(z_fit, zp_fit, z_val, zp_val, z_te, zp_te)
        self.assertLess(result["e_test"], 1e-6)
        true_m, _true_b = true_operator(1.0, 0.0, 8)
        pred = apply_affine(z_te, result["matrix"], result["bias"])
        self.assertLess(normalized_error(pred, apply_affine(z_te, true_m, np.zeros(8))), 1e-6)

    def test_shuffled_pairs_are_worse(self):
        origins = np.stack(np.meshgrid(np.arange(0.0, 8.0), np.arange(0.0, 8.0), indexing="ij"), axis=-1).reshape(-1, 2)
        field = make_latent_field(10, origins, dim=8, seed=2)
        z, zp = pair_features(field, 2.0, 0.0)
        n = len(z)
        cut = (3 * n) // 4
        z_s, zp_s = shuffle_pairs(z, zp, seed=3)
        m, b = fit_affine_ridge(z[:cut], zp[:cut], 1e-6)
        m_s, b_s = fit_affine_ridge(z_s[:cut], zp_s[:cut], 1e-6)
        e = normalized_error(apply_affine(z[cut:], m, b), zp[cut:])
        e_s = normalized_error(apply_affine(z_s[cut:], m_s, b_s), zp_s[cut:])
        self.assertLess(e, 1e-5)
        self.assertGreater(e_s, 0.2)

    def test_random_same_rank_is_not_the_true_map(self):
        m, b = true_operator(1.0, 1.0, 8)
        rm, rb = random_same_rank_operator(m, b, seed=4)
        z = np.random.default_rng(5).standard_normal((32, 8))
        e_true = normalized_error(apply_affine(z, m, b), apply_affine(z, m, b))
        e_rand = normalized_error(apply_affine(z, rm, rb), apply_affine(z, m, b))
        self.assertEqual(e_true, 0.0)
        self.assertGreater(e_rand, 0.3)
