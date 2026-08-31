from __future__ import annotations

import unittest

import numpy as np

from phase3_high_upside.primitives import FAMILIES, extent_px, make_sample, normalize_xy, random_params, render_family


class PrimitiveTests(unittest.TestCase):
    def test_all_families_render_and_keep_generative_center(self):
        rng = np.random.default_rng(11)
        for family in FAMILIES:
            sample = make_sample(family, rng, image_size=96, size_mode="absolute", center=(40.0, 55.0))
            self.assertEqual(sample["image"].shape, (96, 96))
            self.assertEqual(sample["image"].dtype, np.uint8)
            self.assertAlmostEqual(sample["cx_px"], 40.0)
            self.assertAlmostEqual(sample["cy_px"], 55.0)
            u, v = normalize_xy(40.0, 55.0, 96)
            self.assertTrue(np.allclose(sample["xy_norm"], [u, v]))
            self.assertGreater(int(sample["image"].max()), 10)

    def test_relative_size_scales_with_canvas(self):
        rng = np.random.default_rng(12)
        small = random_params("blob", rng, 160, size_mode="relative")
        rng = np.random.default_rng(12)
        large = random_params("blob", rng, 320, size_mode="relative")
        self.assertGreater(large["sigma"], small["sigma"])
        self.assertAlmostEqual(large["sigma"] / small["sigma"], 320.0 / 160.0, places=5)

    def test_absolute_size_independent_of_canvas(self):
        rng = np.random.default_rng(13)
        a = random_params("circle", rng, 160, size_mode="absolute")
        rng = np.random.default_rng(13)
        b = random_params("circle", rng, 320, size_mode="absolute")
        self.assertAlmostEqual(a["radius"], b["radius"])

    def test_object_stays_on_canvas_for_sampled_centers(self):
        rng = np.random.default_rng(14)
        for family in FAMILIES:
            sample = make_sample(family, rng, image_size=96)
            ext = extent_px(family, sample["params"])
            self.assertGreaterEqual(sample["cx_px"], ext - 1e-6)
            self.assertLessEqual(sample["cx_px"], 95.0 - ext + 8.0 + 1e-6)
            self.assertEqual(render_family(family, sample["cx_px"], sample["cy_px"], sample["params"], 96).shape, (96, 96))
