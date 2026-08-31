import unittest

import numpy as np

from position_sources.protocol import load_protocol
from position_sources.renderer import (
    dense_domain_points,
    render_one,
    render_points,
    split_dense_domain,
    split_torus_domain,
    torus_points,
)


class RendererTests(unittest.TestCase):
    def test_complete_dense_domain_and_stratified_split(self):
        cfg = load_protocol()
        domain = dense_domain_points(cfg)
        train, test, metadata = split_dense_domain(cfg)
        self.assertEqual(domain.shape, (16384, 2))
        self.assertEqual(len(train) + len(test), len(domain))
        self.assertTrue(metadata["no_overlap"])
        self.assertTrue(metadata["complete_domain"])
        for split in (metadata["train"], metadata["test"]):
            self.assertEqual(split["quotient_cell_count"], 16)
            self.assertEqual(split["x_residues"], list(range(32)))
            self.assertEqual(split["y_residues"], list(range(32)))
        self.assertEqual(set(map(tuple, train)).intersection(set(map(tuple, test))), set())

    def test_render_types_and_triangle_are_deterministic(self):
        blob = render_one((512.0, 512.0), image_size=1024, sigma_px=6.0, primitive="blob")
        triangle = render_one((512.0, 512.0), image_size=1024, sigma_px=6.0, primitive="triangle")
        self.assertEqual(blob.dtype, np.uint8)
        self.assertEqual(triangle.dtype, np.uint8)
        self.assertFalse(np.array_equal(blob, triangle))
        self.assertTrue(np.array_equal(blob, render_one((512.0, 512.0), image_size=1024, sigma_px=6.0, primitive="blob")))

    def test_torus_wraps_and_has_complete_integer_period(self):
        cfg = load_protocol()
        points = torus_points(cfg)
        self.assertEqual(points.shape, (256 * 256, 2))
        self.assertTrue(np.array_equal(points[0], [0.0, 0.0]))
        left = render_one((0.0, 128.0), image_size=256, sigma_px=6.0, primitive="blob", torus=True)
        right = render_one((256.0, 128.0), image_size=256, sigma_px=6.0, primitive="blob", torus=True)
        self.assertTrue(np.array_equal(left, right))

    def test_empty_batch_shape(self):
        result = render_points([], image_size=64, sigma_px=6.0)
        self.assertEqual(result.shape, (0, 64, 64))


if __name__ == "__main__":
    unittest.main()
