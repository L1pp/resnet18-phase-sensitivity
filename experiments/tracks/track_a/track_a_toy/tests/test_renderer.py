from __future__ import annotations

import numpy as np
import unittest

from track_a_toy.renderer import (
    finite_position_grid,
    grid_points_with_visibility,
    is_fully_visible,
    position_grid,
    render_many,
    render_one,
    triangle_vertices,
)


class RendererTests(unittest.TestCase):
    def test_triangle_visibility_and_grid_are_explicit(self) -> None:
        vertices = triangle_vertices()
        self.assertEqual(vertices.shape, (3, 2))
        self.assertTrue(np.allclose(vertices.mean(axis=0), 0.0))
        self.assertTrue(is_fully_visible((32.0, 32.0)))
        self.assertFalse(is_fully_visible((0.0, 0.0)))

        points, ix, iy = position_grid()
        full_points, full_ix, full_iy, visible = grid_points_with_visibility()
        finite_points, finite_ix, finite_iy = finite_position_grid()
        self.assertEqual(points.shape, (256, 2))
        self.assertTrue(np.array_equal(points, full_points))
        self.assertTrue(np.array_equal(ix, full_ix))
        self.assertTrue(np.array_equal(iy, full_iy))
        self.assertEqual(int(visible.sum()), len(finite_points))
        self.assertTrue(np.array_equal(finite_points, full_points[visible]))
        self.assertTrue(np.array_equal(finite_ix, full_ix[visible]))
        self.assertTrue(np.array_equal(finite_iy, full_iy[visible]))

    def test_finite_and_torus_rendering_rules(self) -> None:
        center = render_one((32.0, 32.0), torus=False)
        self.assertEqual(center.shape, (3, 64, 64))
        self.assertEqual(center.dtype, np.uint8)
        self.assertTrue(np.array_equal(center[0], center[1]))
        with self.assertRaisesRegex(ValueError, "clipped"):
            render_many([(0.0, 0.0)], torus=False, require_fully_visible=True)
        self.assertEqual(render_many([(32.0, 32.0)], torus=False, require_fully_visible=True).shape, (1, 3, 64, 64))

        # The torus renderer is periodic in both coordinates, including the exact seam.
        self.assertTrue(np.array_equal(render_one((0.0, 0.0), torus=True), render_one((64.0, 64.0), torus=True)))
        images = render_many([(0.0, 0.0), (32.0, 32.0)], torus=True)
        self.assertEqual(images.shape, (2, 3, 64, 64))
