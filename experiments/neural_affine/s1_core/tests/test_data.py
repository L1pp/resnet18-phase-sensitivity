"""Lightweight CPU tests for the self-contained S1 data module."""

from __future__ import annotations

import sys
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from s1clean import data  # noqa: E402


class CanonicalDataTests(TestCase):
    def test_protocol_and_grids_are_frozen(self) -> None:
        protocol = data.load_protocol()
        self.assertEqual(protocol["image_size"], 224)
        self.assertEqual(protocol["coord_scale"], 223.0)
        self.assertEqual(protocol["safe_box_px"], {"low": 59.0, "high": 164.0})
        self.assertEqual(protocol["anchor_grid"]["tids"], [0, 7, 56, 63])
        self.assertEqual(protocol["seeds"], [20260816, 20260817, 20260818])
        self.assertEqual(protocol["training"]["steps"], 3000)
        self.assertEqual(protocol["training"]["batch_size"], 64)
        self.assertEqual(protocol["training"]["optimizer"], "AdamW")
        self.assertEqual(protocol["models"], ["vanilla", "coordconv"])
        self.assertEqual(protocol["outputs"]["predictions"], "predictions.npz")
        self.assertEqual(protocol["gates"]["audit_metric_abs"], 1e-6)

        support = data.support_points_px()
        np.testing.assert_array_equal(
            support,
            np.asarray([[59, 59], [59, 164], [164, 59], [164, 164]], dtype=np.float64),
        )
        dense = data.dense_points_px()
        self.assertEqual(dense.shape, (1681, 2))
        np.testing.assert_array_equal(dense[[0, 40, 1640, 1680]], support)

    def test_probe_indices_are_25_unique_dense_indices(self) -> None:
        indices = data.probe_indices()
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(indices.shape, (25,))
        self.assertEqual(len(set(indices.tolist())), 25)
        self.assertTrue(np.all((indices >= 0) & (indices < 1681)))

    def test_renderer_is_uint8_and_worker_order_is_stable(self) -> None:
        points = np.asarray([[59.0, 59.0], [111.5, 111.5], [164.0, 164.0]])
        serial = data.render_points(points, workers=1)
        parallel = data.render_points(points, workers=2)
        self.assertEqual(serial.shape, (3, 224, 224))
        self.assertEqual(serial.dtype, np.uint8)
        np.testing.assert_array_equal(serial, parallel)
        self.assertGreater(int(serial.max()), 200)

    def test_dataset_and_sample_hashes_bind_points_and_pixels(self) -> None:
        points = np.asarray([[59.0, 59.0], [164.0, 164.0]])
        images = data.render_points(points, workers=1)
        digest = data.dataset_hash(images, points)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, data.dataset_hash(images.copy(), points.copy()))
        changed_points = points.copy()
        changed_points[0, 0] += 1.0
        self.assertNotEqual(digest, data.dataset_hash(images, changed_points))
        self.assertEqual(len(data.sample_hashes(images, points)), 2)

    def test_materialize_and_load_cache_manifest_contract(self) -> None:
        # Replace only the renderer in this test.  The canonical 41x41 points
        # and all manifest/probe bookkeeping remain real, while the test stays
        # small enough for a local CPU smoke run.
        def tiny_renderer(points, workers=None, protocol=None):
            count = len(np.asarray(points))
            return np.arange(count * 4, dtype=np.uint8).reshape(count, 2, 2)

        with self.subTest("materialize"), patch.object(data, "render_points", side_effect=tiny_renderer):
            with self._temporary_directory() as output:
                manifest = data.materialize_cache(output, workers=2)
                self.assertEqual(manifest["dense"]["count"], 1681)
                self.assertEqual(len(manifest["dense"]["sample_sha256"]), 1681)
                self.assertEqual(len(manifest["probes"]["indices"]), 25)
                self.assertTrue((output / "images.npy").exists())
                self.assertTrue((output / "points.npy").exists())
                self.assertTrue((output / "cache.npz").exists())
                loaded = data.load_cache(output)
                self.assertEqual(loaded["images"].shape, (1681, 2, 2))
                np.testing.assert_array_equal(loaded["points"], data.dense_points_px())

    @staticmethod
    @contextmanager
    def _temporary_directory():
        # The managed Windows runner may create tempfile directories with an
        # ACL that denies this process.  Use one exact, task-local directory
        # instead of changing system ACLs or relying on %TEMP%.
        output = PACKAGE_ROOT / f".tmp_cache_test_{os.getpid()}"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        try:
            yield output
        finally:
            if output.exists():
                shutil.rmtree(output)


if __name__ == "__main__":
    import unittest

    unittest.main()
