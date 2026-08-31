from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from closeout_sprint.evaluator import audit_headlines, evaluate_arrays, evaluate_npz, fit_affine, load_field_npz


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / f".test_evaluator_{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_evaluator_recomputes_raw_and_affine_residual_without_old_u(self) -> None:
        true = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        pred = true @ np.array([[2.0, 0.0], [0.0, 3.0]]) + np.array([4.0, -2.0])
        payload = evaluate_arrays(pred, true, support_indices=[0, 3])
        self.assertGreater(payload["raw_mae_px"], 0.0)
        self.assertLess(payload["affine_removed_mae_px"], 1e-12)
        self.assertGreater(payload["anchor_mae_px"], 0.0)
        np.testing.assert_allclose(payload["u"], 0.0, atol=1e-12)

    def test_evaluator_ignores_stale_err_and_u_keys(self) -> None:
        true = np.zeros((1, 2, 2, 2), dtype=np.float32)
        pred = np.ones_like(true)
        path = self.root / "field.npz"
        np.savez(path, pred=pred, true=true, err=np.zeros_like(true), u=np.zeros_like(true))
        loaded = load_field_npz(path)
        self.assertIsNotNone(loaded["pred"])
        result = evaluate_npz(path)
        self.assertAlmostEqual(result["raw_mae_px"], np.sqrt(2.0))
        self.assertLess(result["affine_removed_mae_px"], 1e-12)
        np.testing.assert_allclose(result["raw_error"], pred)

    def test_real_rendered_xy_shape_and_output_preserves_metadata(self) -> None:
        true = np.zeros((32, 41, 41, 2), dtype=np.float32)
        pred = np.ones_like(true)
        coords = np.stack(np.meshgrid(np.arange(41), np.arange(41), indexing="ij"), axis=-1)
        support_ids = np.arange(32 * 41 * 41).reshape(32, 41, 41)
        left = self.root / "a" / "same" / "field.npz"
        right = self.root / "b" / "same" / "field.npz"
        left.parent.mkdir(parents=True)
        right.parent.mkdir(parents=True)
        np.savez(left, pred=pred, true=true, coords=coords, support_ids=support_ids)
        np.savez(right, pred=pred * 2, true=true, coords=coords, support_ids=support_ids)
        out = self.root / "results"
        summary = audit_headlines([left, right], out)
        self.assertEqual(len(summary["runs"]), 2)
        files = sorted(out.glob("*_recomputed.npz"))
        self.assertEqual(len(files), 2)
        self.assertNotEqual(files[0].name, files[1].name)
        with np.load(files[0], allow_pickle=False) as blob:
            for key in ("pred", "true", "coords", "support_ids", "raw_error", "u"):
                self.assertIn(key, blob.files)

    def test_fit_affine_returns_pred_to_true_convention(self) -> None:
        pred = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 4.0]])
        true = pred @ np.array([[0.5, 0.0], [0.0, 2.0]]) + np.array([3.0, -1.0])
        a, b = fit_affine(pred, true)
        np.testing.assert_allclose(a, [[0.5, 0.0], [0.0, 2.0]], atol=1e-12)
        np.testing.assert_allclose(b, [3.0, -1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
