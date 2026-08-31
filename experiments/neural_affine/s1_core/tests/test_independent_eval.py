"""Contract tests for the clean-room S1 independent evaluator."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from s1clean.independent_eval import (  # noqa: E402
    audit_run,
    fit_affine_float64,
    recompute_metrics,
)


class IndependentEvaluatorTests(TestCase):
    def setUp(self) -> None:
        self.root = PACKAGE_ROOT / f".tmp_independent_eval_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _protocol() -> dict[str, object]:
        # Use the frozen protocol's field names, but a tiny image/grid so this
        # test exercises parsing and rendering without a CNN or large arrays.
        return {
            "schema_version": 1,
            "image_size": 16,
            "coord_scale": 15.0,
            "safe_box_px": {"low": 2.0, "high": 13.0},
            "anchor_grid": {"nx": 4, "ny": 4, "tids": [0, 3, 12, 15]},
            "dense_grid": {"n": 5},
            "renderer": {"family": "point", "radius": 0.0, "supersample": 1},
            "model": {"variant": "identity", "output_space": "px"},
            "evaluation": {"batch_size": 8},
            "primary_predictions": "primary_predictions.npz",
            "primary_metrics": "primary_metrics.json",
        }

    def _write_identity_run(self) -> tuple[Path, Path]:
        protocol_path = self.root / "protocol.json"
        protocol = self._protocol()
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

        run_dir = self.root / "run"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        torch.save(
            {"model_state": {}, "variant": "identity", "seed": 0},
            checkpoint_dir / "best.pt",
        )

        values = np.linspace(2.0, 13.0, 5, dtype=np.float64)
        dense = np.stack(np.meshgrid(values, values, indexing="ij"), axis=-1).reshape(1, -1, 2)
        support_values = np.linspace(2.0, 13.0, 4, dtype=np.float64)
        support_grid = np.stack(np.meshgrid(support_values, support_values, indexing="ij"), axis=-1).reshape(-1, 2)
        support = support_grid[[0, 3, 12, 15]][None, :, :]
        np.savez_compressed(
            run_dir / "primary_predictions.npz",
            pred_px=dense,
            true_px=dense,
            anchor_pred_px=support,
            anchor_true_px=support,
        )
        (run_dir / "primary_metrics.json").write_text(
            json.dumps(
                {
                    "anchor_mae_px": 0.0,
                    "raw_full_box_mae_px": 0.0,
                    "affine_removed_mae_px": 0.0,
                }
            ),
            encoding="utf-8",
        )
        return run_dir, protocol_path

    def test_identity_run_is_replayed_and_audited(self) -> None:
        run_dir, protocol_path = self._write_identity_run()

        result = audit_run(run_dir, protocol_path)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["prediction_comparison"]["passed"])
        self.assertTrue(all(item["passed"] for item in result["metric_comparison"].values()))
        self.assertEqual(result["prediction_comparison"]["primary_prediction_max_abs_diff_px"], 0.0)
        audit_path = run_dir / "independent_audit.json"
        self.assertTrue(audit_path.exists())
        self.assertEqual(json.loads(audit_path.read_text(encoding="utf-8"))["status"], "passed")

    def test_prediction_tolerance_failure_is_recorded(self) -> None:
        run_dir, protocol_path = self._write_identity_run()
        prediction_path = run_dir / "primary_predictions.npz"
        with np.load(prediction_path) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        arrays["pred_px"] = arrays["pred_px"].copy()
        arrays["pred_px"][0, 0, 0] += 0.002
        np.savez_compressed(prediction_path, **arrays)

        result = audit_run(run_dir, protocol_path)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["prediction_comparison"]["passed"])
        self.assertGreater(result["prediction_comparison"]["primary_prediction_max_abs_diff_px"], 1e-3)

    def test_float64_affine_fit_and_metrics_use_euclidean_point_error(self) -> None:
        pred = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        matrix = np.asarray([[1.2, 0.1], [-0.3, 0.8]])
        bias = np.asarray([0.4, -0.2])
        true = pred @ matrix.T + bias

        fitted_matrix, fitted_bias, residual = fit_affine_float64(pred, true)
        np.testing.assert_allclose(fitted_matrix, matrix, atol=1e-12)
        np.testing.assert_allclose(fitted_bias, bias, atol=1e-12)
        self.assertLess(float(np.max(np.abs(residual))), 1e-12)

        metrics = recompute_metrics(pred, true, pred[:4], true[:4])
        self.assertAlmostEqual(metrics["raw_full_box_mae_px"], float(np.mean(np.linalg.norm(pred - true, axis=1))))
        self.assertLess(metrics["affine_removed_mae_px"], 1e-12)
        self.assertEqual(metrics["dtype"], "float64")


if __name__ == "__main__":
    import unittest

    unittest.main()
