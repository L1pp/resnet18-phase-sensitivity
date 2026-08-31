from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from fsx.records import append_run_catalog, create_attempt
from fsx.schedules import b4_causal_lr_for_update, b7_dense_absolute_loss, b7_relative_loss, common_loss, common_lr_after_update, common_lr_for_update
from fsx.science import b6_ridge, metric_summary, support_gate


class ScienceScheduleRecordTests(unittest.TestCase):
    def test_common_schedule_matches_after_optimizer_semantics(self) -> None:
        self.assertAlmostEqual(common_lr_for_update(1), 1e-3)
        self.assertAlmostEqual(common_lr_after_update(0), 1e-3)
        self.assertAlmostEqual(common_lr_after_update(1500), (1e-3 + 1e-5) / 2)
        self.assertAlmostEqual(common_lr_after_update(3000), 1e-5)

    def test_b4_oracle_schedule(self) -> None:
        self.assertAlmostEqual(b4_causal_lr_for_update(1), 1e-3 / 500)
        self.assertAlmostEqual(b4_causal_lr_for_update(500), 1e-3)
        self.assertLess(b4_causal_lr_for_update(501), 1e-3)
        self.assertGreater(b4_causal_lr_for_update(501), 1e-5)
        self.assertAlmostEqual(b4_causal_lr_for_update(20000), 1e-5)

    def test_loss_forms(self) -> None:
        prediction = np.ones((2,2))
        truth = np.zeros((2,2))
        self.assertAlmostEqual(common_loss(prediction, truth), 1.25)
        endpoints = np.zeros((128,2))
        endpoint_truth = np.zeros_like(endpoints)
        endpoints[64:] = 1.0
        self.assertAlmostEqual(b7_dense_absolute_loss(endpoints, endpoint_truth), 0.5)
        relative = b7_relative_loss(endpoints, endpoint_truth, np.ones((1,2)), np.zeros((1,2)))
        self.assertEqual(relative, {"pair_mean_mse":1.0,"anchor_mean_mse":1.0,"total":2.0})

    def test_b6_ridge_and_rank_zero(self) -> None:
        x = np.asarray([[0.0],[1.0],[2.0],[3.0]])
        y = np.asarray([[0.0,0.0],[2.0,-1.0],[4.0,-2.0],[6.0,-3.0]])
        result = b6_ridge(x, y, np.asarray([[4.0]]))
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["prediction"].shape, (1,2))
        rank_zero = b6_ridge(np.ones((4,2)), np.asarray([[1,2],[3,4],[5,6],[7,8]],dtype=float), np.ones((1,2)))
        self.assertEqual(rank_zero["rank"], 0)
        np.testing.assert_allclose(rank_zero["prediction"], [[4.0,5.0]])

    def test_metrics_and_support(self) -> None:
        metrics = metric_summary(np.asarray([[1.0,3.0],[5.0,7.0]]), np.asarray([[0.0,1.0],[1.0,3.0]]))
        self.assertAlmostEqual(metrics["full_box_raw_mae_px"], 2.75)
        self.assertTrue(support_gate(0.25))
        self.assertFalse(support_gate(0.2500001))

    def test_attempt_paths_never_overwrite(self) -> None:
        root = Path(__file__).resolve().parent / "_work" / f"records_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            (root / "RUN_CATALOG.json").write_text(json.dumps({"schema_version":1,"attempts":[]}), encoding="utf-8")
            first = create_attempt("B1", "gn", 20260816, project_root=root)
            self.assertTrue(Path(first["run_path"]).is_dir())
            self.assertFalse(Path(first["review_path"]).exists())
            second = create_attempt("B1", "gn", 20260816, project_root=root)
            self.assertTrue(second["run_path"].endswith("attempt_002"))
            append_run_catalog({"cell_id":"B1.gn.s20260816","family":"B1","condition":"gn","seed":20260816,"run_path":first["run_path"],"review_path":first["review_path"],"status":"PREPARED"}, project_root=root)
            catalog = json.loads((root / "RUN_CATALOG.json").read_text(encoding="utf-8"))
            self.assertEqual(len(catalog["attempts"]), 1)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
