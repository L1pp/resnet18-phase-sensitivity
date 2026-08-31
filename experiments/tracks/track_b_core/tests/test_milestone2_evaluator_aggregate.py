from __future__ import annotations

import json
import unittest
from pathlib import Path

from fsx.aggregate import aggregate_records
from fsx.config import PACKAGE_ROOT, iter_cells, resolve_cell


class EvaluatorAggregateTests(unittest.TestCase):
    def test_evaluator_source_does_not_import_trainer(self) -> None:
        source = (PACKAGE_ROOT / "fsx" / "evaluator.py").read_text(encoding="utf-8")
        self.assertNotIn("from .trainer", source)
        self.assertNotIn("import fsx.trainer", source)

    @staticmethod
    def _review_records():
        records = []
        for index, cell in enumerate(iter_cells()):
            endpoint = {
                "first_gate": {"full_box_raw_mae_px": float(index + 0.1), "full_box_u_mae": 0.0, "support_eligible": True},
                "step_3000": {"full_box_raw_mae_px": float(index + 0.2), "full_box_u_mae": 0.0, "support_eligible": True},
                "step_20000": {"full_box_raw_mae_px": float(index + 0.3), "full_box_u_mae": 0.0, "support_eligible": True},
                "first_gate_selection_step": 700,
            }
            records.append(
                {
                    "review_status": "PASS",
                    "cell_id": cell["cell_id"],
                    "run_path": f"runs/{cell['cell_id']}",
                    "review_path": f"reviews/{cell['cell_id']}",
                    "support_eligible": True,
                    "primary_metrics": {"full_box_raw_mae_px": float(index), "full_box_u_mae": float(index) / 223.0},
                    "endpoint_metrics": endpoint if cell["training_path"] == "b4_causal" else None,
                }
            )
        return records

    def test_aggregate_requires_exact_passed_84_cell_closure_and_preserves_seeds(self) -> None:
        records = self._review_records()
        result = aggregate_records(records)
        self.assertTrue(result["closure"])
        self.assertEqual(result["aggregate_status"], "PASS_COMPLETE_84_CELL_CLOSURE")
        self.assertEqual(len(result["per_cell"]), 84)
        self.assertEqual(len(result["contrasts"]["B1"]["conditions"]["bn"]["per_seed"]), 3)
        self.assertEqual(len(result["contrasts"]["B4"]["endpoints"]["matched_gate_when_available"]["per_seed"]), 3)
        self.assertNotIn("step_3000", result["contrasts"]["B4"]["endpoints"])

        missing = aggregate_records(records[:-1])
        self.assertFalse(missing["closure"])
        self.assertEqual(len(missing["inventory"]["missing"]), 1)
        self.assertIsNone(missing["contrasts"])

        failed_records = self._review_records()
        failed_records[0]["review_status"] = "FAIL"
        failed = aggregate_records(failed_records)
        self.assertFalse(failed["closure"])
        self.assertEqual(failed["inventory"]["non_pass_reviews"], [failed_records[0]["cell_id"]])

    def test_b8_cnn_index_metadata_is_quadratic_not_identity(self) -> None:
        cell = resolve_cell("B8.cnn.s20260816")
        index = json.loads((PACKAGE_ROOT / "assets" / "indices" / "IDX_G9_BLOB_A8.json").read_text(encoding="utf-8"))
        self.assertEqual(cell["target_kind"], "quadratic_b8")
        self.assertEqual({row["target_kind"] for row in index["rows"]}, {"quadratic_b8"})


if __name__ == "__main__":
    unittest.main()
