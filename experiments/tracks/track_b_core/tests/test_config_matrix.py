from __future__ import annotations

import unittest

from fsx.config import iter_cells, load_matrix, load_protocol, resolve_cell, resolved_scientific_config, validate_matrix


class MatrixTests(unittest.TestCase):
    def test_exact_matrix_counts_and_families(self) -> None:
        matrix = load_matrix()
        self.assertEqual(validate_matrix(matrix), {"logical_cells": 84, "physical_jobs": 82, "optimizer_jobs": 78})
        cells = list(iter_cells(matrix))
        self.assertEqual({family: sum(cell["family"] == family for cell in cells) for family in matrix["families"]}, {"B1":9,"B2":9,"B3":9,"B4":9,"B5":9,"B6":18,"B7":12,"B8":9})

    def test_b8_degree2_is_three_logical_records_one_physical_job(self) -> None:
        cells = [cell for cell in iter_cells() if cell["family"] == "B8" and cell["condition_id"] == "degree2"]
        self.assertEqual(len(cells), 3)
        self.assertEqual({cell["physical_job_id"] for cell in cells}, {"job/B8/degree2/shared"})
        self.assertTrue(all(not cell["optimizer_job"] for cell in cells))

    def test_b7_dense_arms_share_double_draw_stream(self) -> None:
        conditions = ["dense_absolute", "dense_relative_4_neighbor_plus_1_anchor", "dense_relative_4_neighbor_plus_4_corners"]
        for seed in (20260816, 20260817, 20260818):
            sampler_ids = {resolve_cell(f"B7.{condition}.s{seed}")["sampler_id"] for condition in conditions}
            self.assertEqual(sampler_ids, {f"S_B7_DOUBLE_DRAW_T3000_s{seed}"})

    def test_resolved_b4_and_b6_paths(self) -> None:
        b4 = resolved_scientific_config("B4.causal_sgd.s20260816")
        self.assertEqual(b4["training"]["steps"], 20000)
        self.assertEqual(b4["cell"]["optimizer"], "sgd")
        b6 = resolve_cell("B6.lp_ft.s20260817")
        self.assertEqual(b6["upstream_physical_job_id"], "job/B6/g64_upstream/s20260817")

    def test_canonical_query_has_no_renderer_default(self) -> None:
        protocol = load_protocol()
        query = next(item for item in protocol["prepared_inputs"]["query_specs"] if item["id"] == "Q_DENSE41_CANONICAL")
        self.assertIsNone(query["renderer"])
        self.assertIsNone(query["appearance_ids"])
        self.assertIsNone(query["target_kind"])
        self.assertEqual(resolve_cell("B5.line_fixed.s20260816")["index_id"], "IDX_CORNERS4_LINE_FIXED_A1")


if __name__ == "__main__":
    unittest.main()
