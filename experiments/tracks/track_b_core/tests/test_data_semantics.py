from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from fsx.config import load_protocol
from fsx.data import (
    appearance_parameters,
    b7_double_draw_stream,
    b7_endpoint_occurrences,
    canonical_b7_pair_graph,
    common_stream,
    coordinate_target_f64,
    frozen_bn_calibration_row_ids,
    make_index,
    make_query_arrays,
    positions,
    prepare_inputs,
    preprocess_rgb,
    render_query_with_index,
    render_u8,
    resolve_query_target,
    validate_prepared_inputs,
)


class DataSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()
        cls.index_specs = {item["id"]: item for item in cls.protocol["prepared_inputs"]["index_specs"]}
        cls.query_specs = {item["id"]: item for item in cls.protocol["prepared_inputs"]["query_specs"]}

    def test_coordinate_order(self) -> None:
        corners = positions("corners4")
        np.testing.assert_array_equal(corners, [[59.0,59.0],[59.0,164.0],[164.0,59.0],[164.0,164.0]])
        dense = positions("dense41")
        np.testing.assert_array_equal(dense[[0,1,40,41,-1]], [[59.0,59.0],[59.0,61.625],[59.0,164.0],[61.625,59.0],[164.0,164.0]])

    def test_indices_have_exact_rows_and_no_heldout_rows(self) -> None:
        expected = {
            "IDX_CORNERS4_BLOB_A1":4,"IDX_CORNERS4_BLOB_A8":32,"IDX_CORNERS4_BLOB_A64":256,
            "IDX_CORNERS4_BLOB_A16":64,"IDX_GRID4X4_BLOB_A4":64,"IDX_GRID8X8_BLOB_A1":64,
            "IDX_CORNERS4_BLOB_SIGMA6_A1":4,"IDX_CORNERS4_LINE_FIXED_A1":4,
            "IDX_CORNERS4_SCALENE_TRIANGLE_A1":4,"IDX_G64_BLOB_A1":64,"IDX_G9_BLOB_A8":72,
            "SHARD_G9_UNIQUE9":9,
        }
        for artifact_id, rows in expected.items():
            index = make_index(self.index_specs[artifact_id], self.protocol)
            self.assertEqual(index["row_count"], rows)
            self.assertNotIn("heldout_rows", index)
            self.assertTrue(all(row["split"] == "train" for row in index["rows"]))

    def test_b2_is_balanced_and_identity_sets_are_nested_and_disjoint(self) -> None:
        for condition, artifact_id, train_ids, heldout_ids in (
            ("A1", "IDX_CORNERS4_BLOB_A1", set(range(1)), {1,2,3,4}),
            ("A8", "IDX_CORNERS4_BLOB_A8", set(range(8)), {8,9,10,11}),
            ("A64", "IDX_CORNERS4_BLOB_A64", set(range(64)), {64,65,66,67}),
        ):
            index = make_index(self.index_specs[artifact_id], self.protocol)
            observed = {row["appearance_id"] for row in index["rows"]}
            self.assertEqual(observed, train_ids, condition)
            self.assertTrue(observed.isdisjoint(heldout_ids), condition)
            counts = {aid: sum(row["appearance_id"] == aid for row in index["rows"]) for aid in observed}
            self.assertEqual(set(counts.values()), {4})

    def test_appearance_generator_is_namespace_and_call_order_stable(self) -> None:
        first = appearance_parameters(7, "blob", 20260830, self.protocol)
        _ = [appearance_parameters(aid, "blob", 20260830, self.protocol) for aid in reversed(range(20))]
        second = appearance_parameters(7, "blob", 20260830, self.protocol)
        self.assertEqual(first, second)
        self.assertEqual({appearance_parameters(aid, "blob", 20260830, self.protocol)["sigma"] for aid in range(64)}, {4.0, 9.0})
        self.assertNotEqual(first, appearance_parameters(7, "blob", 20260831, self.protocol))
        self.assertNotEqual(first, appearance_parameters(7, "blob_sigma6", 20260830, self.protocol))

    def test_renderers_and_preprocess(self) -> None:
        point = np.asarray([100.0, 120.0])
        images = {name: render_u8(point, 0, name, 20260830, protocol=self.protocol) for name in ("blob", "blob_sigma6", "line_fixed", "scalene_triangle")}
        for image in images.values():
            self.assertEqual(image.shape, (1,224,224))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(int(image[0,120,100]), 255)
            rgb = preprocess_rgb(image)
            self.assertEqual(rgb.shape, (3,224,224))
            self.assertEqual(rgb.dtype, np.float32)
            np.testing.assert_array_equal(rgb[0], rgb[1])
            np.testing.assert_array_equal(rgb[1], rgb[2])
        self.assertFalse(np.array_equal(images["line_fixed"], images["scalene_triangle"]))

    def test_b5_query_is_rendered_by_resolved_index(self) -> None:
        query = make_query_arrays(self.query_specs["Q_DENSE41_CANONICAL"], self.protocol)
        self.assertNotIn("appearance_id", query)
        self.assertNotIn("truth_u", query)
        line = make_index(self.index_specs["IDX_CORNERS4_LINE_FIXED_A1"], self.protocol)
        triangle = make_index(self.index_specs["IDX_CORNERS4_SCALENE_TRIANGLE_A1"], self.protocol)
        one_point = query["xy_px"][:1]
        self.assertFalse(np.array_equal(render_query_with_index(line, one_point, protocol=self.protocol), render_query_with_index(triangle, one_point, protocol=self.protocol)))

    def test_shared_canonical_query_resolves_identity_or_b8_quadratic_from_cell(self) -> None:
        query = make_query_arrays(self.query_specs["Q_DENSE41_CANONICAL"], self.protocol)
        identity = resolve_query_target(query, "identity")
        quadratic = resolve_query_target(query, "quadratic_b8")
        self.assertEqual(identity["truth_u"].shape, (1681,2))
        self.assertEqual(quadratic["truth_u"].shape, (1681,2))
        self.assertFalse(np.array_equal(identity["truth_u"], quadratic["truth_u"]))
        polluted = dict(query)
        polluted["truth_u"] = identity["truth_u"]
        with self.assertRaisesRegex(ValueError, "must not carry"):
            resolve_query_target(polluted, "identity")

    def test_targets(self) -> None:
        identity = coordinate_target_f64(np.asarray([[223.0,0.0],[0.0,223.0]]), "identity")
        np.testing.assert_array_equal(identity, [[1.0,0.0],[0.0,1.0]])
        quadratic = coordinate_target_f64(np.asarray([[223.0,0.0],[0.0,223.0]]), "quadratic_b8")
        np.testing.assert_allclose(quadratic, [[0.76,-0.10],[0.12,0.77]], rtol=0, atol=1e-12)

    def test_b7_graph_stream_and_endpoint_order(self) -> None:
        graph = canonical_b7_pair_graph()
        self.assertEqual(graph.shape, (112,2))
        np.testing.assert_array_equal(graph[:4], [[0,8],[0,1],[1,9],[1,2]])
        stream = b7_double_draw_stream(2, 20260816)
        manual = np.random.default_rng(20260816 + 17)
        for step in range(2):
            np.testing.assert_array_equal(stream["image_ids"][step], manual.integers(0,64,64,dtype=np.int64))
            np.testing.assert_array_equal(stream["pair_ids"][step], manual.integers(0,112,64,dtype=np.int64))
        endpoints = b7_endpoint_occurrences(stream["pair_ids"][0], graph)
        selected = graph[stream["pair_ids"][0]]
        np.testing.assert_array_equal(endpoints[:64], selected[:,0])
        np.testing.assert_array_equal(endpoints[64:], selected[:,1])

    def test_all_common_streams_keep_width_64(self) -> None:
        for population in (4,32,64,72,256):
            stream = common_stream(population, 2, 20260816)
            self.assertEqual(stream.shape, (2,64))
            self.assertTrue(np.all((stream >= 0) & (stream < population)))
        self.assertEqual(frozen_bn_calibration_row_ids().tolist(), [0,1,2,3] * 64)

    def test_full_prepare_and_semantic_validation(self) -> None:
        temporary = Path(__file__).resolve().parent / "_work" / f"prepare_{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            result = prepare_inputs(temporary / "assets")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["counts"], {"indices_or_shards":12,"queries":9,"fixed_streams":21})
            self.assertEqual(validate_prepared_inputs(temporary / "assets")["status"], "PASS")
        finally:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
