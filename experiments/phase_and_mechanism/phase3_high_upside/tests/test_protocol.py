from __future__ import annotations

import unittest

from phase3_high_upside.protocol import (
    ALL_DELTAS,
    EVAL_SHAPE_IDS,
    GEOMETRY_FP,
    build_protocol,
    delta_key,
    origin_split,
    parse_delta_key,
    shape_split,
)


class ProtocolTests(unittest.TestCase):
    def test_shape_split_is_partition_of_eval_ids(self):
        split = shape_split()
        merged = split["fit"] + split["val"] + split["test"]
        self.assertEqual(len(split["fit"]), 16)
        self.assertEqual(len(split["val"]), 8)
        self.assertEqual(len(split["test"]), 8)
        self.assertEqual(sorted(merged), sorted(EVAL_SHAPE_IDS))
        self.assertEqual(len(set(merged)), 32)

    def test_origin_split_covers_all(self):
        split = origin_split(121)
        merged = split["fit"] + split["val"] + split["test"]
        self.assertEqual(sorted(merged), list(range(121)))
        self.assertGreater(len(split["fit"]), len(split["test"]))

    def test_delta_key_roundtrip(self):
        for delta in ALL_DELTAS:
            self.assertEqual(parse_delta_key(delta_key(delta)), (float(delta[0]), float(delta[1])))

    def test_protocol_hash_stable(self):
        a = build_protocol()
        b = build_protocol()
        self.assertEqual(a["protocol_hash"], b["protocol_hash"])
        self.assertEqual(a["geometry_fingerprint"], GEOMETRY_FP)
        self.assertFalse(a["resume_default"])
