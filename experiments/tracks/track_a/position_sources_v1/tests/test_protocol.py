import unittest

from position_sources.protocol import load_protocol, protocol_hash


class ProtocolTests(unittest.TestCase):
    def test_frozen_values(self):
        cfg = load_protocol()
        self.assertEqual(cfg["renderer"]["primary_image_size"], 1024)
        self.assertEqual(cfg["renderer"]["primary_position_box_px"]["low"], 448.0)
        self.assertEqual(cfg["renderer"]["primary_position_box_px"]["high"], 575.0)
        self.assertEqual(cfg["normalization"]["groups"], 32)
        self.assertEqual(cfg["training"]["steps"], 4000)
        self.assertEqual(cfg["training"]["batch_size"], 16)
        self.assertEqual(cfg["training"]["loss"]["l1_weight"], 0.25)
        self.assertEqual(cfg["renderer"]["torus_period_px"], 256)
        self.assertEqual(cfg["reference_only"]["variant"], "zero_s32_bn_reference")
        self.assertFalse(cfg["reference_only"]["primary"])
        self.assertEqual(cfg["evaluation"]["gap_feature"]["field"], "pre_fc_gap")
        self.assertEqual(cfg["evaluation"]["gap_feature"]["feature_dim"], 512)
        self.assertEqual(len(protocol_hash(payload=cfg)), 64)

    def test_variant_families_are_frozen(self):
        cfg = load_protocol()
        self.assertEqual(
            cfg["families"]["padding"]["variants"],
            ["zero_s32", "reflection_s32", "circular_s32", "valid_core_s32", "true_valid_s32"],
        )
        self.assertEqual(cfg["families"]["stride"]["variants"], ["valid_core_s32", "valid_core_aa32", "valid_core_s1"])
        self.assertEqual(cfg["families"]["torus"]["variants"], ["torus_s32", "torus_s1"])


if __name__ == "__main__":
    unittest.main()
