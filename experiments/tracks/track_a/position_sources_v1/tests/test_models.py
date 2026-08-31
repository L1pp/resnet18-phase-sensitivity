import unittest

import torch
from torch import nn

from position_sources.certificate import valid_core_certificate
from position_sources.models import build_model, build_reference_model


class ModelTests(unittest.TestCase):
    def test_padding_variants_have_two_coordinate_outputs(self):
        torch.set_num_threads(2)
        for variant in ("zero_s32", "reflection_s32", "circular_s32"):
            model = build_model(variant, input_size=64)
            with torch.no_grad():
                result = model(torch.zeros(1, 3, 64, 64))
            self.assertEqual(tuple(result.shape), (1, 2))
            self.assertTrue(torch.isfinite(result).all().item())
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in build_model("zero_s32", input_size=64).modules()))

    def test_true_valid_has_symmetric_shortcuts(self):
        model = build_model("true_valid_s32", input_size=512)
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 512, 512))
        self.assertEqual(tuple(result.shape), (1, 2))

    def test_valid_core_certificate_is_computed(self):
        certificate = valid_core_certificate(input_size=1024)
        self.assertEqual(certificate["layer4_feature_shape"], [32, 32])
        self.assertEqual(certificate["receptive_field_size"], 435.0)
        self.assertEqual(certificate["core_shape"], [19, 19])
        self.assertTrue(certificate["checks"]["passed"])
        model = build_model("valid_core_s32", input_size=1024)
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 1024, 1024))
        self.assertEqual(tuple(result.shape), (1, 2))

    def test_torus_outputs_periodic_sin_cos_channels(self):
        model = build_model("torus_s32", input_size=64)
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(result.shape), (1, 4))

    def test_torus_all_conv2d_modules_are_explicitly_circular(self):
        model = build_model("torus_s32", input_size=64)
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertTrue(convolutions)
        self.assertTrue(all(module.padding_mode == "circular" for module in convolutions))

    def test_zero_padding_batchnorm_reference_is_explicitly_non_primary(self):
        model = build_reference_model(input_size=64)
        self.assertEqual(model.variant, "zero_s32_bn_reference")
        self.assertTrue(model.reference_only)
        self.assertEqual(model.normalization, "batchnorm2d")
        self.assertTrue(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))
        with torch.no_grad():
            result = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(result.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
