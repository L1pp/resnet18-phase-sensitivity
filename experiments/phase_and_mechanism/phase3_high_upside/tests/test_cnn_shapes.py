from __future__ import annotations

import unittest

import torch

from phase3_high_upside.controlled_cnn import CHANNELS, ControlledCNN, all_variant_specs, build_controlled
from phase3_high_upside.equivariance import measure_defects, shift_nchw


class ControlledCnnTests(unittest.TestCase):
    def test_param_count_and_output_dim(self):
        specs = all_variant_specs()
        for name, spec in specs.items():
            n = int(spec["n_params"])
            self.assertGreaterEqual(n, 800_000, msg=name)
            self.assertLessEqual(n, 3_000_000, msg=name)
            model = build_controlled(name)
            x = torch.zeros(2, 3, 96, 96)
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 2), msg=name)
            feat = model.forward_features(x)
            self.assertEqual(feat.shape[1], CHANNELS[-1], msg=name)
            if spec["use_stride"]:
                self.assertLess(feat.shape[-1], 96, msg=name)
            else:
                self.assertEqual(feat.shape[-1], 96, msg=name)

    def test_only_symmetry_knobs_change(self):
        s0 = ControlledCNN("S0")
        s2 = ControlledCNN("S2")
        self.assertTrue(s0.circular)
        self.assertFalse(s2.circular)
        self.assertFalse(s0.use_stride)
        self.assertFalse(s2.use_stride)
        self.assertEqual(s0.count_params(), s2.count_params())

    def test_s0_circular_stride1_almost_equivariant_at_init(self):
        torch.manual_seed(0)
        model = build_controlled("S0")
        model.eval()
        images = (torch.rand(4, 96, 96).numpy() * 255).astype("uint8")
        report = measure_defects(model, images, shifts=(2,), batch_size=2)
        self.assertLess(report["mean_d_eq"], 1e-6)
        self.assertLess(report["mean_d_gap"], 1e-6)

    def test_shift_zeros_does_not_wrap(self):
        x = torch.zeros(1, 1, 4, 4)
        x[..., 0, 0] = 1.0
        y = shift_nchw(x, 1, 0, "zeros")
        self.assertEqual(float(y[..., 0, 1]), 1.0)
        self.assertEqual(float(y[..., 0, 0]), 0.0)
        c = shift_nchw(x, -1, 0, "circular")
        self.assertEqual(float(c[..., 0, 3]), 1.0)
