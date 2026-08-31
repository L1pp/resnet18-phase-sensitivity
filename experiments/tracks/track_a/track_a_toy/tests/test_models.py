from __future__ import annotations

import numpy as np
import torch
import unittest

from track_a_toy.config import MODEL_SPECS
from track_a_toy.models import (
    build_models_with_shared_state,
    model_shape_report,
    state_dict_hash,
)
from track_a_toy.renderer import render_one


class ModelTests(unittest.TestCase):
    def test_shared_initialization_and_head_shapes(self) -> None:
        models, shared_state, digest = build_models_with_shared_state(device="cpu")
        self.assertEqual(set(models), set(MODEL_SPECS))
        for name, model in models.items():
            self.assertEqual(state_dict_hash(model.state_dict()), digest, name)
            for key, value in shared_state.items():
                self.assertTrue(torch.equal(model.state_dict()[key].cpu(), value.cpu()), (name, key))
            self.assertEqual(tuple(model.fc.weight.shape), (2, 512))
            self.assertEqual(tuple(model.fc.bias.shape), (2,))
            self.assertFalse(model.training)
            self.assertEqual(model.total_stride, int(MODEL_SPECS[name]["total_stride"]))
            self.assertEqual(model.padding_mode, str(MODEL_SPECS[name]["padding_mode"]))

    def test_spatial_shapes_match_stride_conditions(self) -> None:
        models, _state, _digest = build_models_with_shared_state(device="cpu")
        for name in ("a0_standard", "a1_zero_s1", "a2_torus_s32", "a3_torus_s1"):
            report = model_shape_report(models[name])
            self.assertEqual(report["gap_shape"], [1, 512], name)
            self.assertEqual(report["fc_shape"], [2, 512], name)
            spatial = report["stage_shapes"]["layer4"]
            expected = 2 if int(MODEL_SPECS[name]["total_stride"]) == 32 else 64
            self.assertEqual(spatial[-2:], [expected, expected], (name, spatial))

    def test_strict_torus_translation_is_invariant_at_unit_stride(self) -> None:
        models, _state, _digest = build_models_with_shared_state(device="cpu")
        model = models["a3_torus_s1"]
        left = torch.from_numpy(render_one((3.0, 5.0), torus=True)).unsqueeze(0).float().div_(255.0)
        right = torch.from_numpy(render_one((4.0, 7.0), torus=True)).unsqueeze(0).float().div_(255.0)
        rolled = torch.roll(left, shifts=(2, 1), dims=(2, 3))
        self.assertTrue(torch.equal(rolled, right))
        with torch.no_grad():
            left_features = model.forward_features(left)
            right_features = model.forward_features(right)
        self.assertTrue(torch.allclose(left_features, right_features, atol=1e-5, rtol=1e-5))

    def test_torus_stride32_respects_one_full_stride_period(self) -> None:
        models, _state, _digest = build_models_with_shared_state(device="cpu")
        model = models["a2_torus_s32"]
        left = torch.from_numpy(render_one((7.0, 9.0), torus=True)).unsqueeze(0).float().div_(255.0)
        right = torch.from_numpy(render_one((39.0, 9.0), torus=True)).unsqueeze(0).float().div_(255.0)
        with torch.no_grad():
            left_features = model.forward_features(left)
            right_features = model.forward_features(right)
        self.assertTrue(torch.allclose(left_features, right_features, atol=1e-5, rtol=1e-5))
