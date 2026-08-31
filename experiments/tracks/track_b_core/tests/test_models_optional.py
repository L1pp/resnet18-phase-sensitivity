from __future__ import annotations

import unittest

from fsx import models


@unittest.skipIf(models.TORCH_IMPORT_ERROR is not None, f"Torch/Torchvision unavailable: {models.TORCH_IMPORT_ERROR}")
class ModelFoundationTests(unittest.TestCase):
    def test_explicit_xy_mlp_is_seed_stable(self) -> None:
        first = models.build_explicit_xy_mlp(20260816)
        first_state = models.clone_state(first)
        second = models.build_explicit_xy_mlp(20260816)
        second_state = models.clone_state(second)
        self.assertEqual(first_state.keys(), second_state.keys())
        for key in first_state:
            self.assertTrue(models.torch.equal(first_state[key], second_state[key]))

    def test_trainable_regimes(self) -> None:
        model = models.build_resnet18("gn", 20260816)
        head = models.set_trainable_regime(model, "head_only")
        self.assertTrue(head)
        self.assertTrue(all(name.startswith("fc.") for name in head))
        newly = models.unfreeze_lp_ft(model)
        self.assertTrue(newly)
        self.assertEqual(len([name for name, parameter in model.named_parameters() if parameter.requires_grad]), len(list(model.named_parameters())))

    def test_same_seed_resnet_initialization_is_exact(self) -> None:
        first = models.build_resnet18("gn", 20260816)
        second = models.build_resnet18("gn", 20260816)
        first_state = models.clone_state(first)
        second_state = models.clone_state(second)
        self.assertEqual(first_state.keys(), second_state.keys())
        for key in first_state:
            self.assertTrue(models.torch.equal(first_state[key], second_state[key]), key)

    def test_frozen_bn_one_batch_calibration_then_fixed_stats(self) -> None:
        class Tiny(models.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = models.FrozenBatchNorm2d(2)

            def forward(self, value):
                return self.bn(value)

        model = Tiny()
        images = models.torch.arange(256 * 2 * 2 * 2, dtype=models.torch.float32).reshape(256,2,2,2) / 1000.0
        report = models.calibrate_and_freeze_bn(model, images)
        self.assertEqual(report, {"calibration_rows":256,"forward_calls":1,"frozen_bn_layers":1})
        self.assertFalse(model.bn.training)
        self.assertTrue(model.bn.weight.requires_grad)
        self.assertTrue(model.bn.bias.requires_grad)
        with self.assertRaises(ValueError):
            models.calibrate_and_freeze_bn(model, images)


if __name__ == "__main__":
    unittest.main()
