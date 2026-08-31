import importlib.util
import unittest

import torch

from closeout_sprint.bn_control import (
    BNControlError,
    buffer_hash,
    run_bn_only_control,
)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is unavailable")
class BNControlTests(unittest.TestCase):
    def _model(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(2)
                self.dropout = torch.nn.Dropout(p=0.5)
                self.linear = torch.nn.Conv2d(2, 2, kernel_size=1)

            def forward(self, x):
                return self.linear(self.dropout(self.bn(x)))

        return Net()

    def test_parameters_stay_fixed_and_evaluator_does_not_update_stats(self):
        model = self._model()
        model.train()
        batches = [torch.randn(4, 2, 4, 4) + float(index) for index in range(3)]
        evaluator_hashes = []

        def evaluator(current, exposure):
            self.assertFalse(current.training)
            before = buffer_hash(current)
            with torch.no_grad():
                current(torch.ones(4, 2, 4, 4) * 100.0)
            after = buffer_hash(current)
            evaluator_hashes.append((exposure, before, after))
            self.assertEqual(before, after)
            return {"exposure": exposure}

        report = run_bn_only_control(
            model,
            batches,
            exposure_checkpoints=(0, 1, 3),
            evaluate=evaluator,
        )
        self.assertTrue(report.parameters_unchanged)
        self.assertTrue(report.only_bn_buffers_changed)
        self.assertEqual([record.exposure for record in report.records], [0, 1, 3])
        self.assertEqual(len(evaluator_hashes), 3)
        self.assertTrue(model.training)
        self.assertTrue(any(record.bn_changed_buffer_names for record in report.records[1:]))

    def test_non_integer_checkpoint_is_rejected_before_conversion(self):
        with self.assertRaises(ValueError):
            run_bn_only_control(self._model(), [torch.randn(4, 2, 4, 4)], exposure_checkpoints=(1.5,))

    def test_non_bn_buffer_mutation_is_strict_failure(self):
        class Mutating(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)
                self.register_buffer("counter", torch.zeros(1))

            def forward(self, x):
                self.counter.add_(1)
                return self.bn(x)

        with self.assertRaises(BNControlError):
            run_bn_only_control(
                Mutating(),
                [torch.randn(4, 2)],
                exposure_checkpoints=(1,),
            )


if __name__ == "__main__":
    unittest.main()
