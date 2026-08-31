from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from fsx import models
from fsx.checkpoints import optimizer_parameter_names, restore_checkpoint, save_checkpoint
from fsx.schedules import b4_causal_lr_for_update, common_lr_for_update


@unittest.skipIf(models.TORCH_IMPORT_ERROR is not None, f"Torch unavailable: {models.TORCH_IMPORT_ERROR}")
class CheckpointContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / "_work" / f"resume_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)
        models.torch.set_num_threads(1)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @staticmethod
    def _linear() -> object:
        models.torch.manual_seed(91)
        return models.nn.Linear(2, 2)

    @staticmethod
    def _identity(cell_id: str, training_path: str) -> dict[str, object]:
        return {
            "cell_id": cell_id,
            "run_seed": 91,
            "training_path": training_path,
            "sampler_id": f"toy_{training_path}",
            "index_id": "toy_index",
            "target_kind": "identity",
            "anchors": [],
        }

    @staticmethod
    def _update(model, optimizer, step: int, schedule) -> None:
        x = models.torch.tensor([[0.1, 0.3], [0.7, -0.2]], dtype=models.torch.float32)
        y = models.torch.tensor([[0.4, -0.1], [0.2, 0.8]], dtype=models.torch.float32)
        lr = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = models.torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()

    def _assert_model_exact(self, left, right) -> None:
        for name, value in left.state_dict().items():
            self.assertTrue(models.torch.equal(value, right.state_dict()[name]), name)

    def test_common_continuous_equals_checkpoint_resume(self) -> None:
        continuous = self._linear()
        continuous_optimizer = models.torch.optim.AdamW(continuous.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(1, 5):
            self._update(continuous, continuous_optimizer, step, common_lr_for_update)

        split = self._linear()
        split_optimizer = models.torch.optim.AdamW(split.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(1, 3):
            self._update(split, split_optimizer, step, common_lr_for_update)
        checkpoint = save_checkpoint(
            self.root / "common.pt",
            split,
            split_optimizer,
            completed_step=2,
            stream_cursor=2,
            schedule_kind="common_cosine_after_optimizer",
            resume_identity=self._identity("toy.common", "common"),
        )
        resumed = self._linear()
        resumed_optimizer = models.torch.optim.AdamW(resumed.parameters(), lr=1e-3, weight_decay=1e-4)
        names = optimizer_parameter_names(resumed, resumed_optimizer)
        payload = restore_checkpoint(
            checkpoint,
            resumed,
            resumed_optimizer,
            expected_schedule_kind="common_cosine_after_optimizer",
            expected_resume_identity=self._identity("toy.common", "common"),
            expected_parameter_names=names,
        )
        self.assertEqual((payload["completed_step"], payload["next_step"], payload["stream_cursor"]), (2, 3, 2))
        for step in range(3, 5):
            self._update(resumed, resumed_optimizer, step, common_lr_for_update)
        self._assert_model_exact(continuous, resumed)

    def test_b4_continuous_equals_checkpoint_resume_with_exact_schedule(self) -> None:
        continuous = self._linear()
        continuous_optimizer = models.torch.optim.SGD(continuous.parameters(), lr=1e-3, momentum=0.0, weight_decay=0.0)
        for step in range(498, 503):
            self._update(continuous, continuous_optimizer, step, b4_causal_lr_for_update)

        split = self._linear()
        split_optimizer = models.torch.optim.SGD(split.parameters(), lr=1e-3, momentum=0.0, weight_decay=0.0)
        for step in range(498, 501):
            self._update(split, split_optimizer, step, b4_causal_lr_for_update)
        checkpoint = save_checkpoint(
            self.root / "b4.pt",
            split,
            split_optimizer,
            completed_step=500,
            stream_cursor=500,
            schedule_kind="b4_warmup_then_cosine",
            resume_identity=self._identity("toy.b4", "b4_causal"),
        )
        resumed = self._linear()
        resumed_optimizer = models.torch.optim.SGD(resumed.parameters(), lr=1e-3, momentum=0.0, weight_decay=0.0)
        restore_checkpoint(
            checkpoint,
            resumed,
            resumed_optimizer,
            expected_schedule_kind="b4_warmup_then_cosine",
            expected_resume_identity=self._identity("toy.b4", "b4_causal"),
            expected_parameter_names=optimizer_parameter_names(resumed, resumed_optimizer),
        )
        for step in range(501, 503):
            self._update(resumed, resumed_optimizer, step, b4_causal_lr_for_update)
        self._assert_model_exact(continuous, resumed)

    def test_lpft_500_501_resume_preserves_fc_state_and_zero_initializes_new_state(self) -> None:
        class TinyLPFT(models.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = models.nn.Linear(2, 3)
                self.fc = models.nn.Linear(3, 2)

            def forward(self, value):
                return self.fc(models.torch.tanh(self.body(value)))

        def build():
            models.torch.manual_seed(123)
            model = TinyLPFT()
            for parameter in model.body.parameters():
                parameter.requires_grad_(False)
            return model

        x = models.torch.tensor([[0.2, 0.4], [-0.1, 0.7]], dtype=models.torch.float32)
        y = models.torch.tensor([[0.5, 0.1], [0.3, 0.9]], dtype=models.torch.float32)

        def lp_update(model, optimizer, protocol_step):
            for group in optimizer.param_groups:
                group["lr"] = common_lr_for_update(protocol_step)
            optimizer.zero_grad(set_to_none=True)
            loss = models.torch.mean((model(x) - y) ** 2)
            loss.backward()
            optimizer.step()

        continuous = build()
        continuous_optimizer = models.torch.optim.AdamW(continuous.fc.parameters(), lr=1e-3, weight_decay=1e-4)
        lp_update(continuous, continuous_optimizer, 500)
        before_unfreeze = {name: value.clone() for name, value in continuous.state_dict().items()}
        for parameter in continuous.body.parameters():
            parameter.requires_grad_(True)
        new_parameters = list(continuous.body.parameters())
        continuous_optimizer.add_param_group({"params": new_parameters, "lr": common_lr_for_update(501)})
        self.assertFalse(any(parameter in continuous_optimizer.state for parameter in new_parameters))
        self.assertTrue(all(models.torch.equal(before_unfreeze[name], value) for name, value in continuous.state_dict().items()))
        lp_update(continuous, continuous_optimizer, 501)

        split = build()
        split_optimizer = models.torch.optim.AdamW(split.fc.parameters(), lr=1e-3, weight_decay=1e-4)
        lp_update(split, split_optimizer, 500)
        checkpoint = save_checkpoint(
            self.root / "lp_end.pt",
            split,
            split_optimizer,
            completed_step=500,
            stream_cursor=500,
            schedule_kind="common_cosine_after_optimizer",
            resume_identity=self._identity("toy.lpft", "b6_lp_ft"),
            schedule_position=500,
            stage_state={"stage": "lp_only", "next_protocol_step": 501},
        )
        resumed = build()
        resumed_optimizer = models.torch.optim.AdamW(resumed.fc.parameters(), lr=1e-3, weight_decay=1e-4)
        restore_checkpoint(
            checkpoint,
            resumed,
            resumed_optimizer,
            expected_schedule_kind="common_cosine_after_optimizer",
            expected_resume_identity=self._identity("toy.lpft", "b6_lp_ft"),
            expected_parameter_names=optimizer_parameter_names(resumed, resumed_optimizer),
        )
        for parameter in resumed.body.parameters():
            parameter.requires_grad_(True)
        resumed_new = list(resumed.body.parameters())
        resumed_optimizer.add_param_group({"params": resumed_new, "lr": common_lr_for_update(501)})
        self.assertFalse(any(parameter in resumed_optimizer.state for parameter in resumed_new))
        lp_update(resumed, resumed_optimizer, 501)
        self._assert_model_exact(continuous, resumed)


if __name__ == "__main__":
    unittest.main()
