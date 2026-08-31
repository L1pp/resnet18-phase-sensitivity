import unittest
import copy
import json
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from unittest.mock import patch

import position_sources.experiment as experiment
from position_sources.experiment import (
    _microbatch_slices,
    _microbatch_update,
    _restore_rng_state,
    _resume_identity_mismatches,
    _runtime_identity,
    decode_predictions,
    target_values,
)
from position_sources.protocol import DEFAULT_PROTOCOL_PATH


class ExperimentTests(unittest.TestCase):
    def test_torus_target_round_trip(self):
        points = np.asarray([[0.0, 1.0], [127.0, 255.0]], dtype=np.float64)
        targets = target_values(points, family="torus", image_size=256)
        decoded = decode_predictions(targets, family="torus", image_size=256)
        self.assertTrue(np.allclose(decoded, points, atol=1e-5))

    def test_non_torus_target_round_trip(self):
        points = np.asarray([[448.0, 575.0], [512.0, 512.0]], dtype=np.float64)
        targets = target_values(points, family="padding", image_size=1024)
        decoded = decode_predictions(targets, family="padding", image_size=1024)
        self.assertTrue(np.allclose(decoded, points, atol=1e-5))

    def test_microbatch_accumulates_one_optimizer_step_per_effective_batch(self):
        value, slices = _microbatch_slices(16, 4)
        self.assertEqual(value, 4)
        self.assertEqual(slices, ((0, 4), (4, 8), (8, 12), (12, 16)))
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000, eta_min=1e-5)
        batch_x = torch.ones(16, 2)
        target = torch.zeros(16, 2)
        with patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step, patch.object(
            scheduler, "step", wraps=scheduler.step
        ) as scheduler_step:
            loss = _microbatch_update(
                model,
                optimizer,
                scheduler,
                batch_x,
                target,
                loss_cfg={"mse_weight": 1.0, "l1_weight": 0.25},
                microbatch=value,
            )
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(optimizer_step.call_count, 1)
        self.assertEqual(scheduler_step.call_count, 1)

    def test_microbatch_rejects_non_divisor(self):
        with self.assertRaises(ValueError):
            _microbatch_slices(16, 3)
        with self.assertRaises(ValueError):
            _microbatch_slices(16, 0)

    def test_resume_identity_is_field_by_field_and_rejects_cross_arm(self):
        expected = {
            "protocol_hash": "p",
            "cache_manifest_sha256": "c",
            "family": "padding",
            "primitive": "blob",
            "variant": "zero_s32",
            "seed": 20260823,
            "comparison_block_id": "A1-padding",
            "run_id": "A1-padding-blob-full",
            "effective_batch_size": 16,
            "microbatch": 4,
            "model_stack": "position_sources_resnet18_gap_linear",
            "normalization": "groupnorm",
            "reference_only": False,
        }
        payload = {"run_identity": dict(expected), **expected}
        self.assertEqual(_resume_identity_mismatches(expected, payload), [])
        for field in ("protocol_hash", "cache_manifest_sha256", "family", "primitive", "variant", "seed", "comparison_block_id", "run_id", "microbatch"):
            changed = copy.deepcopy(payload)
            changed["run_identity"][field] = "other" if isinstance(expected[field], str) else -1
            self.assertTrue(_resume_identity_mismatches(expected, changed), field)

    def test_resume_rejects_backend_runtime_and_rng_crossovers_without_gpu(self):
        cpu_identity = _runtime_identity("cpu")
        gpu_identity = _runtime_identity("cuda")
        self.assertEqual(cpu_identity["device_backend"], "cpu")
        self.assertIn(gpu_identity["device_backend"], {"cuda", "rocm"})
        mismatches = _resume_identity_mismatches(
            cpu_identity,
            {"run_identity": gpu_identity, **gpu_identity},
        )
        self.assertTrue(any(item.startswith("device_backend:") for item in mismatches))
        changed_runtime = dict(cpu_identity, torch_version="other-torch")
        self.assertTrue(_resume_identity_mismatches(cpu_identity, changed_runtime))

        sampler = torch.Generator().manual_seed(17)
        cpu_rng = experiment._capture_rng_state(sampler, device="cpu")
        with self.assertRaisesRegex(ValueError, "torch_cuda"):
            _restore_rng_state(cpu_rng, device="cuda")
        gpu_rng = dict(cpu_rng, torch_cuda=[torch.zeros(1, dtype=torch.uint8)])
        with self.assertRaisesRegex(ValueError, "CPU resume refuses"):
            _restore_rng_state(gpu_rng, device="cpu")

    def test_interrupted_resume_matches_one_shot_toy_training(self):
        torch.manual_seed(71)
        base = nn.Linear(3, 2)
        initial_state = copy.deepcopy(base.state_dict())
        x = torch.arange(64, dtype=torch.float32).reshape(16, 4)[:, :3] / 10.0
        target = torch.zeros(16, 2)
        loss_cfg = {"mse_weight": 1.0, "l1_weight": 0.25}

        def make():
            model = nn.Linear(3, 2)
            model.load_state_dict(copy.deepcopy(initial_state))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-5)
            return model, optimizer, scheduler

        def one_step(model, optimizer, scheduler, generator):
            indices = torch.randint(0, 16, (16,), generator=generator)
            _microbatch_update(
                model,
                optimizer,
                scheduler,
                x[indices],
                target[indices],
                loss_cfg=loss_cfg,
                microbatch=4,
            )

        one_shot, opt_one, sch_one = make()
        gen_one = torch.Generator().manual_seed(991)
        for _ in range(4):
            one_step(one_shot, opt_one, sch_one, gen_one)

        interrupted, opt_part, sch_part = make()
        gen_part = torch.Generator().manual_seed(991)
        for _ in range(2):
            one_step(interrupted, opt_part, sch_part, gen_part)
        saved_model = copy.deepcopy(interrupted.state_dict())
        saved_optimizer = copy.deepcopy(opt_part.state_dict())
        saved_scheduler = copy.deepcopy(sch_part.state_dict())
        saved_sampler = gen_part.get_state()
        resumed, opt_resume, sch_resume = make()
        resumed.load_state_dict(saved_model)
        opt_resume.load_state_dict(saved_optimizer)
        sch_resume.load_state_dict(saved_scheduler)
        gen_resume = torch.Generator()
        gen_resume.set_state(saved_sampler)
        for _ in range(2):
            one_step(resumed, opt_resume, sch_resume, gen_resume)
        for left, right in zip(one_shot.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(sch_one.state_dict(), sch_resume.state_dict())

    def test_periodic_checkpoint_resume_short_smoke(self):
        class TinyModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.layer = nn.Linear(8 * 8 * 3, 2)

            def forward(self, value):
                return self.layer(value.reshape(value.shape[0], -1))

        cfg = {
            "seed_records": [{"seed": 1, "data_seed": 1, "init_seed": 11, "sampler_seed": 1}],
            "families": {"padding": {"input_size": 8}},
            "normalization": {"groups": 1},
            "training": {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "scheduler": {"eta_min": 1e-5},
                "loss": {"mse_weight": 1.0, "l1_weight": 0.25},
            },
            "renderer": {"sigma_px": 1.0},
        }
        points = np.asarray([[float(index % 8), float(index // 8)] for index in range(32)], dtype=np.float64)
        images = np.zeros((32, 8, 8), dtype=np.uint8)
        targets = torch.from_numpy(target_values(points, family="padding", image_size=8))
        original_update = experiment._microbatch_update

        def setup(root):
            root.mkdir(parents=True, exist_ok=True)
            (root / "protocol.json").write_text(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            cache = root / "cache" / "padding" / "blob"
            cache.mkdir(parents=True)
            (cache / "CACHE_MANIFEST.json").write_text(json.dumps({"mode": "formal"}), encoding="utf-8")
            return root / "runs" / "formal" / "padding" / "blob" / "short"

        # Keep the short smoke under the package's already-allowlisted test
        # scratch root; the Windows Codex sandbox may deny tempfile's default
        # 0700 directory under the user TEMP tree.
        scratch = Path(__file__).parent / ".test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = scratch / f"resume_smoke_{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        with nullcontext(str(temporary)):
            base = Path(temporary)
            one_root = base / "one"
            one_run = setup(one_root)
            with patch("position_sources.experiment.build_model", side_effect=lambda *args, **kwargs: TinyModel()), patch(
                "position_sources.experiment._CHECKPOINT_INTERVAL", 2
            ):
                one = experiment._run_formal_arm(
                    output=one_root,
                    cfg=cfg,
                    family="padding",
                    primitive="blob",
                    variant="zero_s32",
                    seed=1,
                    device="cpu",
                    requested_steps=4,
                    effective_batch_size=4,
                    microbatch_size=2,
                    comparison_block_id="short-block",
                    run_id="short",
                    train_points=points,
                    train_images=images,
                    targets=targets,
                    run_root=one_run,
                )
            one_payload = torch.load(one["checkpoint"], map_location="cpu", weights_only=False)

            interrupted_root = base / "interrupted"
            interrupted_run = setup(interrupted_root)
            call_count = {"value": 0}

            def fail_after_checkpoint(*args, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 3:
                    raise RuntimeError("simulated interruption")
                return original_update(*args, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"), patch(
                "position_sources.experiment.build_model", side_effect=lambda *args, **kwargs: TinyModel()
            ), patch("position_sources.experiment._microbatch_update", side_effect=fail_after_checkpoint), patch(
                "position_sources.experiment._CHECKPOINT_INTERVAL", 2
            ):
                experiment._run_formal_arm(
                    output=interrupted_root,
                    cfg=cfg,
                    family="padding",
                    primitive="blob",
                    variant="zero_s32",
                    seed=1,
                    device="cpu",
                    requested_steps=4,
                    effective_batch_size=4,
                    microbatch_size=2,
                    comparison_block_id="short-block",
                    run_id="short",
                    train_points=points,
                    train_images=images,
                    targets=targets,
                    run_root=interrupted_run,
                )
            periodic = interrupted_run / "zero_s32" / "1" / "checkpoint_step_0002.pt"
            self.assertTrue(periodic.is_file())
            partial = torch.load(periodic, map_location="cpu", weights_only=False)
            self.assertEqual(partial["status"], "in_progress")
            self.assertEqual(partial["completed_optimizer_steps"], 2)
            with patch("position_sources.experiment.build_model", side_effect=lambda *args, **kwargs: TinyModel()), patch(
                "position_sources.experiment._CHECKPOINT_INTERVAL", 2
            ):
                resumed = experiment._run_formal_arm(
                    output=interrupted_root,
                    cfg=cfg,
                    family="padding",
                    primitive="blob",
                    variant="zero_s32",
                    seed=1,
                    device="cpu",
                    requested_steps=4,
                    effective_batch_size=4,
                    microbatch_size=2,
                    comparison_block_id="short-block",
                    run_id="short",
                    train_points=points,
                    train_images=images,
                    targets=targets,
                    run_root=interrupted_run,
                    resume_from=periodic,
                )
            resumed_payload = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(resumed_payload["completed_optimizer_steps"], 4)
            for key, value in one_payload["model_state"].items():
                self.assertTrue(torch.equal(value, resumed_payload["model_state"][key]), key)


if __name__ == "__main__":
    unittest.main()
