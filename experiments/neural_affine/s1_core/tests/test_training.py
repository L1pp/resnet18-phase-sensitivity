"""Lightweight CPU checks for the standalone S1 model/trainer."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from s1clean.metrics import (
    affine_residual_mae_px,
    apply_affine,
    fit_affine,
    mae_px,
)
from s1clean.models import build_model, seed_all
from s1clean.train import predict_images, train_run


def test_model_shapes_and_coordconv_channels() -> None:
    seed_all(20260816)
    vanilla = build_model("vanilla")
    coordconv = build_model("coordconv")
    assert vanilla.fc.out_features == 2
    assert coordconv.conv1.in_channels == 5
    x = torch.zeros(2, 3, 32, 32)
    assert tuple(vanilla(x).shape) == (2, 2)
    assert tuple(coordconv(x).shape) == (2, 2)


def test_float64_affine_helpers() -> None:
    target = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    predicted = target @ np.asarray([[1.2, 0.1], [-0.2, 0.8]], dtype=np.float64).T + np.asarray([0.4, -0.3])
    matrix, bias = fit_affine(predicted, target)
    np.testing.assert_allclose(apply_affine(predicted, matrix, bias), target, atol=1e-12)
    assert affine_residual_mae_px(predicted, target, coord_scale=223.0) < 1e-9
    assert mae_px(target, target) == 0.0


def test_train_run_smoke_writes_resume_and_predictions() -> None:
    seed_all(20260816)
    model = build_model("vanilla")
    images = np.zeros((4, 32, 32), dtype=np.uint8)
    # Distinguish the four anchors while keeping the smoke computation tiny.
    images[0, 8:12, 8:12] = 255
    images[1, 8:12, 20:24] = 255
    images[2, 20:24, 8:12] = 255
    images[3, 20:24, 20:24] = 255
    targets = np.asarray([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]], dtype=np.float32)
    # Use a known workspace child: the managed Windows runner may deny mkdir
    # operations below a tempfile directory carrying a 0700 ACL.
    temp = Path(__file__).resolve().parent / "_tmp_train_run"
    if temp.exists():
        shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True, exist_ok=True)
    try:
        summary = train_run(
            model,
            images,
            targets,
            images,
            targets,
            Path(temp) / "run",
            {"steps": 2, "batch_size": 4, "anchor_every": 1, "resume_every": 2, "protocol_hash": "test", "code_hash": "test"},
            20260816,
            device="cpu",
        )
        assert summary["steps"] == 2
        run = Path(temp) / "run"
        assert (run / "checkpoints" / "init.pt").exists()
        assert (run / "checkpoints" / "best.pt").exists()
        assert (run / "checkpoints" / "final.pt").exists()
        assert (run / "checkpoints" / "resume.pt").exists()
        pred = predict_images(model, images, "cpu", batch_size=2)
        assert pred.shape == (4, 2)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


class TrainingTests(unittest.TestCase):
    def test_model_shapes_and_coordconv_channels(self) -> None:
        test_model_shapes_and_coordconv_channels()

    def test_float64_affine_helpers(self) -> None:
        test_float64_affine_helpers()

    def test_train_run_smoke_writes_resume_and_predictions(self) -> None:
        test_train_run_smoke_writes_resume_and_predictions()


if __name__ == "__main__":
    unittest.main()
