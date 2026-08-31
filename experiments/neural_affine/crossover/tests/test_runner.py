from __future__ import annotations

import json
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossover.evaluator import evaluate_run
from crossover.models import backbone_parameter_hash, build_model, save_exact_init, seed_all
from crossover.protocol import load_protocol, protocol_hash
from crossover.training import run_condition


def _scratch(name: str) -> Path:
    path = ROOT / "tests" / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _tiny_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, size=(4, 16, 16), dtype=np.uint8)
    points = np.asarray([[59.0, 59.0], [59.0, 164.0], [164.0, 59.0], [164.0, 164.0]], dtype=np.float64)
    return images, points


def test_head_full_and_frozen_use_exact_init_and_support_only() -> None:
    root = _scratch("_runner_scratch")
    try:
        cfg = copy.deepcopy(load_protocol())
        p_hash = protocol_hash(payload=cfg)
        seed = 20260816
        seed_all(seed)
        init_model = build_model(tiny=True)
        init_path = root / "seed.pt"
        save_exact_init(init_model, init_path, seed=seed, protocol_hash=p_hash)
        images, points = _tiny_inputs()
        summaries = {}
        for regime in ("frozen_feature", "head_only", "full"):
            seed_all(seed)
            summaries[regime] = run_condition(
                build_model(tiny=True),
                images,
                points,
                root / regime,
                protocol=cfg,
                seed=seed,
                support_name="corners4",
                regime=regime,
                init_checkpoint=init_path,
                device="cpu",
                steps=2,
                resume=False,
            )
        assert summaries["frozen_feature"]["steps_completed"] == 0
        assert summaries["head_only"]["steps_completed"] == 2
        assert summaries["full"]["steps_completed"] == 2
        assert summaries["head_only"]["backbone_init_hash"] == summaries["head_only"]["backbone_final_hash"]
        assert summaries["full"]["backbone_init_hash"] != summaries["full"]["backbone_final_hash"]
        for regime in summaries:
            receipt = json.loads((root / regime / "receipt.json").read_text(encoding="utf-8"))
            assert receipt["dense_labels_read_by_training"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_backbone_parameter_gate_ignores_bn_buffer_only_changes() -> None:
    model = build_model(tiny=True)
    before = backbone_parameter_hash(model)
    with torch.no_grad():
        model.backbone[1].running_mean.add_(1.0)
        model.backbone[1].running_var.mul_(2.0)
    assert backbone_parameter_hash(model) == before


def test_resume_and_independent_dense_evaluator() -> None:
    root = _scratch("_resume_scratch")
    try:
        cfg = load_protocol()
        p_hash = protocol_hash(payload=cfg)
        seed = 20260817
        seed_all(seed)
        init_path = root / "seed.pt"
        save_exact_init(build_model(tiny=True), init_path, seed=seed, protocol_hash=p_hash)
        images, points = _tiny_inputs()
        run_dir = root / "runs" / "corners4" / str(seed) / "head_only"
        run_condition(
            build_model(tiny=True), images, points, run_dir,
            protocol=cfg, seed=seed, support_name="corners4", regime="head_only",
            init_checkpoint=init_path, device="cpu", steps=2, resume=False,
        )
        try:
            run_condition(
                build_model(tiny=True), images, points, run_dir,
                protocol=cfg, seed=seed, support_name="corners4", regime="head_only",
                init_checkpoint=init_path, device="cpu", steps=3, resume=True,
            )
        except ValueError as exc:
            assert "requested_steps" in str(exc)
        else:
            raise AssertionError("resume must reject a changed requested_steps/T_max")
        resumed = run_condition(
            build_model(tiny=True), images, points, run_dir,
            protocol=cfg, seed=seed, support_name="corners4", regime="head_only",
            init_checkpoint=init_path, device="cpu", steps=2, resume=True,
        )
        assert resumed["resumed"] is True
        assert resumed["steps_completed"] == 2
        # Primary evaluation must ignore a support-selected best checkpoint.
        # Making best invalid proves that the endpoint final.pt is mandatory.
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"diagnostic-best-must-not-be-primary")
        cache = root / "cache"
        cache.mkdir()
        np.save(cache / "dense_images.npy", images, allow_pickle=False)
        np.save(cache / "dense_points.npy", points, allow_pickle=False)
        result = evaluate_run(run_dir, cache, protocol=cfg, device="cpu", model_factory=lambda: build_model(tiny=True), allow_small_cache=True)
        assert result["status"] == "passed"
        assert result["checkpoint"]["selection"] == "endpoint_final"
        assert Path(result["checkpoint"]["path"]).name == "final.pt"
        assert result["dense_labels"]["read_after_prediction"] is True
        assert result["backbone"]["head_only_bitwise_unchanged"] is True
        try:
            evaluate_run(run_dir, cache, protocol=cfg, device="cpu", model_factory=lambda: build_model(tiny=True))
        except ValueError as exc:
            assert "formal dense image shape" in str(exc)
        else:
            raise AssertionError("formal evaluator must reject the synthetic 4-point cache")
    finally:
        shutil.rmtree(root, ignore_errors=True)
