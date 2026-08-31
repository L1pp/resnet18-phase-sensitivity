from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from fsx import models, trainer
from fsx.checkpoints import load_checkpoint_payload
from fsx.config import PACKAGE_ROOT, load_protocol, resolve_cell
from fsx.data import coordinate_target_f64
from fsx.evaluator import review_attempt
from fsx.trainer import SmokeOptions, create_b4_paired_initialization, run_cell


@unittest.skipIf(models.TORCH_IMPORT_ERROR is not None, f"Torch unavailable: {models.TORCH_IMPORT_ERROR}")
class Milestone2Repair1TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / "_work" / f"repair1_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.assets = PACKAGE_ROOT / "assets"
        models.torch.set_num_threads(1)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def assertNestedExact(self, left: Any, right: Any, context: str = "root") -> None:
        if models.torch.is_tensor(left) or models.torch.is_tensor(right):
            self.assertTrue(models.torch.is_tensor(left) and models.torch.is_tensor(right), context)
            self.assertTrue(models.torch.equal(left.detach().cpu(), right.detach().cpu()), context)
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            self.assertTrue(np.array_equal(np.asarray(left), np.asarray(right)), context)
            return
        if isinstance(left, dict) and isinstance(right, dict):
            self.assertEqual(set(left), set(right), context)
            for key in left:
                self.assertNestedExact(left[key], right[key], f"{context}.{key}")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            self.assertEqual(len(left), len(right), context)
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                self.assertNestedExact(left_value, right_value, f"{context}[{index}]")
            return
        self.assertEqual(left, right, context)

    def _run_upstream(self, project_root: Path) -> dict[str, Any]:
        return run_cell(
            "B6.g64_upstream.s20260816",
            project_root=project_root,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=1, query_limit=2, image_side=16),
        )

    def test_b6_ridge_and_b8_degree2_use_direct_float64_targets(self) -> None:
        project = self.root / "analytic"
        upstream = self._run_upstream(project)
        ridge = run_cell(
            "B6.ridge.s20260816",
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=1, query_limit=2, image_side=16),
            upstream_run=upstream["run_path"],
        )
        ridge_run = Path(ridge["run_path"])
        ridge_index = self._json(self.assets / "indices" / "IDX_CORNERS4_BLOB_A1.json")
        ridge_xy = np.asarray([row["position_px"] for row in ridge_index["rows"]], dtype=np.float64)
        expected_ridge_truth = coordinate_target_f64(ridge_xy, "identity")
        with np.load(ridge_run / "ridge_artifact.npz", allow_pickle=False) as artifact:
            saved_truth = np.asarray(artifact["train_truth_u"])
            self.assertEqual(saved_truth.dtype, np.float64)
            self.assertTrue(np.array_equal(saved_truth, expected_ridge_truth))
            for name in ("train_features", "query_features", "beta", "singular_values"):
                self.assertEqual(np.asarray(artifact[name]).dtype, np.float64, name)
        with np.load(ridge_run / "predictions.npz", allow_pickle=False) as predictions:
            for name in predictions.files:
                if name.endswith(("pred_u", "truth_u", "pred_px", "truth_px")):
                    self.assertEqual(np.asarray(predictions[name]).dtype, np.float64, name)
        self.assertEqual(review_attempt(ridge_run, ridge["review_path"])["review_status"], "PASS")

        degree2 = run_cell(
            "B8.degree2.s20260816",
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=1, query_limit=2, image_side=16),
        )
        degree2_run = Path(degree2["run_path"])
        facts = self._json(degree2_run / "run_facts.json")
        expected_coefficients = np.asarray(load_protocol()["targets"]["quadratic_b8"]["coefficients"], dtype=np.float64).T
        observed_coefficients = np.asarray(facts["coefficients"], dtype=np.float64)
        self.assertLessEqual(float(np.max(np.abs(observed_coefficients - expected_coefficients))), 1e-12)
        with np.load(degree2_run / "analytic_degree2.npz", allow_pickle=False) as artifact:
            self.assertEqual(np.asarray(artifact["coefficients"]).dtype, np.float64)
        with np.load(degree2_run / "predictions.npz", allow_pickle=False) as predictions:
            for name in predictions.files:
                if name.endswith(("pred_u", "truth_u", "pred_px", "truth_px")):
                    self.assertEqual(np.asarray(predictions[name]).dtype, np.float64, name)
        self.assertEqual(review_attempt(degree2_run, degree2["review_path"])["review_status"], "PASS")

    def test_lpft_pre_step501_resume_updates_full_model_and_matches_continuous(self) -> None:
        project = self.root / "lpft"
        upstream = self._run_upstream(project)
        source = run_cell(
            "B6.lp_ft.s20260816",
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=1, query_limit=2, image_side=16),
            upstream_run=upstream["run_path"],
        )
        source_run = Path(source["run_path"])
        pre_step501 = source_run / "checkpoints" / "pre_step501.pt"
        pre_payload = load_checkpoint_payload(pre_step501)
        self.assertEqual(pre_payload["stage_state"]["stage"], "full_pre_update")
        new_parameter_ids = set(pre_payload["optimizer_state"]["param_groups"][-1]["params"])
        self.assertFalse(new_parameter_ids.intersection(pre_payload["optimizer_state"]["state"]))

        resumed = run_cell(
            "B6.lp_ft.s20260816",
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=1, query_limit=2, image_side=16),
            resume_from=pre_step501,
            upstream_run=upstream["run_path"],
        )
        resumed_run = Path(resumed["run_path"])
        continuous_final = load_checkpoint_payload(source_run / "checkpoints" / "final.pt")
        resumed_final = load_checkpoint_payload(resumed_run / "checkpoints" / "final.pt")
        self.assertNestedExact(continuous_final["model_state"], resumed_final["model_state"], "model")
        self.assertNestedExact(continuous_final["optimizer_state"], resumed_final["optimizer_state"], "optimizer")
        self.assertNestedExact(
            self._json(source_run / "training_trace.json"),
            self._json(resumed_run / "training_trace.json"),
            "trace",
        )
        final_parameter_ids = {
            parameter_id
            for group in resumed_final["optimizer_state"]["param_groups"]
            for parameter_id in group["params"]
        }
        self.assertTrue(final_parameter_ids.issubset(set(resumed_final["optimizer_state"]["state"])))
        self.assertEqual(resumed_final["completed_step"], 2)
        self.assertEqual(resumed_final["stream_cursor"], 2)
        self.assertEqual(review_attempt(resumed_run, resumed["review_path"])["review_status"], "PASS")

    def test_b7_dense_split_resume_matches_continuous_model_cursor_trace_and_exposure(self) -> None:
        project = self.root / "b7"
        cell_id = "B7.dense_relative_4_neighbor_plus_1_anchor.s20260816"
        continuous = run_cell(
            cell_id,
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=2, query_limit=2, image_side=16),
        )
        split = run_cell(
            cell_id,
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=1, query_limit=2, image_side=16),
        )
        resumed = run_cell(
            cell_id,
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=SmokeOptions(steps=2, query_limit=2, image_side=16),
            resume_from=Path(split["run_path"]) / "checkpoints" / "final.pt",
        )
        continuous_run = Path(continuous["run_path"])
        resumed_run = Path(resumed["run_path"])
        continuous_final = load_checkpoint_payload(continuous_run / "checkpoints" / "final.pt")
        resumed_final = load_checkpoint_payload(resumed_run / "checkpoints" / "final.pt")
        self.assertNestedExact(continuous_final["model_state"], resumed_final["model_state"], "model")
        self.assertNestedExact(continuous_final["optimizer_state"], resumed_final["optimizer_state"], "optimizer")
        self.assertNestedExact(continuous_final["runtime_state"], resumed_final["runtime_state"], "runtime")
        self.assertEqual((resumed_final["completed_step"], resumed_final["stream_cursor"]), (2, 2))
        self.assertNestedExact(
            self._json(continuous_run / "training_trace.json"),
            self._json(resumed_run / "training_trace.json"),
            "trace",
        )
        for filename in ("b7_exposure.npz", "b7_training_raw.npz"):
            with np.load(continuous_run / filename, allow_pickle=False) as left, np.load(resumed_run / filename, allow_pickle=False) as right:
                self.assertEqual(set(left.files), set(right.files))
                for name in left.files:
                    self.assertTrue(np.array_equal(left[name], right[name]), f"{filename}:{name}")
        self.assertEqual(review_attempt(resumed_run, resumed["review_path"])["review_status"], "PASS")

    def test_b4_trainer_resume_before_and_after_first_gate_preserves_full_state(self) -> None:
        cell = resolve_cell("B4.causal_adamw.s20260816")
        index = self._json(self.assets / "indices" / f"{cell['index_id']}.json")
        device = models.torch.device("cpu")

        for label, threshold in (("before_gate", -1.0), ("after_gate", 1e9)):
            case_root = self.root / label
            paired = create_b4_paired_initialization(20260816, project_root=case_root, mode="smoke")
            continuous_path = case_root / "continuous"
            split_path = case_root / "split"
            resumed_path = case_root / "resumed"
            for path in (continuous_path, split_path, resumed_path):
                path.mkdir(parents=True, exist_ok=False)

            continuous_model = models.build_smoke_cnn("gn", 20260816).to(device)
            continuous_trace, continuous_facts = trainer._b4_train(
                continuous_model,
                cell,
                index,
                self.assets,
                device,
                continuous_path,
                mode="smoke",
                smoke=SmokeOptions(steps=2, query_limit=2, image_side=16),
                resume_from=None,
                paired_initialization=paired,
                gate_threshold_px=threshold,
            )
            split_model = models.build_smoke_cnn("gn", 20260816).to(device)
            trainer._b4_train(
                split_model,
                cell,
                index,
                self.assets,
                device,
                split_path,
                mode="smoke",
                smoke=SmokeOptions(steps=1, query_limit=2, image_side=16),
                resume_from=None,
                paired_initialization=paired,
                gate_threshold_px=threshold,
            )
            resumed_model = models.build_smoke_cnn("gn", 20260816).to(device)
            resumed_trace, resumed_facts = trainer._b4_train(
                resumed_model,
                cell,
                index,
                self.assets,
                device,
                resumed_path,
                mode="smoke",
                smoke=SmokeOptions(steps=2, query_limit=2, image_side=16),
                resume_from=split_path / "checkpoints" / "final.pt",
                paired_initialization=paired,
                gate_threshold_px=threshold,
            )
            continuous_final = load_checkpoint_payload(continuous_path / "checkpoints" / "final.pt")
            resumed_final = load_checkpoint_payload(resumed_path / "checkpoints" / "final.pt")
            self.assertNestedExact(continuous_final["model_state"], resumed_final["model_state"], f"{label}.model")
            self.assertNestedExact(continuous_final["optimizer_state"], resumed_final["optimizer_state"], f"{label}.optimizer")
            self.assertNestedExact(continuous_final["runtime_state"], resumed_final["runtime_state"], f"{label}.runtime")
            self.assertNestedExact(continuous_trace, resumed_trace, f"{label}.trace")
            self.assertNestedExact(continuous_facts["first_gate"], resumed_facts["first_gate"], f"{label}.gate")
            with np.load(continuous_path / "b4_support_trajectory.npz", allow_pickle=False) as left, np.load(resumed_path / "b4_support_trajectory.npz", allow_pickle=False) as right:
                for name in left.files:
                    self.assertTrue(np.array_equal(left[name], right[name]), f"{label}:{name}")
            if label == "before_gate":
                self.assertIsNone(resumed_facts["first_gate"]["selection_step"])
            else:
                self.assertEqual(resumed_facts["first_gate"]["selection_step"], 1)
                self.assertTrue((resumed_path / "snapshots" / "first_gate_predictions.npz").is_file())


if __name__ == "__main__":
    unittest.main()
