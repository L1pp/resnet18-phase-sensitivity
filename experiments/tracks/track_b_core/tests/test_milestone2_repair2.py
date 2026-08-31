from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from fsx import models
from fsx.checkpoints import load_checkpoint_payload, resume_identity_from_cell
from fsx.config import PACKAGE_ROOT, resolve_cell
from fsx.trainer import SmokeOptions, run_cell


@unittest.skipIf(models.TORCH_IMPORT_ERROR is not None, f"Torch unavailable: {models.TORCH_IMPORT_ERROR}")
class Milestone2Repair2ResumeIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / "_work" / f"repair2_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.assets = PACKAGE_ROOT / "assets"
        self.options = SmokeOptions(steps=1, query_limit=2, image_side=16)
        models.torch.set_num_threads(1)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @staticmethod
    def _catalog_text(project: Path) -> str:
        return (project / "RUN_CATALOG.json").read_text(encoding="utf-8")

    def _assert_rejected_before_attempt(
        self,
        *,
        project: Path,
        cell_id: str,
        checkpoint: Path,
        expected_error: str,
    ) -> None:
        before = self._catalog_text(project)
        cell = resolve_cell(cell_id)
        target_seed_root = project / "runs" / str(cell["family"]) / str(cell["condition_id"]) / f"seed_{cell['run_seed']}"
        self.assertFalse(target_seed_root.exists())
        with self.assertRaisesRegex(ValueError, expected_error):
            run_cell(
                cell_id,
                project_root=project,
                asset_root=self.assets,
                mode="smoke",
                device="cpu",
                smoke_options=self.options,
                resume_from=checkpoint,
            )
        self.assertEqual(self._catalog_text(project), before)
        self.assertFalse(target_seed_root.exists())

    def test_b7_cross_seed_cross_anchor_condition_and_legacy_checkpoint_fail_before_attempt(self) -> None:
        project = self.root / "b7_identity"
        source_cell_id = "B7.dense_relative_4_neighbor_plus_1_anchor.s20260816"
        source = run_cell(
            source_cell_id,
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=self.options,
        )
        checkpoint = Path(source["run_path"]) / "checkpoints" / "final.pt"
        payload = load_checkpoint_payload(checkpoint)
        self.assertEqual(payload["resume_identity"], resume_identity_from_cell(resolve_cell(source_cell_id)))
        self.assertEqual(payload["resume_identity"]["anchors"], [0])

        self._assert_rejected_before_attempt(
            project=project,
            cell_id="B7.dense_relative_4_neighbor_plus_1_anchor.s20260817",
            checkpoint=checkpoint,
            expected_error="resume_identity mismatch",
        )
        self._assert_rejected_before_attempt(
            project=project,
            cell_id="B7.dense_relative_4_neighbor_plus_4_corners.s20260816",
            checkpoint=checkpoint,
            expected_error="resume_identity mismatch",
        )

        legacy_payload = dict(payload)
        legacy_payload.pop("resume_identity")
        legacy_checkpoint = project / "legacy_without_identity.pt"
        models.torch.save(legacy_payload, legacy_checkpoint)
        before = self._catalog_text(project)
        with self.assertRaisesRegex(ValueError, "no resume_identity"):
            run_cell(
                source_cell_id,
                project_root=project,
                asset_root=self.assets,
                mode="smoke",
                device="cpu",
                smoke_options=SmokeOptions(steps=2, query_limit=2, image_side=16),
                resume_from=legacy_checkpoint,
            )
        self.assertEqual(self._catalog_text(project), before)
        source_seed_root = project / "runs" / "B7" / "dense_relative_4_neighbor_plus_1_anchor" / "seed_20260816"
        self.assertEqual(sorted(path.name for path in source_seed_root.iterdir()), ["attempt_001"])

    def test_global_gate_rejects_common_cross_seed_and_analytic_resume_before_attempt(self) -> None:
        project = self.root / "global_identity"
        source = run_cell(
            "B1.gn.s20260816",
            project_root=project,
            asset_root=self.assets,
            mode="smoke",
            device="cpu",
            smoke_options=self.options,
        )
        checkpoint = Path(source["run_path"]) / "checkpoints" / "final.pt"
        self._assert_rejected_before_attempt(
            project=project,
            cell_id="B1.gn.s20260817",
            checkpoint=checkpoint,
            expected_error="resume_identity mismatch",
        )
        self._assert_rejected_before_attempt(
            project=project,
            cell_id="B8.degree2.s20260816",
            checkpoint=checkpoint,
            expected_error="does not support checkpoint resume",
        )
        self.assertEqual(len(json.loads(self._catalog_text(project))["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
