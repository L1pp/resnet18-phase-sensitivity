import unittest
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from position_sources.cli import build_parser
from position_sources.experiment import prepare_root, preflight_root, run_family, run_reference
from position_sources.packaging import pack_root, verify_package
from position_sources.protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_hash
from position_sources.reviewer import review_block, run_review
from position_sources.renderer import dense_domain_points, split_dense_domain


class ManifestAndCliTests(unittest.TestCase):
    def test_reviewer_cli_import_is_torch_free(self):
        package_root = Path(__file__).resolve().parents[1]
        probe = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import position_sources.cli; print('torch' in sys.modules)"],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(probe.stdout.strip(), "False")

    def test_all_required_commands_parse(self):
        parser = build_parser()
        for argv in (
            ["prepare", "--out", "x"],
            ["preflight", "--root", "x"],
            ["smoke", "--out", "x"],
            ["profile", "--out", "x"],
            [
                "run-family", "--root", "x", "--family", "padding",
                "--comparison-block-id", "A1-padding", "--run-id", "A1-padding-blob-full", "--microbatch", "4",
            ],
            [
                "run-reference", "--root", "x", "--comparison-block-id", "A1-padding-bn-reference",
                "--run-id", "A1-padding-bn-reference-full", "--microbatch", "4",
            ],
            ["evaluate", "--root", "x", "--checkpoint", "x.pt"],
            ["pack", "--root", "x", "--out", "x.zip"],
            ["verify", "--path", "x.zip"],
            [
                "review", "--root", "x", "--out", "independent_review_v1/track_a/static",
                "--allowed-reviewer-root", "independent_review_v1/track_a/static", "--review-id", "static",
            ],
            [
                "review-block",
                "--root", "x",
                "--out", "independent_review_v1/track_a/A1-padding",
                "--allowed-reviewer-root", "independent_review_v1/track_a/A1-padding",
                "--comparison-block-id", "A1-padding",
                "--family", "padding",
                "--input", "inputs.npy",
                "--predictions", "predictions.npy",
                "--coordinates", "coordinates.npy",
                "--gap-train-features", "gap_train_features.npy",
                "--gap-eval-features", "gap_eval_features.npy",
                "--gap-train-coordinates", "gap_train_coordinates.npy",
                "--gap-eval-coordinates", "gap_eval_coordinates.npy",
                "--gap-feature-metadata", "GAP_FEATURE_METADATA.json",
                "--support-coordinates", "support_coordinates.npy",
                "--support-predictions", "support_predictions.npy",
                "--support-targets", "support_targets.npy",
            ],
        ):
            self.assertEqual(parser.parse_args(argv).command, argv[0])

    def test_quick_prepare_preflight_pack_review(self):
        temporary = Path(__file__).parent / ".test_tmp" / f"manifest_{os.getpid()}"
        quick_root = temporary / "quick_prepared"
        prepare_root(quick_root, protocol_path=DEFAULT_PROTOCOL_PATH, families=("padding",), primitives=("blob",), quick=True)
        preflight = preflight_root(quick_root)
        self.assertEqual(preflight["status"], "passed")
        package = temporary / "position_sources_v1.zip"
        receipt = pack_root(quick_root, package)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["package_manifest"]["freeze_provenance"]["freeze_file"], "EXPERIMENT_PREP_FREEZE_20260824.md")
        self.assertEqual(len(receipt["package_manifest"]["freeze_provenance"]["freeze_sha256"]), 64)
        self.assertEqual(verify_package(package)["status"], "passed")
        quick_review_root = temporary / "independent_review_v1" / "track_a" / "quick"
        quick_review = run_review(
            quick_root,
            quick_review_root,
            allowed_reviewer_root=quick_review_root,
            review_id="quick",
        )
        self.assertEqual(quick_review["status"], "failed")

        formal_root = temporary / "formal_prepared"
        prepare_root(formal_root, protocol_path=DEFAULT_PROTOCOL_PATH, quick=False)
        formal_review_root = temporary / "independent_review_v1" / "track_a" / "formal"
        review = run_review(
            formal_root,
            formal_review_root,
            allowed_reviewer_root=formal_review_root,
            review_id="formal",
        )
        self.assertEqual(review["status"], "passed")
        self.assertEqual(review["prepared_caches"]["status"], "passed")

    def test_block_review_recomputes_metrics_and_enforces_isolation(self):
        temporary = Path(__file__).parent / ".test_tmp" / f"block_review_{os.getpid()}"
        executor = temporary / "executor"
        reviewer = temporary / "independent_review_v1" / "track_a" / "A1-padding"
        executor.mkdir(parents=True, exist_ok=True)
        cfg_text = DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8")
        (executor / "protocol.json").write_text(cfg_text, encoding="utf-8")
        points = dense_domain_points()
        np.save(executor / "inputs.npy", np.zeros((len(points), 1), dtype=np.uint8), allow_pickle=False)
        np.save(executor / "predictions.npy", points, allow_pickle=False)
        np.save(executor / "coordinates.npy", points, allow_pickle=False)
        train_points, eval_points, _ = split_dense_domain()
        train_features = np.zeros((len(train_points), 512), dtype=np.float32)
        eval_features = np.zeros((len(eval_points), 512), dtype=np.float32)
        np.save(executor / "gap_train_features.npy", train_features, allow_pickle=False)
        np.save(executor / "gap_eval_features.npy", eval_features, allow_pickle=False)
        np.save(executor / "gap_train_coordinates.npy", train_points, allow_pickle=False)
        np.save(executor / "gap_eval_coordinates.npy", eval_points, allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "kind": "position_sources_gap_feature_field",
            "source": "model.forward_features",
            "layer": "global_average_pool_before_fc",
            "field": "pre_fc_gap",
            "feature_dim": 512,
            "dtype": "float32",
            "head_excluded": True,
            "family": "padding",
            "primitive": "blob",
            "variant": "zero_s32",
            "comparison_block_id": "A1-padding",
            "train_count": len(train_points),
            "eval_count": len(eval_points),
            "protocol_hash": protocol_hash(path=DEFAULT_PROTOCOL_PATH),
            "probe_contract": load_protocol(DEFAULT_PROTOCOL_PATH)["evaluation"]["gap_feature"]["probe"],
            "shift_distances_px": [1, 32, 64],
            "train_features_sha256": hashlib.sha256((executor / "gap_train_features.npy").read_bytes()).hexdigest(),
            "eval_features_sha256": hashlib.sha256((executor / "gap_eval_features.npy").read_bytes()).hexdigest(),
        }
        (executor / "GAP_FEATURE_METADATA.json").write_text(json.dumps(metadata), encoding="utf-8")
        support_points = points[:64]
        np.save(executor / "support_coordinates.npy", support_points, allow_pickle=False)
        np.save(executor / "support_predictions.npy", support_points, allow_pickle=False)
        np.save(executor / "support_targets.npy", support_points, allow_pickle=False)
        result = review_block(
            executor,
            reviewer,
            comparison_block_id="A1-padding",
            allowed_reviewer_root=reviewer,
            config_id="zero_s32_seed20260823",
            family="padding",
            variant="zero_s32",
            input_path="inputs.npy",
            prediction_path="predictions.npy",
            coordinate_path="coordinates.npy",
            gap_train_feature_path="gap_train_features.npy",
            gap_eval_feature_path="gap_eval_features.npy",
            gap_train_coordinate_path="gap_train_coordinates.npy",
            gap_eval_coordinate_path="gap_eval_coordinates.npy",
            gap_feature_metadata_path="GAP_FEATURE_METADATA.json",
            support_coordinate_path="support_coordinates.npy",
            support_prediction_path="support_predictions.npy",
            support_target_path="support_targets.npy",
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["coordinate_coverage"]["complete"])
        self.assertEqual(result["metrics"]["raw_mae_px"], 0.0)
        self.assertTrue(result["support_metrics"]["support_pass"])
        self.assertEqual(len(result["artifacts"]["predictions"]["sha256"]), 64)
        bad_metadata_path = executor / "GAP_FEATURE_METADATA_FAKE.json"
        bad_metadata = dict(metadata)
        bad_metadata["source"] = "prediction_head"
        bad_metadata_path.write_text(json.dumps(bad_metadata), encoding="utf-8")
        fake_feature_review = review_block(
            executor,
            temporary / "independent_review_v1" / "track_a" / "A1-padding-fake-feature",
            comparison_block_id="A1-padding-fake-feature",
            allowed_reviewer_root=temporary / "independent_review_v1" / "track_a" / "A1-padding-fake-feature",
            family="padding",
            input_path="inputs.npy",
            prediction_path="predictions.npy",
            coordinate_path="coordinates.npy",
            gap_train_feature_path="gap_train_features.npy",
            gap_eval_feature_path="gap_eval_features.npy",
            gap_train_coordinate_path="gap_train_coordinates.npy",
            gap_eval_coordinate_path="gap_eval_coordinates.npy",
            gap_feature_metadata_path="GAP_FEATURE_METADATA_FAKE.json",
            support_coordinate_path="support_coordinates.npy",
            support_prediction_path="support_predictions.npy",
            support_target_path="support_targets.npy",
        )
        self.assertEqual(fake_feature_review["status"], "blocked")
        self.assertIn("gap_feature:gap_metadata_source_mismatch", fake_feature_review["errors"])
        np.save(executor / "support_predictions_bad.npy", support_points + 3.0, allow_pickle=False)
        blocked = review_block(
            executor,
            temporary / "independent_review_v1" / "track_a" / "A1-padding-bad-support",
            comparison_block_id="A1-padding-bad-support",
            allowed_reviewer_root=temporary / "independent_review_v1" / "track_a" / "A1-padding-bad-support",
            family="padding",
            input_path="inputs.npy",
            prediction_path="predictions.npy",
            coordinate_path="coordinates.npy",
            gap_train_feature_path="gap_train_features.npy",
            gap_eval_feature_path="gap_eval_features.npy",
            gap_train_coordinate_path="gap_train_coordinates.npy",
            gap_eval_coordinate_path="gap_eval_coordinates.npy",
            gap_feature_metadata_path="GAP_FEATURE_METADATA.json",
            support_coordinate_path="support_coordinates.npy",
            support_prediction_path="support_predictions_bad.npy",
            support_target_path="support_targets.npy",
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("support_gate_failed", blocked["errors"])
        with self.assertRaises(ValueError):
            review_block(
                executor,
                executor / "nested-reviewer",
                comparison_block_id="A1-padding",
                allowed_reviewer_root=executor / "nested-reviewer",
                family="padding",
                input_path="inputs.npy",
                prediction_path="predictions.npy",
                coordinate_path="coordinates.npy",
                gap_train_feature_path="gap_train_features.npy",
                gap_eval_feature_path="gap_eval_features.npy",
                gap_train_coordinate_path="gap_train_coordinates.npy",
                gap_eval_coordinate_path="gap_eval_coordinates.npy",
                gap_feature_metadata_path="GAP_FEATURE_METADATA.json",
                support_coordinate_path="support_coordinates.npy",
                support_prediction_path="support_predictions.npy",
                support_target_path="support_targets.npy",
            )

        with self.assertRaises(ValueError):
            review_block(
                executor,
                temporary / "arbitrary-reviewer-root",
                comparison_block_id="A1-padding",
                allowed_reviewer_root=temporary / "independent_review_v1" / "track_a" / "A1-padding",
                family="padding",
                input_path="inputs.npy",
                prediction_path="predictions.npy",
                coordinate_path="coordinates.npy",
                gap_train_feature_path="gap_train_features.npy",
                gap_eval_feature_path="gap_eval_features.npy",
                gap_train_coordinate_path="gap_train_coordinates.npy",
                gap_eval_coordinate_path="gap_eval_coordinates.npy",
                gap_feature_metadata_path="GAP_FEATURE_METADATA.json",
                support_coordinate_path="support_coordinates.npy",
                support_prediction_path="support_predictions.npy",
                support_target_path="support_targets.npy",
            )

    def test_formal_run_family_rejects_non_protocol_requests_before_cache_access(self):
        temporary = Path(__file__).parent / ".test_tmp" / f"formal_guards_{os.getpid()}"
        root = temporary / "prepared"
        root.mkdir(parents=True, exist_ok=True)
        (root / "protocol.json").write_text(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        common = {
            "family": "padding",
            "primitive": "blob",
            "comparison_block_id": "A1-padding",
            "run_id": "A1-padding-blob-full",
            "allow_formal": True,
        }
        with self.assertRaises(ValueError):
            run_family(root, **common, microbatch=3)
        with self.assertRaises(ValueError):
            run_family(root, **common, steps=1)
        with self.assertRaises(ValueError):
            run_family(root, **common, variants=("torus_s32",))
        with self.assertRaises(ValueError):
            run_family(root, **common, seeds=(999,))
        with self.assertRaises(ValueError):
            run_reference(
                root,
                primitive="blob",
                seeds=(20260823,),
                steps=1,
                comparison_block_id="A1-padding-bn-reference",
                run_id="A1-padding-bn-reference-shard",
                allow_formal=True,
            )


if __name__ == "__main__":
    unittest.main()
