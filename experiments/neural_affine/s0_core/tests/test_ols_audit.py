from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from closeout_sprint.contracts import SchemaError
from closeout_sprint.ols_audit import _metrics, audit_ols, audit_ols_npz, fit_ols_svd, load_design_target_npz, ridge_path


class OlsAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / f".test_ols_{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _formal_payload(self, *, feature_dim: int = 3) -> dict[str, np.ndarray]:
        """Build the historical-size contract without rendering or GPU work."""

        rng = np.random.default_rng(20260821)
        n_train, n_eval = 3584, 32 * 41 * 41
        train_shape = np.concatenate([(np.arange(56) + tid) % 64 for tid in range(64)]).astype(np.int32)
        train_tid = np.repeat(np.arange(64, dtype=np.int32), 56)
        eval_shape = np.repeat(np.arange(32, dtype=np.int32), 41 * 41)
        eval_dense = np.tile(np.arange(41 * 41, dtype=np.int32), 32)
        # Keep a deterministic 64x3x2 geometry table; the two rows above are
        # expanded to three points for each shape.
        shape_q = np.asarray(
            [[[59.0 + sid, 59.0], [164.0 - sid, 59.0], [59.0, 164.0 - sid]] for sid in range(64)],
            dtype=np.float32,
        )
        dense_axis = np.linspace(59.0, 164.0, 41, dtype=np.float32)
        dense_t = np.asarray([[x, y] for x in dense_axis for y in dense_axis], dtype=np.float32)
        train_q = shape_q[train_shape]
        train_t = np.asarray([[float(tid), float((tid * 3) % 64)] for tid in train_tid], dtype=np.float32)
        eval_q = shape_q[eval_shape]
        eval_t = np.tile(dense_t, (32, 1))
        train_p6 = ((train_q + train_t[:, None, :]) / 223.0).reshape(n_train, 6).astype(np.float32)
        eval_p6 = ((eval_q + eval_t[:, None, :]) / 223.0).reshape(n_eval, 6).astype(np.float32)
        train_gap = rng.normal(size=(n_train, feature_dim)).astype(np.float32)
        eval_gap = rng.normal(size=(n_eval, feature_dim)).astype(np.float32)
        provenance = {
            "protocol_id": "a10_materialize_v1",
            "protocol_sha256": "a" * 64,
            "source_assets": {"synthetic": "b" * 64},
            "checkpoint": {"id": "G64", "sha256": "c" * 64},
            "train_pair_split": {"train_count": 3584, "train_per_translation": 56},
            "eval": {"shape_ids": list(range(32)), "row_count": n_eval},
        }
        return {
            "schema_kind": np.asarray("frozen_g64_gap_to_p6_v1"),
            "target_semantics": np.asarray("p6_normalized"),
            "coord_scale": np.asarray(223.0),
            "head_dim": np.asarray(6, dtype=np.int32),
            "train_gap_features": train_gap,
            "eval_gap_features": eval_gap,
            "train_p6_norm": train_p6,
            "train_q_px": train_q,
            "train_t_px": train_t,
            "eval_p6_norm": eval_p6,
            "eval_q_px": eval_q,
            "eval_t_px": eval_t,
            "train_shape_id": train_shape,
            "train_translation_id": train_tid,
            "eval_shape_id": eval_shape,
            "eval_dense_index": eval_dense,
            "train_ids": np.asarray([f"s{int(sid):02d}_t{int(tid):02d}" for sid, tid in zip(train_shape, train_tid)]),
            "eval_ids": np.asarray([f"s{int(sid):02d}_d{int(did):04d}" for sid, did in zip(eval_shape, eval_dense)]),
            "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True, separators=(",", ":"))),
        }

    def test_svd_cutoff_fit_handles_rank_deficiency(self) -> None:
        x = np.column_stack([np.arange(8.0), np.arange(8.0) * 2.0, np.ones(8)])
        y = np.column_stack([x[:, 0] + 1.0, -x[:, 0] + 2.0])
        result = fit_ols_svd(x, y, svd_cutoff=1e-10, fit_intercept=False)
        self.assertLessEqual(result["rank"], 2)
        self.assertIsNotNone(result["condition_number"])
        self.assertLess(result["prediction_mae"], 1e-8)

    def test_ridge_path_contains_standardized_and_whitened_predictions(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(30, 4))
        y = x @ np.array([[1.0, -2.0], [0.5, 0.25], [0.0, 1.0], [2.0, 0.0]])
        rows = ridge_path(x, y, lambdas=[0.0, 1e-3, 1.0], standardize=True, whiten=True)
        self.assertEqual([row["lambda"] for row in rows], [0.0, 1e-3, 1.0])
        self.assertTrue(all(np.isfinite(row["prediction_mae"]) for row in rows))
        self.assertTrue(all("effective_rank" in row for row in rows))

    def test_separated_schema_and_dense_eval_metrics_for_both_precisions(self) -> None:
        rng = np.random.default_rng(11)
        train_x = rng.normal(size=(12, 6)).astype(np.float32)
        train_y = np.column_stack(
            [
                train_x[:, 0] + 0.2 * train_x[:, 1],
                train_x[:, 1] - 0.1 * train_x[:, 2],
                train_x[:, 2] + 0.3 * train_x[:, 3],
                train_x[:, 3] - 0.2 * train_x[:, 4],
                train_x[:, 4] + 0.4 * train_x[:, 5],
                train_x[:, 5] - 0.5 * train_x[:, 0],
            ]
        ).astype(np.float32)
        eval_x = rng.normal(size=(80, 6)).astype(np.float32)
        eval_y = np.column_stack(
            [
                eval_x[:, 0] + 0.2 * eval_x[:, 1],
                eval_x[:, 1] - 0.1 * eval_x[:, 2],
                eval_x[:, 2] + 0.3 * eval_x[:, 3],
                eval_x[:, 3] - 0.2 * eval_x[:, 4],
                eval_x[:, 4] + 0.4 * eval_x[:, 5],
                eval_x[:, 5] - 0.5 * eval_x[:, 0],
            ]
        ).astype(np.float32)
        eval_q = np.zeros((80, 3, 2), dtype=np.float32)
        eval_t = np.zeros((80, 2), dtype=np.float32)
        result = audit_ols(
            train_x,
            train_y,
            eval_features=eval_x,
            eval_targets=eval_y,
            train_support_ids=np.arange(12),
            eval_support_ids=np.arange(12, 92),
            eval_q_px=eval_q,
            eval_t_px=eval_t,
            svd_cutoffs=[1e-12, 1e-8],
            ridge_lambdas=[0.0, 1e-5],
            perturbation_scales=[0.0, 1e-8],
            seed=3,
        )
        self.assertEqual(result["precision_modes"], ["float32", "float64"])
        for precision in result["precision_modes"]:
            branch = result["precision_path"][precision]
            self.assertEqual(len(branch["svd_path"]), 2)
            self.assertEqual(len(branch["ridge_path"]), 2)
            self.assertEqual(len(branch["perturbation_path"]), 2)
            self.assertIn("box_mae_px", branch["svd_path"][0]["eval_metrics"])
            self.assertIn("u_mae_px", branch["svd_path"][0]["eval_metrics"])
            self.assertIn("eval_box_mae_px", branch["svd_path"][0]["eval_metrics"])
            self.assertIn("p6_mae_norm", branch["svd_path"][0]["eval_metrics"])
        self.assertIn(result["stability"]["verdict"], {"stable", "unstable", "inconclusive"})

    def test_npz_requires_separated_train_and_eval_schema(self) -> None:
        old = self.root / "old_ols_head.npz"
        np.savez(old, W=np.zeros((2, 3)), b=np.zeros(2))
        with self.assertRaises(SchemaError):
            load_design_target_npz(old)

    def test_npz_audit_writes_dense_prediction_bundle(self) -> None:
        path = self.root / "design.npz"
        np.savez(path, **self._formal_payload())
        out = self.root / "results"
        result = audit_ols_npz(path, out, svd_cutoffs=[1e-12], ridge_lambdas=[0.0], perturbation_scales=[0.0])
        self.assertIn("precision_path", result)
        self.assertTrue((out / "asset_lock.json").exists())
        self.assertTrue((out / "predictions.npz").exists())
        with np.load(out / "predictions.npz", allow_pickle=False) as blob:
            self.assertIn("eval_true", blob.files)
            self.assertIn("eval_q_px", blob.files)
            self.assertIn("eval_t_px", blob.files)
            self.assertIn("eval_ids", blob.files)
            self.assertIn("pred_float64_svd_0", blob.files)
            self.assertIn("translation_error_px_float64_svd_0", blob.files)
            self.assertIn("shape_residual_px_float64_svd_0", blob.files)

    def test_translation_is_removed_before_box_metric(self) -> None:
        translation = np.asarray([12.0, -7.0])
        q = np.zeros((2, 3, 2), dtype=np.float64)
        pred_px = np.broadcast_to(translation, (2, 3, 2)).copy()
        pred = pred_px.reshape(2, 6) / 223.0
        target = np.zeros_like(pred)
        metrics = _metrics(pred, target, eval_q_px=q, eval_t_px=np.broadcast_to(translation, (2, 2)))
        self.assertGreater(metrics["p6_mae_px_elementwise"], 0.0)
        self.assertAlmostEqual(metrics["eval_box_mae_px"], 0.0, places=12)
        np.testing.assert_allclose(metrics["t_hat_px"], np.broadcast_to(translation, (2, 2)))

    def test_box_metric_uses_euclidean_3_4_5(self) -> None:
        # All three control points share the same [3,4] translation.  The
        # shape residual is zero, but the protocol translation error is 5 px.
        errors = np.asarray([[[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]])
        pred = errors.reshape(1, 6) / 223.0
        target = pred.copy()
        metrics = _metrics(pred, target, eval_q_px=np.zeros((1, 3, 2)), eval_t_px=np.zeros((1, 2)))
        self.assertAlmostEqual(metrics["eval_box_mae_px"], 5.0, places=12)
        self.assertAlmostEqual(metrics["eval_translation_mae_px"], 5.0, places=12)
        self.assertAlmostEqual(metrics["shape_residual_mae_px"], 0.0, places=12)

    def test_npz_rejects_nonfinite_shape_and_id_overlap(self) -> None:
        base = self._formal_payload()
        overlap = self.root / "overlap.npz"
        overlap_ids = np.array(base["eval_ids"], copy=True)
        overlap_ids[: base["train_ids"].size] = base["train_ids"]
        bad = dict(base, eval_ids=overlap_ids)
        np.savez(overlap, **bad)
        with self.assertRaises(SchemaError):
            load_design_target_npz(overlap)
        bad_shape = self.root / "bad_shape.npz"
        bad = dict(base, eval_q_px=np.zeros((base["eval_q_px"].shape[0], 2, 2)))
        np.savez(bad_shape, **bad)
        with self.assertRaises(SchemaError):
            load_design_target_npz(bad_shape)
        nonfinite = self.root / "nonfinite.npz"
        bad_eval = np.array(base["eval_p6_norm"], copy=True)
        bad_eval[0, 0] = np.nan
        bad = dict(base, eval_p6_norm=bad_eval)
        np.savez(nonfinite, **bad)
        with self.assertRaises(SchemaError):
            load_design_target_npz(nonfinite)

    def test_npz_rejects_wrong_formal_metadata(self) -> None:
        path = self.root / "wrong_metadata.npz"
        bad = self._formal_payload()
        bad["schema_kind"] = np.asarray("old_schema")
        np.savez(path, **bad)
        with self.assertRaises(SchemaError):
            load_design_target_npz(path)

    def test_formal_loader_requires_geometry_labels_and_frozen_pair_rows(self) -> None:
        base = self._formal_payload()
        missing_train_geometry = dict(base)
        missing_train_geometry.pop("train_q_px")
        path = self.root / "missing_train_geometry.npz"
        np.savez(path, **missing_train_geometry)
        with self.assertRaises(SchemaError):
            load_design_target_npz(path)

        wrong_label = dict(base)
        wrong_eval = np.array(wrong_label["eval_p6_norm"], copy=True)
        wrong_eval[0, 0] += 0.01
        wrong_label["eval_p6_norm"] = wrong_eval
        path = self.root / "wrong_label.npz"
        np.savez(path, **wrong_label)
        with self.assertRaises(SchemaError):
            load_design_target_npz(path)

        wrong_tid = dict(base)
        tids = np.array(wrong_tid["train_translation_id"], copy=True)
        tids[0] = 1
        wrong_tid["train_translation_id"] = tids
        path = self.root / "wrong_tid_count.npz"
        np.savez(path, **wrong_tid)
        with self.assertRaises(SchemaError):
            load_design_target_npz(path)

        wrong_eval_order = dict(base)
        dense = np.array(wrong_eval_order["eval_dense_index"], copy=True)
        dense[0], dense[1] = dense[1], dense[0]
        wrong_eval_order["eval_dense_index"] = dense
        path = self.root / "wrong_eval_order.npz"
        np.savez(path, **wrong_eval_order)
        with self.assertRaises(SchemaError):
            load_design_target_npz(path)

        generated = dict(base)
        generated.pop("train_ids")
        generated.pop("eval_ids")
        path = self.root / "generated_ids.npz"
        np.savez(path, **generated)
        _, _, _, _, meta = load_design_target_npz(path)
        self.assertEqual(meta["train_ids"].size, 3584)
        self.assertEqual(meta["eval_ids"].size, 32 * 41 * 41)


if __name__ == "__main__":
    unittest.main()
