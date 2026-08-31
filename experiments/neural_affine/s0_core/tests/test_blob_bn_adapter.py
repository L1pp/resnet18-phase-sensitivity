from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import unittest

import numpy as np

from closeout_sprint.blob_bn_adapter import (
    BlobAdapterError,
    AppearanceEvalStream,
    load_appearance_set,
    validate_train_eval_appearance_sets,
    render_evaluation_appearances,
    render_training_appearances,
    CORNERS4_TIDS,
    corners4_coordinates,
    corners4_pack,
    dense_pack,
    integer_grid_px,
    inspect_resnet18_2d_checkpoint,
    make_exposure_batches,
    render_blob_at,
    uint8_to_nchw,
)


class BlobRendererTests(unittest.TestCase):
    def test_renderer_is_repeatable_and_bounded(self) -> None:
        centers = np.asarray([[59.0, 59.0], [111.5, 130.25], [164.0, 164.0]])
        one = render_blob_at(centers)
        two = render_blob_at(centers)
        self.assertEqual(one["images"].shape, (3, 224, 224))
        self.assertEqual(one["images"].dtype, np.uint8)
        self.assertGreaterEqual(int(one["images"].min()), 0)
        self.assertLessEqual(int(one["images"].max()), 255)
        np.testing.assert_array_equal(one["images"], two["images"])
        np.testing.assert_allclose(one["true_px"], centers)

    def test_frozen_support_coordinates_and_dense_shape(self) -> None:
        expected = np.asarray([[59.0, 59.0], [59.0, 164.0], [164.0, 59.0], [164.0, 164.0]])
        np.testing.assert_allclose(corners4_coordinates(), expected)
        np.testing.assert_array_equal(corners4_pack()["tids"], np.asarray(CORNERS4_TIDS))
        dense = dense_pack(n_grid=5)
        self.assertEqual(dense["images"].shape, (5 * 5, 224, 224))
        self.assertEqual(dense["true_px"].shape, (5 * 5, 2))
        self.assertEqual(integer_grid_px().shape, (64, 2))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
    def test_batch_stream_and_tensor_boundary(self) -> None:
        stream = iter(make_exposure_batches(max_exposure=2, batch_size=3))
        batch = next(stream)
        self.assertEqual(tuple(batch.shape), (3, 3, 224, 224))
        self.assertEqual(str(batch.dtype), "torch.float32")
        self.assertGreaterEqual(float(batch.min()), 0.0)
        self.assertLessEqual(float(batch.max()), 1.0)
        self.assertEqual(tuple(uint8_to_nchw(np.zeros((2, 16, 16), dtype=np.uint8)).shape), (2, 3, 16, 16))

    def test_hash_pinned_train_eval_appearances_are_disjoint(self) -> None:
        root = Path.cwd() / f".test_blob_assets_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            train_payload = {"appearances": [{"id": f"tr{i}", "sigma": 4.0 + i * 0.1} for i in range(3)]}
            eval_payload = {"appearances": [{"id": f"ev{i}", "sigma": 8.0 + i * 0.1} for i in range(2)]}
            train_path = root / "appearances.json"
            eval_path = root / "eval_appearances.json"
            train_path.write_text(json.dumps(train_payload), encoding="utf-8")
            eval_path.write_text(json.dumps(eval_payload), encoding="utf-8")
            train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
            eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
            train = load_appearance_set(train_path, expected_sha256=train_sha, role="train", expected_count=3)
            evaluation = load_appearance_set(eval_path, expected_sha256=eval_sha, role="eval", expected_count=2)
            split = validate_train_eval_appearance_sets(train, evaluation, expected_train_count=3, expected_eval_count=2)
            self.assertTrue(split["disjoint_ids"])
            self.assertTrue(split["disjoint_sigmas"])
            self.assertEqual(render_training_appearances(train, image_size=32)["images"].shape, (12, 32, 32))
            self.assertEqual(render_evaluation_appearances(evaluation, n_grid=2, image_size=32)["images"].shape, (8, 32, 32))
            lazy = AppearanceEvalStream(evaluation, n_grid=2, image_size=32)
            self.assertTrue(lazy.metadata()["lazy"])
            self.assertEqual(len(lazy), 8)
            with self.assertRaises(BlobAdapterError):
                load_appearance_set(train_path, expected_sha256="0" * 64, role="train", expected_count=3)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_relative_appearance_path_requires_manifest_base(self) -> None:
        with self.assertRaises(BlobAdapterError):
            load_appearance_set("appearances.json", expected_sha256="0" * 64, role="train")

    def test_parallel_appearance_ids_and_sigmas_must_have_equal_lengths(self) -> None:
        root = Path.cwd() / f".test_blob_parallel_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            path = root / "appearances.json"
            path.write_text(
                json.dumps(
                    {
                        "appearance_ids": [f"a{i}" for i in range(48)],
                        "sigmas": [4.0 + i * 0.1 for i in range(49)],
                    }
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(BlobAdapterError):
                load_appearance_set(path, expected_sha256=digest, role="train")
        finally:
            shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class CheckpointBoundaryTests(unittest.TestCase):
    def test_6d_checkpoint_is_rejected_by_2d_loader(self) -> None:
        root = Path(__file__).resolve().parents[2]
        path = root / "ckpts" / "phase2" / "resnet18__G64__20260820" / "checkpoints" / "best_slim.pt"
        if not path.is_file():
            self.skipTest("local 6D checkpoint is unavailable")
        with self.assertRaises(BlobAdapterError):
            inspect_resnet18_2d_checkpoint(path)

    def test_2d_checkpoint_inspection_if_available(self) -> None:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "results" / "error_extension" / "amd" / "unfreeze_blob_g64_corners4" / "l4" / "best_slim.pt",
            root / "results" / "phase3_high_upside_discovery" / "track_c" / "models" / "lofo_blob__20260830" / "checkpoints" / "best_slim.pt",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            self.skipTest("local 2D checkpoint is unavailable")
        info = inspect_resnet18_2d_checkpoint(path)
        self.assertEqual(info["head_dim"], 2)
        self.assertEqual(info["head_in"], 512)


if __name__ == "__main__":
    unittest.main()
