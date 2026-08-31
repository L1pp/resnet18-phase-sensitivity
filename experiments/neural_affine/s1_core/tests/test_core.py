from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from s1clean.baselines import closed_form_identity
from s1clean.config import canonical_json_bytes
from s1clean.manifest import build_manifest
from s1clean.status import StatusStore


class CoreTests(unittest.TestCase):
    @contextmanager
    def workspace_temp(self):
        path = Path(__file__).resolve().parent / f"_scratch_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_closed_form_identity(self) -> None:
        support = np.array([[59, 59], [59, 164], [164, 59], [164, 164]], dtype=np.float64)
        dense = np.stack(np.meshgrid(np.linspace(59, 164, 11), np.linspace(59, 164, 11), indexing="ij"), -1).reshape(-1, 2)
        result = closed_form_identity(support, dense)
        self.assertLessEqual(result["raw_mae_px"], 1e-10)
        self.assertLessEqual(result["affine_residual_mae_px"], 1e-10)

    def test_status_is_atomic_json(self) -> None:
        with self.workspace_temp() as temp:
            store = StatusStore(Path(temp))
            store.update("TEST", value=3)
            payload = json.loads((Path(temp) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "TEST")
            self.assertEqual(payload["value"], 3)
            self.assertTrue((Path(temp) / "TEST").exists())

    def test_status_rotates_known_current_marker(self) -> None:
        with self.workspace_temp() as temp:
            root = Path(temp)
            store = StatusStore(root)
            store.update("READY_FOR_GPU")
            store.update("NO_GO", reason="test")
            self.assertFalse((root / "READY_FOR_GPU").exists())
            self.assertTrue((root / "NO_GO").exists())
            self.assertEqual(store.read()["state"], "NO_GO")

    def test_manifest_ignores_outputs(self) -> None:
        with self.workspace_temp() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "outputs").mkdir()
            (root / "outputs" / "ignored.txt").write_text("x", encoding="utf-8")
            payload = build_manifest(root)
            self.assertEqual([row["path"] for row in payload["files"]], ["a.txt"])

    def test_canonical_json(self) -> None:
        self.assertEqual(canonical_json_bytes({"b": 1, "a": 2}), b'{"a":2,"b":1}')


if __name__ == "__main__":
    unittest.main()
