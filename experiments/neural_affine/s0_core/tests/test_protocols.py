from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid
import unittest

from closeout_sprint.protocols import (
    AssetManifestError,
    ProtocolChangedError,
    StageError,
    load_asset_manifest,
    load_protocol,
    reject_training,
    validate_device,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / f".test_protocol_{uuid.uuid4().hex}"
        self.root.mkdir()
        self.asset = self.root / "asset.bin"
        self.asset.write_bytes(b"locked asset")
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol_id": "synthetic",
                    "current_stage": "local_s0",
                    "allowed_devices": ["cpu", "cuda:0"],
                    "asset_manifest": [
                        {
                            "id": "asset",
                            "path": str(self.asset),
                            "sha256": hashlib.sha256(self.asset.read_bytes()).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_asset_hash_is_rechecked_after_manifest_load(self) -> None:
        protocol = load_protocol(self.protocol_path)
        manifest = load_asset_manifest(protocol)
        self.asset.write_bytes(b"changed asset")
        with self.assertRaises(ProtocolChangedError):
            manifest.assert_unchanged()

    def test_unpinned_assets_are_rejected_for_formal_runs(self) -> None:
        protocol = load_protocol(self.protocol_path)
        with self.assertRaises(AssetManifestError):
            load_asset_manifest(
                protocol,
                entries=[{"id": "unlocked", "path": str(self.asset)}],
            )

    def test_local_s0_device_gate_and_training_guard(self) -> None:
        protocol = load_protocol(self.protocol_path)
        self.assertEqual(validate_device(protocol, "cuda"), "cuda:0")
        with self.assertRaises(StageError):
            validate_device(protocol, "cuda:1")
        with self.assertRaises(StageError):
            reject_training(protocol, self.root / "approved.json")

    def test_remote_cpu_preflight_overrides_a_misconfigured_cuda_allowlist(self) -> None:
        payload = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        payload["current_stage"] = "remote_cpu_preflight"
        payload["allowed_devices"] = ["cpu", "cuda:0"]
        remote = self.root / "remote.json"
        remote.write_text(json.dumps(payload), encoding="utf-8")
        protocol = load_protocol(remote)
        with self.assertRaises(StageError):
            validate_device(protocol, "cuda:0")


if __name__ == "__main__":
    unittest.main()
