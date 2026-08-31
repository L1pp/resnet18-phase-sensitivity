from __future__ import annotations

import shutil
import sys
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossover.protocol import load_protocol
from crossover.runner import prepare_assets, run_matrix
from crossover.assets import AssetValidationError, _read_package_identity, validate_assets


def _fixture_root() -> Path:
    root = ROOT / "tests" / "_assets_fixture"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    # This test intentionally exercises the real prepare contract.  It does
    # no training and is skipped by callers that do not have the corrected
    # exact-init handoff available.
    prepare_assets(root, workers=4)
    return root


def test_prepare_assets_validate_and_bundle_are_bound() -> None:
    source = Path(__file__).resolve().parents[2] / "handoff" / "20260823_crossover_inputs" / "neural_affine_autorun_20260822" / "assets" / "corrected_init"
    if not source.exists():
        return
    root = _fixture_root()
    try:
        result = validate_assets(root, load_protocol())
        assert result["status"] == "passed"
        assert len(result["records"]["streams"]) == 12
        assert len(result["records"]["exact_inits"]) == 3
        assert len(result["asset_bundle_sha256"]) == 64
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cache_or_stream_tamper_is_rejected_before_first_arm() -> None:
    source = Path(__file__).resolve().parents[2] / "handoff" / "20260823_crossover_inputs" / "neural_affine_autorun_20260822" / "assets" / "corrected_init"
    if not source.exists():
        return
    root = _fixture_root()
    try:
        cache_path = root / "cache" / "dense_points.npy"
        original = cache_path.read_bytes()
        cache_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        try:
            validate_assets(root, load_protocol())
        except AssetValidationError:
            pass
        else:
            raise AssertionError("tampered cache must be rejected")
        cache_path.write_bytes(original)
        stream_path = root / "batch_streams" / "G64" / "seed_20260816.npy"
        original_stream = stream_path.read_bytes()
        stream_path.write_bytes(original_stream[:-1] + bytes([original_stream[-1] ^ 1]))
        called = {"training": False}

        def forbidden_training(*args, **kwargs):
            called["training"] = True
            raise AssertionError("training arm must not start after asset tamper")

        try:
            with patch("crossover.runner.run_condition", side_effect=forbidden_training):
                run_matrix(root, device="cpu", steps=1, supports=("corners4",), seeds=(20260816,), regimes=("head_only",))
        except AssetValidationError:
            pass
        else:
            raise AssertionError("tampered stream must be rejected before matrix arm")
        assert called["training"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_package_identity_is_independent_of_asset_bundle_hash() -> None:
    root = ROOT / "tests" / "_package_identity_fixture" / "assets"
    if root.parent.exists():
        shutil.rmtree(root.parent, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        bundle = "a" * 64
        manifest = {"schema_version": 1, "package_id": "formal-crossover-test", "facts": {"asset_bundle_sha256": bundle}}
        manifest_path = root.parent / "PACKAGE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (root.parent / "PACKAGE_MANIFEST.json.sha256").write_text(f"{manifest_sha}  PACKAGE_MANIFEST.json\n", encoding="utf-8")
        identity, errors = _read_package_identity(root, bundle, require=True)
        assert not errors
        assert identity["package_id"] == "formal-crossover-test"
        assert identity["package_manifest_sha256"] == manifest_sha
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)
