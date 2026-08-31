from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossover.aggregate import aggregate_runs
from crossover.protocol import SUPPORT_NAMES, load_protocol, protocol_hash


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(root: Path, *, package_identity: dict | None = None) -> None:
    cfg = load_protocol()
    p_hash = protocol_hash(payload=cfg)
    bundle = "a" * 64
    _write(root / "formal_asset_validation.json", {
        "status": "passed",
        "protocol_hash": p_hash,
        "asset_bundle_sha256": bundle,
        "package_identity": package_identity,
        "records": {"exact_inits": [
            {"seed": seed, "init_hash": f"init-{seed}", "sha256": f"file-{seed}", "source_protocol_hash": "8f8e47b6947f393cd4e246e30ff2a97c0190f76baf053ba45d3ff83d68ec05d3"}
            for seed in (20260816, 20260817, 20260818)
        ]},
    })
    for support in SUPPORT_NAMES:
        for seed in (20260816, 20260817, 20260818):
            init_hash = f"init-{seed}"
            stream_hash = f"stream-{support}-{seed}"
            for regime in ("frozen_feature", "head_only", "full"):
                steps = 0 if regime == "frozen_feature" else 3000
                run = root / "formal_runs" / support / str(seed) / regime
                summary = {
                    "schema_version": 1,
                    "protocol_id": cfg["protocol_id"],
                    "protocol_hash": p_hash,
                    "package_id": "neural_affine_crossover_clean" if package_identity is None else package_identity["package_id"],
                    "package_manifest_sha256": None if package_identity is None else package_identity["package_manifest_sha256"],
                    "asset_bundle_sha256": bundle,
                    "support": support,
                    "support_count": len(cfg["supports"][support]),
                    "seed": seed,
                    "regime": regime,
                    "status": "completed",
                    "steps_completed": steps,
                    "support_mae_px": 0.0,
                    "init_hash": init_hash,
                    "init_state_hash": init_hash,
                    "init_source_protocol_hash": "8f8e47b6947f393cd4e246e30ff2a97c0190f76baf053ba45d3ff83d68ec05d3",
                    "init_source_file_sha256": f"file-{seed}",
                    "stream_digest": stream_hash,
                }
                evaluator = {
                    "protocol_hash": p_hash,
                    "asset_bundle_sha256": bundle,
                    "package_id": summary["package_id"],
                    "package_manifest_sha256": summary["package_manifest_sha256"],
                    "status": "passed",
                    "metrics": {"raw_mae_px": 1.0, "affine_removed_mae_px": 1.0},
                    "backbone": {
                        "head_only_bitwise_unchanged": True if regime == "head_only" else None,
                        "full_backbone_changed": True if regime == "full" else None,
                        "full_backbone_parameter_changed": True if regime == "full" else None,
                    },
                }
                receipt = {
                    "support_input_hash": f"support-{support}",
                    "asset_bundle_sha256": bundle,
                    "package_id": summary["package_id"],
                    "package_manifest_sha256": summary["package_manifest_sha256"],
                    "init_state_hash": init_hash,
                    "init_source_protocol_hash": summary["init_source_protocol_hash"],
                    "init_source_file_sha256": summary["init_source_file_sha256"],
                }
                _write(run / "summary.json", summary)
                _write(run / "independent_evaluator.json", evaluator)
                _write(run / "receipt.json", receipt)


def test_pair_mismatch_cannot_produce_scientific_decision() -> None:
    root = ROOT / "tests" / "_aggregate_pair_fixture"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        _fixture(root)
        path = root / "formal_runs" / "G64" / "20260816" / "full" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["stream_digest"] = "tampered-stream"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = aggregate_runs(root)
        assert result["global_invalid"] is True
        assert result["decision"] in {"incomplete", "invalid_protocol"}
        assert result["decision"] not in {"stable_crossover", "catch_up", "no_stable_crossover"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_package_identity_mismatch_is_invalid_protocol() -> None:
    root = ROOT / "tests" / "_aggregate_package_fixture"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        identity = {"package_id": "formal-package", "package_manifest_sha256": "b" * 64}
        _fixture(root, package_identity=identity)
        path = root / "formal_runs" / "corners4" / "20260816" / "head_only" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["package_id"] = "wrong-package"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = aggregate_runs(root)
        assert result["global_invalid"] is True
        assert result["decision"] == "invalid_protocol"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_triplet_consistent_but_wrong_init_is_rejected() -> None:
    root = ROOT / "tests" / "_aggregate_init_fixture"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        _fixture(root)
        for regime in ("frozen_feature", "head_only", "full"):
            path = root / "formal_runs" / "G16" / "20260817" / regime / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["init_hash"] = "wrong-init"
            summary["init_state_hash"] = "wrong-init"
            path.write_text(json.dumps(summary), encoding="utf-8")
        result = aggregate_runs(root)
        assert result["global_invalid"] is True
        assert result["decision"] in {"invalid_protocol", "incomplete"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_failed_asset_validation_blocks_self_consistent_cells() -> None:
    root = ROOT / "tests" / "_aggregate_failed_asset_fixture"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        _fixture(root)
        validation_path = root / "formal_asset_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["status"] = "failed"
        validation["errors"] = ["tampered stream"]
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        result = aggregate_runs(root)
        assert result["global_invalid"] is True
        assert result["decision"] == "invalid_protocol"
    finally:
        shutil.rmtree(root, ignore_errors=True)
