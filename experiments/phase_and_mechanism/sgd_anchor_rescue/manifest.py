"""Cloud-agnostic manifest schema and deterministic data-order fingerprints."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import (
    CANDIDATES,
    BATCH_SIZE,
    ETA_MIN,
    LOW_LR_PEAK_LR,
    LOW_LR_WARMUP_STEPS,
    PEAK_LR,
    PROTOCOL_NAME,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    TOTAL_STEPS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from .guard import validate_anchor_only_record, find_forbidden_stage1_keys


MANIFEST_SCHEMA_VERSION = "1.0"
REQUIRED_MANIFEST_FIELDS = (
    "protocol",
    "protocol_version",
    "status",
    "stage",
    "candidate",
    "seed",
    "batch_size",
    "weight_decay",
    "loss",
    "optimizer",
    "schedule",
    "theta0",
    "anchors",
    "data_order_fingerprint",
)


def manifest_schema() -> dict[str, Any]:
    """Return a JSON-Schema-like description for local/remote runners."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{PROTOCOL_NAME}/manifest/{MANIFEST_SCHEMA_VERSION}",
        "type": "object",
        "required": list(REQUIRED_MANIFEST_FIELDS),
        "properties": {
            "protocol": {"const": PROTOCOL_NAME},
            "protocol_version": {"type": "string"},
            "status": {"type": "string"},
            "stage": {"enum": ["prepare", "anchor_only", "dense_reveal"]},
            "candidate": {"enum": ["A", "B", "low_lr_restart", None]},
            "seed": {"type": "integer"},
            "batch_size": {"const": BATCH_SIZE},
            "weight_decay": {"const": WEIGHT_DECAY},
            "loss": {"type": "string"},
            "optimizer": {"type": "object"},
            "schedule": {"type": "object"},
            "theta0": {"type": "object"},
            "anchors": {"type": "object"},
            "data_order_fingerprint": {"type": "string"},
            "device": {"type": "string"},
            "dimensions": {"type": "integer"},
            "code_hash": {"type": "string"},
            "dependency_lock_hash": {"type": "string"},
            "output_directory": {"type": "string"},
            "notes": {"type": "string"},
            "anchor_only_guard": {"type": "boolean"},
            "stage1_records": {"type": "array"},
        },
        "additionalProperties": True,
    }


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalise_order_value(value: Any) -> Any:
    if isinstance(value, bool):
        raise TypeError("data-order indices cannot be bool")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError):
        pass
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("data-order values must be integer indices or nested batches")
    # numpy arrays/tensors expose tolist without forcing this package to import
    # numpy or torch.  A list result is recursively normalized below.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _normalise_order_value(tolist())
    if isinstance(value, Mapping):
        raise TypeError("data-order values cannot be mappings")
    try:
        return [_normalise_order_value(item) for item in value]
    except TypeError as exc:
        raise TypeError("data-order values must be integer indices or nested batches") from exc


def normalize_data_order(order: Iterable[Any]) -> list[Any]:
    """Materialize and normalize a batch-index stream without global RNG state."""

    if isinstance(order, (str, bytes, bytearray, Mapping)):
        raise TypeError("order must be an iterable of integer indices/batches")
    return [_normalise_order_value(item) for item in order]


def fingerprint_data_order(order: Iterable[Any], *, algorithm: str = "sha256") -> str:
    """Hash exact order and batch boundaries into a reproducible fingerprint."""

    normalized = normalize_data_order(order)
    payload = canonical_json(
        {
            "fingerprint_schema": "sgd_anchor_rescue.data_order.v1",
            "order": normalized,
        }
    ).encode("utf-8")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"unsupported hash algorithm: {algorithm}") from exc
    digest.update(payload)
    return digest.hexdigest()


# A shorter alias is useful in runner code and makes the intent explicit.
data_order_fingerprint = fingerprint_data_order


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local file without importing or depending on a cloud runtime."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_optimizer_schedule(candidate: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the protocol-frozen optimizer/schedule pair for a run label."""

    if candidate in CANDIDATES:
        candidate_config = CANDIDATES[candidate]
        peak_lr = candidate_config.peak_lr
        warmup_steps = candidate_config.warmup_steps
        momentum = candidate_config.momentum
    elif candidate == "low_lr_restart":
        # The one-shot low-LR branch retains the non-momentum SGD baseline;
        # only its peak LR and warmup are changed by the protocol.
        peak_lr = LOW_LR_PEAK_LR
        warmup_steps = LOW_LR_WARMUP_STEPS
        momentum = 0.0
    else:
        raise ValueError("candidate must be A, B, or low_lr_restart")
    optimizer = {
        "name": "SGD",
        "momentum": momentum,
        "dampening": 0.0,
        "nesterov": False,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "amp": False,
    }
    schedule = {
        "kind": "linear_warmup_cosine",
        "peak_lr": peak_lr,
        "warmup_steps": warmup_steps,
        "total_steps": TOTAL_STEPS,
        "eta_min": ETA_MIN,
        "extension_steps": TOTAL_STEPS if candidate == "low_lr_restart" else 40_000,
    }
    return optimizer, schedule


def _field_mismatches(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, prefix: str
) -> list[str]:
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"{prefix}.{key} is missing")
        elif actual[key] != expected_value:
            errors.append(
                f"{prefix}.{key} must be {expected_value!r}, got {actual[key]!r}"
            )
    return errors


def build_manifest(
    *,
    stage: str,
    candidate: str | None,
    seed: int,
    theta0: Mapping[str, Any],
    anchors: Mapping[str, Any],
    data_order_fingerprint: str,
    device: str | None = None,
    dimensions: int | None = None,
    status: str = PROTOCOL_STATUS,
    optimizer: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    loss: str = "MSE + 0.25 * L1",
    output_directory: str | None = None,
    code_hash: str | None = None,
    dependency_lock_hash: str | None = None,
    notes: str | None = None,
    anchor_only_guard: bool | None = None,
    stage1_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a manifest using only experiment-level, not cloud-specific, data."""

    if candidate in {"A", "B", "low_lr_restart"}:
        expected_optimizer, expected_schedule = expected_optimizer_schedule(candidate)
    else:
        expected_optimizer, expected_schedule = {}, {}
    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "stage": stage,
        "candidate": candidate,
        "seed": int(seed),
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "loss": loss,
        "optimizer": dict(expected_optimizer if optimizer is None else optimizer),
        "schedule": dict(expected_schedule if schedule is None else schedule),
        "theta0": dict(theta0),
        "anchors": dict(anchors),
        "data_order_fingerprint": str(data_order_fingerprint),
    }
    optional = {
        "device": device,
        "dimensions": dimensions,
        "output_directory": output_directory,
        "code_hash": code_hash,
        "dependency_lock_hash": dependency_lock_hash,
        "notes": notes,
    }
    manifest.update({key: value for key, value in optional.items() if value is not None})
    if stage == "anchor_only":
        guard_used = True if anchor_only_guard is None else bool(anchor_only_guard)
        manifest["anchor_only_guard"] = guard_used
        if stage1_records is not None:
            guarded_records: list[dict[str, Any]] = []
            for record in stage1_records:
                validate_anchor_only_record(record)
                guarded_records.append(dict(record))
            manifest["stage1_records"] = guarded_records
    elif anchor_only_guard is not None:
        manifest["anchor_only_guard"] = bool(anchor_only_guard)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic validation errors; an empty tuple means valid."""

    errors: list[str] = []
    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in manifest:
            errors.append(f"missing field: {key}")
    if manifest.get("protocol") != PROTOCOL_NAME:
        errors.append(f"protocol must be {PROTOCOL_NAME}")
    if "stage" in manifest and manifest["stage"] not in {"prepare", "anchor_only", "dense_reveal"}:
        errors.append("stage must be prepare, anchor_only, or dense_reveal")
    if "candidate" in manifest and manifest["candidate"] not in {"A", "B", "low_lr_restart", None}:
        errors.append("candidate must be A, B, low_lr_restart, or null")
    if "seed" in manifest and (isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int)):
        errors.append("seed must be an integer")
    if "batch_size" in manifest and manifest["batch_size"] != BATCH_SIZE:
        errors.append(f"batch_size must be {BATCH_SIZE}")
    if "weight_decay" in manifest and manifest["weight_decay"] != WEIGHT_DECAY:
        errors.append(f"weight_decay must be {WEIGHT_DECAY}")
    for key in ("optimizer", "schedule", "theta0", "anchors"):
        if key in manifest and not isinstance(manifest[key], Mapping):
            errors.append(f"{key} must be an object")
    if "data_order_fingerprint" in manifest:
        value = manifest["data_order_fingerprint"]
        if not isinstance(value, str) or not value:
            errors.append("data_order_fingerprint must be a non-empty string")
    candidate = manifest.get("candidate")
    if candidate in {"A", "B", "low_lr_restart"}:
        optimizer = manifest.get("optimizer")
        schedule = manifest.get("schedule")
        if isinstance(optimizer, Mapping) and isinstance(schedule, Mapping):
            expected_optimizer, expected_schedule = expected_optimizer_schedule(candidate)
            errors.extend(_field_mismatches(optimizer, expected_optimizer, prefix="optimizer"))
            errors.extend(_field_mismatches(schedule, expected_schedule, prefix="schedule"))
        if manifest.get("loss") != "MSE + 0.25 * L1":
            errors.append("loss must be 'MSE + 0.25 * L1'")
    if manifest.get("stage") == "anchor_only":
        if manifest.get("anchor_only_guard") is not True:
            errors.append("anchor_only_guard must be true for anchor_only manifests")
        forbidden = find_forbidden_stage1_keys(manifest)
        if forbidden:
            errors.append("anchor_only contains forbidden stage-1 key(s): " + ", ".join(forbidden))
        records = manifest.get("stage1_records")
        if records is not None:
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
                errors.append("stage1_records must be an array")
            else:
                for index, record in enumerate(records):
                    try:
                        validate_anchor_only_record(record)
                    except (TypeError, ValueError) as exc:
                        errors.append(f"stage1_records[{index}] invalid: {exc}")
    return tuple(errors)


SHARED_MANIFEST_FIELDS = (
    "protocol",
    "protocol_version",
    "status",
    "stage",
    "seed",
    "batch_size",
    "weight_decay",
    "loss",
    "theta0",
    "anchors",
    "data_order_fingerprint",
)


def compare_shared_manifests(
    manifest_a: Mapping[str, Any], manifest_b: Mapping[str, Any]
) -> tuple[str, ...]:
    """Check that A/B share data, seed, theta0, and order identity."""

    errors: list[str] = []
    for label, manifest in (("A", manifest_a), ("B", manifest_b)):
        errors.extend(f"{label}: {error}" for error in validate_manifest(manifest))
    if manifest_a.get("candidate") != "A":
        errors.append("first manifest candidate must be A")
    if manifest_b.get("candidate") != "B":
        errors.append("second manifest candidate must be B")
    for key in SHARED_MANIFEST_FIELDS:
        if manifest_a.get(key) != manifest_b.get(key):
            errors.append(f"shared field mismatch: {key}")
    if manifest_a.get("data_order_fingerprint") != manifest_b.get("data_order_fingerprint"):
        errors.append("data_order_fingerprint mismatch")
    return tuple(dict.fromkeys(errors))


def assert_shared_manifests(manifest_a: Mapping[str, Any], manifest_b: Mapping[str, Any]) -> None:
    errors = compare_shared_manifests(manifest_a, manifest_b)
    if errors:
        raise ValueError("incompatible A/B manifests: " + "; ".join(errors))


def assert_valid_manifest(manifest: Mapping[str, Any]) -> None:
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "REQUIRED_MANIFEST_FIELDS",
    "assert_valid_manifest",
    "assert_shared_manifests",
    "build_manifest",
    "canonical_json",
    "data_order_fingerprint",
    "fingerprint_data_order",
    "manifest_schema",
    "compare_shared_manifests",
    "expected_optimizer_schedule",
    "normalize_data_order",
    "sha256_file",
    "validate_manifest",
]
