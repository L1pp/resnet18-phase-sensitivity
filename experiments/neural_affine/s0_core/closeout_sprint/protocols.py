"""Protocol and provenance locks for the clean-room S0 runners.

The scientific audit modules intentionally remain small numerical libraries.
This module supplies the execution boundary around them: a protocol is read
once, hashed, and checked again before a result is committed; assets are
resolved from an explicit manifest and their hashes/schema are verified before
any audit starts.

No historical experiment module is imported here.  In particular, protocol
files are data, not Python code, and adapter specifications are never passed
to ``eval``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .assets import asset_record, sha256_file
from .contracts import SCHEMA_VERSION, json_safe, read_json, write_json


class ProtocolError(ValueError):
    """The immutable protocol or its manifest is malformed."""


class ProtocolChangedError(ProtocolError):
    """A lock file changed while a run was in progress."""


class StageError(ProtocolError):
    """The requested operation is outside the current execution stage."""


class AssetManifestError(ProtocolError):
    """An asset is missing, unpinned, changed, or has the wrong schema."""


DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "protocols" / "s0_v1.json"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a deterministic hash for an in-memory JSON value."""

    return sha256_bytes(_canonical_bytes(value))


@dataclass(frozen=True)
class ProtocolLock:
    """A protocol payload plus the bytes hash used for the run."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def assert_unchanged(self) -> None:
        current = sha256_file(self.path)
        if current != self.sha256:
            raise ProtocolChangedError(
                f"protocol lock changed during run: {self.path} "
                f"({self.sha256} -> {current})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AssetSpec:
    """One manifest entry resolved to an absolute path."""

    asset_id: str
    path: Path
    role: str | None
    expected_sha256: str | None
    source_machine: str
    group: str | None = None
    expected_shape: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.asset_id,
            "path": str(self.path),
            "role": self.role,
            "sha256": self.expected_sha256,
            "source_machine": self.source_machine,
        }
        if self.group is not None:
            result["group"] = self.group
        if self.expected_shape is not None:
            result["shape"] = list(self.expected_shape)
        return result


@dataclass(frozen=True)
class AssetManifest:
    """Validated manifest and the source JSON lock (when one was supplied)."""

    specs: tuple[AssetSpec, ...]
    records: tuple[Mapping[str, Any], ...]
    source_path: Path | None
    sha256: str

    def by_id(self, asset_id: str) -> AssetSpec:
        for spec in self.specs:
            if spec.asset_id == asset_id:
                return spec
        raise AssetManifestError(f"asset id not found in manifest: {asset_id}")

    def matching(self, *, roles: Iterable[str] = (), group: str | None = None) -> tuple[AssetSpec, ...]:
        role_set = {str(role) for role in roles}
        values = tuple(
            spec
            for spec in self.specs
            if (not role_set or spec.role in role_set)
            and (group is None or spec.group == group)
        )
        return values

    def assert_unchanged(self) -> None:
        if self.source_path is not None:
            current = sha256_file(self.source_path)
            if current != self.sha256:
                raise ProtocolChangedError(
                    f"asset manifest changed during run: {self.source_path} "
                    f"({self.sha256} -> {current})"
                )

        # A manifest hash only protects the list of paths and expected
        # digests.  Re-hash every referenced file as well so a file modified
        # after the initial preflight cannot silently enter a formal result.
        for spec in self.specs:
            if not spec.expected_sha256:
                raise ProtocolChangedError(f"asset is not hash-pinned: {spec.asset_id}")
            if not spec.path.is_file():
                raise ProtocolChangedError(f"asset disappeared during run: {spec.path}")
            current = sha256_file(spec.path)
            if current != spec.expected_sha256:
                raise ProtocolChangedError(
                    f"asset changed during run: {spec.asset_id} "
                    f"({spec.expected_sha256} -> {current})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "clean_room_asset_manifest",
            "sha256": self.sha256,
            "source_path": None if self.source_path is None else str(self.source_path),
            "assets": [dict(record) for record in self.records],
        }


def load_protocol(path: Path | str | None = None) -> ProtocolLock:
    """Read and validate an immutable JSON protocol lock."""

    protocol_path = Path(path or DEFAULT_PROTOCOL_PATH).expanduser().resolve()
    if not protocol_path.is_file():
        raise ProtocolError(f"protocol lock does not exist: {protocol_path}")
    raw = protocol_path.read_bytes()
    digest = sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolError(f"invalid protocol JSON: {protocol_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("protocol lock must contain a JSON object")
    stage = payload.get("current_stage", payload.get("stage"))
    if str(stage).lower() not in {"cpu", "local_s0", "remote_cpu_preflight"}:
        raise StageError(
            "clean-room runner only accepts CPU preparation stages "
            f"(cpu/local_s0/remote_cpu_preflight), got {stage!r}"
        )
    if str(stage).lower() == "local_s0" and str(payload.get("execution_target", "local")).lower() != "local":
        raise StageError("local_s0 protocol must bind execution_target=local")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ProtocolError(
            f"unsupported protocol schema_version={payload.get('schema_version')!r}"
        )
    if not payload.get("protocol_id") and not payload.get("id"):
        raise ProtocolError("protocol lock requires protocol_id")
    return ProtocolLock(path=protocol_path, sha256=digest, payload=payload)


def _manifest_entries(raw: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw, Mapping):
        entries = raw.get("assets", raw.get("manifest"))
    else:
        entries = raw
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise AssetManifestError("asset manifest must contain an 'assets' list")
    result: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise AssetManifestError(f"asset manifest entry {index} is not an object")
        result.append(entry)
    if not result:
        raise AssetManifestError("asset manifest cannot be empty")
    return tuple(result)


def _resolve_manifest_path(value: Any, *, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise AssetManifestError("asset manifest entry requires a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _entry_shape(entry: Mapping[str, Any]) -> tuple[int, ...] | None:
    shape = entry.get("shape", entry.get("expected_shape"))
    if shape is None:
        return None
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise AssetManifestError(f"asset shape must be a sequence: {entry}")
    try:
        return tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise AssetManifestError(f"asset shape contains a non-integer: {entry}") from exc


def _specs_from_entries(entries: Sequence[Mapping[str, Any]], *, base_dir: Path) -> tuple[AssetSpec, ...]:
    specs: list[AssetSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        asset_id = str(entry.get("id", entry.get("asset_id", f"asset_{index}"))).strip()
        if not asset_id:
            raise AssetManifestError(f"asset manifest entry {index} has an empty id")
        if asset_id in seen:
            raise AssetManifestError(f"duplicate asset id: {asset_id}")
        seen.add(asset_id)
        path = _resolve_manifest_path(entry.get("path"), base_dir=base_dir)
        expected = entry.get("sha256", entry.get("expected_sha256"))
        expected_sha256 = None if expected in (None, "") else str(expected).lower()
        if expected_sha256 is not None and (len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise AssetManifestError(f"asset {asset_id} has invalid sha256")
        specs.append(
            AssetSpec(
                asset_id=asset_id,
                path=path,
                role=None if entry.get("role") is None else str(entry.get("role")),
                expected_sha256=expected_sha256,
                source_machine=str(entry.get("source_machine", "local")),
                group=None if entry.get("group") is None else str(entry.get("group")),
                expected_shape=_entry_shape(entry),
            )
        )
    return tuple(specs)


def _validate_spec(spec: AssetSpec, *, require_hash: bool) -> Mapping[str, Any]:
    if require_hash and not spec.expected_sha256:
        raise AssetManifestError(f"asset {spec.asset_id} is not hash-pinned")
    try:
        record = asset_record(spec.path, role=spec.role, source_machine=spec.source_machine)
    except (OSError, ValueError) as exc:
        raise AssetManifestError(f"asset {spec.asset_id} failed validation: {exc}") from exc
    actual = str(record.get("sha256", "")).lower()
    if spec.expected_sha256 and actual != spec.expected_sha256:
        raise AssetManifestError(
            f"asset {spec.asset_id} sha256 mismatch: expected {spec.expected_sha256}, got {actual}"
        )
    if spec.expected_shape is not None:
        shapes = record.get("npz_shapes", {})
        candidate = shapes.get("pred") if "pred" in shapes else shapes.get("features")
        if candidate is None and record.get("head_out") is not None:
            candidate = [record["head_out"]]
        if candidate is None or tuple(int(value) for value in candidate) != spec.expected_shape:
            raise AssetManifestError(
                f"asset {spec.asset_id} shape mismatch: expected {spec.expected_shape}, got {candidate}"
            )
    return record


def load_asset_manifest(
    protocol: ProtocolLock,
    path: Path | str | None = None,
    *,
    entries: Sequence[Mapping[str, Any]] | None = None,
    require_hash: bool = True,
) -> AssetManifest:
    """Load, hash, and validate an explicit asset manifest.

    If ``path`` is omitted, the protocol's ``asset_manifest``/``assets`` list
    is used.  Callers supplying command-line assets may pass ``entries``;
    those entries are still immediately hash-pinned by the returned lock.
    """

    if path is not None and entries is not None:
        raise AssetManifestError("pass either an asset manifest path or entries, not both")
    source_path: Path | None = None
    if path is not None:
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise AssetManifestError(f"asset manifest does not exist: {source_path}")
        raw = read_json(source_path)
        raw_entries = _manifest_entries(raw)
        base_dir = source_path.parent
        source_hash = sha256_file(source_path)
    elif entries is not None:
        raw_entries = _manifest_entries(entries)
        base_dir = protocol.path.parent
        source_hash = sha256_json(list(raw_entries))
    else:
        raw = protocol.payload.get("asset_manifest", protocol.payload.get("assets"))
        raw_entries = _manifest_entries(raw)
        base_dir = protocol.path.parent
        source_hash = sha256_json(list(raw_entries))
    specs = _specs_from_entries(raw_entries, base_dir=base_dir)
    records: list[Mapping[str, Any]] = []
    for spec in specs:
        records.append(_validate_spec(spec, require_hash=require_hash))
    return AssetManifest(specs=specs, records=tuple(records), source_path=source_path, sha256=source_hash)


def load_adapter(spec: Any, *, label: str) -> Callable[..., Any]:
    """Resolve a ``module:function`` adapter without executing arbitrary code."""

    if not isinstance(spec, str) or ":" not in spec:
        raise ProtocolError(f"{label} adapter must be a module:function string")
    module_name, function_name = spec.split(":", 1)
    module_name, function_name = module_name.strip(), function_name.strip()
    if not module_name or not function_name or not function_name.isidentifier():
        raise ProtocolError(f"{label} adapter has invalid module:function syntax: {spec!r}")
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise ProtocolError(f"cannot load {label} adapter {spec!r}: {exc}") from exc
    if not callable(function):
        raise ProtocolError(f"{label} adapter {spec!r} is not callable")
    return function


def require_cpu_stage(protocol: ProtocolLock) -> None:
    protocol.assert_unchanged()
    stage = protocol.payload.get("current_stage", protocol.payload.get("stage"))
    if str(stage).lower() not in {"cpu", "local_s0", "remote_cpu_preflight"}:
        raise StageError(f"operation requires a CPU preparation stage, got {stage!r}")


def validate_device(protocol: ProtocolLock, device: Any) -> str | None:
    """Validate an optional local device against the protocol allow-list."""

    if device is None:
        return None
    requested = str(device).lower()
    if requested == "cuda":
        requested = "cuda:0"
    stage = str(protocol.payload.get("current_stage", protocol.payload.get("stage"))).lower()
    if stage == "remote_cpu_preflight" and requested != "cpu":
        raise StageError("remote_cpu_preflight permits CPU only; CUDA is deferred until an explicit GPU stage")
    allowed = protocol.payload.get("allowed_devices")
    if allowed is None:
        allowed_values = ("cpu",)
    elif isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes)):
        allowed_values = tuple(str(value).lower() for value in allowed)
    else:
        raise StageError("protocol allowed_devices must be a list")
    if requested not in allowed_values:
        raise StageError(
            f"device {device!r} is not allowed by protocol; allowed_devices={list(allowed_values)}"
        )
    return requested


def reject_training(protocol: ProtocolLock, review_gate: Path | str | None = None) -> None:
    """Hard-stop training while the clean-room protocol is in CPU preparation."""

    require_cpu_stage(protocol)
    # CPU is the only stage implemented by this package.  Even an accidentally
    # present/approved gate must not turn a CPU preparation process into a
    # training process; a future GPU protocol must explicitly change the stage
    # and be reviewed separately.
    raise StageError(
        "train is unavailable during the CPU/local_s0 preparation stage; a signed review gate and "
        "an explicit future GPU protocol are required before any training"
    )


def write_protocol_copy(output_dir: Path | str, protocol: ProtocolLock, manifest: AssetManifest) -> dict[str, Path]:
    """Write immutable lock copies into a result bundle for later review."""

    out = Path(output_dir)
    return {
        "protocol_lock": write_json(out / "protocol_lock.json", protocol.to_dict()),
        "asset_manifest": write_json(out / "asset_manifest.json", manifest.to_dict()),
    }


__all__ = [
    "AssetManifest",
    "AssetManifestError",
    "AssetSpec",
    "DEFAULT_PROTOCOL_PATH",
    "ProtocolChangedError",
    "ProtocolError",
    "ProtocolLock",
    "StageError",
    "load_adapter",
    "load_asset_manifest",
    "load_protocol",
    "reject_training",
    "require_cpu_stage",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_device",
    "write_protocol_copy",
]
