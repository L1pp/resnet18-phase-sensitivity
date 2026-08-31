"""Formal asset manifest, bundle hashing and pre-run validation.

The validator is intentionally read-only with respect to experimental assets:
it only writes the diagnostic ``formal_asset_validation.json``.  A formal
matrix must pass it before constructing or training its first arm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import grid_points_px, hash_array, sha256_file, support_tids
from .models import build_model, load_exact_init, state_dict_hash
from .protocol import EXPECTED_SEEDS, SUPPORT_NAMES, canonical_json_bytes, load_protocol, protocol_hash, write_json
from .training import _stream_seed


KNOWN_EXACT_INIT_SHA256 = {
    20260816: "e2edf8c57b8201ed5c973c4e57f9620e129a5cc62c033d93953c44a0ce0f0b2d",
    20260817: "1695d625c07b2232bfa53d98008342759a2253c595fcf42c4c1868ca13fe24ef",
    20260818: "01bae741789811bcb2e31bea270784187e144f323fc27540ce03e42dcc18b794",
}
KNOWN_SOURCE_PROTOCOL_HASH = "8f8e47b6947f393cd4e246e30ff2a97c0190f76baf053ba45d3ff83d68ec05d3"
EXPECTED_CACHE_FILES = (
    "support_bank_images.npy",
    "support_bank_points.npy",
    "dense_images.npy",
    "dense_points.npy",
)


class AssetValidationError(ValueError):
    """Raised when formal assets are absent, altered or inconsistent."""


def _support_input_hash(images: np.ndarray, points: np.ndarray) -> str:
    return hash_array(images, "support_images") + hash_array(points, "support_points")


def _json_sha256(path: Path) -> str:
    return sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssetValidationError(f"JSON object required: {path}")
    return value


def _npy_header(path: Path) -> tuple[list[int], str]:
    """Read only an NPY header, without mapping or touching array payload.

    The formal preflight runs before any model arm.  In particular, it must
    not read dense labels while checking the cache.  Header inspection is
    sufficient for the preflight geometry/dtype gate; the evaluator performs
    the value-level Cartesian-grid check only after the prediction archive is
    atomically committed.
    """

    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        elif version == (3, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY header version {version}")
    return list(shape), str(np.dtype(dtype))


def _package_manifest_paths(root: Path) -> tuple[Path | None, Path | None]:
    """Locate the optional runtime package identity next to an asset root.

    Formal packages use uppercase names.  The two sidecar spellings are
    accepted because existing packagers have emitted both
    ``PACKAGE_MANIFEST.json.sha256`` and ``PACKAGE_MANIFEST.sha256``.
    """

    parent = root.parent
    manifest = parent / "PACKAGE_MANIFEST.json"
    sidecars = (parent / "PACKAGE_MANIFEST.json.sha256", parent / "PACKAGE_MANIFEST.sha256")
    sidecar = next((path for path in sidecars if path.exists()), None)
    return (manifest if manifest.exists() else None), sidecar


def _read_package_identity(root: Path, expected_bundle: str, *, require: bool) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the independent package manifest identity, if present.

    Package identity deliberately remains outside the canonical asset bundle:
    the package manifest commonly records that bundle hash, so including it
    would create a hash cycle.
    """

    manifest_path, sidecar_path = _package_manifest_paths(root)
    errors: list[str] = []
    if manifest_path is None and sidecar_path is None:
        if require:
            errors.append("formal package identity is missing")
        return None, errors
    if manifest_path is None:
        return None, ["package manifest sidecar exists without PACKAGE_MANIFEST.json"]
    if sidecar_path is None:
        errors.append("PACKAGE_MANIFEST.json.sha256 is missing")
    manifest_sha = sha256_file(manifest_path)
    package_id: str | None = None
    manifest: dict[str, Any] = {}
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        errors.append(f"invalid package manifest: {exc}")
    raw_package_id = manifest.get("package_id")
    if raw_package_id is None:
        raw_package_id = manifest.get("package")
    if raw_package_id is not None and str(raw_package_id).strip():
        package_id = str(raw_package_id)
    else:
        errors.append("package manifest package_id is missing")

    declared_bundle = None
    facts = manifest.get("facts")
    if isinstance(facts, Mapping):
        declared_bundle = facts.get("asset_bundle_sha256")
    if declared_bundle is None:
        declared_bundle = manifest.get("asset_bundle_sha256")
    if str(declared_bundle) != str(expected_bundle):
        errors.append("package manifest facts.asset_bundle_sha256 mismatch")

    if sidecar_path is not None:
        try:
            sidecar_text = sidecar_path.read_text(encoding="utf-8").strip()
            sidecar_value: Any = sidecar_text
            if sidecar_text.startswith("{"):
                sidecar_value = json.loads(sidecar_text)
            if isinstance(sidecar_value, Mapping):
                sidecar_value = sidecar_value.get("sha256", sidecar_value.get("manifest_sha256"))
            elif sidecar_text:
                sidecar_value = sidecar_text.split()[0]
            if str(sidecar_value).lower() != manifest_sha.lower():
                errors.append("package manifest sidecar hash mismatch")
        except Exception as exc:
            errors.append(f"invalid package manifest sidecar: {exc}")

    identity = {
        "package_id": package_id,
        "package_manifest_sha256": manifest_sha,
        "package_manifest_path": str(manifest_path.resolve()),
        "package_manifest_sidecar_sha256": None if sidecar_path is None else sha256_file(sidecar_path),
        "asset_bundle_sha256": expected_bundle,
    }
    return identity, errors


def _stream_row(root: Path, support: str, seed: int) -> dict[str, Any]:
    path = root / "batch_streams" / support / f"seed_{seed}.npy"
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        raise AssetValidationError(f"missing paired stream or sidecar: {path}")
    payload = _load_json(sidecar)
    return {
        "key": f"{support}/{seed}",
        "support": support,
        "seed": int(seed),
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sidecar_path": str(sidecar.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "sidecar_sha256": sha256_file(sidecar),
        "sidecar": payload,
    }


def _asset_records(root: Path, cfg: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Collect canonical current-file records and validation errors."""

    errors: list[str] = []
    p_hash = protocol_hash(payload=cfg)
    cache = root / "cache"
    manifest_path = cache / "manifest.json"
    records: dict[str, Any] = {"schema_version": 1, "protocol_hash": p_hash}
    if not manifest_path.exists():
        return records, ["missing cache/manifest.json"]
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        return records, [f"invalid cache manifest: {exc}"]
    records["cache_manifest_sha256"] = _json_sha256(manifest_path)
    records["cache_files"] = {}
    records["supports"] = {}
    records["streams"] = []
    records["exact_inits"] = []
    if manifest.get("protocol_hash") != p_hash:
        errors.append("cache manifest protocol_hash mismatch")
    file_entries = manifest.get("files")
    if not isinstance(file_entries, Mapping) or set(file_entries) != set(EXPECTED_CACHE_FILES):
        errors.append("cache manifest must contain exactly four NPY files")
        file_entries = {} if not isinstance(file_entries, Mapping) else file_entries
    arrays: dict[str, np.ndarray] = {}
    expected_shapes = {
        "support_bank_images.npy": ((64, int(cfg["image_size"]), int(cfg["image_size"])), "uint8"),
        "support_bank_points.npy": ((64, 2), "float64"),
        "dense_images.npy": ((int(cfg["dense_grid"]["n"]) ** 2, int(cfg["image_size"]), int(cfg["image_size"])), "uint8"),
        "dense_points.npy": ((int(cfg["dense_grid"]["n"]) ** 2, 2), "float64"),
    }
    for name in EXPECTED_CACHE_FILES:
        path = cache / name
        entry = file_entries.get(name)
        if not path.exists():
            errors.append(f"missing cache file: {name}")
            continue
        actual_sha = sha256_file(path)
        actual_bytes = int(path.stat().st_size)
        try:
            # Dense labels are header-checked only.  Their values remain
            # inaccessible to formal preflight until independent evaluation
            # has atomically archived predictions.
            if name in ("dense_images.npy", "dense_points.npy"):
                actual_shape, actual_dtype = _npy_header(path)
            else:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                arrays[name] = array
                actual_shape, actual_dtype = list(array.shape), str(array.dtype)
        except Exception as exc:
            errors.append(f"cannot load {name}: {exc}")
            continue
        records["cache_files"][name] = {
            "sha256": actual_sha,
            "bytes": actual_bytes,
            "shape": actual_shape,
            "dtype": actual_dtype,
        }
        expected_shape, expected_dtype = expected_shapes[name]
        if tuple(actual_shape) != tuple(expected_shape) or actual_dtype != expected_dtype:
            errors.append(f"cache geometry/dtype mismatch: {name}")
        if not isinstance(entry, Mapping) or entry.get("sha256") != actual_sha or int(entry.get("bytes", -1)) != actual_bytes or list(entry.get("shape", [])) != actual_shape or str(entry.get("dtype")) != actual_dtype:
            errors.append(f"cache manifest entry mismatch: {name}")

    if {"support_bank_images.npy", "support_bank_points.npy"}.issubset(arrays):
        expected_bank_points = grid_points_px(cfg)
        if not np.array_equal(np.asarray(arrays["support_bank_points.npy"]), expected_bank_points):
            errors.append("support bank points are not the protocol 8x8 grid")
        bank_images, bank_points = arrays["support_bank_images.npy"], arrays["support_bank_points.npy"]
        for support in SUPPORT_NAMES:
            tids = support_tids(support, cfg)
            image_hash = hash_array(np.asarray(bank_images[tids]), "support_images")
            point_hash = hash_array(np.asarray(bank_points[tids]), "support_points")
            input_hash = image_hash + point_hash
            current = {"tids": tids.tolist(), "count": int(len(tids)), "input_hash": input_hash, "image_hash": image_hash, "point_hash": point_hash}
            records["supports"][support] = current
            manifest_current = manifest.get("support_bank", {}).get(support)
            if not isinstance(manifest_current, Mapping) or manifest_current.get("tids") != current["tids"] or manifest_current.get("input_hash") != input_hash:
                errors.append(f"support manifest/input hash mismatch: {support}")

    for support in SUPPORT_NAMES:
        for seed in EXPECTED_SEEDS:
            try:
                row = _stream_row(root, support, seed)
            except AssetValidationError as exc:
                errors.append(str(exc))
                continue
            path = root / row["path"]
            sidecar = root / row["sidecar_path"]
            stream = np.load(path, allow_pickle=False, mmap_mode="r")
            support_count = len(cfg["supports"][support])
            side = row["sidecar"]
            if stream.dtype != np.int64 or tuple(stream.shape) != (int(cfg["training"]["steps"]), int(cfg["training"]["batch_size"])):
                errors.append(f"stream shape/dtype mismatch: {row['key']}")
            indices_in_range = not stream.size or (int(np.min(stream)) >= 0 and int(np.max(stream)) < support_count)
            if not indices_in_range:
                errors.append(f"stream index range mismatch: {row['key']}")
            # Never pass a corrupted int64 into bincount: a single flipped
            # high bit could request a petabyte-scale allocation before the
            # validator gets to report the tamper.
            counts = np.bincount(np.asarray(stream).reshape(-1), minlength=support_count) if indices_in_range else np.zeros(support_count, dtype=np.int64)
            if int(counts.max() - counts.min()) > 1:
                errors.append(f"stream balance mismatch: {row['key']}")
            expected_stream_seed = _stream_seed(seed, support)
            if side.get("sha256") != row["sha256"] or side.get("balanced_max_minus_min") != int(counts.max() - counts.min()) or side.get("shape") != list(stream.shape) or side.get("count") != support_count or side.get("batch_size") != int(cfg["training"]["batch_size"]) or side.get("batches") != int(cfg["training"]["steps"]) or side.get("support") != support or int(side.get("experiment_seed", -1)) != int(seed) or int(side.get("stream_seed", side.get("seed", -1))) != int(expected_stream_seed):
                errors.append(f"stream sidecar mismatch: {row['key']}")
            row.pop("sidecar", None)
            records["streams"].append(row)

    init_dir = root / "exact_inits"
    for seed in EXPECTED_SEEDS:
        path = init_dir / f"seed_{seed}.pt"
        if not path.exists():
            errors.append(f"missing exact init: {path.name}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != KNOWN_EXACT_INIT_SHA256[seed]:
            errors.append(f"exact init SHA not on whitelist: {seed}")
        try:
            model = build_model()
            payload = load_exact_init(path, model, expected_seed=seed)
            if str(payload.get("protocol_hash")) != KNOWN_SOURCE_PROTOCOL_HASH:
                errors.append(f"exact init source protocol hash mismatch: {seed}")
            state_hash = state_dict_hash(payload["model_state"])
            if state_hash != payload.get("init_hash"):
                errors.append(f"exact init payload hash mismatch: {seed}")
            if str(payload.get("variant")) != "vanilla":
                errors.append(f"exact init variant mismatch: {seed}")
            records["exact_inits"].append({"seed": int(seed), "path": str(path.relative_to(root)).replace("\\", "/"), "sha256": actual_sha, "init_hash": str(payload.get("init_hash")), "variant": str(payload.get("variant")), "source_protocol_hash": str(payload.get("protocol_hash"))})
        except Exception as exc:
            errors.append(f"exact init payload invalid {seed}: {exc}")

    records["streams"].sort(key=lambda value: value["key"])
    records["exact_inits"].sort(key=lambda value: value["seed"])
    return records, errors


def asset_bundle_sha256(records: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(records)).hexdigest()


def validate_assets(
    root_dir: str | Path,
    protocol: Mapping[str, Any] | None = None,
    *,
    require_package_identity: bool = False,
) -> dict[str, Any]:
    """Validate all formal assets and bind them to one canonical bundle hash."""

    root = Path(root_dir).resolve()
    cfg = load_protocol() if protocol is None else dict(protocol)
    p_hash = protocol_hash(payload=cfg)
    records, errors = _asset_records(root, cfg)
    bundle = asset_bundle_sha256(records)
    receipt_path = root / "prepare_receipt.json"
    receipt = _load_json(receipt_path) if receipt_path.exists() else {}
    if not receipt_path.exists():
        errors.append("missing prepare_receipt.json")
    if receipt.get("protocol_hash") != p_hash:
        errors.append("prepare receipt protocol_hash mismatch")
    if receipt.get("cache_manifest_sha256") != records.get("cache_manifest_sha256"):
        errors.append("prepare receipt cache manifest SHA mismatch")
    if receipt.get("asset_bundle_sha256") != bundle:
        errors.append("prepare receipt asset_bundle_sha256 mismatch")
    receipt_streams = {(str(row.get("support")), int(row.get("seed", -1))): row for row in receipt.get("batch_streams", []) if isinstance(row, Mapping)}
    for row in records.get("streams", []):
        expected = receipt_streams.get((row["support"], row["seed"]))
        if expected is None or expected.get("sha256") != row["sha256"] or expected.get("sidecar_sha256") != row["sidecar_sha256"]:
            errors.append(f"prepare receipt stream binding mismatch: {row['key']}")
    receipt_inits = {int(row.get("seed", -1)): row for row in receipt.get("exact_inits", []) if isinstance(row, Mapping)}
    for row in records.get("exact_inits", []):
        expected = receipt_inits.get(row["seed"])
        if expected is None or expected.get("target_sha256") != row["sha256"] or str(expected.get("source_protocol_hash")) != str(row.get("source_protocol_hash")) or str(expected.get("init_hash")) != str(row.get("init_hash")):
            errors.append(f"prepare receipt exact-init binding mismatch: {row['seed']}")
    package_identity, package_errors = _read_package_identity(root, bundle, require=require_package_identity)
    errors.extend(package_errors)
    result = {
        "schema_version": 1,
        "kind": "formal_asset_validation",
        "status": "passed" if not errors else "failed",
        "root": str(root),
        "protocol_hash": p_hash,
        "asset_bundle_sha256": bundle,
        "package_identity": package_identity,
        "package_id": None if package_identity is None else package_identity.get("package_id"),
        "package_manifest_sha256": None if package_identity is None else package_identity.get("package_manifest_sha256"),
        "dense_labels_preflight": {
            "values_read": False,
            "header_only": True,
            "value_geometry_deferred_to_evaluator": True,
        },
        "errors": errors,
        "records": records,
    }
    write_json(root / "formal_asset_validation.json", result)
    if errors:
        raise AssetValidationError("formal asset validation failed: " + "; ".join(errors[:8]))
    return result


__all__ = ["AssetValidationError", "KNOWN_EXACT_INIT_SHA256", "KNOWN_SOURCE_PROTOCOL_HASH", "asset_bundle_sha256", "validate_assets"]
