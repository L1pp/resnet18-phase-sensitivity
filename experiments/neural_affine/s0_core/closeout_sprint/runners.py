"""Protocol-bound CPU runners for the clean-room S0 audit.

The functions in this module are the only supported orchestration boundary for
the S0 command line.  They load a hash-locked protocol and asset manifest,
validate every input before computation, call one numerical audit, and emit a
reviewable result bundle containing ``environment.json``, ``manifest.json``,
``metrics.json`` and ``decision.json``.  The old experiment directories are
never imported or written.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .assets import sha256_file, write_run_bundle
from .bn_control import run_bn_only_control
from .contracts import SCHEMA_VERSION, json_safe, read_json, write_json
from .evaluator import evaluate_npz
from .kernel_audit import evaluate_kernel_gate, fit_fixed_log_error_predictor
from .ols_audit import audit_ols_npz
from .partial_out import partial_out_groups
from .protocols import (
    AssetManifest,
    AssetManifestError,
    AssetSpec,
    DEFAULT_PROTOCOL_PATH,
    ProtocolError,
    ProtocolLock,
    StageError,
    load_adapter,
    load_asset_manifest,
    load_protocol,
    require_cpu_stage,
    sha256_json,
    validate_device,
    write_protocol_copy,
)


class RunnerError(ProtocolError):
    """A command input cannot satisfy its formal runner contract."""


DEFAULT_SOURCE_MACHINE = "local"


# The A10 materializer writes one atomic bridge for the production kernel
# audit.  Keep this contract local to the runner so loading it never imports
# the materializer (and therefore cannot execute historical experiment code).
KERNEL_FORMAL_SCHEMA = "kernel_runner_input_v1"
# This schema is frozen against the A10 materializer protocol.  A future
# protocol must publish a new bridge schema instead of silently changing the
# meaning of an existing locked input.
KERNEL_FORMAL_PROTOCOL_SHA256 = "62ad9a43c540896a8031186a1bda9107d8c68ca58e79d0c1036d5d48a2548ac1"
KERNEL_FORMAL_FIT_NAMES = (
    "G64",
    "G32",
    "G16",
    "G9",
    "maximin9",
    "boundary8_center",
)
KERNEL_FORMAL_HELDOUT_NAMES = ("random9", "cross5", "diagonal8")
KERNEL_FORMAL_FEATURE_NAMES = (
    "euclidean_covering_radius_px",
    "gap_kernel_posterior_variance",
)
KERNEL_PREDICTOR_FEATURE_NAMES = (
    "euclidean_covering_radius",
    "gap_kernel_posterior_variance",
)
KERNEL_FORMAL_SOURCE_ASSETS = ("shapes", "translations", "split", "train", "val", "test")
KERNEL_FORMAL_CHECKPOINTS = KERNEL_FORMAL_FIT_NAMES + KERNEL_FORMAL_HELDOUT_NAMES
KERNEL_FORMAL_PROVENANCE_DEFINITION = (
    "sha256(canonical_json({protocol_sha256,checkpoint_hashes}))"
)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _dynamic_entries(paths: Sequence[Path | str], *, roles: Sequence[str] | None = None) -> list[dict[str, Any]]:
    role_values = list(roles or ())
    if role_values and len(role_values) != len(paths):
        raise RunnerError(f"roles has {len(role_values)} entries for {len(paths)} inputs")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(paths):
        resolved = _path(raw)
        asset_id = f"input_{index}_{resolved.stem or 'asset'}"
        while asset_id in seen:
            asset_id += "_x"
        seen.add(asset_id)
        entries.append(
            {
                "id": asset_id,
                "path": str(resolved),
                "role": role_values[index] if role_values else None,
                # A command-line input is pinned at the start of this run and
                # subsequently checked again by AssetManifest.assert_unchanged.
                "sha256": sha256_file(resolved),
                "source_machine": DEFAULT_SOURCE_MACHINE,
            }
        )
    return entries


def _prepare_context(
    *,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    asset_entries: Sequence[Mapping[str, Any]] | None = None,
    input_paths: Sequence[Path | str] | None = None,
    input_roles: Sequence[str] | None = None,
    require_hash: bool = True,
) -> tuple[ProtocolLock, AssetManifest]:
    protocol = load_protocol(protocol_path or DEFAULT_PROTOCOL_PATH)
    require_cpu_stage(protocol)
    if asset_manifest_path is not None and asset_entries is not None:
        raise RunnerError("asset_manifest_path and asset_entries are mutually exclusive")
    if input_paths is not None:
        if asset_manifest_path is not None or asset_entries is not None:
            raise RunnerError("input paths cannot be combined with an explicit asset manifest")
        asset_entries = _dynamic_entries(input_paths, roles=input_roles)
    manifest = load_asset_manifest(
        protocol,
        asset_manifest_path,
        entries=asset_entries,
        require_hash=require_hash,
    )
    return protocol, manifest


def _asset_paths(manifest: AssetManifest, paths: Sequence[Path | str]) -> tuple[AssetSpec, ...]:
    by_path = {spec.path: spec for spec in manifest.specs}
    selected: list[AssetSpec] = []
    for raw in paths:
        resolved = _path(raw)
        if resolved not in by_path:
            raise RunnerError(f"input is not present in the locked asset manifest: {resolved}")
        selected.append(by_path[resolved])
    return tuple(selected)


def _assert_roles(specs: Sequence[AssetSpec], allowed: set[str], stage: str) -> None:
    invalid = [f"{spec.asset_id}:{spec.role}" for spec in specs if spec.role not in allowed]
    if invalid:
        raise RunnerError(
            f"asset role is not allowed for stage {stage}: {invalid}; allowed={sorted(allowed)}"
        )


def _begin_bundle(
    output_dir: Path | str,
    protocol: ProtocolLock,
    manifest: AssetManifest,
    *,
    command: Sequence[str],
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    extra_manifest: Mapping[str, Any] | None = None,
) -> Path:
    # Re-check both lock files and every referenced asset immediately before
    # any formal result directory is written.
    protocol.assert_unchanged()
    manifest.assert_unchanged()
    out = _path(output_dir)
    role_map = {str(spec.path): spec.role for spec in manifest.specs if spec.role}
    extra = {
        "protocol_id": protocol.payload.get("protocol_id", protocol.payload.get("id")),
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
        "current_stage": protocol.payload.get("current_stage", protocol.payload.get("stage")),
        "allowed_devices": protocol.payload.get("allowed_devices", ["cpu"]),
        "formal_experiment_started": False,
    }
    if extra_manifest:
        extra.update(dict(extra_manifest))
    write_run_bundle(
        out,
        [spec.path for spec in manifest.specs],
        command=command,
        source_machine=source_machine,
        role_map=role_map,
        extra_manifest=extra,
    )
    write_protocol_copy(out, protocol, manifest)
    return out


def _refresh_bundle(
    output_dir: Path | str,
    protocol: ProtocolLock,
    manifest: AssetManifest,
    *,
    command: Sequence[str],
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    extra_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Refresh provenance after a legacy audit wrote its own bundle files."""

    protocol.assert_unchanged()
    manifest.assert_unchanged()
    _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=command,
        source_machine=source_machine,
        extra_manifest=extra_manifest,
    )


def _finish(
    output_dir: Path | str,
    protocol: ProtocolLock,
    manifest: AssetManifest,
    *,
    metrics: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Path]:
    protocol.assert_unchanged()
    manifest.assert_unchanged()
    out = _path(output_dir)
    return {
        "metrics": write_json(out / "metrics.json", {"schema_version": SCHEMA_VERSION, **dict(metrics)}),
        "decision": write_json(out / "decision.json", {"schema_version": SCHEMA_VERSION, **dict(decision)}),
    }


def _result_without_arrays(value: Any) -> Any:
    """Keep JSON summaries compact while retaining shapes/dtypes as evidence."""

    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _result_without_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_result_without_arrays(item) for item in value]
    return value


def _safe_id(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "input"
    return f"{stem}_{index}_{sha256_file(path)[:12]}"


def _explicit_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise RunnerError(f"run_id is empty after sanitization: {value!r}")
    return result


def _default_stage_specs(manifest: AssetManifest, stage: str) -> tuple[AssetSpec, ...]:
    aliases = {
        "headlines": {"field_rendered_xy", "field", "field_2d", "rendered_xy"},
        "ols": {"ols", "ols_features", "frozen_g64_ols"},
        "kernel": {"kernel_input", "kernel_npz", "gap_features"},
        "bn": {"bn_config", "bn_adapter_config"},
        "partial": {"partial_input", "partial_npz", "partial_group"},
    }
    selected = manifest.matching(roles=aliases.get(stage, set()))
    if not selected:
        raise RunnerError(f"no locked assets found for S0 stage {stage!r}")
    return selected


def _stage_specs(protocol: ProtocolLock, manifest: AssetManifest, stage: str) -> tuple[AssetSpec, ...]:
    stages = protocol.payload.get("stages", {})
    config = stages.get(stage, {}) if isinstance(stages, Mapping) else {}
    if not isinstance(config, Mapping):
        raise RunnerError(f"protocol stages.{stage} must be an object")
    ids = config.get("asset_ids", config.get("assets"))
    if ids is not None:
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise RunnerError(f"protocol stages.{stage}.asset_ids must be a list")
        return tuple(manifest.by_id(str(asset_id)) for asset_id in ids)
    roles = config.get("roles")
    if roles is not None:
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
            raise RunnerError(f"protocol stages.{stage}.roles must be a list")
        selected = manifest.matching(roles=[str(role) for role in roles])
        if not selected:
            raise RunnerError(f"no assets match protocol stages.{stage}.roles")
        return selected
    return _default_stage_specs(manifest, stage)


def run_check(
    *,
    output_dir: Path | str,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    asset_entries: Sequence[Mapping[str, Any]] | None = None,
    input_paths: Sequence[Path | str] | None = None,
    input_roles: Sequence[str] | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    _context: tuple[ProtocolLock, AssetManifest] | None = None,
) -> dict[str, Any]:
    """Run the read-only asset/environment preflight."""

    protocol, manifest = _context or _prepare_context(
        protocol_path=protocol_path,
        asset_manifest_path=asset_manifest_path,
        asset_entries=asset_entries,
        input_paths=input_paths,
        input_roles=input_roles,
    )
    out = _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=["check"],
        source_machine=source_machine,
        extra_manifest={"stage": "check", "read_only": True},
    )
    assets = [dict(record) for record in manifest.records]
    metrics = {
        "stage": "check",
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
        "asset_count": len(assets),
        "assets": assets,
        "environment_recorded": True,
    }
    decision = {
        "stage": "check",
        "status": "passed",
        "passed": True,
        "reason": "all protocol and asset checks passed",
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    write_json(out / "check.json", {"schema_version": SCHEMA_VERSION, **metrics})
    return {"metrics": metrics, "decision": decision, "output_dir": str(out)}


def run_headlines(
    input_paths: Sequence[Path | str],
    *,
    output_dir: Path | str,
    support_indices: Any = None,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    run_ids: Sequence[str] | None = None,
    _context: tuple[ProtocolLock, AssetManifest] | None = None,
) -> dict[str, Any]:
    """Recompute field headlines from pred/true under a lock."""

    paths = tuple(_path(path) for path in input_paths)
    if not paths:
        raise RunnerError("audit-headlines requires at least one field input")
    if run_ids is not None and len(run_ids) != len(paths):
        raise RunnerError(f"run_ids has {len(run_ids)} entries for {len(paths)} inputs")
    if run_ids is not None:
        ids = tuple(_explicit_id(value) for value in run_ids)
        if len(set(ids)) != len(ids):
            raise RunnerError(f"run_id collision detected: {ids}")
    else:
        ids = tuple(_safe_id(path, index) for index, path in enumerate(paths))
    protocol, manifest = _context or _prepare_context(
        protocol_path=protocol_path,
        asset_manifest_path=asset_manifest_path,
        input_paths=None if asset_manifest_path else paths,
        input_roles=["field_rendered_xy"] * len(paths) if asset_manifest_path is None else None,
    )
    selected = _asset_paths(manifest, paths)
    _assert_roles(selected, {"field_rendered_xy", "field", "field_2d", "rendered_xy"}, "headlines")
    out = _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=["audit-headlines", *[str(path) for path in paths]],
        source_machine=source_machine,
        extra_manifest={"stage": "headlines", "uses_old_metrics": False, "uses_old_u": False},
    )
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        result = evaluate_npz(path, support_indices=support_indices)
        run_id = ids[index]
        summary = _result_without_arrays(
            {key: value for key, value in result.items() if key not in {"pred", "true", "coords", "support_ids", "raw_error", "u", "affine_pred"}}
        )
        summary["run_id"] = run_id
        write_json(out / f"{run_id}.json", summary)
        np.savez_compressed(
            out / f"{run_id}_recomputed.npz",
            pred=result["pred"],
            true=result["true"],
            coords=result["coords"],
            support_ids=result["support_ids"],
            raw_error=result["raw_error"],
            u=result["u"],
            affine_pred=result["affine_pred"],
        )
        rows.append(summary)
    metrics = {
        "stage": "headlines",
        "runs": rows,
        "run_count": len(rows),
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
    }
    decision = {
        "stage": "headlines",
        "status": "complete",
        "passed": True,
        "reason": "independent pred/true recomputation completed",
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    return {"metrics": metrics, "decision": decision, "output_dir": str(out)}


def run_ols(
    input_path: Path | str,
    *,
    output_dir: Path | str,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the separated-train/eval OLS audit and preserve predictions."""

    path = _path(input_path)
    protocol, manifest = _prepare_context(
        protocol_path=protocol_path,
        asset_manifest_path=asset_manifest_path,
        input_paths=None if asset_manifest_path else [path],
        input_roles=None if asset_manifest_path else ["ols"],
    )
    _enforce_analysis_output_root(protocol, "ols", output_dir)
    selected = _asset_paths(manifest, [path])
    _assert_roles(selected, {"ols", "ols_features", "frozen_g64_ols"}, "ols")
    out = _path(output_dir)
    # audit_ols_npz writes its own predictions/OLS JSON and provenance trio;
    # refresh that trio after the numerical step with our protocol metadata.
    result = audit_ols_npz(path, out, source_machine=source_machine, **kwargs)
    _refresh_bundle(
        out,
        protocol,
        manifest,
        command=["audit-ols", str(path)],
        source_machine=source_machine,
        extra_manifest={"stage": "ols", "audit": "svd_ridge_whiten_perturbation"},
    )
    metrics = {
        "stage": "ols",
        "stability": result.get("stability"),
        "n_train": result.get("n_train"),
        "n_eval": result.get("n_eval"),
        "n_features": result.get("n_features"),
        "precision_modes": result.get("precision_modes"),
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
    }
    decision = {
        "stage": "ols",
        "status": "complete",
        "passed": True,
        "numerical_stability_verdict": result.get("stability", {}).get("verdict"),
        "reason": "OLS audit completed; stability verdict is reported separately",
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    return {"metrics": metrics, "decision": decision, "result": result, "output_dir": str(out)}


def _text_scalar(value: Any, *, label: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise RunnerError(f"{label} must be a scalar string")
    item = array.item()
    if isinstance(item, (bytes, np.bytes_)):
        try:
            item = bytes(item).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RunnerError(f"{label} is not valid UTF-8 text") from exc
    if not isinstance(item, (str, np.str_)):
        raise RunnerError(f"{label} must be a scalar string")
    return str(item)


def _text_vector(value: Any, *, label: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise RunnerError(f"{label} must be a one-dimensional string array")
    result: list[str] = []
    for index, item in enumerate(array.tolist()):
        try:
            result.append(_text_scalar(np.asarray(item), label=f"{label}[{index}]"))
        except RunnerError:
            raise
    return tuple(result)


def _json_scalar(value: Any, *, label: str) -> Any:
    text = _text_scalar(value, label=label)
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} is not valid JSON") from exc


def _digest_mapping(value: Any, *, label: str, required: Sequence[str]) -> dict[str, str]:
    parsed = _json_scalar(value, label=label)
    if not isinstance(parsed, Mapping):
        raise RunnerError(f"{label} must encode a JSON object")
    result: dict[str, str] = {}
    for key, digest in parsed.items():
        if not isinstance(key, str) or not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise RunnerError(f"{label} values must all be 64-hex digests")
        result[key] = digest
    missing = sorted(set(required) - set(result))
    if missing:
        raise RunnerError(f"{label} is missing required entries: {missing}")
    return result


def _load_formal_kernel_npz(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the A10 ``kernel_runner_input_v1`` bridge without fitting."""

    required = {
        "fit_features",
        "fit_errors",
        "heldout_features",
        "heldout_errors",
        "heldout_euclidean_baseline_errors",
        "feature_names",
        "fit_names",
        "heldout_names",
        "partition_json",
        "protocol_sha256",
        "provenance_token_sha256",
        "provenance_definition",
        "source_asset_sha256_json",
        "checkpoint_sha256_json",
        "schema_kind",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RunnerError(f"formal kernel input is missing required arrays: {missing}")

    schema_kind = _text_scalar(arrays["schema_kind"], label="schema_kind")
    if schema_kind != KERNEL_FORMAL_SCHEMA:
        raise RunnerError(f"unsupported formal kernel schema_kind: {schema_kind!r}")
    protocol_hash = _text_scalar(arrays["protocol_sha256"], label="protocol_sha256")
    if _HEX64.fullmatch(protocol_hash) is None:
        raise RunnerError("protocol_sha256 must be a 64-hex digest")
    if protocol_hash.lower() != KERNEL_FORMAL_PROTOCOL_SHA256:
        raise RunnerError(
            "kernel_runner_input_v1 is bound to the frozen A10 materializer "
            f"protocol {KERNEL_FORMAL_PROTOCOL_SHA256}"
        )
    if expected_protocol_sha256 is not None and protocol_hash.lower() != str(expected_protocol_sha256).lower():
        raise RunnerError("kernel runner protocol hash mismatch")

    fit_names = _text_vector(arrays["fit_names"], label="fit_names")
    heldout_names = _text_vector(arrays["heldout_names"], label="heldout_names")
    if fit_names != KERNEL_FORMAL_FIT_NAMES:
        raise RunnerError(f"kernel fit partition mismatch: {fit_names}")
    if heldout_names != KERNEL_FORMAL_HELDOUT_NAMES:
        raise RunnerError(f"kernel heldout partition mismatch: {heldout_names}")
    feature_names = _text_vector(arrays["feature_names"], label="feature_names")
    if feature_names != KERNEL_FORMAL_FEATURE_NAMES:
        raise RunnerError(f"unexpected kernel feature names: {feature_names}")

    partition = _json_scalar(arrays["partition_json"], label="partition_json")
    if not isinstance(partition, Mapping):
        raise RunnerError("partition_json must encode a JSON object")
    partition_fit = partition.get("fit_names")
    partition_heldout = partition.get("heldout_names")
    if not isinstance(partition_fit, list) or tuple(partition_fit) != KERNEL_FORMAL_FIT_NAMES:
        raise RunnerError("partition_json fit_names do not match the formal fit array order")
    if not isinstance(partition_heldout, list) or tuple(partition_heldout) != KERNEL_FORMAL_HELDOUT_NAMES:
        raise RunnerError("partition_json heldout_names do not match the formal heldout array order")
    if partition.get("fit_target_source") != "fit_geometry_only":
        raise RunnerError("kernel baseline/fit source must be declared fit_geometry_only")
    if partition.get("heldout_target_source") != "post_fit_scoring_only":
        raise RunnerError("heldout target source must be declared post_fit_scoring_only")

    source_hashes = _digest_mapping(
        arrays["source_asset_sha256_json"],
        label="source_asset_sha256_json",
        required=KERNEL_FORMAL_SOURCE_ASSETS,
    )
    checkpoint_hashes = _digest_mapping(
        arrays["checkpoint_sha256_json"],
        label="checkpoint_sha256_json",
        required=KERNEL_FORMAL_CHECKPOINTS,
    )
    provenance_definition = _text_scalar(
        arrays["provenance_definition"], label="provenance_definition"
    )
    if provenance_definition != KERNEL_FORMAL_PROVENANCE_DEFINITION:
        raise RunnerError("unsupported provenance token definition")
    provenance_token = _text_scalar(
        arrays["provenance_token_sha256"], label="provenance_token_sha256"
    )
    if _HEX64.fullmatch(provenance_token) is None:
        raise RunnerError("provenance_token_sha256 must be a 64-hex digest")
    expected_token = sha256_json(
        {"protocol_sha256": protocol_hash, "checkpoint_hashes": checkpoint_hashes}
    )
    if provenance_token.lower() != expected_token:
        raise RunnerError("provenance_token_sha256 does not match protocol/checkpoint hashes")

    try:
        fit_features_raw = np.asarray(arrays["fit_features"])
        heldout_features_raw = np.asarray(arrays["heldout_features"])
        fit_errors_raw = np.asarray(arrays["fit_errors"])
        heldout_errors_raw = np.asarray(arrays["heldout_errors"])
        baseline_raw = np.asarray(arrays["heldout_euclidean_baseline_errors"])
        if fit_features_raw.shape != (6, 2) or heldout_features_raw.shape != (3, 2):
            raise RunnerError(
                "formal kernel feature shapes must be exactly (6,2)/(3,2), "
                f"got {fit_features_raw.shape}/{heldout_features_raw.shape}"
            )
        if fit_errors_raw.shape != (6,) or heldout_errors_raw.shape != (3,) or baseline_raw.shape != (3,):
            raise RunnerError("formal kernel scalar arrays must have shapes (6,)/(3,)/(3,)")
        fit_features = np.asarray(fit_features_raw, dtype=np.float64)
        heldout_features = np.asarray(heldout_features_raw, dtype=np.float64)
        fit_errors = np.asarray(fit_errors_raw, dtype=np.float64)
        heldout_errors = np.asarray(heldout_errors_raw, dtype=np.float64)
        baseline = np.asarray(baseline_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RunnerError("formal kernel numeric arrays are not valid finite numeric values") from exc
    if not all(
        np.all(np.isfinite(array))
        for array in (fit_features, heldout_features, fit_errors, heldout_errors, baseline)
    ):
        raise RunnerError("formal kernel input contains non-finite values")
    if np.any(fit_errors <= 0) or np.any(heldout_errors <= 0) or np.any(baseline <= 0):
        raise RunnerError("formal kernel errors and baselines must be strictly positive")

    return {
        "fit_features": fit_features,
        "fit_errors": fit_errors,
        "heldout_features": heldout_features,
        "heldout_errors": heldout_errors,
        "baseline": baseline,
        # Keep the materializer's explicit names in the result bundle.  The
        # numerical predictor predates the ``_px`` suffix, so callers use the
        # separate compatibility names only at the fitting boundary.
        "feature_names": feature_names,
        "predictor_feature_names": KERNEL_PREDICTOR_FEATURE_NAMES,
        "formal": True,
        "schema_kind": schema_kind,
        "protocol_sha256": protocol_hash,
        "partition": dict(partition),
        "source_asset_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "provenance_token_sha256": provenance_token,
        "baseline_source": "fit_geometry_only",
    }


def _load_legacy_kernel_npz(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Load the pre-bridge synthetic format for low-level compatibility only."""

    keys = set(arrays)
    required = {"fit_features", "fit_errors", "heldout_features", "heldout_errors"}
    missing = sorted(required - keys)
    if missing:
        raise RunnerError(f"kernel input is missing required arrays: {missing}")
    baseline_key = "heldout_euclidean_baseline_errors"
    if baseline_key not in keys:
        baseline_key = "euclidean_baseline_errors"
    if baseline_key not in keys:
        raise RunnerError(
            "kernel input requires heldout_euclidean_baseline_errors; "
            "the Euclidean baseline is not allowed to be inferred from held-out targets"
        )
    if "fit_feature_names" in keys:
        feature_names = _text_vector(arrays["fit_feature_names"], label="fit_feature_names")
    elif "feature_names" in keys:
        feature_names = _text_vector(arrays["feature_names"], label="feature_names")
    else:
        raise RunnerError("kernel input requires feature_names/fit_feature_names for array predictors")
    try:
        fit_features = np.asarray(arrays["fit_features"], dtype=np.float64)
        heldout_features = np.asarray(arrays["heldout_features"], dtype=np.float64)
        fit_errors = np.asarray(arrays["fit_errors"], dtype=np.float64).reshape(-1)
        heldout_errors = np.asarray(arrays["heldout_errors"], dtype=np.float64).reshape(-1)
        baseline = np.asarray(arrays[baseline_key], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RunnerError("kernel input numeric arrays are invalid") from exc
    if fit_features.ndim != 2 or heldout_features.ndim != 2 or fit_features.shape[1] != heldout_features.shape[1]:
        raise RunnerError("kernel fit/held-out feature matrices must be 2-D with matching columns")
    if fit_features.shape[0] != fit_errors.size:
        raise RunnerError("kernel fit feature rows and fit_errors do not match")
    if heldout_features.shape[0] != heldout_errors.size or heldout_errors.size != baseline.size:
        raise RunnerError("kernel held-out rows, targets, and baseline do not match")
    if not all(np.all(np.isfinite(arr)) for arr in (fit_features, heldout_features, fit_errors, heldout_errors, baseline)):
        raise RunnerError("kernel input contains non-finite values")
    if np.any(fit_errors <= 0) or np.any(heldout_errors <= 0) or np.any(baseline <= 0):
        raise RunnerError("kernel errors and baselines must be strictly positive")
    return {
        "fit_features": fit_features,
        "fit_errors": fit_errors,
        "heldout_features": heldout_features,
        "heldout_errors": heldout_errors,
        "baseline": baseline,
        "feature_names": feature_names,
        "predictor_feature_names": feature_names,
        "formal": False,
        "schema_kind": None,
    }


def _load_kernel_npz(
    path: Path,
    *,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a kernel NPZ, strictly validating the formal A10 bridge when present."""

    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: np.array(blob[key], copy=True) for key in blob.files}
    if "schema_kind" in arrays:
        schema_kind = _text_scalar(arrays["schema_kind"], label="schema_kind")
        if schema_kind != KERNEL_FORMAL_SCHEMA:
            raise RunnerError(f"unsupported formal kernel schema_kind: {schema_kind!r}")
        return _load_formal_kernel_npz(
            arrays,
            expected_protocol_sha256=expected_protocol_sha256,
        )
    return _load_legacy_kernel_npz(arrays)


def _require_formal_kernel_payload(payload: Mapping[str, Any]) -> None:
    if not bool(payload.get("formal")) or payload.get("schema_kind") != KERNEL_FORMAL_SCHEMA:
        raise RunnerError(
            "production kernel runner requires formal kernel_runner_input_v1; "
            "the legacy synthetic loader is not eligible for a locked result"
        )


def _enforce_analysis_output_root(
    protocol: ProtocolLock,
    stage: str,
    output_dir: Path | str,
) -> Path:
    """Enforce the fresh, protocol-frozen output root for the A10 analysis lock.

    Other protocols deliberately retain their existing output behavior.  The
    imported A10 analysis protocol is the one exception: its OLS/kernel
    outputs are unique, must stay below ``closeout_sprint_clean/results``, and
    must not be appended to or overwritten.
    """

    if protocol.payload.get("protocol_id") != "s0_a10_analysis_v1":
        return _path(output_dir)
    stages = protocol.payload.get("stages")
    stage_config = stages.get(stage) if isinstance(stages, Mapping) else None
    if not isinstance(stage_config, Mapping):
        raise RunnerError(f"analysis protocol is missing stages.{stage}")
    declared = stage_config.get("output_root")
    if not isinstance(declared, str) or not declared.strip():
        raise RunnerError(f"analysis protocol requires stages.{stage}.output_root")
    declared_path = Path(declared)
    if declared_path.is_absolute() or any(part == ".." for part in declared_path.parts):
        raise RunnerError(f"analysis stages.{stage}.output_root must be a safe relative path")

    package_root = protocol.path.parent.parent.resolve(strict=False)
    results_root = (package_root / "results").resolve(strict=False)
    expected = (package_root / declared_path).resolve(strict=False)
    try:
        relative = expected.relative_to(results_root)
    except ValueError as exc:
        raise RunnerError(
            f"analysis stages.{stage}.output_root must be below {results_root}"
        ) from exc
    if not relative.parts:
        raise RunnerError(f"analysis stages.{stage}.output_root cannot be the results root")

    actual = _path(output_dir)
    if actual != expected:
        raise RunnerError(
            f"analysis {stage} output must exactly match the frozen path {expected}; got {actual}"
        )
    if actual.exists():
        raise RunnerError(
            f"analysis {stage} output already exists; choose a new protocol version instead: {actual}"
        )
    return actual


def run_kernel(
    input_path: Path | str,
    *,
    output_dir: Path | str,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    ridge_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit kernel geometry only on declared fit rows, then gate held-out rows."""

    path = _path(input_path)
    protocol, manifest = _prepare_context(
        protocol_path=protocol_path,
        asset_manifest_path=asset_manifest_path,
        input_paths=None if asset_manifest_path else [path],
        input_roles=None if asset_manifest_path else ["kernel_input"],
    )
    _enforce_analysis_output_root(protocol, "kernel", output_dir)
    selected = _asset_paths(manifest, [path])
    _assert_roles(selected, {"kernel_input", "kernel_npz", "gap_features"}, "kernel")
    payload = _load_kernel_npz(path)
    _require_formal_kernel_payload(payload)
    out = _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=["audit-kernel", str(path)],
        source_machine=source_machine,
        extra_manifest={"stage": "kernel", "heldout_targets_used_for_fit": False},
    )
    predictor = fit_fixed_log_error_predictor(
        payload["fit_features"],
        payload["fit_errors"],
        feature_names=payload["predictor_feature_names"],
        ridge_grid=tuple(ridge_grid) if ridge_grid is not None else (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
    )
    # No held-out array is passed to fit_fixed_log_error_predictor.  This
    # sequencing is deliberate and recorded in the result manifest.
    predicted_log = predictor.predict_log(payload["heldout_features"])
    gate = evaluate_kernel_gate(predicted_log, payload["heldout_errors"], payload["baseline"])
    np.savez_compressed(
        out / "predictions.npz",
        heldout_predicted_log_error=predicted_log,
        heldout_predicted_error=np.exp(predicted_log),
        heldout_observed_error=payload["heldout_errors"],
        heldout_euclidean_baseline_error=payload["baseline"],
    )
    metrics = {
        "stage": "kernel",
        "fit_count": int(payload["fit_errors"].size),
        "heldout_count": int(payload["heldout_errors"].size),
        "predictor": predictor.to_dict(),
        "gate": gate.to_dict(),
        "heldout_targets_used_for_fit": False,
        "schema_kind": payload["schema_kind"],
        "feature_names": list(payload["feature_names"]),
        "protocol_sha256": payload["protocol_sha256"],
        "partition": payload["partition"],
        "baseline_source": payload["baseline_source"],
        "provenance_token_sha256": payload["provenance_token_sha256"],
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
    }
    decision = {
        "stage": "kernel",
        "status": "passed" if gate.passed else "closed",
        "passed": bool(gate.passed),
        "reason": gate.reason,
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    return {"metrics": metrics, "decision": decision, "output_dir": str(out)}


def _load_bn_config(path: Path) -> Mapping[str, Any]:
    try:
        payload = read_json(path)
    except Exception as exc:
        raise RunnerError(f"invalid BN adapter config: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RunnerError("BN adapter config must be a JSON object")
    # These names are intentionally mandatory even when the API caller injects
    # the objects directly.  A bare model plus a generic eval callback is not a
    # reproducible BN protocol.
    aliases = {
        "model": ("model_adapter", "model"),
        "data": ("data_adapter", "data", "batch_adapter"),
        "evaluator": ("evaluator_adapter", "evaluator"),
    }
    missing = [label for label, names in aliases.items() if not any(payload.get(name) for name in names)]
    if missing:
        raise RunnerError(f"BN adapter config requires explicit model/data/evaluator adapters: {missing}")
    for label, names in aliases.items():
        value = next((payload.get(name) for name in names if payload.get(name)), None)
        if not isinstance(value, str):
            raise RunnerError(f"BN {label} adapter must be an explicit module:function string")
        # Resolve the symbol during static preflight.  This imports only the
        # declared adapter module; it does not construct a model or run data.
        load_adapter(value, label=label)
    return payload


def _resolve_bn_asset_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RunnerError(f"BN {label} requires a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _validate_bn_static_bindings(config_path: Path, config: Mapping[str, Any]) -> None:
    """Validate real blob BN bindings without constructing/rendering anything.

    Synthetic adapters used by unit tests do not have blob asset kwargs and
    are intentionally left at the generic schema check.  The formal clean
    blob adapters, however, must prove checkpoint head=2 and the exact 48/8
    disjoint appearance files before ``run_s0`` creates its first output.
    """

    adapter_values = [
        config.get("model_adapter", config.get("model")),
        config.get("data_adapter", config.get("data", config.get("batch_adapter"))),
        config.get("evaluator_adapter", config.get("evaluator")),
    ]
    is_blob = any("blob_bn_adapter" in str(value) for value in adapter_values)
    kwargs_by_label: dict[str, Mapping[str, Any]] = {}
    for label in ("model", "data", "evaluator"):
        raw = config.get(f"{label}_kwargs", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise RunnerError(f"BN {label}_kwargs must be an object")
        kwargs_by_label[label] = raw
    if not is_blob:
        return

    model_kwargs = kwargs_by_label["model"]
    if "checkpoint_path" not in model_kwargs or not model_kwargs.get("expected_sha256"):
        raise RunnerError("formal blob BN model_kwargs require checkpoint_path and expected_sha256")
    model_base_value = model_kwargs.get("manifest_base", config.get("manifest_base", str(config_path.parent)))
    model_base = Path(model_base_value).expanduser()
    if not model_base.is_absolute():
        model_base = Path.cwd() / model_base
    checkpoint = _resolve_bn_asset_path(
        model_kwargs.get("checkpoint_path"), base=model_base.resolve(), label="checkpoint_path"
    )
    try:
        from .blob_bn_adapter import inspect_resnet18_2d_checkpoint

        info = inspect_resnet18_2d_checkpoint(
            checkpoint, expected_sha256=str(model_kwargs.get("expected_sha256"))
        )
    except Exception as exc:
        raise RunnerError(f"formal blob BN checkpoint preflight failed: {exc}") from exc
    if int(info.get("head_dim", -1)) != 2:
        raise RunnerError(f"formal blob BN checkpoint must have head_dim=2, got {info.get('head_dim')}")

    bindings: dict[str, tuple[Path, str, int]] = {}
    for label in ("data", "evaluator"):
        kwargs = kwargs_by_label[label]
        base_value = kwargs.get("manifest_base", config.get("manifest_base", str(config_path.parent)))
        base = Path(base_value).expanduser()
        if not base.is_absolute():
            base = (Path.cwd() / base).resolve()
        for role, count in (("train", 48), ("eval", 8)):
            path_key = f"{role}_appearance_asset_path"
            hash_key = f"{role}_appearance_sha256"
            if path_key not in kwargs or not kwargs.get(hash_key):
                raise RunnerError(f"formal blob BN {label}_kwargs require {path_key} and {hash_key}")
            path = _resolve_bn_asset_path(kwargs[path_key], base=base, label=path_key)
            digest = str(kwargs[hash_key]).strip().lower()
            if len(digest) != 64:
                raise RunnerError(f"formal blob BN {label} {hash_key} is not a SHA256")
            key = f"{role}"
            if key in bindings and (bindings[key][0] != path or bindings[key][1] != digest):
                raise RunnerError(f"formal blob BN data/evaluator {role} appearance bindings disagree")
            try:
                from .blob_bn_adapter import load_appearance_set

                load_appearance_set(
                    path,
                    expected_sha256=digest,
                    role=role,
                    manifest_base=None,
                    expected_count=count,
                )
            except Exception as exc:
                raise RunnerError(f"formal blob BN {role} appearance preflight failed: {exc}") from exc
            bindings[key] = (path, digest, count)
        for key in ("expected_train_count", "expected_eval_count"):
            expected = 48 if key.endswith("train_count") else 8
            if key in kwargs and int(kwargs[key]) != expected:
                raise RunnerError(f"formal blob BN {label} {key} must be {expected}")
        if "max_exposure" in kwargs:
            try:
                max_exposure = int(kwargs["max_exposure"])
            except (TypeError, ValueError) as exc:
                raise RunnerError(f"formal blob BN {label} max_exposure must be an integer") from exc
            checkpoints = config.get("exposure_checkpoints", ())
            if checkpoints and max_exposure < max(int(value) for value in checkpoints):
                raise RunnerError(
                    f"formal blob BN {label} max_exposure={max_exposure} is below the largest checkpoint"
                )
    # The two adapter files must also be disjoint by IDs/sigmas.  Loading them
    # again is cheap and still read-only; no image rendering occurs here.
    try:
        from .blob_bn_adapter import load_appearance_set, validate_train_eval_appearance_sets

        train = load_appearance_set(bindings["train"][0], expected_sha256=bindings["train"][1], role="train", expected_count=48)
        evaluation = load_appearance_set(bindings["eval"][0], expected_sha256=bindings["eval"][1], role="eval", expected_count=8)
        validate_train_eval_appearance_sets(train, evaluation, expected_train_count=48, expected_eval_count=8)
    except Exception as exc:
        raise RunnerError(f"formal blob BN train/eval appearance split preflight failed: {exc}") from exc


def _adapter_value(config: Mapping[str, Any], label: str, *, direct: Any = None) -> Any:
    names = {
        "model": ("model_adapter", "model"),
        "data": ("data_adapter", "data", "batch_adapter"),
        "evaluator": ("evaluator_adapter", "evaluator"),
    }[label]
    value = next((config.get(name) for name in names if config.get(name)), None)
    if callable(direct):
        # Direct injection is only accepted alongside the named adapter lock.
        return direct
    if isinstance(value, str):
        return load_adapter(value, label=label)
    raise RunnerError(f"BN {label} adapter must be a module:function string or injected callable")


def _call_adapter(
    adapter: Callable[..., Any],
    config: Mapping[str, Any],
    label: str,
    *,
    device_override: str | None = None,
) -> Any:
    kwargs_key = f"{label}_kwargs"
    kwargs = config.get(kwargs_key, {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise RunnerError(f"BN {kwargs_key} must be an object")
    original_kwargs = dict(kwargs)
    if label == "model" and device_override is not None:
        # The validated runner device is authoritative.  In particular, do
        # not let a repeated model_kwargs.device value reach the constructor.
        kwargs = dict(kwargs)
        kwargs["device"] = device_override
    try:
        return adapter(**dict(kwargs))
    except TypeError:
        # Some adapters deliberately take no kwargs.  Do not retry arbitrary
        # signatures; only an explicitly empty kwargs object gets this path.
        # A few test/dummy adapters intentionally take no arguments.  We still
        # attempt the authoritative device first; only a completely empty
        # original model_kwargs may use this compatibility fallback.  Any
        # declared kwargs (including a declared device) must be accepted.
        if original_kwargs or label != "model" or device_override is None:
            raise
        return adapter()


def _resolve_bn_device(protocol: ProtocolLock, config: Mapping[str, Any]) -> str:
    """Resolve the one device allowed to cross the BN adapter boundary.

    ``device`` is deliberately required at the top level.  A nested
    ``model_kwargs.device`` is checked as a duplicate declaration rather than
    being trusted as an override.  This is important for the remote CPU
    preflight: allowing a nested CUDA value to reach a model constructor would
    bypass the stage gate before the control itself has a chance to inspect it.
    """

    if "device" not in config or config.get("device") in (None, ""):
        raise RunnerError("BN adapter config requires an explicit top-level device")
    device = validate_device(protocol, config.get("device"))
    if device is None:  # defensive: validate_device currently returns None only for None
        raise RunnerError("BN adapter config device resolved to no device")
    model_kwargs = config.get("model_kwargs", {})
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, Mapping):
        raise RunnerError("BN model_kwargs must be an object")
    if "device" in model_kwargs:
        nested = model_kwargs.get("device")
        if nested in (None, ""):
            raise RunnerError("BN model_kwargs.device cannot be empty")
        nested_device = validate_device(protocol, nested)
        if nested_device != device:
            raise StageError(
                "BN top-level device and model_kwargs.device disagree: "
                f"{device!r} != {nested_device!r}"
            )
    return device


def _bn_scientific_thresholds(protocol: ProtocolLock, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the frozen BN scientific threshold, without inventing one silently."""

    protocol_stage = protocol.payload.get("stages", {})
    protocol_bn = protocol_stage.get("bn", {}) if isinstance(protocol_stage, Mapping) else {}
    if not isinstance(protocol_bn, Mapping):
        protocol_bn = {}
    config_thresholds = config.get("scientific_thresholds", config.get("bn_scientific", {}))
    if config_thresholds is None:
        config_thresholds = {}
    if not isinstance(config_thresholds, Mapping):
        raise RunnerError("BN scientific_thresholds must be an object")
    protocol_thresholds = protocol_bn.get("scientific_thresholds", protocol_bn.get("bn_scientific", {}))
    if protocol_thresholds is None:
        protocol_thresholds = {}
    if not isinstance(protocol_thresholds, Mapping):
        raise RunnerError("protocol stages.bn scientific_thresholds must be an object")

    # A config override is allowed only when it agrees with the protocol lock;
    # otherwise a run could change the scientific gate without a new protocol.
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key in ("target_sgd_ratio", "min_explained_fraction"):
        if key in protocol_thresholds:
            values[key] = protocol_thresholds[key]
            sources[key] = "protocol"
        if key in config_thresholds:
            if key in values and float(config_thresholds[key]) != float(values[key]):
                raise RunnerError(f"BN {key} differs from the protocol lock")
            values[key] = config_thresholds[key]
            sources[key] = "config+protocol" if key in sources else "config"
    missing = [key for key in ("target_sgd_ratio", "min_explained_fraction") if key not in values]
    if missing:
        return {
            "available": False,
            "missing": missing,
            "source": sources,
            "target_sgd_ratio": None,
            "min_explained_fraction": None,
            "threshold_ratio": None,
        }
    try:
        target = float(values["target_sgd_ratio"])
        fraction = float(values["min_explained_fraction"])
    except (TypeError, ValueError) as exc:
        raise RunnerError("BN scientific thresholds must be numeric") from exc
    if not np.isfinite(target) or target <= 1.0:
        raise RunnerError("BN target_sgd_ratio must be finite and > 1")
    if not np.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise RunnerError("BN min_explained_fraction must be finite in (0, 1]")
    threshold = 1.0 + (target - 1.0) * fraction
    return {
        "available": True,
        "missing": [],
        "source": sources,
        "target_sgd_ratio": target,
        "min_explained_fraction": fraction,
        "threshold_ratio": threshold,
    }


def _bn_scientific_audit(report_payload: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Extract exposure ratios from evaluator records and apply the frozen gate."""

    records = report_payload.get("records", [])
    ratios: list[dict[str, Any]] = []
    baseline: float | None = None
    missing_reason: str | None = None
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        missing_reason = "BN report records are missing"
        records = []
    for item in records:
        if not isinstance(item, Mapping):
            missing_reason = "BN report contains a malformed record"
            continue
        evaluation = item.get("evaluation")
        value: Any = evaluation.get("affine_removed_mae_px") if isinstance(evaluation, Mapping) else None
        exposure = item.get("exposure")
        try:
            exposure_int = int(exposure)
            value_float = float(value)
        except (TypeError, ValueError, OverflowError):
            missing_reason = "affine_removed_mae_px is missing from one or more evaluator records"
            continue
        if not np.isfinite(value_float) or value_float < 0.0:
            missing_reason = "affine_removed_mae_px contains a non-finite or negative value"
            continue
        if exposure_int == 0:
            baseline = value_float
        ratios.append({"exposure": exposure_int, "affine_removed_mae_px": value_float})
    ratios.sort(key=lambda row: int(row["exposure"]))
    if baseline is None:
        missing_reason = missing_reason or "exposure-zero affine_removed_mae_px baseline is missing"
    elif baseline <= 0.0:
        missing_reason = "exposure-zero affine_removed_mae_px baseline must be positive"
    u_ratios: list[dict[str, Any]] = []
    if missing_reason is None and baseline is not None:
        for row in ratios:
            u_ratios.append(
                {
                    "exposure": row["exposure"],
                    "u_ratio": float(row["affine_removed_mae_px"] / baseline),
                    "affine_removed_mae_px": row["affine_removed_mae_px"],
                }
            )
    max_ratio = max((float(row["u_ratio"]) for row in u_ratios), default=None)
    final_ratio = float(u_ratios[-1]["u_ratio"]) if u_ratios else None
    explained_fraction = None
    scientific_status = "inconclusive"
    if missing_reason is None and max_ratio is not None and thresholds.get("available"):
        target = float(thresholds["target_sgd_ratio"])
        explained_fraction = float((max_ratio - 1.0) / (target - 1.0))
        scientific_status = (
            "explains_material_fraction"
            if max_ratio >= float(thresholds["threshold_ratio"])
            else "small"
        )
    elif missing_reason is None and not thresholds.get("available"):
        missing_reason = "BN scientific thresholds are not frozen in protocol/config"
    return {
        "baseline_affine_removed_mae_px": baseline,
        "u_ratios": u_ratios,
        "max_u_ratio": max_ratio,
        "final_u_ratio": final_ratio,
        "explained_fraction_at_max": explained_fraction,
        "scientific_status": scientific_status,
        "scientific_reason": missing_reason,
        "thresholds": dict(thresholds),
    }


def run_bn(
    config_path: Path | str,
    *,
    output_dir: Path | str,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    model: Any = None,
    batches: Iterable[Any] | None = None,
    evaluate: Callable[[Any, int], Any] | None = None,
    _context: tuple[ProtocolLock, AssetManifest] | None = None,
) -> dict[str, Any]:
    """Run an explicit BN-only control with model/data/evaluator adapters."""

    config = _load_bn_config(_path(config_path))
    config_path = _path(config_path)
    protocol, manifest = _context or _prepare_context(
        protocol_path=protocol_path,
        asset_manifest_path=asset_manifest_path,
        input_paths=None if asset_manifest_path else [config_path],
        input_roles=None if asset_manifest_path else ["bn_adapter_config"],
    )
    selected = _asset_paths(manifest, [config_path])
    _assert_roles(selected, {"bn_adapter_config", "bn_config"}, "bn")
    # Device validation is intentionally before any adapter is constructed or
    # an output bundle is created.  This also prevents nested model kwargs from
    # bypassing a remote CPU-only stage.
    device = _resolve_bn_device(protocol, config)
    _validate_bn_static_bindings(config_path, config)
    model_adapter = _adapter_value(config, "model", direct=model)
    data_adapter = _adapter_value(config, "data", direct=batches if callable(batches) else None)
    evaluator_adapter = _adapter_value(config, "evaluator", direct=evaluate)
    model_value = model if model is not None else _call_adapter(
        model_adapter, config, "model", device_override=device
    )
    batch_value = batches if batches is not None and not callable(batches) else _call_adapter(data_adapter, config, "data")
    evaluate_value = evaluate if evaluate is not None else _call_adapter(evaluator_adapter, config, "evaluator")
    if not callable(evaluate_value):
        raise RunnerError("BN evaluator adapter must return a callable (model, exposure) evaluator")
    checkpoints = config.get("exposure_checkpoints", (0, 1, 10, 50, 100, 500, 3000))
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
        raise RunnerError("BN exposure_checkpoints must be a list")
    scientific_thresholds = _bn_scientific_thresholds(protocol, config)
    out = _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=["audit-bn", str(config_path)],
        source_machine=source_machine,
        extra_manifest={
            "stage": "bn",
            "bn_adapter_config": str(config_path),
            "bn_explicit_adapters": ["model", "data", "evaluator"],
        },
    )
    report = run_bn_only_control(
        model_value,
        batch_value,
        exposure_checkpoints=tuple(checkpoints),
        evaluate=evaluate_value,
        batch_to_inputs=None,
        device=device,
        strict=bool(config.get("strict", True)),
    )
    report_payload = report.to_dict()
    write_json(out / "bn_report.json", report_payload)
    scientific = _bn_scientific_audit(report_payload, scientific_thresholds)
    positive_records = tuple(record for record in report.records if int(record.exposure) > 0)
    has_bn_buffers = bool(report.bn_buffer_names)
    has_exposure_update = any(bool(record.bn_changed_buffer_names) for record in positive_records)
    control_valid = bool(
        report.parameters_unchanged
        and report.only_bn_buffers_changed
        and has_bn_buffers
        and has_exposure_update
    )
    metrics = {
        "stage": "bn",
        "report": report_payload,
        "parameters_unchanged": report.parameters_unchanged,
        "only_bn_buffers_changed": report.only_bn_buffers_changed,
        "control_valid": control_valid,
        "has_bn_running_buffers": has_bn_buffers,
        "has_exposure_bn_update": has_exposure_update,
        "scientific": scientific,
        "device": device,
        "explicit_adapters": ["model", "data", "evaluator"],
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
    }
    decision = {
        "stage": "bn",
        "status": (
            "passed"
            if control_valid and scientific["scientific_status"] == "explains_material_fraction"
            else ("failed" if not control_valid else scientific["scientific_status"])
        ),
        "passed": bool(
            control_valid and scientific["scientific_status"] == "explains_material_fraction"
        ),
        "control_valid": control_valid,
        "scientific_status": scientific["scientific_status"],
        "max_u_ratio": scientific["max_u_ratio"],
        "final_u_ratio": scientific["final_u_ratio"],
        "thresholds": scientific["thresholds"],
        "reason": (
            "only BN running buffers changed and the frozen scientific threshold was met"
            if control_valid and scientific["scientific_status"] == "explains_material_fraction"
            else (
                "BN-only invariant failed"
                if not control_valid
                else scientific["scientific_reason"] or "frozen scientific threshold was not met"
            )
        ),
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    return {"metrics": metrics, "decision": decision, "output_dir": str(out)}


def _load_partial_group(path: Path) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    with np.load(path, allow_pickle=False) as blob:
        if "anchors" not in blob.files:
            raise RunnerError(
                f"partial group input requires explicit anchors (no corner fallback): {path}"
            )
        anchors = np.array(blob["anchors"], copy=True)
        if "signals_vector" in blob.files:
            vector = np.array(blob["signals_vector"], copy=True)
            if vector.ndim != 5 or vector.shape[-1] != 2:
                raise RunnerError(
                    "partial signals_vector must have shape "
                    "[appearance,y,x,seed,2]"
                )
            appearance, ny, nx, seed_count, _ = vector.shape
            if seed_count < 2:
                raise RunnerError("partial signals_vector requires at least two seed columns")
            saved_coords = np.asarray(blob["coords"], dtype=np.float64) if "coords" in blob.files else None
            x_for_output: np.ndarray | None = None
            y_for_output: np.ndarray | None = None
            if saved_coords is not None:
                if saved_coords.shape == (appearance * ny * nx, 2):
                    coord_grid = saved_coords.reshape(appearance, ny, nx, 2)
                elif saved_coords.shape == (appearance, ny, nx, 2):
                    coord_grid = saved_coords
                elif saved_coords.shape == (ny, nx, 2):
                    coord_grid = np.broadcast_to(
                        saved_coords[None, ...], (appearance, ny, nx, 2)
                    )
                else:
                    raise RunnerError(
                        "partial signals_vector coords must be [y,x,2], "
                        "[appearance,y,x,2], or repeated [appearance*y*x,2]"
                    )
                if not np.allclose(coord_grid, coord_grid[0][None, ...], rtol=0.0, atol=0.0):
                    raise RunnerError(
                        "partial signals_vector coordinates differ across appearances"
                    )
                base_coords = np.asarray(coord_grid[0], dtype=np.float64)
                if "x" in blob.files or "y" in blob.files:
                    if "x" not in blob.files or "y" not in blob.files:
                        raise RunnerError("partial signals_vector coords require both x and y when either is present")
                    x_raw = np.asarray(blob["x"], dtype=np.float64).reshape(-1)
                    y_raw = np.asarray(blob["y"], dtype=np.float64).reshape(-1)
                    expected_x = coord_grid[..., 0].reshape(-1)
                    expected_y = coord_grid[..., 1].reshape(-1)
                    if x_raw.size == expected_x.size and y_raw.size == expected_y.size:
                        if not np.allclose(x_raw, expected_x, rtol=0.0, atol=0.0) or not np.allclose(y_raw, expected_y, rtol=0.0, atol=0.0):
                            raise RunnerError("partial signals_vector x/y disagree with validated coords")
                        x_for_output, y_for_output = expected_x, expected_y
                    elif x_raw.shape == (ny * nx,) and y_raw.shape == (ny * nx,):
                        expected_base_x = base_coords[..., 0].reshape(-1)
                        expected_base_y = base_coords[..., 1].reshape(-1)
                        if not np.allclose(x_raw, expected_base_x, rtol=0.0, atol=0.0) or not np.allclose(y_raw, expected_base_y, rtol=0.0, atol=0.0):
                            raise RunnerError("partial signals_vector x/y disagree with validated coords")
                        x_for_output, y_for_output = expected_base_x, expected_base_y
                    else:
                        raise RunnerError("partial signals_vector x/y shape does not match coords")
            else:
                if "x" not in blob.files or "y" not in blob.files:
                    raise RunnerError("partial signals_vector requires explicit coords or x/y arrays")
                x_value = np.asarray(blob["x"], dtype=np.float64)
                y_value = np.asarray(blob["y"], dtype=np.float64)
                if x_value.ndim == 1 and y_value.ndim == 1:
                    if x_value.size == appearance * ny * nx and y_value.size == appearance * ny * nx:
                        x_grid = x_value.reshape(appearance, ny, nx)
                        y_grid = y_value.reshape(appearance, ny, nx)
                        if not np.allclose(x_grid, x_grid[0][None, ...], rtol=0.0, atol=0.0) or not np.allclose(y_grid, y_grid[0][None, ...], rtol=0.0, atol=0.0):
                            raise RunnerError("partial signals_vector x/y differ across appearances")
                        base_coords = np.stack([x_grid[0], y_grid[0]], axis=-1)
                        x_for_output, y_for_output = x_value.reshape(-1), y_value.reshape(-1)
                    elif x_value.size == nx and y_value.size == ny:
                        x_grid, y_grid = np.meshgrid(x_value, y_value, indexing="xy")
                        base_coords = np.stack([x_grid, y_grid], axis=-1)
                        x_for_output, y_for_output = base_coords[..., 0].reshape(-1), base_coords[..., 1].reshape(-1)
                    else:
                        raise RunnerError(
                            "partial signals_vector x/y lengths do not match its y/x dimensions"
                        )
                elif x_value.shape == (ny, nx) and y_value.shape == (ny, nx):
                    base_coords = np.stack([x_value, y_value], axis=-1)
                    x_for_output, y_for_output = x_value.reshape(-1), y_value.reshape(-1)
                else:
                    raise RunnerError(
                        "partial signals_vector x/y must be 1-D grid axes, repeated flat arrays, or [y,x] arrays"
                    )
            if not np.all(np.isfinite(base_coords)):
                raise RunnerError("partial signals_vector coordinates contain non-finite values")
            # Vector partial-out expects one shared spatial coordinate grid;
            # appearance is already an explicit axis of signals_vector.
            coords = base_coords.reshape(-1, 2)
            # Preserve the vector contract for partial_out_vector_group.  Only
            # the spatial grid axes are flattened; appearance, seed and the
            # final (x,y) component remain explicit dimensions.
            signals_vector = vector.reshape(appearance, ny * nx, seed_count, 2)
            signals = signals_vector
            payload: dict[str, Any] = {
                "coords": coords,
                "signals": signals,
                "signals_vector": signals_vector,
                "anchors": anchors,
                "x": (coords[:, 0].copy() if x_for_output is None else np.asarray(x_for_output, dtype=np.float64).copy()),
                "y": (coords[:, 1].copy() if y_for_output is None else np.asarray(y_for_output, dtype=np.float64).copy()),
                "vector_contract": "signals_vector[appearance,y,x,seed,2]",
                "gate_signal": "pooled_vector_flatten",
            }
        elif "coords" in blob.files and "signals" in blob.files:
            coords = np.array(blob["coords"], copy=True)
            signals = np.array(blob["signals"], copy=True)
            payload = {
                "coords": coords,
                "signals": signals,
                "anchors": anchors,
                "x": np.asarray(coords)[..., 0].copy(),
                "y": np.asarray(coords)[..., 1].copy(),
                "vector_contract": None,
            }
        else:
            raise RunnerError(
                f"partial group input must contain coords/signals or signals_vector: {path}"
            )
        if "domain" in blob.files:
            payload["domain"] = np.array(blob["domain"], copy=True)
    return coords, signals, payload


def run_partial(
    group_inputs: Mapping[str, Path | str],
    *,
    output_dir: Path | str,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
    block_size: float | None = None,
    _context: tuple[ProtocolLock, AssetManifest] | None = None,
) -> dict[str, Any]:
    """Run partial-out independently for each named task group."""

    if not group_inputs:
        raise RunnerError("audit-partial requires at least one named group input")
    paths = [_path(path) for path in group_inputs.values()]
    roles = ["partial_input"] * len(paths)
    protocol, manifest = _context or _prepare_context(
        protocol_path=protocol_path,
        asset_manifest_path=asset_manifest_path,
        input_paths=None if asset_manifest_path else paths,
        input_roles=None if asset_manifest_path else roles,
    )
    selected = _asset_paths(manifest, paths)
    _assert_roles(selected, {"partial_input", "partial_npz", "partial_group"}, "partial")
    loaded: dict[str, Mapping[str, Any]] = {}
    for group_name, raw_path in group_inputs.items():
        if not str(group_name).strip():
            raise RunnerError("partial group names cannot be empty")
        coords, signals, payload = _load_partial_group(_path(raw_path))
        loaded[str(group_name)] = payload
    out = _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=["audit-partial", *[f"{name}={path}" for name, path in group_inputs.items()]],
        source_machine=source_machine,
        extra_manifest={"stage": "partial", "groups": list(loaded)},
    )
    kwargs: dict[str, Any] = {}
    if block_size is not None:
        kwargs["block_size"] = float(block_size)
    results = partial_out_groups(loaded, **kwargs)
    metrics_groups: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for group_name, result in results.items():
        payload = result.to_dict()
        source_payload = loaded[group_name]
        # These are explicit audit outputs, not inferred from the correlation
        # matrix: reviewers can trace every pooled row back to its x/y point.
        payload["x"] = np.asarray(source_payload["x"], dtype=np.float64).reshape(-1).tolist()
        payload["y"] = np.asarray(source_payload["y"], dtype=np.float64).reshape(-1).tolist()
        payload["pooled_original_correlations"] = result.original_correlations.tolist()
        payload["pooled_partial_correlations"] = result.partial_correlations.tolist()
        payload["anchors"] = np.asarray(source_payload["anchors"], dtype=np.float64).tolist()
        payload["gate_signal"] = payload.get("gate_signal", "pooled_vector_flatten" if source_payload.get("signals_vector") is not None else "scalar_seed_matrix")
        if source_payload.get("vector_contract"):
            payload["vector_contract"] = source_payload["vector_contract"]
        metrics_groups[group_name] = _result_without_arrays(payload)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_name).strip("._") or "group"
        write_json(out / f"group_{safe}.json", payload)
        predictions[f"{safe}_original_correlations"] = result.original_correlations
        predictions[f"{safe}_partial_correlations"] = result.partial_correlations
        predictions[f"{safe}_residual_signals"] = result.residual_signals
        predictions[f"{safe}_x"] = np.asarray(source_payload["x"], dtype=np.float64).reshape(-1)
        predictions[f"{safe}_y"] = np.asarray(source_payload["y"], dtype=np.float64).reshape(-1)
    np.savez_compressed(out / "predictions.npz", **predictions)
    metrics = {
        "stage": "partial",
        "groups": metrics_groups,
        "group_count": len(metrics_groups),
        "groups_independent": True,
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
    }
    group_outcomes = {name: result.gate.to_dict() for name, result in results.items()}
    passed_groups = [name for name, result in results.items() if result.gate.status == "survived"]
    closed_groups = [name for name, result in results.items() if result.gate.status == "closed"]
    inconclusive_groups = [
        name for name, result in results.items() if result.gate.status == "inconclusive"
    ]
    if passed_groups and not closed_groups and not inconclusive_groups:
        aggregate_status = "all_survived"
    elif closed_groups and not passed_groups and not inconclusive_groups:
        aggregate_status = "all_closed"
    else:
        aggregate_status = "mixed"
    decision = {
        "stage": "partial",
        # Computation completed even when one group is closed/inconclusive;
        # scientific outcome is carried by aggregate_status and group lists.
        "status": "complete",
        "aggregate_status": aggregate_status,
        "passed": bool(all(result.gate.passed for result in results.values())),
        "group_outcomes": group_outcomes,
        "groups": group_outcomes,
        "passed_groups": passed_groups,
        "closed_groups": closed_groups,
        "inconclusive_groups": inconclusive_groups,
        "reason": "each task group was fitted and reported separately",
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    return {"metrics": metrics, "decision": decision, "output_dir": str(out)}


def run_s0(
    *,
    output_dir: Path | str,
    protocol_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    source_machine: str = DEFAULT_SOURCE_MACHINE,
) -> dict[str, Any]:
    """Execute CPU S0 in the fixed check→headlines/OLS→kernel→BN→partial order."""

    protocol, manifest = _prepare_context(protocol_path=protocol_path, asset_manifest_path=asset_manifest_path)

    # Resolve and validate every stage before creating any S0 output.  This
    # prevents a half-run (for example check/headlines output left behind
    # before a missing OLS/kernel/BN asset is discovered).
    headline_specs = _stage_specs(protocol, manifest, "headlines")
    if not headline_specs or any(spec.role not in {"field_rendered_xy", "field", "field_2d", "rendered_xy"} for spec in headline_specs):
        raise RunnerError("S0 headlines stage requires field_rendered_xy assets")
    ols_specs = _stage_specs(protocol, manifest, "ols")
    if len(ols_specs) != 1 or ols_specs[0].role not in {"ols", "ols_features", "frozen_g64_ols"}:
        raise RunnerError("S0 protocol requires exactly one separated OLS asset")
    from .ols_audit import load_design_target_npz

    load_design_target_npz(ols_specs[0].path)
    kernel_specs = _stage_specs(protocol, manifest, "kernel")
    if len(kernel_specs) != 1 or kernel_specs[0].role not in {"kernel_input", "kernel_npz", "gap_features"}:
        raise RunnerError("S0 protocol requires exactly one kernel_input asset")
    kernel_preflight_payload = _load_kernel_npz(kernel_specs[0].path)
    _require_formal_kernel_payload(kernel_preflight_payload)
    bn_specs = _stage_specs(protocol, manifest, "bn")
    if len(bn_specs) != 1 or bn_specs[0].role not in {"bn_adapter_config", "bn_config"}:
        raise RunnerError("S0 protocol requires exactly one BN adapter config")
    bn_config = _load_bn_config(bn_specs[0].path)
    # Static S0 validation must apply the same top-level/nested device rule as
    # run_bn, before the composite output bundle is created.
    _resolve_bn_device(protocol, bn_config)
    _validate_bn_static_bindings(bn_specs[0].path, bn_config)
    partial_stages = protocol.payload.get("stages", {})
    partial_cfg = partial_stages.get("partial", {}) if isinstance(partial_stages, Mapping) else {}
    partial_block_size = None
    if isinstance(partial_cfg, Mapping) and partial_cfg.get("block_size") is not None:
        try:
            partial_block_size = float(partial_cfg.get("block_size"))
        except (TypeError, ValueError) as exc:
            raise RunnerError("protocol stages.partial.block_size must be numeric") from exc
        if not np.isfinite(partial_block_size) or partial_block_size <= 0:
            raise RunnerError("protocol stages.partial.block_size must be finite and positive")
    partial_groups: dict[str, Path] = {}
    if isinstance(partial_cfg, Mapping) and isinstance(partial_cfg.get("groups"), Mapping):
        for group_name, asset_id in partial_cfg["groups"].items():
            spec = manifest.by_id(str(asset_id))
            if spec.role not in {"partial_input", "partial_npz", "partial_group"}:
                raise RunnerError(f"partial group asset has invalid role: {spec.asset_id}")
            _load_partial_group(spec.path)
            partial_groups[str(group_name)] = spec.path
    else:
        partial_specs = _stage_specs(protocol, manifest, "partial")
        if not partial_specs:
            raise RunnerError("S0 partial stage requires at least one group asset")
        for spec in partial_specs:
            if spec.role not in {"partial_input", "partial_npz", "partial_group"}:
                raise RunnerError(f"partial asset has invalid role: {spec.asset_id}")
            _load_partial_group(spec.path)
            partial_groups[spec.group or spec.asset_id] = spec.path
    if not partial_groups:
        raise RunnerError("S0 partial stage requires at least one named group")

    out = _begin_bundle(
        output_dir,
        protocol,
        manifest,
        command=["run-s0"],
        source_machine=source_machine,
        extra_manifest={"stage": protocol.payload.get("current_stage", protocol.payload.get("stage")), "run_order": ["check", "headlines", "ols", "kernel", "bn", "partial"]},
    )
    context = (protocol, manifest)
    stage_results: dict[str, Any] = {}
    order: list[str] = []

    # The fixed order is intentionally explicit rather than iterating a user
    # supplied list.  Any later failure propagates and prevents later stages.
    check_result = run_check(output_dir=out / "check", source_machine=source_machine, _context=context)
    stage_results["check"] = check_result["decision"]
    order.append("check")

    headline_result = run_headlines(
        [spec.path for spec in headline_specs],
        output_dir=out / "headlines",
        source_machine=source_machine,
        _context=context,
    )
    stage_results["headlines"] = headline_result["decision"]
    order.append("headlines")

    # ``run_ols`` accepts a protocol path for public use.  For the composite
    # runner, use the same lock and avoid reloading it by executing the small
    # deterministic operation inline; this prevents accidental lock
    # replacement between stages.
    ols_path = ols_specs[0].path
    result = audit_ols_npz(ols_path, out / "ols", source_machine=source_machine)
    _refresh_bundle(out / "ols", protocol, manifest, command=["audit-ols", str(ols_path)], source_machine=source_machine, extra_manifest={"stage": "ols"})
    ols_metrics = {"stage": "ols", "stability": result.get("stability"), "n_train": result.get("n_train"), "n_eval": result.get("n_eval"), "n_features": result.get("n_features"), "precision_modes": result.get("precision_modes"), "protocol_lock_sha256": protocol.sha256, "asset_manifest_sha256": manifest.sha256}
    ols_result = {"metrics": ols_metrics, "decision": {"stage": "ols", "status": "complete", "passed": True, "numerical_stability_verdict": result.get("stability", {}).get("verdict")}}
    _finish(out / "ols", protocol, manifest, metrics=ols_metrics, decision=ols_result["decision"])
    stage_results["ols"] = ols_result["decision"]
    order.append("ols")

    # Reuse the locked context through the internal equivalent by calling the
    # public runner with a temporary manifest would be unsafe; execute its
    # deterministic input/fit/gate sequence here.
    kernel_path = kernel_specs[0].path
    kernel_payload = _load_kernel_npz(kernel_path)
    _require_formal_kernel_payload(kernel_payload)
    kernel_out = _begin_bundle(out / "kernel", protocol, manifest, command=["audit-kernel", str(kernel_path)], source_machine=source_machine, extra_manifest={"stage": "kernel", "heldout_targets_used_for_fit": False})
    predictor = fit_fixed_log_error_predictor(kernel_payload["fit_features"], kernel_payload["fit_errors"], feature_names=kernel_payload["predictor_feature_names"])
    predicted_log = predictor.predict_log(kernel_payload["heldout_features"])
    gate = evaluate_kernel_gate(predicted_log, kernel_payload["heldout_errors"], kernel_payload["baseline"])
    np.savez_compressed(kernel_out / "predictions.npz", heldout_predicted_log_error=predicted_log, heldout_predicted_error=np.exp(predicted_log), heldout_observed_error=kernel_payload["heldout_errors"], heldout_euclidean_baseline_error=kernel_payload["baseline"])
    kernel_metrics = {"stage": "kernel", "fit_count": int(kernel_payload["fit_errors"].size), "heldout_count": int(kernel_payload["heldout_errors"].size), "predictor": predictor.to_dict(), "gate": gate.to_dict(), "heldout_targets_used_for_fit": False, "schema_kind": kernel_payload["schema_kind"], "feature_names": list(kernel_payload["feature_names"]), "protocol_sha256": kernel_payload["protocol_sha256"], "partition": kernel_payload["partition"], "baseline_source": kernel_payload["baseline_source"], "provenance_token_sha256": kernel_payload["provenance_token_sha256"], "protocol_lock_sha256": protocol.sha256, "asset_manifest_sha256": manifest.sha256}
    kernel_decision = {"stage": "kernel", "status": "passed" if gate.passed else "closed", "passed": bool(gate.passed), "reason": gate.reason}
    _finish(kernel_out, protocol, manifest, metrics=kernel_metrics, decision=kernel_decision)
    stage_results["kernel"] = kernel_decision
    order.append("kernel")

    # The composite runner deliberately requires the config to expose direct
    # adapters.  Its execution is delegated to run_bn after preserving the
    # fixed stage order; public callers get the same validation.
    bn_result = run_bn(bn_specs[0].path, output_dir=out / "bn", source_machine=source_machine, _context=context)
    stage_results["bn"] = bn_result["decision"]
    order.append("bn")

    partial_result = run_partial(partial_groups, output_dir=out / "partial", source_machine=source_machine, block_size=partial_block_size, _context=context)
    stage_results["partial"] = partial_result["decision"]
    order.append("partial")

    metrics = {
        "stage": protocol.payload.get("current_stage", protocol.payload.get("stage")),
        "run_order": order,
        "required_run_order": ["check", "headlines", "ols", "kernel", "bn", "partial"],
        "stages": stage_results,
        "protocol_lock_sha256": protocol.sha256,
        "asset_manifest_sha256": manifest.sha256,
    }
    decision = {
        "stage": protocol.payload.get("current_stage", protocol.payload.get("stage")),
        "status": "complete",
        "passed": order == ["check", "headlines", "ols", "kernel", "bn", "partial"],
        "reason": "S0 completed in the locked order",
    }
    _finish(out, protocol, manifest, metrics=metrics, decision=decision)
    return {"metrics": metrics, "decision": decision, "output_dir": str(out)}


__all__ = [
    "RunnerError",
    "run_bn",
    "run_check",
    "run_headlines",
    "run_kernel",
    "run_ols",
    "run_partial",
    "run_s0",
]
