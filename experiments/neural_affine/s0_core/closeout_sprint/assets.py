"""Read-only provenance and environment helpers for the clean-room sprint."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import SCHEMA_VERSION, write_json


CHUNK_BYTES = 1024 * 1024


def _path(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Hash a file without loading an old checkpoint or archive into memory."""

    p = _path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        while True:
            block = handle.read(CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _npz_summary(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as blob:
            return {
                "npz_keys": list(blob.files),
                "npz_shapes": {key: list(blob[key].shape) for key in blob.files},
                "npz_dtypes": {key: str(blob[key].dtype) for key in blob.files},
            }
    except Exception as exc:  # provenance must still identify a malformed file
        return {"npz_error": f"{type(exc).__name__}: {exc}"}


def _torch_summary(path: Path) -> dict[str, Any]:
    """Safe-only checkpoint inspection, never importing project model code.

    ``weights_only=True`` is intentional.  Falling back to pickle loading
    would let a provenance check execute arbitrary objects from an old asset.
    """

    if importlib.util.find_spec("torch") is None:
        return {"torch_available": False}
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            return {
                "torch_available": True,
                "safe_load": False,
                "safe_load_error": f"{type(exc).__name__}: {exc}",
                "head_check": "unavailable",
            }
        state = payload.get("model_state", payload) if isinstance(payload, Mapping) else payload
        tensors: dict[str, list[int]] = {}
        if isinstance(state, Mapping):
            for key, value in state.items():
                if hasattr(value, "shape"):
                    tensors[str(key)] = [int(v) for v in value.shape]
        out: dict[str, Any] = {
            "torch_available": True,
            "safe_load": True,
            "tensor_count": len(tensors),
            "tensor_shapes": tensors,
        }
        # Do not infer a head from whichever 2-D tensor happens to be last.
        # Slim ResNet checkpoints in this workspace use fc.weight/fc.bias;
        # payloads that carry an explicit head_dim are also accepted.
        explicit_dim = payload.get("head_dim") if isinstance(payload, Mapping) else None
        state_keys = set(state.keys()) if isinstance(state, Mapping) else set()
        for weight_key, bias_key in (("fc.weight", "fc.bias"), ("head.weight", "head.bias")):
            if weight_key in state_keys and bias_key in state_keys:
                weight_shape = tensors.get(weight_key)
                bias_shape = tensors.get(bias_key)
                if weight_shape and bias_shape and len(weight_shape) == 2 and bias_shape == [weight_shape[0]]:
                    out["head_key"] = weight_key.rsplit(".", 1)[0]
                    out["head_out"] = int(weight_shape[0])
                    out["head_in"] = int(weight_shape[1])
                    out["head_check"] = "explicit_state_keys"
                    break
        else:
            if isinstance(explicit_dim, (int, np.integer)):
                out["head_out"] = int(explicit_dim)
                out["head_check"] = "explicit_payload_head_dim"
            else:
                out["head_check"] = "unresolved"
        if isinstance(payload, Mapping):
            out["payload_keys"] = [str(k) for k in payload.keys()]
        return out
    except Exception as exc:
        return {"torch_available": True, "safe_load": False, "torch_error": f"{type(exc).__name__}: {exc}"}


def asset_record(
    path: os.PathLike[str] | str,
    *,
    role: str | None = None,
    source_machine: str = "local",
) -> dict[str, Any]:
    p = _path(path)
    record: dict[str, Any] = {
        "path": str(p),
        "role": role,
        "source_machine": source_machine,
        "exists": p.is_file(),
    }
    if not p.is_file():
        _validate_asset_record(record)
        return record
    record.update({"size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    if not record["sha256"]:
        raise ValueError(f"empty sha256 for asset: {p}")
    if p.suffix.lower() == ".npz":
        record.update(_npz_summary(p))
        if record.get("role") is None:
            keys = set(record.get("npz_keys", []))
            if {"pred", "true"}.issubset(keys):
                record["role"] = "field_rendered_xy"
            elif {
                "schema_kind", "target_semantics", "coord_scale", "head_dim",
                "train_gap_features", "eval_gap_features", "train_p6_norm",
                "train_q_px", "train_t_px", "eval_p6_norm", "eval_q_px", "eval_t_px",
                "train_shape_id", "train_translation_id", "eval_shape_id", "eval_dense_index",
                "provenance_json",
            }.issubset(keys):
                record["role"] = "ols"
            else:
                raise ValueError(
                    f"cannot infer clean-room NPZ schema/role for {p}; "
                    "expected pred/true or separated train/eval OLS arrays"
                )
    elif p.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        if record.get("role") is None:
            raise ValueError(f"checkpoint role must be explicit (checkpoint_2d or checkpoint_6d): {p}")
        record.update(_torch_summary(p))
    _validate_asset_record(record)
    return record


def _validate_asset_record(record: Mapping[str, Any]) -> None:
    """Fail fast on missing files, schema/role errors, and 2D/6D mixing."""

    if not record.get("exists"):
        raise FileNotFoundError(str(record.get("path")))
    role = str(record.get("role") or "")
    suffix = Path(str(record["path"])).suffix.lower()
    if suffix == ".npz":
        keys = set(record.get("npz_keys", []))
        if "npz_error" in record:
            raise ValueError(f"invalid NPZ schema: {record['path']}: {record['npz_error']}")
        if role in {"field", "field_2d", "field_rendered_xy", "rendered_xy"} and not {"pred", "true"}.issubset(keys):
            raise ValueError(f"field NPZ must contain pred and true: {record['path']}")
        if role in {"ols", "ols_features", "frozen_g64_ols"} and not ({
            "schema_kind", "target_semantics", "coord_scale", "head_dim",
            "train_gap_features", "eval_gap_features", "train_p6_norm",
            "train_q_px", "train_t_px", "eval_p6_norm", "eval_q_px", "eval_t_px",
            "train_shape_id", "train_translation_id", "eval_shape_id", "eval_dense_index",
            "provenance_json",
        }.issubset(keys)):
            raise ValueError(
                "OLS NPZ must contain formal frozen_g64_gap_to_p6_v1 fields: "
                f"{record['path']}"
            )
        if role in {"field", "field_2d", "field_rendered_xy", "rendered_xy"}:
            shapes = record.get("npz_shapes", {})
            shape = shapes.get("pred", [])
            expected = 2
            if not shape or int(shape[-1]) != expected:
                raise ValueError(f"field role {role} expects rendered output dimension {expected}: {record['path']}")
    if suffix in {".pt", ".pth", ".ckpt"}:
        if not record.get("safe_load", False):
            raise ValueError(f"checkpoint cannot be inspected with safe weights-only loading: {record['path']}")
        if record.get("head_check") == "unresolved":
            raise ValueError(f"checkpoint head role is unresolved (need fc.weight/fc.bias or head_dim): {record['path']}")
        if role.endswith("2d") or role.endswith("6d"):
            expected = 2 if role.endswith("2d") else 6
            if record.get("head_out") != expected:
                raise ValueError(f"checkpoint role {role} expects head_out={expected}, got {record.get('head_out')}: {record['path']}")


def build_asset_lock(
    paths: Iterable[os.PathLike[str] | str],
    *,
    source_machine: str = "local",
    role_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    role_map = role_map or {}
    normalized_roles = {str(_path(key)): value for key, value in role_map.items()}
    unique = sorted({_path(p) for p in paths}, key=str)
    assets = []
    for p in unique:
        role = normalized_roles.get(str(p))
        assets.append(asset_record(p, role=role, source_machine=source_machine))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "clean_room_asset_lock",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_machine": source_machine,
        "assets": assets,
    }


def environment_snapshot() -> dict[str, Any]:
    """Record versions without importing any historical project module."""

    snap: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "cwd": str(Path.cwd().resolve()),
        "cuda_available": None,
        "torch_version": None,
    }
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            snap["torch_version"] = torch.__version__
            snap["cuda_built"] = getattr(torch.version, "cuda", None)
            snap["cuda_available"] = bool(torch.cuda.is_available())
            snap["cuda_device_count"] = int(torch.cuda.device_count()) if snap["cuda_available"] else 0
        except Exception as exc:
            snap["torch_error"] = f"{type(exc).__name__}: {exc}"
    return snap


def _assert_output_is_new(output_dir: Path, inputs: Sequence[Path]) -> None:
    out = output_dir.resolve(strict=False)
    for inp in inputs:
        # Writing an output below an input directory would mutate a historical
        # asset tree.  A sibling/new results directory is the intended shape.
        if inp.is_dir() and (out == inp or inp in out.parents):
            raise ValueError(f"refuse output inside input asset directory: {out}")
        if inp.is_file() and out == inp.parent:
            raise ValueError(f"refuse output beside input without a new directory: {out}")


def write_run_bundle(
    output_dir: os.PathLike[str] | str,
    input_paths: Iterable[os.PathLike[str] | str],
    *,
    command: Sequence[str],
    source_machine: str = "local",
    role_map: Mapping[str, str] | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Create the mandatory asset_lock/environment/manifest trio."""

    inputs = [_path(p) for p in input_paths]
    out = _path(output_dir)
    _assert_output_is_new(out, inputs)
    out.mkdir(parents=True, exist_ok=True)
    lock = build_asset_lock(inputs, source_machine=source_machine, role_map=role_map)
    env = environment_snapshot()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "clean_room_run_manifest",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "output_dir": str(out),
        "input_paths": [str(p) for p in inputs],
        "input_sha256": {str(item["path"]): item.get("sha256") for item in lock["assets"]},
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    paths = {
        "asset_lock": write_json(out / "asset_lock.json", lock),
        "environment": write_json(out / "environment.json", env),
        "manifest": write_json(out / "manifest.json", manifest),
    }
    return paths
