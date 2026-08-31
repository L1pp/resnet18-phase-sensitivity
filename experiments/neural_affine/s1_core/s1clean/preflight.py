from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .baselines import closed_form_identity, run_explicit_xy_mlp
from .config import protocol_hash
from .environment import write_environment
from .manifest import sha256_file, verify_manifest
from .status import StatusStore


SEEDS = (20260816, 20260817, 20260818)
EXPECTED_INIT_HASHES = {
    20260816: "481f381c5a0acfa62809a584b19a09a5b3466d25214d8d256111782351b437bf",
    20260817: "ab2c09678d1fb3bfd25d8db1ac106fb43fb69af4966204e168f04e128c12cc15",
    20260818: "29050987fcb3cb7bb86ed5ff1d5896f2c3dc6cd4b4d8438df445ee0a83af76c4",
}


def state_dict_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        value = state[key]
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_checkpoint_state(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, Mapping) and "model_state" in payload:
        payload = payload["model_state"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path} does not contain a state dict")
    return payload


def historical_init_candidates(historical_root: str | Path, seed: int) -> list[Path]:
    base = Path(historical_root) / "results" / "today_shortcycle" / "a10" / "affine" / f"corners4_s{seed}"
    return [base / "checkpoints" / "init.pt", base / "init.pt"]


def isolate_exact_inits(historical_root: str | Path, input_root: str | Path) -> dict[str, Any]:
    destination = Path(input_root) / "legacy_init"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    all_exact = True
    for seed in SEEDS:
        source = next((item for item in historical_init_candidates(historical_root, seed) if item.is_file()), None)
        if source is None:
            rows.append({"seed": seed, "status": "missing"})
            all_exact = False
            continue
        actual = state_dict_hash(load_checkpoint_state(source))
        expected = EXPECTED_INIT_HASHES[seed]
        if actual != expected:
            rows.append({"seed": seed, "status": "hash_mismatch", "actual": actual, "expected": expected})
            all_exact = False
            continue
        target = destination / f"s{seed}_init.pt"
        if target.exists():
            # A successful earlier preflight deliberately seals this copy as
            # read-only.  Re-running the CPU gate must verify and reuse it,
            # never attempt to overwrite the sealed artifact.
            target_actual = state_dict_hash(load_checkpoint_state(target))
            if target_actual != expected:
                rows.append(
                    {
                        "seed": seed,
                        "status": "sealed_copy_hash_mismatch",
                        "actual": target_actual,
                        "expected": expected,
                        "copy": str(target),
                    }
                )
                all_exact = False
                continue
        else:
            shutil.copy2(source, target)
            try:
                target.chmod(0o444)
            except OSError:
                pass
        rows.append(
            {
                "seed": seed,
                "status": "exact",
                "source": str(source),
                "copy": str(target),
                "state_dict_sha256": actual,
                "file_bytes": target.stat().st_size,
            }
        )
    mode = "exact_init_replay" if all_exact else "legacy_order_simulation"
    payload = {"mode": mode, "runs": rows}
    (destination / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def cpu_preflight(
    package_root: str | Path,
    run_root: str | Path,
    historical_root: str | Path,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    from .data import dense_points_px, load_cache, materialize_cache, support_points_px

    package = Path(package_root).resolve()
    run = Path(run_root).resolve()
    status = StatusStore(run / "status")
    status.update("CPU_PREFLIGHT", package_root=str(package), run_root=str(run))
    run.mkdir(parents=True, exist_ok=True)
    ok, failures = verify_manifest(package)
    if not ok:
        status.update("NO_GO", reason="package_manifest", failures=failures)
        raise RuntimeError(f"package manifest verification failed: {failures}")
    package_manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    environment = write_environment(run / "cpu_environment.json")
    init_manifest = isolate_exact_inits(historical_root, run / "inputs")
    if init_manifest["mode"] != "exact_init_replay":
        status.update("NO_GO", reason="exact_init_preflight", legacy_init=init_manifest)
        raise RuntimeError(f"exact-init preflight failed: {init_manifest}")
    cache_manifest = materialize_cache(run / "cache_cpu", workers=workers)
    verified_cache = load_cache(run / "cache_cpu", verify=True)
    support = support_points_px()
    dense = dense_points_px()
    closed = closed_form_identity(support, dense)
    smoke = run_explicit_xy_mlp(
        support,
        dense,
        run / "cpu_smoke" / "mlp",
        steps=20,
        batch_size=64,
        device="cpu",
    )
    if closed["raw_mae_px"] > 1e-6 or closed["affine_residual_mae_px"] > 1e-6:
        status.update("NO_GO", reason="closed_form_baseline", metrics=closed)
        raise RuntimeError(f"closed form baseline failed: {closed}")
    payload = {
        "package_aggregate_sha256": package_manifest["aggregate_sha256"],
        "package_manifest_sha256": sha256_file(package / "package_manifest.json"),
        "protocol_sha256": protocol_hash(package / "protocol.json"),
        "environment": environment,
        "legacy_init": init_manifest,
        "cache": cache_manifest,
        "cache_verified": {
            "dataset_sha256": verified_cache["manifest"]["dataset_sha256"],
            "dense_count": int(len(verified_cache["dense_points"])),
            "support_count": int(len(verified_cache["support_points"])),
        },
        "closed_form": closed,
        "mlp_smoke": smoke,
    }
    (run / "cpu_preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status.update(
        "READY_FOR_GPU",
        package_aggregate_sha256=payload["package_aggregate_sha256"],
        package_manifest_sha256=payload["package_manifest_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        legacy_mode=init_manifest["mode"],
        cache_manifest=str(run / "cache_cpu" / "manifest.json"),
    )
    return payload
