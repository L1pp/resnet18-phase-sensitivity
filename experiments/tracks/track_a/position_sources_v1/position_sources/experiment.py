"""Local preparation, smoke, profile, training and evaluator helpers."""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .certificate import valid_core_certificate
from .manifest import build_manifest, sha256_file, verify_manifest, write_manifest
from .models import REFERENCE_VARIANTS, VARIANTS, build_model, build_reference_model, seed_all
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_hash, write_json
from .renderer import load_cache, materialize_family_cache, render_points


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    array = np.asarray(images)
    if array.ndim != 3 or array.dtype != np.uint8:
        raise ValueError(f"expected uint8 [N,H,W] images, got {array.shape}/{array.dtype}")
    result = torch.from_numpy(np.array(array, dtype=np.uint8, copy=True)).to(dtype=torch.float32).div_(255.0)
    return result[:, None].expand(-1, 3, -1, -1).contiguous()


def target_values(points_px: np.ndarray, *, family: str, image_size: int) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    if family == "torus":
        phase = 2.0 * np.pi * points / float(image_size)
        return np.ascontiguousarray(
            np.stack((np.sin(phase[:, 0]), np.cos(phase[:, 0]), np.sin(phase[:, 1]), np.cos(phase[:, 1])), axis=1),
            dtype=np.float32,
        )
    return np.ascontiguousarray(points / float(image_size), dtype=np.float32)


def decode_predictions(predictions: np.ndarray, *, family: str, image_size: int) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if family == "torus":
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("torus predictions must have four sin/cos channels")
        x = np.mod(np.arctan2(values[:, 0], values[:, 1]) / (2.0 * np.pi) * image_size, image_size)
        y = np.mod(np.arctan2(values[:, 2], values[:, 3]) / (2.0 * np.pi) * image_size, image_size)
        return np.ascontiguousarray(np.stack((x, y), axis=1), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("non-torus predictions must have two channels")
    return np.ascontiguousarray(values * float(image_size), dtype=np.float64)


def mixed_loss(prediction: torch.Tensor, target: torch.Tensor, *, mse_weight: float = 1.0, l1_weight: float = 0.25) -> torch.Tensor:
    return float(mse_weight) * nn.functional.mse_loss(prediction, target) + float(l1_weight) * nn.functional.l1_loss(prediction, target)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(path)


def _cache_root(root: Path, family: str, primitive: str) -> Path:
    return root / "cache" / family / primitive


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_CHECKPOINT_INTERVAL = 500


def _safe_evidence_id(value: str | None, name: str) -> str:
    text = str(value or "").strip()
    if not text or _SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"{name} must be a non-empty path-safe pre-registered identifier")
    return text


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "formal evidence already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _microbatch_slices(effective_batch_size: int, microbatch: int | None) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Validate the physical microbatch and return one logical-batch plan.

    The frozen protocol measures one optimizer/scheduler update over an
    effective batch of 16.  Splitting that same pre-sampled batch into
    microbatches changes only the physical memory footprint; it must not add
    optimizer or scheduler steps.
    """

    effective = int(effective_batch_size)
    if effective <= 0:
        raise ValueError("effective batch size must be positive")
    value = effective if microbatch is None else int(microbatch)
    if value <= 0 or effective % value != 0:
        raise ValueError(f"microbatch must be a positive divisor of effective batch {effective}")
    return value, tuple((start, start + value) for start in range(0, effective, value))


def _microbatch_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    batch_x: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_cfg: Mapping[str, Any],
    microbatch: int,
) -> float:
    """Apply one effective-batch update using gradient accumulation.

    Each chunk contributes its sample fraction to the mean loss, so the
    resulting gradient equals the logical batch mean (up to floating-point
    reduction order).  Exactly one optimizer and one scheduler step occur for
    this function call.
    """

    if batch_x.ndim < 1 or target.ndim < 1 or int(batch_x.shape[0]) != int(target.shape[0]):
        raise ValueError("microbatch inputs and targets must share a non-empty leading batch dimension")
    effective = int(batch_x.shape[0])
    _, slices = _microbatch_slices(effective, microbatch)
    optimizer.zero_grad(set_to_none=True)
    aggregate_loss = 0.0
    for start, end in slices:
        prediction = model(batch_x[start:end])
        loss = mixed_loss(
            prediction,
            target[start:end],
            mse_weight=float(loss_cfg["mse_weight"]),
            l1_weight=float(loss_cfg["l1_weight"]),
        )
        weight = float(end - start) / float(effective)
        (loss * weight).backward()
        aggregate_loss += float(loss.detach().cpu()) * weight
    optimizer.step()
    scheduler.step()
    return aggregate_loss


def _run_identity(
    *,
    protocol_digest: str,
    cache_digest: str,
    family: str,
    primitive: str,
    variant: str,
    seed: int,
    comparison_block_id: str,
    run_id: str,
    effective_batch_size: int,
    microbatch: int,
    normalization: str,
    reference_only: bool,
    device: str,
) -> dict[str, Any]:
    """Return the immutable identity carried by every formal checkpoint."""

    runtime = _runtime_identity(device)
    return {
        "protocol_hash": str(protocol_digest),
        "cache_manifest_sha256": str(cache_digest),
        "family": str(family),
        "primitive": str(primitive),
        "variant": str(variant),
        "seed": int(seed),
        "comparison_block_id": str(comparison_block_id),
        "run_id": str(run_id),
        "effective_batch_size": int(effective_batch_size),
        "microbatch": int(microbatch),
        "model_stack": "position_sources_resnet18_gap_linear",
        "normalization": str(normalization),
        "reference_only": bool(reference_only),
        **runtime,
    }


def _runtime_identity(device: str) -> dict[str, Any]:
    """Describe the selected backend and the torch runtime for resume binding."""

    selected = torch.device(str(device))
    if selected.type == "cpu":
        backend = "cpu"
        runtime_type = "torch_cpu"
    elif selected.type == "cuda":
        # ROCm exposes the CUDA device API but carries a HIP build marker.
        if getattr(torch.version, "hip", None):
            backend = "rocm"
            runtime_type = "torch_rocm"
        else:
            backend = "cuda"
            runtime_type = "torch_cuda"
    else:
        backend = str(selected.type)
        runtime_type = f"torch_{selected.type}"
    return {
        "device_backend": backend,
        "runtime_type": runtime_type,
        "torch_version": str(torch.__version__),
        "cuda_version": None if getattr(torch.version, "cuda", None) is None else str(torch.version.cuda),
        "hip_version": None if getattr(torch.version, "hip", None) is None else str(torch.version.hip),
    }


def _resume_identity_mismatches(expected: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> list[str]:
    """Compare resume identity field-by-field, never silently coalescing arms."""

    recorded = checkpoint.get("run_identity")
    if not isinstance(recorded, Mapping):
        recorded = checkpoint
    mismatches: list[str] = []
    for field, value in expected.items():
        if field not in recorded:
            mismatches.append(f"{field}:missing")
        elif recorded[field] != value:
            mismatches.append(f"{field}:expected={value!r},actual={recorded[field]!r}")
        # Keep the legacy/top-level fields bound too.  A hand-edited nested
        # identity must not be enough to pass a resume check.
        if field in checkpoint and checkpoint[field] != value:
            mismatches.append(f"top_level.{field}:expected={value!r},actual={checkpoint[field]!r}")
    return mismatches


def _capture_rng_state(generator: torch.Generator, *, device: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "sampler": generator.get_state(),
    }
    if torch.device(str(device)).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU checkpoint requires an available CUDA/ROCm runtime")
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any], *, device: str) -> None:
    required = {"python", "numpy", "torch_cpu", "sampler"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"resume checkpoint RNG state is incomplete: {missing}")
    gpu_device = torch.device(str(device)).type == "cuda"
    if gpu_device and "torch_cuda" not in state:
        raise ValueError("GPU resume checkpoint is missing torch_cuda RNG state")
    if not gpu_device and "torch_cuda" in state:
        raise ValueError("CPU resume refuses a checkpoint containing torch_cuda RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if gpu_device:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU resume requires an available CUDA/ROCm runtime")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    generator: torch.Generator,
    rng_state: Mapping[str, Any],
    identity: Mapping[str, Any],
    requested_steps: int,
    optimizer_steps: int,
    status: str,
    last_loss: float,
    data_seed: int,
    init_seed: int,
    sampler_seed: int,
    last_batch_indices: torch.Tensor | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "kind": "position_sources_checkpoint",
        "status": str(status),
        "run_identity": dict(identity),
        # Retain explicit top-level fields for independent, field-wise audit.
        **dict(identity),
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "sampler_seed": int(sampler_seed),
        "steps": int(requested_steps),
        "optimizer_steps": int(optimizer_steps),
        "completed_optimizer_steps": int(optimizer_steps),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": dict(rng_state),
        "sampler_state": generator.get_state(),
        "last_batch_indices": None if last_batch_indices is None else last_batch_indices.detach().cpu(),
        "last_loss": float(last_loss),
        "checkpoint_interval": int(_CHECKPOINT_INTERVAL),
    }
    return payload


def prepare_root(
    out: str | Path,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    families: Iterable[str] | None = None,
    primitives: Iterable[str] | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    cfg = load_protocol(protocol_path)
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not quick and any(path.is_file() for path in output.rglob("*")):
        raise FileExistsError(f"formal prepared root is not empty; refusing to overwrite evidence: {output}")
    protocol_target = output / "protocol.json"
    protocol_target.write_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    family_names = tuple(families or cfg["families"].keys())
    primitive_names = tuple(primitives or cfg["renderer"]["primitives"])
    for family in family_names:
        if family not in cfg["families"]:
            raise ValueError(f"unknown family: {family}")
        for primitive in primitive_names:
            materialize_family_cache(
                _cache_root(output, family, primitive),
                family=family,
                protocol=cfg,
                quick=quick,
                primitive=primitive,
            )
    certificate_path = output / "certificates" / "valid_core_s32.json"
    write_json(certificate_path, valid_core_certificate(input_size=1024))
    declared_files = [
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"MANIFEST.json", "PREPARE_RECEIPT.json"}
        and "__pycache__" not in path.parts
        and not any(part in {"reports", "runs", "reviewer_outputs", "independent_review_v1", "outputs"} for part in path.relative_to(output).parts)
    ]
    manifest = build_manifest(
        output,
        protocol_path=protocol_target,
        files=declared_files,
        mode="quick" if quick else "formal",
        metadata={"families": list(family_names), "primitives": list(primitive_names)},
    )
    write_manifest(output, manifest)
    receipt = {
        "schema_version": 1,
        "kind": "position_sources_prepare_receipt",
        "status": "prepared",
        "root": str(output),
        "protocol_hash": protocol_hash(path=protocol_target),
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": "quick" if quick else "formal",
        "families": list(family_names),
        "primitives": list(primitive_names),
        "formal_training_started": False,
        "remote_access": False,
    }
    write_json(output / "PREPARE_RECEIPT.json", receipt)
    return receipt


def preflight_root(root: str | Path) -> dict[str, Any]:
    output = Path(root).resolve()
    manifest_result = verify_manifest(output)
    cfg = load_protocol(output / "protocol.json")
    certificate = json.loads((output / "certificates" / "valid_core_s32.json").read_text(encoding="utf-8"))
    certificate_result = valid_core_certificate(input_size=1024)
    if certificate != certificate_result:
        raise ValueError("stored valid-core certificate differs from computed certificate")
    cache_rows = []
    missing_caches = []
    for family, family_cfg in cfg["families"].items():
        for primitive in cfg["renderer"]["primitives"]:
            cache_path = _cache_root(output, family, primitive)
            if not (cache_path / "CACHE_MANIFEST.json").exists():
                missing_caches.append(f"{family}/{primitive}")
                continue
            train_points, train_images, eval_points, eval_images, cache_manifest = load_cache(cache_path)
            cache_rows.append(
                {
                    "family": family,
                    "primitive": primitive,
                    "mode": cache_manifest.get("mode"),
                    "train_count": int(len(train_points)),
                    "eval_count": int(len(eval_points)),
                    "images_materialized": train_images is not None and eval_images is not None,
                    "train_image_shape": None if train_images is None else list(train_images.shape),
                    "eval_image_shape": None if eval_images is None else list(eval_images.shape),
                    "input_size": int(family_cfg["input_size"]),
                    "split_metadata": cache_manifest.get("split_metadata"),
                }
            )
    result = {
        "schema_version": 1,
        "kind": "position_sources_preflight",
        "status": "passed",
        "root": str(output),
        "protocol_hash": protocol_hash(path=output / "protocol.json"),
        "manifest": manifest_result,
        "valid_core_certificate": certificate_result,
        "caches": cache_rows,
        "missing_caches": missing_caches,
        "partial_scope": bool(missing_caches),
        "formal_training_started": False,
        "remote_access": False,
    }
    write_json(output / "reports" / "preflight.json", result)
    return result


def _input_size_for_variant(variant: str, cfg: Mapping[str, Any], override: int | None = None) -> int:
    if override is not None:
        return int(override)
    if variant.startswith("torus_"):
        return int(cfg["families"]["torus"]["input_size"])
    if variant == "true_valid_s32":
        return max(512, int(cfg["families"]["padding"]["input_size"]))
    if variant == "valid_core_s1":
        return int(cfg["families"]["stride"]["input_size"])
    return int(cfg["families"]["padding"]["input_size"])


def smoke(
    out: str | Path,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    variants: Iterable[str] | None = None,
    input_size_override: int | None = None,
) -> dict[str, Any]:
    cfg = load_protocol(protocol_path)
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = tuple(variants or VARIANTS)
    rows: list[dict[str, Any]] = []
    for variant in names:
        input_size = _input_size_for_variant(variant, cfg, input_size_override)
        seed_all(1000 + names.index(variant))
        model = build_model(variant, input_size=input_size, groups=int(cfg["normalization"]["groups"]))
        # A single compact input catches model shape/padding errors without
        # creating a formal train stream or loading any historical asset.
        image = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
        image[..., input_size // 2 - 2 : input_size // 2 + 2, input_size // 2 - 2 : input_size // 2 + 2] = 1.0
        start = time.perf_counter()
        with torch.no_grad():
            prediction = model(image)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "variant": variant,
                "input_size": input_size,
                "output_shape": list(prediction.shape),
                "finite": bool(torch.isfinite(prediction).all().item()),
                "forward_seconds": elapsed,
                "valid_core": model.valid_core_info(input_size=input_size) if variant.startswith("valid_core") else None,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "position_sources_smoke",
        "status": "passed" if all(row["finite"] for row in rows) else "failed",
        "protocol_hash": protocol_hash(path=protocol_path),
        "rows": rows,
        "formal_training_started": False,
        "remote_access": False,
    }
    write_json(output / "SMOKE_RECEIPT.json", result)
    return result


def profile(
    out: str | Path,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    variants: Iterable[str] | None = None,
    input_size_override: int | None = None,
    backward: bool = False,
) -> dict[str, Any]:
    cfg = load_protocol(protocol_path)
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = tuple(variants or ("zero_s32", "reflection_s32", "circular_s32", "true_valid_s32"))
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(names):
        input_size = _input_size_for_variant(variant, cfg, input_size_override)
        seed_all(2000 + index)
        model = build_model(variant, input_size=input_size, groups=int(cfg["normalization"]["groups"]))
        image = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32, requires_grad=bool(backward))
        start = time.perf_counter()
        prediction = model(image)
        if backward:
            mixed_loss(prediction, torch.zeros_like(prediction)).backward()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "variant": variant,
                "input_size": input_size,
                "seconds": elapsed,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "backward": bool(backward),
            }
        )
    result = {
        "schema_version": 1,
        "kind": "position_sources_profile",
        "status": "passed",
        "protocol_hash": protocol_hash(path=protocol_path),
        "rows": rows,
        "formal_training_started": False,
        "remote_access": False,
    }
    write_json(output / "PROFILE_RECEIPT.json", result)
    return result


def _family_for_variant(variant: str) -> str:
    if variant.startswith("torus_"):
        return "torus"
    if variant in {"valid_core_aa32", "valid_core_s1"}:
        return "stride"
    return "padding"


def _loss_from_cfg(cfg: Mapping[str, Any], prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss_cfg = cfg["training"]["loss"]
    return mixed_loss(
        prediction,
        target,
        mse_weight=float(loss_cfg["mse_weight"]),
        l1_weight=float(loss_cfg["l1_weight"]),
    )


def _run_formal_arm(
    *,
    output: Path,
    cfg: Mapping[str, Any],
    family: str,
    primitive: str,
    variant: str,
    seed: int,
    device: str,
    requested_steps: int,
    effective_batch_size: int,
    microbatch_size: int,
    comparison_block_id: str,
    run_id: str,
    train_points: np.ndarray,
    train_images: np.ndarray | None,
    targets: torch.Tensor,
    run_root: Path,
    reference_only: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    seed_record = next(row for row in cfg["seed_records"] if int(row["seed"]) == int(seed))
    cache_path = _cache_root(output, family, primitive)
    cache_digest = sha256_file(cache_path / "CACHE_MANIFEST.json")
    protocol_digest = protocol_hash(path=output / "protocol.json")
    normalization = "batchnorm2d" if reference_only else "groupnorm"
    identity = _run_identity(
        protocol_digest=protocol_digest,
        cache_digest=cache_digest,
        family=family,
        primitive=primitive,
        variant=variant,
        seed=seed,
        comparison_block_id=comparison_block_id,
        run_id=run_id,
        effective_batch_size=effective_batch_size,
        microbatch=microbatch_size,
        normalization=normalization,
        reference_only=reference_only,
        device=device,
    )
    arm_root = run_root / variant / str(seed)
    final_checkpoint = arm_root / "checkpoint.pt"
    resume_payload: Mapping[str, Any] | None = None
    resumed_from_step = 0
    if resume_from is None:
        _ensure_absent([arm_root, final_checkpoint])
        seed_all(int(seed_record["init_seed"]))
    else:
        resume_path = Path(resume_from).resolve()
        if not resume_path.exists() or not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        if resume_path.parent.resolve() != arm_root.resolve():
            raise ValueError("--resume-from must be the periodic checkpoint for the requested primitive/arm/seed")
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        mismatches = _resume_identity_mismatches(identity, resume_payload)
        if mismatches:
            raise ValueError("resume identity mismatch: " + "; ".join(mismatches))
        if str(resume_payload.get("status")) != "in_progress":
            raise ValueError("--resume-from must point to an in_progress periodic checkpoint")
        resumed_from_step = int(resume_payload.get("completed_optimizer_steps", -1))
        if resumed_from_step <= 0 or resumed_from_step >= requested_steps:
            raise ValueError("resume checkpoint completed_optimizer_steps must be between 1 and 3999")
        _ensure_absent([final_checkpoint])
        seed_all(int(seed_record["init_seed"]))

    model_factory = build_reference_model if reference_only else build_model
    if reference_only:
        model = model_factory(variant, input_size=int(cfg["families"][family]["input_size"]), groups=int(cfg["normalization"]["groups"]))
    else:
        model = model_factory(variant, input_size=int(cfg["families"][family]["input_size"]), groups=int(cfg["normalization"]["groups"]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=requested_steps,
        eta_min=float(cfg["training"]["scheduler"]["eta_min"]),
    )
    generator = torch.Generator().manual_seed(int(seed_record["sampler_seed"]))
    last_loss = math.nan
    optimizer_steps = 0
    last_batch_indices: torch.Tensor | None = None
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        rng_state = resume_payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("resume checkpoint is missing complete rng_state")
        _restore_rng_state(rng_state, device=device)
        sampler_state = resume_payload.get("sampler_state", rng_state.get("sampler"))
        if sampler_state is None:
            raise ValueError("resume checkpoint is missing sampler_state")
        generator.set_state(sampler_state)
        optimizer_steps = resumed_from_step
        last_loss = float(resume_payload.get("last_loss", math.nan))
        previous_indices = resume_payload.get("last_batch_indices")
        if previous_indices is not None:
            last_batch_indices = torch.as_tensor(previous_indices).detach().cpu()

    image_size = int(cfg["families"][family]["input_size"])
    model.train()
    for _ in range(optimizer_steps, requested_steps):
        indices = torch.randint(0, len(train_points), (effective_batch_size,), generator=generator)
        last_batch_indices = indices.detach().cpu()
        if train_images is None:
            batch_images = render_points(
                train_points[indices.numpy()],
                image_size=image_size,
                sigma_px=float(cfg["renderer"]["sigma_px"]),
                primitive=primitive,
                torus=family == "torus",
            )
            batch_x = images_to_tensor(batch_images)
        else:
            batch_x = images_to_tensor(train_images[indices.numpy()])
        target = targets[indices]
        last_loss = _microbatch_update(
            model,
            optimizer,
            scheduler,
            batch_x.to(device),
            target.to(device),
            loss_cfg=cfg["training"]["loss"],
            microbatch=microbatch_size,
        )
        optimizer_steps += 1
        if optimizer_steps < requested_steps and optimizer_steps % _CHECKPOINT_INTERVAL == 0:
            periodic = arm_root / f"checkpoint_step_{optimizer_steps:04d}.pt"
            _ensure_absent([periodic])
            _atomic_torch_save(
                periodic,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    generator=generator,
                    rng_state=_capture_rng_state(generator, device=device),
                    identity=identity,
                    requested_steps=requested_steps,
                    optimizer_steps=optimizer_steps,
                    status="in_progress",
                    last_loss=last_loss,
                    data_seed=int(seed_record["data_seed"]),
                    init_seed=int(seed_record["init_seed"]),
                    sampler_seed=int(seed_record["sampler_seed"]),
                    last_batch_indices=last_batch_indices,
                ),
            )

    model.eval()
    _atomic_torch_save(
        final_checkpoint,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            rng_state=_capture_rng_state(generator, device=device),
            identity=identity,
            requested_steps=requested_steps,
            optimizer_steps=optimizer_steps,
            status="completed",
            last_loss=last_loss,
            data_seed=int(seed_record["data_seed"]),
            init_seed=int(seed_record["init_seed"]),
            sampler_seed=int(seed_record["sampler_seed"]),
            last_batch_indices=last_batch_indices,
        ),
    )
    return {
        "family": family,
        "primitive": primitive,
        "variant": variant,
        "seed": int(seed),
        "data_seed": int(seed_record["data_seed"]),
        "init_seed": int(seed_record["init_seed"]),
        "sampler_seed": int(seed_record["sampler_seed"]),
        "steps": requested_steps,
        "optimizer_steps": optimizer_steps,
        "completed_optimizer_steps": optimizer_steps,
        "microbatch": microbatch_size,
        "effective_batch_size": effective_batch_size,
        "last_loss": last_loss,
        "checkpoint": str(final_checkpoint),
        "comparison_block_id": comparison_block_id,
        "run_id": run_id,
        "reference_only": bool(reference_only),
        "resumed": resume_payload is not None,
        "resumed_from_step": int(resumed_from_step),
        "checkpoint_interval": int(_CHECKPOINT_INTERVAL),
    }


def _run_formal(
    root: str | Path,
    *,
    family: str,
    primitive: str,
    names: tuple[str, ...],
    seed_values: tuple[int, ...],
    device: str,
    steps: int | None,
    microbatch: int | None,
    allow_formal: bool,
    comparison_block_id: str | None,
    run_id: str | None,
    reference_only: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    if not allow_formal:
        raise PermissionError("formal training is disabled; pass --allow-formal explicitly after review")
    output = Path(root).resolve()
    cfg = load_protocol(output / "protocol.json")
    if family not in cfg["families"]:
        raise ValueError(f"unknown family: {family}")
    if primitive not in tuple(cfg["renderer"]["primitives"]):
        raise ValueError(f"unknown primitive: {primitive}")
    if reference_only:
        expected_variants = tuple(cfg["reference_only"]["variant"] for _ in [0])
        if names != expected_variants:
            raise ValueError(f"reference-only run accepts exactly {expected_variants}")
        allowed_seeds = tuple(int(value) for value in cfg["reference_only"]["seeds"])
    else:
        expected_variants = tuple(cfg["families"][family]["variants"])
        allowed_seeds = tuple(int(value) for value in cfg["seeds"])
        if not names or len(set(names)) != len(names) or any(name not in expected_variants for name in names):
            raise ValueError(f"variants must be a non-empty subset of {family}: {expected_variants}")
    if not seed_values or len(set(seed_values)) != len(seed_values) or any(seed not in allowed_seeds for seed in seed_values):
        raise ValueError(f"seeds must be a non-empty subset of the three frozen seeds: {allowed_seeds}")
    requested_steps = int(cfg["training"]["steps"] if steps is None else steps)
    if requested_steps != int(cfg["training"]["steps"]):
        raise ValueError(f"formal training requires exactly {cfg['training']['steps']} steps; shards may restrict arms/seeds, not steps")
    configured_effective_batch = int(cfg["training"].get("effective_batch_size", cfg["training"]["batch_size"]))
    if int(cfg["training"]["batch_size"]) != configured_effective_batch:
        raise ValueError("frozen protocol batch_size must equal effective_batch_size")
    microbatch_size, microbatch_plan = _microbatch_slices(configured_effective_batch, microbatch)
    block_id = _safe_evidence_id(comparison_block_id, "comparison_block_id")
    registered_run_id = _safe_evidence_id(run_id, "run_id")
    if resume_from is not None and (len(names) != 1 or len(seed_values) != 1):
        raise ValueError("resume is restricted to one explicitly selected variant and seed")
    train_points, train_images, _, _, cache_manifest = load_cache(_cache_root(output, family, primitive))
    if cache_manifest.get("mode") != "formal":
        raise ValueError("formal training requires a formal prepared cache, not a quick subset")
    image_size = int(cfg["families"][family]["input_size"])
    targets = torch.from_numpy(target_values(train_points, family=family, image_size=image_size))
    if reference_only:
        run_root = output / "runs" / "formal" / "reference_only" / family / primitive / registered_run_id
    else:
        run_root = output / "runs" / "formal" / family / primitive / registered_run_id
    receipt_path = run_root / "RUN_RECEIPT.json"
    _ensure_absent([receipt_path])
    if resume_from is None:
        _ensure_absent([run_root / variant / str(seed) for variant in names for seed in seed_values])
    rows = []
    for variant in names:
        for seed in seed_values:
            rows.append(
                _run_formal_arm(
                    output=output,
                    cfg=cfg,
                    family=family,
                    primitive=primitive,
                    variant=variant,
                    seed=seed,
                    device=device,
                    requested_steps=requested_steps,
                    effective_batch_size=configured_effective_batch,
                    microbatch_size=microbatch_size,
                    comparison_block_id=block_id,
                    run_id=registered_run_id,
                    train_points=train_points,
                    train_images=train_images,
                    targets=targets,
                    run_root=run_root,
                    reference_only=reference_only,
                    resume_from=resume_from,
                )
            )
    complete_scope = set(names) == set(expected_variants) and set(seed_values) == set(allowed_seeds)
    primitive_scope = {
        "primitive": primitive,
        "status": "complete" if complete_scope else "shard",
        "variants": list(names),
        "seeds": list(seed_values),
        "expected_variants": list(expected_variants),
        "expected_seeds": list(allowed_seeds),
    }
    receipt = {
        "schema_version": 2,
        "kind": "position_sources_run_reference" if reference_only else "position_sources_run_family",
        "status": "shard" if not complete_scope else "completed",
        "scope": "shard" if not complete_scope else "family_primitive_complete",
        "primitive_scope": primitive_scope,
        "reference_only": bool(reference_only),
        "primary": not reference_only,
        "normalization": "batchnorm2d" if reference_only else "groupnorm",
        "base_variant": "zero_s32" if reference_only else None,
        "protocol_hash": protocol_hash(path=output / "protocol.json"),
        "family": family,
        "primitive": primitive,
        "comparison_block_id": block_id,
        "run_id": registered_run_id,
        "variants": list(names),
        "seeds": list(seed_values),
        "steps": requested_steps,
        "optimizer_steps_per_arm": requested_steps,
        "microbatch": microbatch_size,
        "effective_batch_size": configured_effective_batch,
        "microbatch_plan": [list(item) for item in microbatch_plan],
        "checkpoint_interval": int(_CHECKPOINT_INTERVAL),
        "device": str(device),
        "rows": rows,
        "formal_training_started": True,
        "remote_access": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def run_family(
    root: str | Path,
    *,
    family: str,
    primitive: str = "blob",
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    device: str = "cpu",
    steps: int | None = None,
    microbatch: int | None = None,
    allow_formal: bool = False,
    comparison_block_id: str | None = None,
    run_id: str | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    if not allow_formal:
        raise PermissionError("formal training is disabled; pass --allow-formal explicitly after review")
    cfg = load_protocol(Path(root).resolve() / "protocol.json")
    if family not in cfg["families"]:
        raise ValueError(f"unknown family: {family}")
    expected = tuple(cfg["families"][family]["variants"])
    names = expected if variants is None else tuple(str(value) for value in variants)
    seed_values = tuple(int(value) for value in (cfg["seeds"] if seeds is None else seeds))
    return _run_formal(
        root,
        family=family,
        primitive=primitive,
        names=names,
        seed_values=seed_values,
        device=device,
        steps=steps,
        microbatch=microbatch,
        allow_formal=allow_formal,
        comparison_block_id=comparison_block_id,
        run_id=run_id,
        resume_from=resume_from,
    )


def run_reference(
    root: str | Path,
    *,
    primitive: str = "blob",
    seeds: Iterable[int] | None = None,
    device: str = "cpu",
    steps: int | None = None,
    microbatch: int | None = None,
    allow_formal: bool = False,
    comparison_block_id: str | None = None,
    run_id: str | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    if not allow_formal:
        raise PermissionError("formal training is disabled; pass --allow-formal explicitly after review")
    cfg = load_protocol(Path(root).resolve() / "protocol.json")
    variant = str(cfg["reference_only"]["variant"])
    seed_values = tuple(int(value) for value in (cfg["reference_only"]["seeds"] if seeds is None else seeds))
    return _run_formal(
        root,
        family="padding",
        primitive=primitive,
        names=(variant,),
        seed_values=seed_values,
        device=device,
        steps=steps,
        microbatch=microbatch,
        allow_formal=allow_formal,
        comparison_block_id=comparison_block_id,
        run_id=run_id,
        reference_only=True,
        resume_from=resume_from,
    )


def _metric_report(predictions_px: np.ndarray, points_px: np.ndarray, *, family: str, image_size: int) -> dict[str, Any]:
    delta = np.asarray(predictions_px, dtype=np.float64) - np.asarray(points_px, dtype=np.float64)
    absolute = np.abs(delta)
    report: dict[str, Any] = {
        "raw_mae_px": float(absolute.mean()),
        "axis_mae_px": [float(item) for item in absolute.mean(axis=0)],
        "raw_max_px": float(absolute.max()),
        "finite": bool(np.isfinite(predictions_px).all()),
    }
    constant_prediction = np.broadcast_to(points_px.mean(axis=0, keepdims=True), points_px.shape)
    constant_mae = float(np.abs(constant_prediction - points_px).mean())
    report["global_improvement_px"] = float(constant_mae - absolute.mean())

    def component_r2(features: np.ndarray) -> float:
        design = np.column_stack((np.ones(len(features), dtype=np.float64), np.asarray(features, dtype=np.float64)))
        values: list[float] = []
        for axis in range(predictions_px.shape[1]):
            target = predictions_px[:, axis]
            fitted = design @ np.linalg.lstsq(design, target, rcond=1e-8)[0]
            denominator = float(np.sum((target - target.mean()) ** 2))
            values.append(0.0 if denominator <= 1e-12 else float(1.0 - np.sum((target - fitted) ** 2) / denominator))
        return float(np.mean(values))

    if family == "torus":
        period = float(image_size)
        quotient_features = np.column_stack((np.floor(points_px[:, 0] / 32.0), np.floor(points_px[:, 1] / 32.0)))
        phase_features = np.column_stack((np.mod(points_px[:, 0], 32.0), np.mod(points_px[:, 1], 32.0)))
    else:
        offsets = points_px - 448.0
        quotient_features = np.floor(offsets / 32.0)
        phase_features = np.mod(offsets, 32.0)
    report["quotient_r2"] = component_r2(quotient_features)
    report["phase_r2"] = component_r2(phase_features)
    if family != "torus":
        modulo_delta = np.abs((delta + 16.0) % 32.0 - 16.0)
        report["modulo32_mae_px"] = float(modulo_delta.mean())
        report["stride_cell_accuracy"] = float(np.mean(np.floor(predictions_px / 32.0) == np.floor(points_px / 32.0)))
    else:
        report["modulo_period_mae_px"] = float(np.abs((delta + image_size / 2.0) % image_size - image_size / 2.0).mean())

    lookup = {tuple(np.round(point, 6)): index for index, point in enumerate(points_px)}
    collision_rows: dict[str, Any] = {}
    for shift in (1.0, 32.0, 64.0):
        values: list[float] = []
        for index, point in enumerate(points_px):
            candidate = point + np.asarray([shift, 0.0])
            if family == "torus":
                candidate %= float(image_size)
            target_index = lookup.get(tuple(np.round(candidate, 6)))
            if target_index is not None:
                values.append(float(np.abs(predictions_px[index] - predictions_px[target_index]).mean()))
        collision_rows[str(int(shift))] = {"pair_count": len(values), "mean_prediction_distance_px": None if not values else float(np.mean(values))}
    report["shift_collision"] = collision_rows
    collision_values = [
        row["mean_prediction_distance_px"]
        for key, row in collision_rows.items()
        if key in {"32", "64"} and row["mean_prediction_distance_px"] is not None
    ]
    report["collision_32_64_max_px"] = None if not collision_values else float(max(collision_values))
    if family == "torus":
        report["explanation_decision"] = "inconclusive"
    elif report["global_improvement_px"] >= 2.0 and report["quotient_r2"] >= 0.10:
        report["explanation_decision"] = "global_quotient"
    elif (
        report["phase_r2"] >= 0.50
        and report["quotient_r2"] <= 0.05
        and report["collision_32_64_max_px"] is not None
        and report["collision_32_64_max_px"] <= 1.0
    ):
        report["explanation_decision"] = "phase_only"
    elif report["global_improvement_px"] <= 1.0 and report["phase_r2"] <= 0.05 and report["quotient_r2"] <= 0.05:
        report["explanation_decision"] = "null"
    else:
        report["explanation_decision"] = "inconclusive"

    n = int(round(math.sqrt(len(points_px))))
    if n * n == len(points_px) and n >= 2:
        grid = predictions_px.reshape(n, n, 2)
        dx = np.diff(grid, axis=0)
        dy = np.diff(grid, axis=1)
        report["local_jacobian_slope"] = {
            "pred_x_vs_grid_x": float(np.mean(dx[..., 0])),
            "pred_y_vs_grid_y": float(np.mean(dy[..., 1])),
            "pred_y_vs_grid_x": float(np.mean(dx[..., 1])),
            "pred_x_vs_grid_y": float(np.mean(dy[..., 0])),
        }
        report["monotonicity"] = {
            "x_axis": float(np.mean(dx[..., 0] >= 0.0)),
            "y_axis": float(np.mean(dy[..., 1] >= 0.0)),
        }
    return report


def _ridge_probe(train_features: np.ndarray, train_targets: np.ndarray, eval_features: np.ndarray, eval_targets: np.ndarray, *, alpha: float) -> dict[str, Any]:
    features_train = np.asarray(train_features, dtype=np.float64)
    features_eval = np.asarray(eval_features, dtype=np.float64)
    targets_train = np.asarray(train_targets, dtype=np.float64)
    targets_eval = np.asarray(eval_targets, dtype=np.float64)
    if features_train.ndim != 2 or features_eval.ndim != 2 or features_train.shape[1] != features_eval.shape[1]:
        raise ValueError("GAP features must be rank-2 with a shared feature dimension")
    if targets_train.ndim != 2 or targets_eval.ndim != 2 or targets_train.shape[1] != targets_eval.shape[1]:
        raise ValueError("GAP probe targets must be rank-2 with a shared output dimension")
    if features_train.shape[0] != targets_train.shape[0] or features_eval.shape[0] != targets_eval.shape[0]:
        raise ValueError("GAP feature/target row counts must match")
    if not np.isfinite(features_train).all() or not np.isfinite(features_eval).all():
        raise ValueError("GAP features must be finite")
    design_train = np.column_stack((np.ones(len(features_train), dtype=np.float64), features_train))
    design_eval = np.column_stack((np.ones(len(features_eval), dtype=np.float64), features_eval))
    regularizer = np.eye(design_train.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    system = design_train.T @ design_train + float(alpha) * regularizer
    try:
        coefficients = np.linalg.solve(system, design_train.T @ targets_train)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, design_train.T @ targets_train, rcond=1e-12)[0]
    predictions = design_eval @ coefficients
    residual = predictions - targets_eval
    sse = np.sum(residual * residual, axis=0)
    centered = targets_eval - targets_eval.mean(axis=0, keepdims=True)
    sst = np.sum(centered * centered, axis=0)
    r2 = [0.0 if float(total) <= 1e-12 else float(1.0 - error / total) for error, total in zip(sse, sst)]
    return {
        "method": "ridge",
        "alpha": float(alpha),
        "fit_intercept": True,
        "train_count": int(len(features_train)),
        "eval_count": int(len(features_eval)),
        "r2_by_axis": r2,
        "r2_mean": float(np.mean(r2)),
    }


def _gap_targets(points: np.ndarray, *, family: str) -> tuple[np.ndarray, np.ndarray] | None:
    if family == "torus":
        return None
    values = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int64)
    offsets = values - 448
    quotient = np.floor_divide(offsets, 32).astype(np.float64)
    phase = np.mod(offsets, 32).astype(np.float64)
    return phase, quotient


def _feature_shift_metrics(features: np.ndarray, points: np.ndarray, *, family: str, image_size: int, shifts: Iterable[int]) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    coordinates = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int64)
    lookup = {tuple(row): index for index, row in enumerate(coordinates.tolist())}
    result: dict[str, Any] = {}
    for shift in shifts:
        l2_values: list[float] = []
        cosine_values: list[float] = []
        zero_norm_pairs = 0
        for index, point in enumerate(coordinates):
            candidate = point + np.asarray([int(shift), 0], dtype=np.int64)
            if family == "torus":
                candidate %= int(image_size)
            target_index = lookup.get(tuple(candidate.tolist()))
            if target_index is None:
                continue
            left = values[index]
            right = values[target_index]
            l2_values.append(float(np.linalg.norm(left - right)))
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 1e-12:
                zero_norm_pairs += 1
            else:
                cosine_values.append(float(1.0 - np.dot(left, right) / denominator))
        result[str(int(shift))] = {
            "pair_count": len(l2_values),
            "mean_l2": None if not l2_values else float(np.mean(l2_values)),
            "mean_cosine_distance": None if not cosine_values else float(np.mean(cosine_values)),
            "zero_norm_pairs": int(zero_norm_pairs),
        }
    return result


def _gap_feature_metrics(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    train_points: np.ndarray,
    eval_points: np.ndarray,
    *,
    family: str,
    image_size: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    features_train = np.asarray(train_features, dtype=np.float32)
    features_eval = np.asarray(eval_features, dtype=np.float32)
    points_train = np.asarray(train_points, dtype=np.float64)
    points_eval = np.asarray(eval_points, dtype=np.float64)
    if features_train.ndim != 2 or features_eval.ndim != 2 or features_train.shape[1] != features_eval.shape[1]:
        raise ValueError("GAP feature fields must be rank-2 with matching dimensions")
    if features_train.shape[0] != points_train.shape[0] or features_eval.shape[0] != points_eval.shape[0]:
        raise ValueError("GAP feature fields must match their physical split coordinates")
    if int(features_train.shape[1]) != int(protocol["evaluation"]["gap_feature"]["feature_dim"]):
        raise ValueError("GAP feature dimension does not match the frozen pre-fc field")
    if not np.isfinite(features_train).all() or not np.isfinite(features_eval).all():
        raise ValueError("GAP feature fields must be finite")
    gap_cfg = protocol["evaluation"]["gap_feature"]
    shifts = tuple(int(value) for value in gap_cfg["shift_distances_px"])
    result: dict[str, Any] = {
        "source": gap_cfg["source"],
        "layer": gap_cfg["layer"],
        "feature_dim": int(features_train.shape[1]),
        "train_count": int(len(features_train)),
        "eval_count": int(len(features_eval)),
        "shift_distances": {
            "train": _feature_shift_metrics(features_train, points_train, family=family, image_size=image_size, shifts=shifts),
            "eval": _feature_shift_metrics(features_eval, points_eval, family=family, image_size=image_size, shifts=shifts),
        },
    }
    target_train = _gap_targets(points_train, family=family)
    target_eval = _gap_targets(points_eval, family=family)
    if target_train is None or target_eval is None:
        result["probe_available"] = False
        result["probe_reason"] = "quotient_phase_targets_are_primary-domain_only"
        return result
    alpha = float(gap_cfg["probe"]["alpha"])
    phase_probe = _ridge_probe(features_train, target_train[0], features_eval, target_eval[0], alpha=alpha)
    quotient_probe = _ridge_probe(features_train, target_train[1], features_eval, target_eval[1], alpha=alpha)
    result.update(
        {
            "probe_available": True,
            "probe": {
                "fit_split": "train",
                "eval_split": "test",
                "phase": phase_probe,
                "quotient": quotient_probe,
                "phase_r2": float(phase_probe["r2_mean"]),
                "quotient_r2": float(quotient_probe["r2_mean"]),
                "targets": {
                    "phase": gap_cfg["probe"]["phase_target"],
                    "quotient": gap_cfg["probe"]["quotient_target"],
                },
            },
        }
    )
    phase_r2 = float(phase_probe["r2_mean"])
    quotient_r2 = float(quotient_probe["r2_mean"])
    explanation = protocol["gates"]["explanation"]
    if phase_r2 >= float(explanation["phase_only_phase_r2_min"]) and quotient_r2 <= float(explanation["phase_only_quotient_r2_max"]):
        readout_decision = "phase_only"
    elif quotient_r2 >= float(explanation["quotient_r2_min"]):
        readout_decision = "quotient_readout"
    elif phase_r2 <= float(explanation["null_r2_max"]) and quotient_r2 <= float(explanation["null_r2_max"]):
        readout_decision = "null_readout"
    else:
        readout_decision = "inconclusive"
    result["probe"]["explanation_thresholds"] = {
        "phase_only_phase_r2_min": float(explanation["phase_only_phase_r2_min"]),
        "phase_only_quotient_r2_max": float(explanation["phase_only_quotient_r2_max"]),
        "quotient_r2_min": float(explanation["quotient_r2_min"]),
        "null_r2_max": float(explanation["null_r2_max"]),
    }
    result["probe"]["feature_readout_decision"] = readout_decision
    return result


def _infer_predictions_and_features(
    model: nn.Module,
    points: np.ndarray,
    images: np.ndarray | None,
    *,
    image_size: int,
    sigma_px: float,
    primitive: str,
    torus: bool,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_batches: list[np.ndarray] = []
    feature_batches: list[np.ndarray] = []
    for start in range(0, len(points), int(batch_size)):
        batch_points = points[start : start + int(batch_size)]
        if images is None:
            batch_images = render_points(
                batch_points,
                image_size=image_size,
                sigma_px=sigma_px,
                primitive=primitive,
                torus=torus,
            )
        else:
            batch_images = images[start : start + int(batch_size)]
        batch_x = images_to_tensor(batch_images).to(device)
        with torch.no_grad():
            features = model.forward_features(batch_x)
            predictions = model.fc(features)
        feature_batches.append(features.detach().cpu().numpy())
        prediction_batches.append(predictions.detach().cpu().numpy())
    return np.concatenate(prediction_batches, axis=0), np.concatenate(feature_batches, axis=0)


def evaluate_checkpoint(
    root: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    output = Path(root).resolve()
    cfg = load_protocol(output / "protocol.json")
    checkpoint_path = Path(checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = str(payload["variant"])
    family = str(payload["family"])
    primitive = str(payload["primitive"])
    reference_only = bool(payload.get("reference_only", variant in REFERENCE_VARIANTS))
    if family not in cfg["families"]:
        raise ValueError(f"checkpoint family is not registered: {family}")
    if primitive not in tuple(cfg["renderer"]["primitives"]):
        raise ValueError(f"checkpoint primitive is not registered: {primitive}")
    if reference_only:
        reference_cfg = cfg["reference_only"]
        if family != "padding" or variant != str(reference_cfg["variant"]):
            raise ValueError("reference-only checkpoint must use the frozen zero-padding BatchNorm arm")
        if payload.get("normalization") != str(reference_cfg["normalization"]):
            raise ValueError("reference-only checkpoint normalization is not BatchNorm2d")
    elif variant not in tuple(cfg["families"][family]["variants"]):
        raise ValueError(f"checkpoint variant is not registered for {family}: {variant}")
    expected_protocol_hash = protocol_hash(path=output / "protocol.json")
    if str(payload.get("protocol_hash")) != expected_protocol_hash:
        raise ValueError("checkpoint protocol hash does not match the prepared root")
    if int(payload.get("seed", -1)) not in {int(seed) for seed in cfg["seeds"]}:
        raise ValueError("checkpoint seed is not one of the three frozen seeds")
    if int(payload.get("steps", -1)) != int(cfg["training"]["steps"]):
        raise ValueError("formal checkpoint steps do not match the frozen 4000-step protocol")
    configured_effective_batch = int(cfg["training"].get("effective_batch_size", cfg["training"]["batch_size"]))
    checkpoint_effective_batch = int(payload.get("effective_batch_size", -1))
    checkpoint_microbatch = int(payload.get("microbatch", -1))
    if checkpoint_effective_batch != configured_effective_batch:
        raise ValueError("checkpoint effective batch does not match the frozen protocol")
    _microbatch_slices(configured_effective_batch, checkpoint_microbatch)
    if int(payload.get("optimizer_steps", -1)) != int(payload.get("steps", -1)):
        raise ValueError("checkpoint optimizer-step count must equal formal training steps")
    if int(payload.get("completed_optimizer_steps", payload.get("optimizer_steps", -1))) != int(payload.get("optimizer_steps", -1)):
        raise ValueError("checkpoint completed_optimizer_steps must equal optimizer_steps")
    comparison_block_id = _safe_evidence_id(payload.get("comparison_block_id"), "comparison_block_id")
    run_id = _safe_evidence_id(payload.get("run_id"), "run_id")
    cache_digest = sha256_file(_cache_root(output, family, primitive) / "CACHE_MANIFEST.json")
    if str(payload.get("cache_manifest_sha256")) != cache_digest:
        raise ValueError("checkpoint cache manifest hash does not match the prepared root")
    report_family = "reference_only" if reference_only else family
    report_dir = output / "reports" / "evaluations" / report_family / family / primitive / run_id / variant / str(payload["seed"])
    prediction_path = report_dir / "predictions_raw.npy"
    gap_train_path = report_dir / "gap_features_train.npy"
    gap_eval_path = report_dir / "gap_features_eval.npy"
    gap_metadata_path = report_dir / "GAP_FEATURE_METADATA.json"
    evaluation_path = report_dir / "EVALUATION.json"
    _ensure_absent([prediction_path, gap_train_path, gap_eval_path, gap_metadata_path, evaluation_path])
    image_size = int(cfg["families"][family]["input_size"])
    train_points, train_images, eval_points, eval_images, cache_manifest = load_cache(_cache_root(output, family, primitive))
    model = (
        build_reference_model(variant, input_size=image_size, groups=int(cfg["normalization"]["groups"]))
        if reference_only
        else build_model(variant, input_size=image_size, groups=int(cfg["normalization"]["groups"]))
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    _, train_features = _infer_predictions_and_features(
        model,
        train_points,
        train_images,
        image_size=image_size,
        sigma_px=float(cfg["renderer"]["sigma_px"]),
        primitive=primitive,
        torus=family == "torus",
        batch_size=int(cfg["training"]["batch_size"]),
        device=device,
    )
    raw_predictions, eval_features = _infer_predictions_and_features(
        model,
        eval_points,
        eval_images,
        image_size=image_size,
        sigma_px=float(cfg["renderer"]["sigma_px"]),
        primitive=primitive,
        torus=family == "torus",
        batch_size=int(cfg["training"]["batch_size"]),
        device=device,
    )
    # Commit prediction and both split-specific feature fields before loading
    # the coordinate labels for metric/probe calculation.  The coordinates
    # are needed as renderer inputs above, but are not used as labels until
    # every evidence field has been physically materialized and hash-bound.
    _atomic_npy(prediction_path, raw_predictions)
    _atomic_npy(gap_train_path, train_features)
    _atomic_npy(gap_eval_path, eval_features)
    feature_metadata = {
        "schema_version": 1,
        "kind": "position_sources_gap_feature_field",
        "source": cfg["evaluation"]["gap_feature"]["source"],
        "layer": cfg["evaluation"]["gap_feature"]["layer"],
        "field": cfg["evaluation"]["gap_feature"]["field"],
        "feature_dim": int(eval_features.shape[1]),
        "dtype": str(eval_features.dtype),
        "head_excluded": bool(cfg["evaluation"]["gap_feature"]["head_excluded"]),
        "reference_only": bool(reference_only),
        "primary": not reference_only,
        "normalization": "batchnorm2d" if reference_only else "groupnorm",
        "family": family,
        "primitive": primitive,
        "variant": variant,
        "seed": int(payload["seed"]),
        "comparison_block_id": comparison_block_id,
        "run_id": run_id,
        "protocol_hash": protocol_hash(path=output / "protocol.json"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_count": int(len(train_features)),
        "eval_count": int(len(eval_features)),
        "train_features_sha256": sha256_file(gap_train_path),
        "eval_features_sha256": sha256_file(gap_eval_path),
        "probe_contract": cfg["evaluation"]["gap_feature"]["probe"],
        "shift_distances_px": cfg["evaluation"]["gap_feature"]["shift_distances_px"],
    }
    write_json(gap_metadata_path, feature_metadata)
    committed_predictions = np.load(prediction_path, allow_pickle=False)
    committed_train_features = np.load(gap_train_path, allow_pickle=False)
    committed_eval_features = np.load(gap_eval_path, allow_pickle=False)
    train_labels = np.load(_cache_root(output, family, primitive) / "train_points.npy", allow_pickle=False)
    labels = np.load(_cache_root(output, family, primitive) / "eval_points.npy", allow_pickle=False)
    decoded = decode_predictions(committed_predictions, family=family, image_size=image_size)
    metrics = _metric_report(decoded, labels, family=family, image_size=image_size)
    gap_metrics = _gap_feature_metrics(
        committed_train_features,
        committed_eval_features,
        train_labels,
        labels,
        family=family,
        image_size=image_size,
        protocol=cfg,
    )
    result = {
        "schema_version": 1,
        "kind": "position_sources_evaluation",
        "status": "passed" if metrics["finite"] else "failed",
        "protocol_hash": protocol_hash(path=output / "protocol.json"),
        "family": family,
        "variant": variant,
        "reference_only": bool(reference_only),
        "primary": not reference_only,
        "normalization": "batchnorm2d" if reference_only else "groupnorm",
        "primitive": primitive,
        "comparison_block_id": comparison_block_id,
        "run_id": run_id,
        "seed": int(payload["seed"]),
        "microbatch": checkpoint_microbatch,
        "effective_batch_size": checkpoint_effective_batch,
        "optimizer_steps": int(payload["optimizer_steps"]),
        "checkpoint": str(checkpoint_path),
        "prediction_path": str(prediction_path),
        "gap_feature": {
            "train_path": str(gap_train_path),
            "eval_path": str(gap_eval_path),
            "metadata_path": str(gap_metadata_path),
            "train_sha256": sha256_file(gap_train_path),
            "eval_sha256": sha256_file(gap_eval_path),
            "metadata_sha256": sha256_file(gap_metadata_path),
            "metadata": feature_metadata,
            "metrics": gap_metrics,
        },
        "cache_manifest_sha256": cache_digest,
        "metrics": metrics,
        "prediction_commit_before_label_read": True,
        "gap_feature_commit_before_label_read": True,
        "formal_training_started": False,
        "remote_access": False,
    }
    write_json(evaluation_path, result)
    return result


__all__ = [
    "decode_predictions",
    "evaluate_checkpoint",
    "images_to_tensor",
    "mixed_loss",
    "preflight_root",
    "prepare_root",
    "profile",
    "run_reference",
    "run_family",
    "smoke",
    "target_values",
]
