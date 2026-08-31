"""Frozen JSON protocol and canonical hashing helpers.

The protocol is intentionally readable without importing PyTorch.  Every
prepared root copies this JSON and records its SHA-256 in the root manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"

PADDING_VARIANTS = (
    "zero_s32",
    "reflection_s32",
    "circular_s32",
    "valid_core_s32",
    "true_valid_s32",
)
STRIDE_VARIANTS = ("valid_core_s32", "valid_core_aa32", "valid_core_s1")
TORUS_VARIANTS = ("torus_s32", "torus_s1")
FAMILIES = ("padding", "stride", "torus")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _as_float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair")
    result = (float(value[0]), float(value[1]))
    if not all(map(lambda item: item == item and abs(item) != float("inf"), result)):
        raise ValueError(f"{name} must be finite")
    if result[0] >= result[1]:
        raise ValueError(f"{name} low must be less than high")
    return result


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol_id",
        "task",
        "seeds",
        "coordinate",
        "normalization",
        "renderer",
        "model",
        "families",
        "training",
        "checkpoint",
        "evaluation",
        "gates",
        "safety",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"protocol missing keys: {missing}")
    if int(payload["schema_version"]) != 1:
        raise ValueError("schema_version must be 1")
    if str(payload["protocol_id"]) != "position_sources_v1":
        raise ValueError("protocol_id drifted")
    if str(payload["task"]) != "absolute_position_source_decomposition":
        raise ValueError("task drifted")

    seeds = tuple(int(seed) for seed in payload["seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct seeds are required")
    seed_records = payload.get("seed_records")
    if not isinstance(seed_records, list) or tuple(int(row.get("seed", -1)) for row in seed_records) != seeds:
        raise ValueError("seed_records must cover the frozen seeds in order")
    for row in seed_records:
        if int(row.get("data_seed", -1)) != int(row["seed"]) or int(row.get("sampler_seed", -1)) != int(row["seed"]):
            raise ValueError("data/sampler seed contract drifted")
        if int(row.get("init_seed", -1)) != int(row["seed"]) + 8000000:
            raise ValueError("init seed contract drifted")

    coordinate = payload["coordinate"]
    if coordinate.get("space") != "pixel" or coordinate.get("model_output") != "normalized_by_image_size":
        raise ValueError("coordinate convention drifted")
    if coordinate.get("axis_order") != "x_y" or coordinate.get("torus_target") != "periodic_sin_cos_xy":
        raise ValueError("axis order must be x_y")

    norm = payload["normalization"]
    if norm.get("name") != "groupnorm" or int(norm.get("groups", 0)) != 32 or bool(norm.get("batch_statistics")):
        raise ValueError("Track A primary normalization must be GroupNorm without batch statistics")

    renderer = payload["renderer"]
    image_size = int(renderer.get("primary_image_size", 0))
    if image_size < 448:
        raise ValueError("primary_image_size must leave room for ResNet18 valid-core receptive fields")
    primary_box = renderer["primary_position_box_px"]
    low, high = _as_float_pair(
        [primary_box["low"], primary_box["high"]],
        "renderer.primary_position_box_px",
    )
    if low < 0 or high > image_size:
        raise ValueError("primary position box must be inside the primary image")
    domain = renderer.get("primary_domain", {})
    if domain.get("axis_values") != "all_integers_inclusive" or int(domain.get("count", -1)) != 16384:
        raise ValueError("primary domain must be the complete 128x128 integer box")
    if domain.get("quotient_definition") != "q=floor((p-448)/32)" or domain.get("residue_definition") != "r=(p-448) mod 32":
        raise ValueError("quotient/residue definitions drifted")
    if renderer.get("grid_order") != "x_major_y_minor":
        raise ValueError("grid order drifted")
    if renderer.get("dtype") != "uint8" or int(renderer.get("channels", 0)) != 1:
        raise ValueError("renderer dtype/channels drifted")
    if float(renderer.get("sigma_px", 0)) <= 0 or int(renderer.get("supersample", 0)) < 1:
        raise ValueError("renderer sigma/supersample invalid")
    if tuple(renderer.get("primitives", ())) != ("blob", "triangle"):
        raise ValueError("renderer primitives drifted")
    if int(renderer.get("torus_image_size", 0)) != int(renderer.get("torus_period_px", -1)):
        raise ValueError("torus period must equal torus image size")
    if renderer.get("torus_domain") != "all_integer_points_inclusive_0_to_period_minus_1":
        raise ValueError("torus domain must be complete integer period")
    dense_split = renderer.get("dense_split", {})
    if dense_split.get("method") != "quotient_cell_stratified_sha256" or float(dense_split.get("train_fraction", 0)) != 0.8:
        raise ValueError("dense split must be quotient-cell stratified 80/20")
    if int(dense_split.get("hash_seed", -1)) != 20260824:
        raise ValueError("dense split hash seed drifted")
    expected_coverage = {
        "all_quotient_cells_each_split",
        "all_x_residues_each_split",
        "all_y_residues_each_split",
    }
    if set(dense_split.get("coverage_requirements", ())) != expected_coverage:
        raise ValueError("dense split coverage requirements drifted")

    model = payload["model"]
    if model.get("family") != "resnet18_gap_linear" or tuple(model.get("blocks", ())) != (2, 2, 2, 2):
        raise ValueError("model family/blocks drifted")
    if int(model.get("output_dim", -1)) != 2 or int(model.get("torus_output_dim", -1)) != 4:
        raise ValueError("model output_dim must be 2")
    if tuple(model.get("base_channels", ())) != (64, 128, 256, 512):
        raise ValueError("model base channels drifted")
    if model.get("true_valid", {}).get("all_spatial_padding") != 0:
        raise ValueError("true-valid must have all spatial padding zero")

    reference = payload.get("reference_only", {})
    if reference != {
        "variant": "zero_s32_bn_reference",
        "base_variant": "zero_s32",
        "normalization": "batchnorm2d",
        "primary": False,
        "status": "reference_only",
        "seeds": [20260823, 20260824, 20260825],
        "steps": 4000,
        "effective_batch_size": 16,
    }:
        raise ValueError("zero-padding BatchNorm reference contract drifted")

    families = payload["families"]
    expected = {
        "padding": PADDING_VARIANTS,
        "stride": STRIDE_VARIANTS,
        "torus": TORUS_VARIANTS,
    }
    if set(families) != set(FAMILIES):
        raise ValueError("family names drifted")
    for family, variants in expected.items():
        actual = tuple(families[family].get("variants", ()))
        if actual != variants:
            raise ValueError(f"{family} variants drifted: {actual!r}")
        size = int(families[family].get("input_size", 0))
        if size < 32:
            raise ValueError(f"{family}.input_size is too small")
        _as_float_pair(families[family].get("position_box", ()), f"families.{family}.position_box")

    training = payload["training"]
    if training.get("objective") != "mean_squared_error" or training.get("optimizer") != "AdamW":
        raise ValueError("training objective/optimizer drifted")
    if float(training.get("lr", 0)) <= 0 or float(training.get("weight_decay", -1)) < 0:
        raise ValueError("training optimizer values invalid")
    if int(training.get("steps", 0)) != 4000 or int(training.get("batch_size", 0)) != 16 or int(training.get("effective_batch_size", 0)) != 16:
        raise ValueError("training steps/batch_size invalid")
    if training.get("dtype") != "float32" or bool(training.get("amp")):
        raise ValueError("Track A training must be fp32 with AMP disabled")
    if float(training.get("loss", {}).get("mse_weight", 0)) != 1.0 or float(training.get("loss", {}).get("l1_weight", 0)) != 0.25:
        raise ValueError("loss must be MSE + 0.25*L1")
    if training.get("scheduler", {}).get("name") != "CosineAnnealingLR" or float(training.get("scheduler", {}).get("eta_min", -1)) != 1e-5:
        raise ValueError("scheduler must be cosine with eta_min=1e-5")
    if not bool(training.get("formal_requires_explicit_flag")):
        raise ValueError("formal training safety flag must be enabled")

    checkpoint = payload["checkpoint"]
    expected_checkpoint = {
        "schema_version": 2,
        "periodic_interval_optimizer_steps": 500,
        "identity_fields": [
            "protocol_hash",
            "cache_manifest_sha256",
            "family",
            "primitive",
            "variant",
            "seed",
            "comparison_block_id",
            "run_id",
            "effective_batch_size",
            "microbatch",
            "model_stack",
            "normalization",
            "reference_only",
            "device_backend",
            "runtime_type",
            "torch_version",
            "cuda_version",
            "hip_version",
        ],
        "rng_fields": {
            "python": "python",
            "numpy": "numpy",
            "torch_cpu": "torch_cpu",
            "torch_cuda": "torch_cuda",
            "sampler": "sampler",
        },
        "gpu_rng_required": True,
        "cpu_gpu_resume_isolation": True,
    }
    if checkpoint != expected_checkpoint:
        raise ValueError("checkpoint/resume identity contract drifted")

    evaluation = payload["evaluation"]
    if evaluation.get("primary_domain") != "fixed_protocol_position_box":
        raise ValueError("evaluation primary domain drifted")
    if not bool(evaluation.get("prediction_commit_before_label_read")):
        raise ValueError("prediction commit ordering must be enabled")
    gap_feature = evaluation.get("gap_feature", {})
    expected_gap_feature = {
        "field": "pre_fc_gap",
        "source": "model.forward_features",
        "layer": "global_average_pool_before_fc",
        "feature_dim": 512,
        "dtype": "float32",
        "head_excluded": True,
        "probe": {
            "method": "ridge",
            "alpha": 0.000001,
            "fit_split": "train",
            "eval_split": "test",
            "fit_intercept": True,
            "r2_definition": "1-SSE/SST on heldout eval targets only",
            "phase_target": "r=(p-448) mod 32 in pixel units, axis_order=x_y",
            "quotient_target": "q=floor((p-448)/32) in cell units, axis_order=x_y",
            "primary_domain_only": True,
        },
        "shift_distances_px": [1, 32, 64],
        "cosine_distance": "1-cosine_similarity; zero-norm pairs report null",
    }
    if gap_feature != expected_gap_feature:
        raise ValueError("GAP feature/readout contract drifted")

    safety = payload["safety"]
    if bool(safety.get("remote_access")) or bool(safety.get("upload")) or bool(safety.get("dependency_mutation")):
        raise ValueError("local preparation protocol cannot permit remote/upload/dependency mutation")
    if safety.get("reviewer_output_subdir") != "independent_review_v1":
        raise ValueError("reviewer output directory drifted")
    explanation = payload["gates"].get("explanation", {})
    expected_explanation = {
        "global_improvement_min_px": 2.0,
        "quotient_r2_min": 0.1,
        "phase_only_phase_r2_min": 0.5,
        "phase_only_quotient_r2_max": 0.05,
        "collision_max_px": 1.0,
        "null_improvement_max_px": 1.0,
        "null_r2_max": 0.05,
        "otherwise": "inconclusive",
    }
    if explanation != expected_explanation:
        raise ValueError("explanation gates drifted")
    return dict(payload)


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("protocol root must be an object")
    return validate_protocol(value)


def protocol_hash(*, path: str | Path | None = None, payload: Mapping[str, Any] | None = None) -> str:
    value = load_protocol(path) if payload is None else validate_protocol(payload)
    return sha256_bytes(canonical_json_bytes(value))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "FAMILIES",
    "PADDING_VARIANTS",
    "STRIDE_VARIANTS",
    "TORUS_VARIANTS",
    "canonical_json_bytes",
    "load_protocol",
    "protocol_hash",
    "sha256_bytes",
    "validate_protocol",
    "write_json",
]
