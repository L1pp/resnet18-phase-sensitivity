from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .models import require_torch, torch


RESUME_IDENTITY_FIELDS = (
    "cell_id",
    "run_seed",
    "training_path",
    "sampler_id",
    "index_id",
    "target_kind",
    "anchors",
)


def canonical_resume_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one frozen identity shape accepted by checkpoint resume."""
    if not isinstance(value, Mapping):
        raise ValueError("resume_identity must be a mapping")
    missing = [field for field in RESUME_IDENTITY_FIELDS if field not in value]
    extra = sorted(set(value) - set(RESUME_IDENTITY_FIELDS))
    if missing or extra:
        raise ValueError(f"resume_identity fields mismatch: missing={missing}, extra={extra}")
    anchors_value = value["anchors"]
    if anchors_value is None:
        anchors: list[int] = []
    elif isinstance(anchors_value, (list, tuple)) and not isinstance(anchors_value, (str, bytes)):
        if any(isinstance(anchor, bool) or not isinstance(anchor, (int, np.integer)) for anchor in anchors_value):
            raise ValueError("resume_identity anchors must contain only integers")
        anchors = [int(anchor) for anchor in anchors_value]
    else:
        raise ValueError("resume_identity anchors must be a list/tuple or null")
    if len(set(anchors)) != len(anchors):
        raise ValueError("resume_identity anchors must not contain duplicates")
    run_seed = value["run_seed"]
    if isinstance(run_seed, bool) or not isinstance(run_seed, (int, np.integer)):
        raise ValueError("resume_identity run_seed must be an integer")
    sampler = value["sampler_id"]
    if sampler is not None and not isinstance(sampler, str):
        raise ValueError("resume_identity sampler_id must be a string or null")
    identity = {
        "cell_id": str(value["cell_id"]),
        "run_seed": int(run_seed),
        "training_path": str(value["training_path"]),
        "sampler_id": sampler,
        "index_id": str(value["index_id"]),
        "target_kind": str(value["target_kind"]),
        "anchors": anchors,
    }
    for field in ("cell_id", "training_path", "index_id", "target_kind"):
        if not identity[field]:
            raise ValueError(f"resume_identity {field} must be nonempty")
    return identity


def resume_identity_from_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_resume_identity(
        {
            "cell_id": cell["cell_id"],
            "run_seed": cell["run_seed"],
            "training_path": cell["training_path"],
            "sampler_id": cell.get("sampler_id"),
            "index_id": cell["index_id"],
            "target_kind": cell["target_kind"],
            "anchors": cell.get("anchors", []),
        }
    )


def validate_resume_identity(observed: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    if observed is None:
        raise ValueError("checkpoint has no resume_identity and is not resumable")
    candidate = canonical_resume_identity(observed)
    frozen_expected = canonical_resume_identity(expected)
    if candidate != frozen_expected:
        mismatches = [field for field in RESUME_IDENTITY_FIELDS if candidate[field] != frozen_expected[field]]
        raise ValueError(f"resume_identity mismatch: {mismatches}")
    return candidate


def optimizer_parameter_names(model: Any, optimizer: Any) -> list[list[str]]:
    """Return names in optimizer group order so resume can reject group drift."""
    require_torch()
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter that is not in the model")
            names.append(name)
        groups.append(names)
    return groups


def capture_rng_state() -> dict[str, Any]:
    require_torch()
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: Mapping[str, Any]) -> None:
    require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_payload(
    model: Any,
    optimizer: Any | None,
    *,
    completed_step: int,
    stream_cursor: int,
    schedule_kind: str,
    resume_identity: Mapping[str, Any],
    schedule_position: int | None = None,
    stage_state: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture all mutable state needed for the next update.

    ``stream_cursor`` is the zero-based row used by the next update and therefore
    normally equals ``completed_step``.  Keeping both makes a mismatch explicit.
    """
    require_torch()
    completed = int(completed_step)
    cursor = int(stream_cursor)
    if completed < 0 or cursor < 0:
        raise ValueError("completed step and stream cursor must be nonnegative")
    return {
        "format": "trackb_checkpoint_v1",
        "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "optimizer_parameter_names": None if optimizer is None else optimizer_parameter_names(model, optimizer),
        "completed_step": completed,
        "next_step": completed + 1,
        "stream_cursor": cursor,
        "schedule_kind": str(schedule_kind),
        "resume_identity": canonical_resume_identity(resume_identity),
        "schedule_position": completed if schedule_position is None else int(schedule_position),
        "stage_state": dict(stage_state or {}),
        "runtime_state": dict(runtime_state or {}),
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None,
    **state: Any,
) -> Path:
    require_torch()
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(checkpoint_payload(model, optimizer, **state), temporary)
    temporary.replace(target)
    return target


def load_checkpoint_payload(path: str | Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    require_torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "trackb_checkpoint_v1":
        raise ValueError("unsupported checkpoint")
    if int(payload["next_step"]) != int(payload["completed_step"]) + 1:
        raise ValueError("checkpoint next-step relation is invalid")
    return payload


def validate_resume_checkpoint(
    path: str | Path,
    expected_resume_identity: Mapping[str, Any],
    *,
    map_location: str | Any = "cpu",
) -> dict[str, Any]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    validate_resume_identity(payload.get("resume_identity"), expected_resume_identity)
    return payload


def restore_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None,
    *,
    expected_schedule_kind: str,
    expected_resume_identity: Mapping[str, Any],
    expected_parameter_names: Sequence[Sequence[str]] | None = None,
    map_location: str | Any = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    validate_resume_identity(payload.get("resume_identity"), expected_resume_identity)
    if payload["schedule_kind"] != expected_schedule_kind:
        raise ValueError("checkpoint schedule kind does not match the resumed path")
    model.load_state_dict(payload["model_state"], strict=True)
    observed_names = payload["optimizer_parameter_names"]
    if optimizer is None:
        if payload["optimizer_state"] is not None:
            raise ValueError("checkpoint has optimizer state but resume path supplied none")
    else:
        expected = [list(group) for group in (expected_parameter_names or optimizer_parameter_names(model, optimizer))]
        if observed_names != expected:
            raise ValueError("optimizer parameter groups changed across resume")
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
