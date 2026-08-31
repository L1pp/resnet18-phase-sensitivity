"""BatchNorm-only exposure control.

The control intentionally does not construct an optimizer or call backward.
It places only BatchNorm modules in training mode, keeps every other module in
evaluation mode, runs forwards under ``torch.no_grad()``, and verifies that
parameter tensors did not change.  This isolates running-statistics exposure
from optimizer and gradient effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


class BNControlError(RuntimeError):
    """Raised when a BN-only control changes something outside BN buffers."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("BN-only control requires PyTorch") from exc
    return torch


def _tensor_bytes(value: Any) -> tuple[bytes, str, tuple[int, ...]]:
    tensor = value.detach().cpu().contiguous()
    # numpy conversion is supported for ordinary dense model parameters and
    # buffers.  Keeping dtype and shape in the digest prevents concatenation
    # ambiguities between differently shaped tensors.
    array = tensor.numpy()
    return array.tobytes(order="C"), str(array.dtype), tuple(array.shape)


def _hash_named_tensors(named_tensors: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        raw, dtype, shape = _tensor_bytes(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
    return digest.hexdigest()


def parameter_hash(model: Any) -> str:
    """SHA-256 digest of all named parameter tensors."""

    return _hash_named_tensors(model.named_parameters())


def buffer_hash(model: Any) -> str:
    """SHA-256 digest of all named buffers, including BN running statistics."""

    return _hash_named_tensors(model.named_buffers())


def _batch_norm_types():
    torch = _torch()
    return (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )


def batch_norm_buffer_names(model: Any) -> tuple[str, ...]:
    """List named buffers owned by BatchNorm modules."""

    bn_types = _batch_norm_types()
    names: list[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, bn_types):
            prefix = f"{module_name}." if module_name else ""
            for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                if getattr(module, suffix, None) is not None:
                    names.append(prefix + suffix)
    return tuple(sorted(names))


def _named_buffer_values(model: Any) -> dict[str, Any]:
    """Snapshot buffers as detached CPU clones, never live model references."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
    }


def changed_buffer_names(before: Mapping[str, Any], after_model: Any) -> tuple[str, ...]:
    """Return buffer names whose exact values differ from a saved snapshot."""

    after = _named_buffer_values(after_model)
    changed: list[str] = []
    for name, old in before.items():
        if name not in after:
            changed.append(name)
            continue
        torch = _torch()
        if not bool(torch.equal(old.detach().cpu(), after[name].detach().cpu())):
            changed.append(name)
    changed.extend(name for name in after if name not in before)
    return tuple(sorted(set(changed)))


def _snapshot_training_flags(model: Any) -> dict[int, bool]:
    return {id(module): bool(module.training) for module in model.modules()}


def _restore_training_flags(model: Any, snapshot: Mapping[int, bool]) -> None:
    # Calling .train recursively would overwrite child states, so set each
    # module's flag directly and restore the exact pre-control state.
    for module in model.modules():
        if id(module) in snapshot:
            module.training = bool(snapshot[id(module)])


def _set_bn_only_training(model: Any) -> None:
    """Set BN modules to train and all other modules to eval."""

    model.eval()
    for module in model.modules():
        if isinstance(module, _batch_norm_types()):
            module.train(True)


def _move_to_device(value: Any, device: Any) -> Any:
    if device is None:
        return value
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _default_model_inputs(batch: Any) -> Any:
    """Extract model inputs from common dataloader batch shapes.

    A mapping prefers ``inputs`` then ``x``.  For a tuple/list, the first item
    is treated as inputs (the common ``(x, y)`` case); callers with a multi-input
    model should supply an explicit ``batch_to_inputs`` adapter.
    """

    if isinstance(batch, Mapping):
        for key in ("inputs", "x", "image", "images"):
            if key in batch:
                return batch[key]
        return batch
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("empty batch cannot be passed to model")
        return batch[0]
    return batch


def _evaluate_without_bn_updates(
    model: Any,
    exposure: int,
    evaluate: Callable[[Any, int], Any] | None,
    exposure_training_flags: Mapping[int, bool],
) -> Any:
    """Run an evaluator in eval mode, then restore BN-only exposure mode."""

    if evaluate is None:
        return None
    torch = _torch()
    # ``model.eval()`` is deliberately scoped to the evaluator. A plain
    # forward in train mode would update running statistics and contaminate
    # the named exposure checkpoint.
    model.eval()
    try:
        with torch.no_grad():
            return evaluate(model, exposure)
    finally:
        _restore_training_flags(model, exposure_training_flags)


@dataclass(frozen=True)
class BNExposureRecord:
    exposure: int
    evaluation: Any
    parameter_hash: str
    buffer_hash: str
    changed_buffer_names: tuple[str, ...]
    bn_changed_buffer_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["changed_buffer_names"] = list(self.changed_buffer_names)
        result["bn_changed_buffer_names"] = list(self.bn_changed_buffer_names)
        return result


@dataclass(frozen=True)
class BNControlReport:
    parameter_hash_before: str
    parameter_hash_after: str
    buffer_hash_before: str
    buffer_hash_after: str
    bn_buffer_names: tuple[str, ...]
    records: tuple[BNExposureRecord, ...]

    @property
    def parameters_unchanged(self) -> bool:
        return self.parameter_hash_before == self.parameter_hash_after

    @property
    def only_bn_buffers_changed(self) -> bool:
        return all(
            set(record.changed_buffer_names).issubset(set(self.bn_buffer_names))
            for record in self.records
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_hash_before": self.parameter_hash_before,
            "parameter_hash_after": self.parameter_hash_after,
            "buffer_hash_before": self.buffer_hash_before,
            "buffer_hash_after": self.buffer_hash_after,
            "bn_buffer_names": list(self.bn_buffer_names),
            "parameters_unchanged": self.parameters_unchanged,
            "only_bn_buffers_changed": self.only_bn_buffers_changed,
            "records": [record.to_dict() for record in self.records],
        }


def run_bn_only_control(
    model: Any,
    batches: Iterable[Any],
    *,
    exposure_checkpoints: Sequence[int] = (0, 1, 10, 50, 100, 500, 3000),
    evaluate: Callable[[Any, int], Any] | None = None,
    batch_to_inputs: Callable[[Any], Any] | None = None,
    device: Any = None,
    strict: bool = True,
) -> BNControlReport:
    """Expose a model to batches while updating only BN running statistics.

    Exposure ``n`` means exactly ``n`` forward batches.  Checkpoint zero is
    evaluated before the first batch.  A checkpoint above the available batch
    count raises ``ValueError`` instead of silently returning incomplete data.
    In strict mode, any changed non-BN buffer or parameter raises
    :class:`BNControlError` after the model's original training flags are
    restored.
    """

    torch = _torch()
    raw_checkpoints = tuple(exposure_checkpoints)
    if not raw_checkpoints:
        raise ValueError("exposure_checkpoints must contain non-negative integers")
    validated_checkpoints: list[int] = []
    for value in raw_checkpoints:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean exposure checkpoints are not valid integers")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("exposure checkpoints must be integers") from exc
        try:
            exact = bool(value == integer)
        except Exception:
            exact = False
        if not exact:
            raise ValueError("exposure checkpoints must be integers; 1.5 is invalid")
        validated_checkpoints.append(integer)
    checkpoints = tuple(sorted(set(validated_checkpoints)))
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("exposure_checkpoints must contain non-negative integers")
    max_exposure = checkpoints[-1]
    before_parameters = parameter_hash(model)
    before_buffers = _named_buffer_values(model)
    before_buffer_hash = buffer_hash(model)
    bn_names = batch_norm_buffer_names(model)
    training_flags = _snapshot_training_flags(model)
    records: list[BNExposureRecord] = []
    requested = set(checkpoints)
    iterator = iter(batches)
    adapter = batch_to_inputs or _default_model_inputs

    def record(exposure: int, evaluation: Any) -> None:
        current_parameters = parameter_hash(model)
        current_buffer_hash = buffer_hash(model)
        changed = changed_buffer_names(before_buffers, model)
        changed_bn = tuple(name for name in changed if name in bn_names)
        if current_parameters != before_parameters and strict:
            raise BNControlError("parameter hash changed during BN-only control")
        if set(changed) - set(bn_names) and strict:
            raise BNControlError(
                "non-BN buffers changed during BN-only control: "
                + ", ".join(sorted(set(changed) - set(bn_names)))
            )
        records.append(
            BNExposureRecord(
                exposure=exposure,
                evaluation=evaluation,
                parameter_hash=current_parameters,
                buffer_hash=current_buffer_hash,
                changed_buffer_names=changed,
                bn_changed_buffer_names=changed_bn,
            )
        )

    try:
        _set_bn_only_training(model)
        exposure_training_flags = _snapshot_training_flags(model)
        with torch.no_grad():
            if 0 in requested:
                evaluation = _evaluate_without_bn_updates(
                    model, 0, evaluate, exposure_training_flags
                )
                record(0, evaluation)
            for exposure in range(1, max_exposure + 1):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"batches ended at exposure {exposure - 1}; "
                        f"checkpoint {max_exposure} requires {max_exposure} batches"
                    ) from exc
                inputs = _move_to_device(adapter(batch), device)
                if isinstance(inputs, Mapping):
                    model(**inputs)
                elif isinstance(inputs, tuple):
                    model(*inputs)
                else:
                    model(inputs)
                if exposure in requested:
                    evaluation = _evaluate_without_bn_updates(
                        model, exposure, evaluate, exposure_training_flags
                    )
                    record(exposure, evaluation)
    finally:
        _restore_training_flags(model, training_flags)

    after_parameters = parameter_hash(model)
    after_buffer_hash = buffer_hash(model)
    if after_parameters != before_parameters and strict:
        raise BNControlError("parameter hash changed after BN-only control")
    final_changed = changed_buffer_names(before_buffers, model)
    if set(final_changed) - set(bn_names) and strict:
        raise BNControlError(
            "non-BN buffers changed after BN-only control: "
            + ", ".join(sorted(set(final_changed) - set(bn_names)))
        )
    return BNControlReport(
        parameter_hash_before=before_parameters,
        parameter_hash_after=after_parameters,
        buffer_hash_before=before_buffer_hash,
        buffer_hash_after=after_buffer_hash,
        bn_buffer_names=bn_names,
        records=tuple(records),
    )


run_bn_control = run_bn_only_control
bn_only_control = run_bn_only_control


__all__ = [
    "BNControlError",
    "BNControlReport",
    "BNExposureRecord",
    "batch_norm_buffer_names",
    "bn_only_control",
    "buffer_hash",
    "changed_buffer_names",
    "parameter_hash",
    "run_bn_control",
    "run_bn_only_control",
]
