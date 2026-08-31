"""Independent, read-only evaluator for the S1 Neural Affine runs.

This module deliberately does not import the S1 implementation.  It owns a
small copy of the renderer and the two model constructors needed to replay a
run.  The public entry point is :func:`audit_run`.

The evaluator expects a run directory containing a checkpoint payload with a
``model_state`` member and a primary prediction NPZ with the keys
``pred_px``, ``true_px``, ``anchor_pred_px`` and ``anchor_true_px``.  The
protocol may override those filenames through ``primary_predictions`` and
``primary_metrics``.  A small ``identity`` model variant is supported for
lightweight contract tests; production runs use ``vanilla`` or ``coordconv``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_COORD_SCALE = 223.0
_DEFAULT_SAFE_BOX = (59.0, 164.0)
_DEFAULT_DENSE_GRID = 41
_DEFAULT_GRID_LEVELS = 8
_DEFAULT_SUPPORT_TIDS = (0, 7, 56, 63)
_DEFAULT_SUPERSAMPLE = 4
_PREDICTION_TOL_PX = 1e-3
_METRIC_TOL = 1e-6
_TRUTH_TOL_PX = 1e-6
_DATASET_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _nested_first(mapping: Mapping[str, Any], paths: Iterable[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        current: Any = mapping
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return default


def _as_float_pair(value: Any, name: str) -> Tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite numbers")
    return float(array[0]), float(array[1])


def _protocol_config(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    data = protocol.get("data") if isinstance(protocol.get("data"), Mapping) else {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), Mapping) else {}
    anchor_grid = protocol.get("anchor_grid") if isinstance(protocol.get("anchor_grid"), Mapping) else {}
    dense_grid_value = protocol.get("dense_grid")
    dense_grid_mapping = dense_grid_value if isinstance(dense_grid_value, Mapping) else {}
    renderer = protocol.get("renderer") if isinstance(protocol.get("renderer"), Mapping) else {}
    model = protocol.get("model") if isinstance(protocol.get("model"), Mapping) else {}
    evaluation = protocol.get("evaluation") if isinstance(protocol.get("evaluation"), Mapping) else {}

    image_size = int(_first(data, "image_size", default=_first(protocol, "image_size", default=_DEFAULT_IMAGE_SIZE)))
    coord_scale = float(
        _first(
            data,
            "coord_scale",
            default=_first(protocol, "coord_scale", default=image_size - 1.0),
        )
    )
    safe_box_value = _first(
        grid,
        "safe_box",
        default=_first(
            data,
            "safe_box",
            default=_first(protocol, "safe_box", "safe_box_px", default=_DEFAULT_SAFE_BOX),
        ),
    )
    if isinstance(safe_box_value, Mapping):
        safe_box = (
            float(safe_box_value.get("lo", safe_box_value.get("low", _DEFAULT_SAFE_BOX[0]))),
            float(safe_box_value.get("hi", safe_box_value.get("high", _DEFAULT_SAFE_BOX[1]))),
        )
    else:
        safe_box = _as_float_pair(safe_box_value, "safe_box")
    dense_grid_candidate = _first(
        grid,
        "dense_grid",
        "n_dense",
        default=_first(
            evaluation,
            "dense_grid",
            "n_grid",
            default=_first(protocol, "dense_grid", "n_dense", default=dense_grid_mapping or _DEFAULT_DENSE_GRID),
        ),
    )
    if isinstance(dense_grid_candidate, Mapping):
        dense_grid_candidate = _first(dense_grid_candidate, "n", "size", "count", default=_DEFAULT_DENSE_GRID)
    dense_grid = int(dense_grid_candidate)
    grid_levels = int(
        _first(
            grid,
            "levels",
            "grid_levels",
            "n_levels",
            default=_first(
                data,
                "grid_levels",
                "n_levels",
                default=_first(
                    anchor_grid,
                    "levels",
                    "n_levels",
                    "nx",
                    default=_first(protocol, "grid_levels", "n_levels", default=_DEFAULT_GRID_LEVELS),
                ),
            ),
        )
    )
    support_tids_value = _first(
        grid,
        "support_tids",
        default=_first(
            data,
            "support_tids",
            default=_first(
                anchor_grid,
                "tids",
                "support_tids",
                default=_first(protocol, "support_tids", default=_DEFAULT_SUPPORT_TIDS),
            ),
        ),
    )
    support_tids = tuple(int(value) for value in np.asarray(support_tids_value).reshape(-1).tolist())
    if not support_tids:
        raise ValueError("support_tids must not be empty")

    template_value = _first(
        data,
        "template",
        default=_first(protocol, "template", default=None),
    )
    if template_value is None:
        template_value = _first(protocol, "appearance", default=None)
    if isinstance(template_value, Mapping):
        family = str(_first(template_value, "family", "kind", default=_first(data, "family", default="blob")))
        params_value = _first(template_value, "params", "parameters", default={})
    else:
        family = str(
            _first(
                data,
                "family",
                default=_first(renderer, "family", "kind", default=_first(protocol, "family", default="blob")),
            )
        )
        params_value = _first(data, "template_params", "params", default=_first(protocol, "template_params", "params", default={}))
    params = dict(params_value) if isinstance(params_value, Mapping) else {}
    # The frozen S1 protocol stores renderer parameters directly alongside
    # ``family`` (for example ``renderer.sigma``), whereas small contract
    # protocols commonly use ``template.params``.  Merge only renderer
    # drawing parameters and keep explicit template parameters authoritative.
    for key, value in renderer.items():
        if key not in {"family", "kind", "supersample", "downsample", "dtype", "mode", "workers", "formula"}:
            params.setdefault(str(key), value)

    variant = str(
        _first(
            model,
            "variant",
            "architecture",
            "arch",
            default=_first(protocol, "variant", "architecture", "arch", default="vanilla"),
        )
    ).lower()
    output_space = str(
        _first(
            model,
            "output_space",
            "target_space",
            default=_first(protocol, "output_space", default="normalized"),
        )
    ).lower()
    batch_size = int(
        _first(
            evaluation,
            "batch_size",
            default=_first(protocol, "eval_batch_size", "batch_size", default=64),
        )
    )
    supersample = int(
        _first(
            data,
            "supersample",
            default=_first(renderer, "supersample", default=_first(protocol, "supersample", default=_DEFAULT_SUPERSAMPLE)),
        )
    )
    render_workers = int(_first(renderer, "workers", default=_first(data, "workers", default=1)))
    support_px_value = _first(
        grid,
        "support_px",
        default=_first(
            data,
            "support_px",
            default=_first(
                anchor_grid,
                "support_px",
                "points_px",
                default=_first(protocol, "support_px", default=None),
            ),
        ),
    )
    support_px = None if support_px_value is None else np.asarray(support_px_value, dtype=np.float64).reshape(-1, 2)
    if support_px is not None and len(support_px) != len(support_tids):
        raise ValueError("support_px length must match support_tids")
    if image_size < 2 or coord_scale <= 0 or dense_grid < 1 or grid_levels < 2:
        raise ValueError("invalid image/grid protocol")
    if batch_size < 1 or supersample < 1 or render_workers < 1:
        raise ValueError("invalid evaluator batch/supersample")
    return {
        "image_size": image_size,
        "coord_scale": coord_scale,
        "safe_box": safe_box,
        "dense_grid": dense_grid,
        "grid_levels": grid_levels,
        "support_tids": support_tids,
        "support_px": support_px,
        "family": family,
        "params": params,
        "variant": variant,
        "output_space": output_space,
        "batch_size": batch_size,
        "supersample": supersample,
        "render_workers": render_workers,
        "protocol": protocol,
    }


def _integer_grid(cfg: Mapping[str, Any]) -> np.ndarray:
    lo, hi = cfg["safe_box"]
    levels = int(cfg["grid_levels"])
    xs = np.linspace(float(lo), float(hi), levels, dtype=np.float64)
    return np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)


def _dense_grid(cfg: Mapping[str, Any]) -> np.ndarray:
    lo, hi = cfg["safe_box"]
    n = int(cfg["dense_grid"])
    xs = np.linspace(float(lo), float(hi), n, dtype=np.float64)
    return np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1)


def _resampling_lanczos() -> Any:
    from PIL import Image

    return getattr(Image, "Resampling", Image).LANCZOS


def _downsample(high_image: Any, image_size: int) -> np.ndarray:
    from PIL import Image

    out = high_image.resize((int(image_size), int(image_size)), resample=_resampling_lanczos())
    return np.asarray(out, dtype=np.uint8)


@lru_cache(maxsize=8)
def _high_resolution_grid(image_size: int, supersample: int) -> Tuple[np.ndarray, np.ndarray]:
    high = int(image_size) * int(supersample)
    yy, xx = np.mgrid[0:high, 0:high].astype(np.float64)
    return np.ascontiguousarray(xx), np.ascontiguousarray(yy)


def _render_blob(cx: float, cy: float, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    from PIL import Image

    sigma = float(params.get("sigma", 6.0))
    if sigma <= 0:
        raise ValueError("blob sigma must be positive")
    xx, yy = _high_resolution_grid(int(image_size), int(supersample))
    dx = xx - float(cx) * supersample
    dy = yy - float(cy) * supersample
    value = np.exp(-(dx * dx + dy * dy) / (2.0 * (sigma * supersample) ** 2))
    image = Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")
    return _downsample(image, image_size)


def _render_blob_batch(
    points: np.ndarray,
    params: Mapping[str, Any],
    image_size: int,
    supersample: int,
) -> np.ndarray:
    """Render a point batch while retaining the scalar renderer's exact steps."""

    from PIL import Image

    sigma = float(params.get("sigma", 6.0))
    if sigma <= 0:
        raise ValueError("blob sigma must be positive")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    output = np.empty((len(points), int(image_size), int(image_size)), dtype=np.uint8)
    xx, yy = _high_resolution_grid(int(image_size), int(supersample))
    # Keep the temporary broadcast bounded.  At the formal 224x224/4x
    # protocol, 32 samples use about 200 MB for the two coordinate deltas and
    # the value array, instead of materialising the whole dense set at once.
    for start in range(0, len(points), 32):
        batch = points[start : start + 32]
        dx = xx[None, :, :] - batch[:, 0, None, None] * supersample
        dy = yy[None, :, :] - batch[:, 1, None, None] * supersample
        values = np.exp(-(dx * dx + dy * dy) / (2.0 * (sigma * supersample) ** 2))
        for offset, value in enumerate(values):
            image = Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")
            output[start + offset] = _downsample(image, image_size)
    return output


def _render_blob_points(
    points: np.ndarray,
    params: Mapping[str, Any],
    image_size: int,
    supersample: int,
    workers: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if workers <= 1 or len(points) <= 32:
        return _render_blob_batch(points, params, image_size, supersample)
    chunks = [points[start : start + 32] for start in range(0, len(points), 32)]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        rendered = list(pool.map(lambda chunk: _render_blob_batch(chunk, params, image_size, supersample), chunks))
    return np.concatenate(rendered, axis=0) if rendered else np.empty((0, image_size, image_size), dtype=np.uint8)


def _render_point(cx: float, cy: float, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    high = int(image_size) * int(supersample)
    image = Image.new("L", (high, high), color=0)
    draw = ImageDraw.Draw(image)
    radius = max(0, int(round(float(params.get("radius", 0.0)) * supersample)))
    x = int(round(float(cx) * supersample))
    y = int(round(float(cy) * supersample))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return _downsample(image, image_size)


def _stroke_polyline(points: np.ndarray, image_size: int, supersample: int, width: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    high = int(image_size) * int(supersample)
    image = Image.new("L", (high, high), color=0)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * supersample, float(y) * supersample) for x, y in np.asarray(points, dtype=np.float64)]
    line_width = max(1, int(round(float(width) * supersample)))
    try:
        draw.line(xy, fill=255, width=line_width, joint="curve")
    except TypeError:
        draw.line(xy, fill=255, width=line_width)
    return _downsample(image, image_size)


def _render_line(cx: float, cy: float, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    half = float(params.get("length", 28.0)) * 0.5
    theta = np.deg2rad(float(params.get("theta_deg", 0.0)))
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    points = np.asarray([np.array([cx, cy]) - half * direction, np.array([cx, cy]) + half * direction])
    return _stroke_polyline(points, image_size, supersample, float(params.get("width", 3.0)))


def _render_circle(cx: float, cy: float, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    high = int(image_size) * int(supersample)
    image = Image.new("L", (high, high), color=0)
    draw = ImageDraw.Draw(image)
    radius = float(params.get("radius", 16.0)) * supersample
    width = max(1, int(round(float(params.get("width", 3.0)) * supersample)))
    draw.ellipse(((cx - float(params.get("radius", 16.0))) * supersample,
                  (cy - float(params.get("radius", 16.0))) * supersample,
                  (cx + float(params.get("radius", 16.0))) * supersample,
                  (cy + float(params.get("radius", 16.0))) * supersample),
                 outline=255, width=width)
    del radius
    return _downsample(image, image_size)


def _render_arc(cx: float, cy: float, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    high = int(image_size) * int(supersample)
    image = Image.new("L", (high, high), color=0)
    draw = ImageDraw.Draw(image)
    radius = float(params.get("radius", 18.0))
    width = max(1, int(round(float(params.get("width", 3.0)) * supersample)))
    box = ((cx - radius) * supersample, (cy - radius) * supersample,
           (cx + radius) * supersample, (cy + radius) * supersample)
    draw.arc(box, start=float(params.get("start_deg", 20.0)), end=float(params.get("end_deg", 200.0)), fill=255, width=width)
    return _downsample(image, image_size)


def _render_polygon(cx: float, cy: float, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    high = int(image_size) * int(supersample)
    image = Image.new("L", (high, high), color=0)
    draw = ImageDraw.Draw(image)
    n_sides = max(3, min(int(params.get("n_sides", 3)), 6))
    radius = float(params.get("radius", 18.0))
    theta = np.deg2rad(float(params.get("theta_deg", 0.0)))
    angles = theta + np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    vertices = [(float(cx + radius * np.cos(angle)) * supersample,
                float(cy + radius * np.sin(angle)) * supersample) for angle in angles]
    draw.polygon(vertices, fill=255)
    return _downsample(image, image_size)


def _render_family(cx: float, cy: float, family: str, params: Mapping[str, Any], image_size: int, supersample: int) -> np.ndarray:
    family = str(family).lower()
    if family == "blob":
        return _render_blob(cx, cy, params, image_size, supersample)
    if family in {"point", "identity"}:
        return _render_point(cx, cy, params, image_size, supersample)
    if family == "line":
        return _render_line(cx, cy, params, image_size, supersample)
    if family == "circle":
        return _render_circle(cx, cy, params, image_size, supersample)
    if family == "arc":
        return _render_arc(cx, cy, params, image_size, supersample)
    if family == "polygon":
        return _render_polygon(cx, cy, params, image_size, supersample)
    raise ValueError(f"unsupported independent renderer family: {family}")


def _appearance_specs(cfg: Mapping[str, Any]) -> list[Tuple[str, Dict[str, Any]]]:
    protocol = cfg["protocol"]
    data = protocol.get("data") if isinstance(protocol.get("data"), Mapping) else {}
    raw = _first(data, "appearances", default=_first(protocol, "appearances", default=None))
    if raw is None:
        return [(str(cfg["family"]), dict(cfg["params"]))]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("appearances must be a list")
    specs: list[Tuple[str, Dict[str, Any]]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("each appearance must be an object")
        family = str(_first(item, "family", "kind", default=cfg["family"]))
        params = _first(item, "params", "parameters", default={})
        specs.append((family, dict(params) if isinstance(params, Mapping) else {}))
    if not specs:
        raise ValueError("appearances must not be empty")
    return specs


def _render_dataset(cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Recreate support and dense images from the protocol, without S1 imports."""

    image_size = int(cfg["image_size"])
    supersample = int(cfg["supersample"])
    dense_grid = _dense_grid(cfg)
    dense_points = dense_grid.reshape(-1, 2)
    if cfg["support_px"] is None:
        integer = _integer_grid(cfg)
        support_points = integer[np.asarray(cfg["support_tids"], dtype=np.int64)]
    else:
        support_points = np.asarray(cfg["support_px"], dtype=np.float64)
    specs = _appearance_specs(cfg)
    dense_images: list[np.ndarray] = []
    anchor_images: list[np.ndarray] = []
    for family, params in specs:
        if str(family).lower() == "blob":
            workers = int(cfg.get("render_workers", 1))
            dense_images.append(_render_blob_points(dense_points, params, image_size, supersample, workers))
            anchor_images.append(_render_blob_points(support_points, params, image_size, supersample, workers))
        else:
            dense_images.append(
                np.stack([_render_family(float(x), float(y), family, params, image_size, supersample) for x, y in dense_points])
            )
            anchor_images.append(
                np.stack([_render_family(float(x), float(y), family, params, image_size, supersample) for x, y in support_points])
            )
    dense_images_array = np.stack(dense_images, axis=0)
    anchor_images_array = np.stack(anchor_images, axis=0)
    dense_true = np.broadcast_to(dense_points[None, :, :], (len(specs), len(dense_points), 2)).copy()
    anchor_true = np.broadcast_to(support_points[None, :, :], (len(specs), len(support_points), 2)).copy()
    return {
        "dense_images": dense_images_array,
        "anchor_images": anchor_images_array,
        "dense_true_px": dense_true,
        "anchor_true_px": anchor_true,
        "dense_grid_shape": np.asarray([len(specs), int(cfg["dense_grid"]), int(cfg["dense_grid"]), 2], dtype=np.int64),
    }


def _cached_render_dataset(cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Render once per frozen protocol within the independent audit process.

    All nine checkpoint audits use identical protocol pixels.  Reusing this
    independently generated, read-only dataset avoids nine CPU rerenders while
    retaining a separate implementation from the training-side cache.
    """

    key = json.dumps(_jsonable(dict(cfg)), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if key not in _DATASET_CACHE:
        if len(_DATASET_CACHE) >= 4:
            _DATASET_CACHE.pop(next(iter(_DATASET_CACHE)))
        dataset = _render_dataset(cfg)
        for value in dataset.values():
            value.setflags(write=False)
        _DATASET_CACHE[key] = dataset
    return _DATASET_CACHE[key]


class _CoordConvResNet18(nn.Module):
    """Independent copy of the historical 5-channel CoordConv wrapper."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as models

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 2)
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            5,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        yy = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1).expand(batch, 1, height, width)
        xx = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width).expand(batch, 1, height, width)
        return self.backbone(torch.cat([x, xx, yy], dim=1))


class _IdentityModel(nn.Module):
    """Contract-test marker; production S1 protocols must use a CNN variant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - handled by the special path
        raise RuntimeError("identity model is evaluated from protocol targets")


def _build_model(variant: str) -> nn.Module:
    variant = str(variant).lower()
    if variant in {"identity", "synthetic_identity"}:
        return _IdentityModel()
    import torchvision.models as models

    if variant in {"vanilla", "standard", "resnet18", "cnn"}:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        return model
    if variant in {"coordconv", "coord_conv", "resnet18_coordconv"}:
        return _CoordConvResNet18()
    raise ValueError(f"unsupported model variant: {variant}")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _checkpoint_path(run_dir: Path, protocol: Mapping[str, Any]) -> Path:
    candidates: list[Path] = []
    for key in ("checkpoint", "checkpoint_path", "primary_checkpoint"):
        value = protocol.get(key)
        if isinstance(value, str):
            candidates.append(Path(value))
    model = protocol.get("model") if isinstance(protocol.get("model"), Mapping) else {}
    value = _first(model, "checkpoint", "checkpoint_path", default=None)
    if isinstance(value, str):
        candidates.append(Path(value))
    candidates.extend(
        run_dir / name
        for name in ("best_slim.pt", "checkpoint.pt", "best.pt", "model.pt", "identity.pt")
    )
    candidates.extend(
        run_dir / "checkpoints" / name
        for name in ("best_slim.pt", "checkpoint.pt", "best.pt", "model.pt", "identity.pt", "final.pt")
    )
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else run_dir / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"no checkpoint found under {run_dir}")


def _state_dict_from_payload(payload: Any) -> Tuple[MutableMapping[str, Any], str | None]:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    variant = payload.get("variant")
    state = payload.get("model_state")
    if state is None:
        state = payload.get("state_dict")
    if state is None and all(isinstance(key, str) for key in payload.keys()):
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain model_state/state_dict")
    normalized: Dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        normalized[name[7:] if name.startswith("module.") else name] = value
    return normalized, None if variant is None else str(variant)


def _load_model(checkpoint: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = _torch_load(checkpoint)
    state, payload_variant = _state_dict_from_payload(payload)
    variant = str(payload_variant or cfg["variant"]).lower()
    model = _build_model(variant)
    if variant not in {"identity", "synthetic_identity"}:
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, {"variant": variant, "payload_keys": sorted(str(key) for key in payload.keys()) if isinstance(payload, Mapping) else []}


@torch.no_grad()
def _forward_images(model: nn.Module, images: np.ndarray, cfg: Mapping[str, Any], device: torch.device) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim != 3:
        raise ValueError(f"images must be [N,H,W], got {array.shape}")
    batch_size = int(cfg["batch_size"])
    outputs: list[np.ndarray] = []
    for start in range(0, len(array), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(array[start : start + batch_size]))
        tensor = batch.to(device=device, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1).div_(255.0)
        value = model(tensor)
        if isinstance(value, Mapping):
            value = value.get("pred", value.get("output"))
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not torch.is_tensor(value):
            raise ValueError("model output must be a tensor")
        outputs.append(value.detach().float().cpu().numpy())
    if not outputs:
        return np.empty((0, 2), dtype=np.float64)
    result = np.concatenate(outputs, axis=0)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"model output must be [N,2], got {result.shape}")
    return result.astype(np.float64, copy=False)


def _to_pixel_space(pred: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(pred, dtype=np.float64)
    if str(cfg["output_space"]).lower() in {"px", "pixel", "pixels"}:
        return value
    return value * float(cfg["coord_scale"])


def _regenerated_predictions(cfg: Mapping[str, Any], checkpoint: Path, device: torch.device) -> Dict[str, np.ndarray | Dict[str, Any]]:
    dataset = _cached_render_dataset(cfg)
    model, model_meta = _load_model(checkpoint, cfg, device)
    variant = str(model_meta["variant"])
    if variant in {"identity", "synthetic_identity"}:
        dense_pred = dataset["dense_true_px"].copy()
        anchor_pred = dataset["anchor_true_px"].copy()
    else:
        dense_norm = _forward_images(model, dataset["dense_images"].reshape(-1, cfg["image_size"], cfg["image_size"]), cfg, device)
        anchor_norm = _forward_images(model, dataset["anchor_images"].reshape(-1, cfg["image_size"], cfg["image_size"]), cfg, device)
        dense_pred = _to_pixel_space(dense_norm, cfg).reshape(dataset["dense_true_px"].shape)
        anchor_pred = _to_pixel_space(anchor_norm, cfg).reshape(dataset["anchor_true_px"].shape)
    return {
        "pred_px": dense_pred,
        "true_px": dataset["dense_true_px"],
        "anchor_pred_px": anchor_pred,
        "anchor_true_px": dataset["anchor_true_px"],
        "model": model_meta,
    }


def _rows(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or array.shape[-1:] != (2,):
        raise ValueError(f"{name} must end in dimension 2, got {array.shape}")
    result = array.reshape(-1, 2)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _mean_vector_error(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.shape != true.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} != {true.shape}")
    return float(np.mean(np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64), axis=1)))


def fit_affine_float64(pred_px: Any, true_px: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``A @ pred + b -> true`` in float64 and return ``A, b, residual``."""

    pred = _rows(pred_px, "pred_px")
    true = _rows(true_px, "true_px")
    if pred.shape != true.shape or len(pred) < 3:
        raise ValueError("affine fit needs matching arrays with at least three rows")
    design = np.concatenate([pred, np.ones((len(pred), 1), dtype=np.float64)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, true, rcond=None)
    matrix = np.asarray(coef[:2, :].T, dtype=np.float64)
    bias = np.asarray(coef[2, :], dtype=np.float64)
    fitted = pred @ matrix.T + bias
    residual = fitted - true
    return matrix, bias, residual


def recompute_metrics(pred_px: Any, true_px: Any, anchor_pred_px: Any, anchor_true_px: Any) -> Dict[str, Any]:
    pred = _rows(pred_px, "pred_px")
    true = _rows(true_px, "true_px")
    anchor_pred = _rows(anchor_pred_px, "anchor_pred_px")
    anchor_true = _rows(anchor_true_px, "anchor_true_px")
    if pred.shape != true.shape:
        raise ValueError("full-box prediction/target shape mismatch")
    if anchor_pred.shape != anchor_true.shape:
        raise ValueError("anchor prediction/target shape mismatch")
    matrix, bias, residual = fit_affine_float64(pred, true)
    return {
        "anchor_mae_px": _mean_vector_error(anchor_pred, anchor_true),
        "raw_full_box_mae_px": _mean_vector_error(pred, true),
        "affine_removed_mae_px": float(np.mean(np.linalg.norm(residual, axis=1))),
        "affine_A": matrix,
        "affine_b": bias,
        "n_full_box": int(len(pred)),
        "n_anchor": int(len(anchor_pred)),
        "dtype": "float64",
    }


def _prediction_path(run_dir: Path, protocol: Mapping[str, Any]) -> Path:
    candidates: list[Path] = []
    for key in ("primary_predictions", "primary_predictions_path", "predictions_path"):
        value = protocol.get(key)
        if isinstance(value, str):
            candidates.append(Path(value))
    evaluation = protocol.get("evaluation") if isinstance(protocol.get("evaluation"), Mapping) else {}
    value = _first(evaluation, "primary_predictions", "predictions_path", default=None)
    if isinstance(value, str):
        candidates.append(Path(value))
    candidates.extend(run_dir / name for name in ("primary_predictions.npz", "predictions.npz", "field.npz"))
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else run_dir / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"no primary predictions NPZ found under {run_dir}")


def _metrics_path(run_dir: Path, protocol: Mapping[str, Any]) -> Path | None:
    candidates: list[Path] = []
    for key in ("primary_metrics", "primary_metrics_path", "metrics_path"):
        value = protocol.get(key)
        if isinstance(value, str):
            candidates.append(Path(value))
    evaluation = protocol.get("evaluation") if isinstance(protocol.get("evaluation"), Mapping) else {}
    value = _first(evaluation, "primary_metrics", "metrics_path", default=None)
    if isinstance(value, str):
        candidates.append(Path(value))
    candidates.extend(
        run_dir / name
        for name in ("primary_metrics.json", "metrics.json", "dense_official.json", "train_summary.json", "summary.json", "evaluation.json", "results.json")
    )
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else run_dir / candidate
        if resolved.exists():
            return resolved
    return None


def _recursive_metric(value: Any, aliases: Sequence[str]) -> float | None:
    if isinstance(value, Mapping):
        for alias in aliases:
            candidate = value.get(alias)
            if isinstance(candidate, (int, float, np.number)) and math.isfinite(float(candidate)):
                return float(candidate)
        for child in value.values():
            result = _recursive_metric(child, aliases)
            if result is not None:
                return result
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            result = _recursive_metric(child, aliases)
            if result is not None:
                return result
    return None


def _primary_metric_values(path: Path | None) -> Dict[str, float | None]:
    if path is None:
        return {"anchor_mae_px": None, "raw_full_box_mae_px": None, "affine_removed_mae_px": None}
    payload = _read_json(path)
    return {
        "anchor_mae_px": _recursive_metric(
            payload,
            ("anchor_mae_px", "train_anchor_mae_px", "mean_anchor_mae_px", "best_anchor_mae_px", "final_anchor_mae_px"),
        ),
        "raw_full_box_mae_px": _recursive_metric(
            payload,
            ("raw_full_box_mae_px", "raw_mae_px", "mean_box_mae_px", "full_box_mae_px"),
        ),
        "affine_removed_mae_px": _recursive_metric(
            payload,
            ("affine_removed_mae_px", "affine_residual_mae_px", "affine_removed_residual_mae_px", "residual_mae_px"),
        ),
    }


def _compare_scalar(expected: float | None, actual: float, tolerance: float) -> Dict[str, Any]:
    if expected is None:
        return {"available": False, "passed": False, "expected": None, "actual": actual, "abs_diff": None, "tolerance": tolerance}
    diff = abs(float(expected) - float(actual))
    return {"available": True, "passed": bool(diff <= tolerance), "expected": float(expected), "actual": float(actual), "abs_diff": diff, "tolerance": tolerance}


def _load_primary_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("pred_px", "true_px", "anchor_pred_px", "anchor_true_px")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"primary prediction NPZ missing keys: {missing}")
        return {key: np.asarray(payload[key], dtype=np.float64) for key in required}


def _compare_predictions(primary: Mapping[str, np.ndarray], regenerated: Mapping[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    all_passed = True
    for name in ("pred_px", "true_px", "anchor_pred_px", "anchor_true_px"):
        left = _rows(primary[name], f"primary.{name}")
        right = _rows(regenerated[name], f"regenerated.{name}")
        if left.shape != right.shape:
            fields[name] = {"passed": False, "shape_primary": list(left.shape), "shape_regenerated": list(right.shape), "max_abs_diff_px": None}
            all_passed = False
            continue
        diff = float(np.max(np.abs(left - right))) if left.size else 0.0
        tolerance = _TRUTH_TOL_PX if name.endswith("true_px") else _PREDICTION_TOL_PX
        passed = bool(diff <= tolerance)
        fields[name] = {"passed": passed, "shape": list(left.shape), "max_abs_diff_px": diff, "tolerance_px": tolerance}
        all_passed = all_passed and passed
    prediction_diffs = [
        fields[name]["max_abs_diff_px"]
        for name in ("pred_px", "anchor_pred_px")
        if fields.get(name, {}).get("max_abs_diff_px") is not None
    ]
    truth_diffs = [
        fields[name]["max_abs_diff_px"]
        for name in ("true_px", "anchor_true_px")
        if fields.get(name, {}).get("max_abs_diff_px") is not None
    ]
    return {
        "passed": all_passed,
        "fields": fields,
        "primary_prediction_max_abs_diff_px": max(prediction_diffs) if prediction_diffs else None,
        "primary_truth_max_abs_diff_px": max(truth_diffs) if truth_diffs else None,
    }


def _audit_impl(run_dir: Path, protocol_path: Path, device: str) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = _read_json(protocol_path)
    cfg = _protocol_config(protocol)
    # Formal checkpoints carry ``variant`` themselves.  The run config is a
    # useful fallback for older payloads, and is deliberately consulted only
    # for architecture selection; geometry still comes solely from protocol.
    run_cfg_path = run_dir / "config.json"
    if run_cfg_path.exists():
        run_cfg = _read_json(run_cfg_path)
        if isinstance(run_cfg.get("variant"), str):
            cfg = dict(cfg)
            cfg["variant"] = str(run_cfg["variant"])
    prediction_path = _prediction_path(run_dir, protocol)
    metrics_path = _metrics_path(run_dir, protocol)
    checkpoint = _checkpoint_path(run_dir, protocol)
    device_obj = torch.device(device)
    regenerated = _regenerated_predictions(cfg, checkpoint, device_obj)
    primary = _load_primary_arrays(prediction_path)
    recomputed = recompute_metrics(primary["pred_px"], primary["true_px"], primary["anchor_pred_px"], primary["anchor_true_px"])
    independent_recomputed = recompute_metrics(regenerated["pred_px"], regenerated["true_px"], regenerated["anchor_pred_px"], regenerated["anchor_true_px"])
    prediction_comparison = _compare_predictions(primary, regenerated)
    primary_metrics = _primary_metric_values(metrics_path)
    metric_comparison = {
        name: _compare_scalar(primary_metrics[name], float(recomputed[name]), _METRIC_TOL)
        for name in ("anchor_mae_px", "raw_full_box_mae_px", "affine_removed_mae_px")
    }
    metrics_passed = all(bool(value["passed"]) for value in metric_comparison.values())
    status = "passed" if prediction_comparison["passed"] and metrics_passed else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "run_dir": str(run_dir),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint), **dict(regenerated["model"])},
        "primary_predictions": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        "primary_metrics": {"path": None if metrics_path is None else str(metrics_path), "values": primary_metrics},
        "recomputed_metrics": recomputed,
        "independent_forward_metrics": independent_recomputed,
        "prediction_comparison": prediction_comparison,
        "metric_comparison": metric_comparison,
        "tolerances": {"prediction_max_abs_px": _PREDICTION_TOL_PX, "metric_abs": _METRIC_TOL, "truth_max_abs_px": _TRUTH_TOL_PX},
    }


def audit_run(run_dir: str | os.PathLike[str], protocol_path: str | os.PathLike[str], device: str = "cpu") -> Dict[str, Any]:
    """Re-evaluate one run and write ``independent_audit.json``.

    Errors are represented in the returned/written JSON so a caller can
    archive a failed audit without losing the failure reason.
    """

    run_path = Path(run_dir).resolve()
    protocol = Path(protocol_path).resolve()
    try:
        result = _audit_impl(run_path, protocol, device)
    except Exception as exc:  # the audit artifact is itself part of the contract
        result = {
            "schema_version": 1,
            "status": "error",
            "run_dir": str(run_path),
            "protocol_path": str(protocol),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    output = run_path / "independent_audit.json"
    result["audit_path"] = str(output)
    _write_json(output, result)
    return result


__all__ = ["audit_run", "fit_affine_float64", "recompute_metrics"]
