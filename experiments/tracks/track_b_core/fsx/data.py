from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import PACKAGE_ROOT, load_protocol


def positions(layout: str, safe_box_px: Sequence[float] = (59.0, 164.0)) -> np.ndarray:
    sizes = {"corners4": 2, "grid4x4": 4, "grid8x8": 8, "g64": 8, "g9": 3, "dense41": 41}
    if layout not in sizes:
        raise ValueError(f"unknown layout {layout}")
    xs = np.linspace(float(safe_box_px[0]), float(safe_box_px[1]), sizes[layout], dtype=np.float64)
    return np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)


def appearance_parameters(
    appearance_id: int,
    renderer: str,
    appearance_seed: int,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    protocol = protocol or load_protocol()
    spec = protocol["data"]["appearance"]
    namespaces = spec["renderer_namespaces"]
    if renderer not in namespaces:
        raise ValueError(f"unknown renderer {renderer}")
    aid = int(appearance_id)
    if aid < 0:
        raise ValueError("appearance_id must be nonnegative")
    seed_sequence = np.random.SeedSequence(
        [int(appearance_seed), aid, int(namespaces[renderer]), 1]
    )
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    angle = float(rng.uniform(0.0, np.pi))
    aspect = float(rng.uniform(float(spec["aspect_range"][0]), float(spec["aspect_range"][1])))
    sigma_pool = [float(value) for value in spec["sigma_pool"]]
    sigma = sigma_pool[(aid + 1) % len(sigma_pool)]
    if renderer == "blob_sigma6":
        sigma = 6.0
    if renderer == "line_fixed":
        angle = float(np.deg2rad(30.0))
        aspect = 1.0
    if renderer == "scalene_triangle":
        angle = 0.0
        aspect = 1.0
    return {"sigma": sigma, "angle_radians": angle, "aspect": aspect}


@lru_cache(maxsize=4)
def _pixel_grid(image_size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:image_size, :image_size].astype(np.float64)
    return yy, xx


def render_u8(
    point_xy_px: Sequence[float],
    appearance_id: int,
    renderer: str,
    appearance_seed: int,
    image_size: int = 224,
    protocol: Mapping[str, Any] | None = None,
) -> np.ndarray:
    protocol = protocol or load_protocol()
    point = np.asarray(point_xy_px, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point_xy_px must be one finite [x,y] row")
    params = appearance_parameters(appearance_id, renderer, appearance_seed, protocol)
    yy, xx = _pixel_grid(int(image_size))
    dx, dy = xx - point[0], yy - point[1]
    angle = params["angle_radians"]
    ca, sa = np.cos(angle), np.sin(angle)
    u, v = ca * dx + sa * dy, -sa * dx + ca * dy
    if renderer in {"blob", "blob_sigma6"}:
        sigma = params["sigma"]
        value = np.exp(
            -0.5
            * (
                (u / (sigma * params["aspect"])) ** 2
                + (v / sigma) ** 2
            )
        )
    elif renderer == "line_fixed":
        value = np.exp(
            -0.5
            * (
                (v / 1.5) ** 2
                + (np.maximum(np.abs(u) - 10.5, 0.0) / 1.5) ** 2
            )
        )
    elif renderer == "scalene_triangle":
        area = -282.0
        w1 = (15.0 * (u + 2.0) - 12.0 * (v + 10.0)) / area
        w2 = (-16.0 * (u + 2.0) - 6.0 * (v + 10.0)) / area
        w3 = 1.0 - w1 - w2
        value = (np.minimum(np.minimum(w1, w2), w3) >= 0.0).astype(np.float64)
    else:
        raise ValueError(f"unknown renderer {renderer}")
    return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)[None, :, :]


def preprocess_rgb(image_u8: np.ndarray) -> np.ndarray:
    image = np.asarray(image_u8)
    if image.dtype != np.uint8 or image.ndim not in {3, 4}:
        raise ValueError("preprocess expects uint8 [1,H,W] or [N,1,H,W]")
    channel_axis = 0 if image.ndim == 3 else 1
    if image.shape[channel_axis] != 1:
        raise ValueError("source image must have exactly one channel")
    scaled = image.astype(np.float32) / np.float32(255.0)
    return np.repeat(scaled, 3, axis=channel_axis)


def render_query_with_index(
    index: Mapping[str, Any],
    xy_px: np.ndarray,
    *,
    appearance_id: int = 0,
    image_size: int = 224,
    protocol: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Render coordinate-only query rows using the resolved cell's index renderer.

    This is intentionally index-driven: Q_DENSE41_CANONICAL never defaults to
    blob, so B5 line/triangle evaluations cannot silently use the wrong image.
    """
    protocol = protocol or load_protocol()
    renderer = index.get("renderer")
    if not isinstance(renderer, str):
        raise ValueError("coordinate-only shards cannot render image queries")
    matching = [row for row in index.get("rows", []) if row.get("appearance_id") == int(appearance_id)]
    if not matching:
        raise ValueError(f"appearance {appearance_id} is not declared by the resolved index")
    seeds = {int(row["appearance_seed"]) for row in matching}
    if len(seeds) != 1:
        raise ValueError("resolved index appearance seed is ambiguous")
    query = np.asarray(xy_px, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("xy_px must be [N,2]")
    return np.stack(
        [render_u8(point, appearance_id, renderer, next(iter(seeds)), image_size, protocol) for point in query]
    )


def coordinate_target(
    xy_px: np.ndarray,
    target_kind: str,
    coordinate_scale: float = 223.0,
) -> np.ndarray:
    xy = np.asarray(xy_px, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy_px must be finite [N,2]")
    return coordinate_target_f64(xy, target_kind, coordinate_scale).astype(np.float32)


def coordinate_target_f64(
    xy_px: np.ndarray,
    target_kind: str,
    coordinate_scale: float = 223.0,
) -> np.ndarray:
    xy = np.asarray(xy_px, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy_px must be finite [N,2]")
    u = xy / float(coordinate_scale)
    if target_kind == "identity":
        return u
    if target_kind == "quadratic_b8":
        x, y = u[:, 0], u[:, 1]
        return np.stack(
            (
                0.70 * x + 0.08 * y + 0.06 * x * x - 0.05 * x * y + 0.04 * y * y,
                -0.06 * x + 0.72 * y - 0.04 * x * x + 0.06 * x * y + 0.05 * y * y,
            ),
            axis=1,
        )
    raise ValueError(f"unknown target kind {target_kind}")


def canonical_b7_pair_graph(grid_xy_px: np.ndarray | None = None) -> np.ndarray:
    xy = positions("grid8x8") if grid_xy_px is None else np.asarray(grid_xy_px, dtype=np.float64)
    if xy.shape != (64, 2):
        raise ValueError("B7 pair graph requires canonical [64,2] grid")
    lookup = {tuple(point.tolist()): index for index, point in enumerate(xy)}
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    pairs: list[tuple[int, int]] = []
    for index, (x, y) in enumerate(xy):
        for neighbor in ((x + dx, y), (x, y + dy)):
            other = lookup.get(tuple(neighbor))
            if other is not None:
                pairs.append((index, other))
    result = np.asarray(pairs, dtype=np.int64)
    if result.shape != (112, 2):
        raise AssertionError(f"B7 graph shape drifted: {result.shape}")
    return result


def b7_endpoint_occurrences(pair_ids: np.ndarray, pair_graph: np.ndarray | None = None) -> np.ndarray:
    graph = canonical_b7_pair_graph() if pair_graph is None else np.asarray(pair_graph, dtype=np.int64)
    ids = np.asarray(pair_ids, dtype=np.int64)
    if ids.shape[-1] != 64 or ids.size == 0 or ids.min() < 0 or ids.max() >= len(graph):
        raise ValueError("pair_ids must end in 64 values within [0,111]")
    selected = graph[ids]
    return np.concatenate((selected[..., 0], selected[..., 1]), axis=-1)


def frozen_bn_calibration_row_ids() -> np.ndarray:
    """The one frozen 256-row B1 calibration batch in canonical row order."""
    return np.tile(np.arange(4, dtype=np.int64), 64)


def make_index(spec: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or load_protocol()
    safe_box = protocol["data"]["safe_box_px"]
    coords = positions(str(spec["layout"]), safe_box)
    train_seed = int(protocol["data"]["appearance"]["train_seed"])
    renderer = spec.get("renderer")
    rows: list[dict[str, Any]] = []
    if renderer is None:
        for position_index, point in enumerate(coords):
            rows.append(
                {
                    "row_id": position_index,
                    "position_index": position_index,
                    "position_px": point.tolist(),
                    "split": "train",
                    "target_kind": str(spec.get("target_kind", "identity")),
                }
            )
    else:
        for appearance_id in range(int(spec["appearance_count"])):
            params = appearance_parameters(appearance_id, str(renderer), train_seed, protocol)
            for position_index, point in enumerate(coords):
                rows.append(
                    {
                        "row_id": len(rows),
                        "position_index": position_index,
                        "position_px": point.tolist(),
                        "appearance_id": appearance_id,
                        "appearance_seed": train_seed,
                        "appearance_parameters": params,
                        "split": "train",
                        "target_kind": str(spec.get("target_kind", "identity")),
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": str(spec["id"]),
        "kind": "coordinate_shard" if renderer is None else "training_index",
        "role": str(spec["role"]),
        "layout": str(spec["layout"]),
        "renderer": renderer,
        "coordinate_order": "appearance_outer_then_x_outer_y_inner",
        "rows": rows,
        "row_count": len(rows),
    }
    if spec.get("include_b7_pair_graph"):
        graph = canonical_b7_pair_graph(coords)
        payload["b7_pair_graph"] = {
            "order": "point_row_order_then_plus_x_then_plus_y",
            "pair_count": 112,
            "pairs": graph.tolist(),
        }
    return payload


def make_query_arrays(spec: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    protocol = protocol or load_protocol()
    coords = positions(str(spec["layout"]), protocol["data"]["safe_box_px"])
    declared_appearances = spec.get("appearance_ids")
    if declared_appearances is None:
        xy = coords.copy()
        field_index = np.zeros(len(coords), dtype=np.int64)
        position_index = np.arange(len(coords), dtype=np.int64)
    else:
        appearances = np.asarray(declared_appearances, dtype=np.int64)
        xy = np.concatenate([coords for _ in appearances], axis=0)
        field_index = np.repeat(np.arange(len(appearances), dtype=np.int64), len(coords))
        position_index = np.tile(np.arange(len(coords), dtype=np.int64), len(appearances))
    result = {
        "xy_px": xy,
        "field_index": field_index,
        "position_index": position_index,
    }
    if spec.get("target_kind") is not None:
        result["truth_u"] = coordinate_target(xy, str(spec["target_kind"]), float(protocol["data"]["coordinate_scale"]))
    if declared_appearances is not None:
        result["appearance_id"] = np.repeat(appearances, len(coords))
        result["appearance_seed"] = np.full(len(xy), int(spec["appearance_seed"]), dtype=np.int64)
    return result


def resolve_query_target(
    query_arrays: Mapping[str, np.ndarray],
    target_kind: str,
    coordinate_scale: float = 223.0,
) -> dict[str, np.ndarray]:
    """Bind a shared coordinate query to the resolved cell target."""
    if "xy_px" not in query_arrays:
        raise ValueError("query arrays contain no xy_px")
    resolved = {key: np.asarray(value) for key, value in query_arrays.items()}
    if "truth_u" in resolved:
        raise ValueError("shared coordinate query must not carry a default truth field")
    resolved["truth_u"] = coordinate_target(resolved["xy_px"], target_kind, coordinate_scale)
    return resolved


def common_stream(population: int, steps: int, seed: int, width: int = 64) -> np.ndarray:
    if population <= 0 or steps <= 0 or width != 64:
        raise ValueError("fixed common stream requires positive population/steps and width 64")
    rng = np.random.default_rng(int(seed) + 17)
    return rng.integers(0, int(population), size=(int(steps), 64), dtype=np.int64)


def b7_double_draw_stream(steps: int, seed: int, width: int = 64) -> dict[str, np.ndarray]:
    if steps <= 0 or width != 64:
        raise ValueError("B7 stream requires positive steps and width 64")
    rng = np.random.default_rng(int(seed) + 17)
    image_ids = np.empty((int(steps), 64), dtype=np.int64)
    pair_ids = np.empty((int(steps), 64), dtype=np.int64)
    for step in range(int(steps)):
        image_ids[step] = rng.integers(0, 64, size=64, dtype=np.int64)
        pair_ids[step] = rng.integers(0, 112, size=64, dtype=np.int64)
    return {"image_ids": image_ids, "pair_ids": pair_ids}


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _write_json_if_missing_or_equal(path: Path, value: Any) -> None:
    text = _json_text(value)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != value:
            raise ValueError(f"existing prepared input differs semantically: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_npy_if_missing_or_equal(path: Path, expected: np.ndarray) -> None:
    if path.exists():
        observed = np.load(path, allow_pickle=False)
        if not np.array_equal(observed, expected):
            raise ValueError(f"existing stream differs semantically: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, expected, allow_pickle=False)


def _write_npz_if_missing_or_equal(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as observed:
            if set(observed.files) != set(expected) or any(
                not np.array_equal(observed[key], expected[key]) for key in expected
            ):
                raise ValueError(f"existing array bundle differs semantically: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **expected)


def expected_streams(protocol: Mapping[str, Any] | None = None) -> list[tuple[str, str, dict[str, Any]]]:
    protocol = protocol or load_protocol()
    seeds = [int(value) for value in protocol["discovery_seeds"]]
    result: list[tuple[str, str, dict[str, Any]]] = []
    for group in protocol["prepared_inputs"]["stream_groups"]:
        for seed in seeds:
            kind = str(group["kind"])
            if kind == "common":
                artifact_id = f"S_COMMON_N{group['population']}_T{group['steps']}_s{seed}"
                result.append((artifact_id, "npy", {"group": group, "seed": seed}))
            elif kind == "b4_causal":
                artifact_id = f"S_B4_CAUSAL_N4_T{group['steps']}_s{seed}"
                result.append((artifact_id, "npy", {"group": group, "seed": seed}))
            elif kind == "b7_double_draw":
                artifact_id = f"S_B7_DOUBLE_DRAW_T{group['steps']}_s{seed}"
                result.append((artifact_id, "npz", {"group": group, "seed": seed}))
            else:
                raise ValueError(f"unknown stream group {kind}")
    return result


def _stream_value(metadata: Mapping[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
    group, seed = metadata["group"], int(metadata["seed"])
    if group["kind"] in {"common", "b4_causal"}:
        return common_stream(int(group["population"]), int(group["steps"]), seed, int(group["width"]))
    return b7_double_draw_stream(int(group["steps"]), seed, int(group["width"]))


def _catalog(protocol: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for spec in protocol["prepared_inputs"]["index_specs"]:
        payload = make_index(spec, protocol)
        entries.append(
            {
                "artifact_id": spec["id"],
                "kind": "index_or_shard",
                "relative_path": f"indices/{spec['id']}.json",
                "rows": payload["row_count"],
                "role": spec["role"],
            }
        )
    for spec in protocol["prepared_inputs"]["query_specs"]:
        arrays = make_query_arrays(spec, protocol)
        entries.append(
            {
                "artifact_id": spec["id"],
                "kind": "query",
                "relative_path": f"queries/{spec['id']}.npz",
                "rows": int(len(arrays["xy_px"])),
                "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
                "role": spec["role"],
            }
        )
    for artifact_id, extension, metadata in expected_streams(protocol):
        group = metadata["group"]
        if group["kind"] == "b7_double_draw":
            fields = {
                "image_ids": {"shape": [int(group["steps"]), 64], "dtype": "int64"},
                "pair_ids": {"shape": [int(group["steps"]), 64], "dtype": "int64"},
            }
        else:
            fields = {"ids": {"shape": [int(group["steps"]), 64], "dtype": "int64"}}
        entries.append(
            {
                "artifact_id": artifact_id,
                "kind": "fixed_stream",
                "relative_path": f"streams/{artifact_id}.{extension}",
                "fields": fields,
                "role": metadata["group"]["kind"],
            }
        )
    return {
        "schema_version": 1,
        "release_id": protocol["release_id"],
        "counts": protocol["prepared_inputs"]["expected_counts"],
        "entries": entries,
        "integrity_policy": "semantic_validation_only_no_internal_digests",
    }


def prepare_inputs(asset_root: str | Path = PACKAGE_ROOT / "assets") -> dict[str, Any]:
    root = Path(asset_root)
    protocol = load_protocol()
    for spec in protocol["prepared_inputs"]["index_specs"]:
        _write_json_if_missing_or_equal(root / "indices" / f"{spec['id']}.json", make_index(spec, protocol))
    for spec in protocol["prepared_inputs"]["query_specs"]:
        _write_npz_if_missing_or_equal(root / "queries" / f"{spec['id']}.npz", make_query_arrays(spec, protocol))
    for artifact_id, extension, metadata in expected_streams(protocol):
        value = _stream_value(metadata)
        path = root / "streams" / f"{artifact_id}.{extension}"
        if isinstance(value, np.ndarray):
            _write_npy_if_missing_or_equal(path, value)
        else:
            _write_npz_if_missing_or_equal(path, value)
    catalog = _catalog(protocol)
    _write_json_if_missing_or_equal(root / "INPUT_CATALOG.json", catalog)
    return validate_prepared_inputs(root)


def validate_prepared_inputs(asset_root: str | Path = PACKAGE_ROOT / "assets") -> dict[str, Any]:
    root = Path(asset_root)
    protocol = load_protocol()
    expected_catalog = _catalog(protocol)
    catalog_path = root / "INPUT_CATALOG.json"
    if not catalog_path.is_file() or json.loads(catalog_path.read_text(encoding="utf-8")) != expected_catalog:
        raise ValueError("prepared input catalog is missing or semantically wrong")
    for spec in protocol["prepared_inputs"]["index_specs"]:
        path = root / "indices" / f"{spec['id']}.json"
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != make_index(spec, protocol):
            raise ValueError(f"index semantic validation failed: {spec['id']}")
    for spec in protocol["prepared_inputs"]["query_specs"]:
        path = root / "queries" / f"{spec['id']}.npz"
        expected = make_query_arrays(spec, protocol)
        if not path.is_file():
            raise ValueError(f"query missing: {spec['id']}")
        with np.load(path, allow_pickle=False) as observed:
            if set(observed.files) != set(expected) or any(not np.array_equal(observed[key], expected[key]) for key in expected):
                raise ValueError(f"query semantic validation failed: {spec['id']}")
    for artifact_id, extension, metadata in expected_streams(protocol):
        path = root / "streams" / f"{artifact_id}.{extension}"
        expected = _stream_value(metadata)
        if not path.is_file():
            raise ValueError(f"stream missing: {artifact_id}")
        if isinstance(expected, np.ndarray):
            if not np.array_equal(np.load(path, allow_pickle=False), expected):
                raise ValueError(f"stream semantic validation failed: {artifact_id}")
        else:
            with np.load(path, allow_pickle=False) as observed:
                if set(observed.files) != set(expected) or any(not np.array_equal(observed[key], expected[key]) for key in expected):
                    raise ValueError(f"stream semantic validation failed: {artifact_id}")
    counts = {
        "indices_or_shards": len(protocol["prepared_inputs"]["index_specs"]),
        "queries": len(protocol["prepared_inputs"]["query_specs"]),
        "fixed_streams": len(expected_streams(protocol)),
    }
    if counts != {key: int(value) for key, value in protocol["prepared_inputs"]["expected_counts"].items()}:
        raise ValueError(f"prepared input count mismatch: {counts}")
    return {"status": "PASS", "counts": counts, "asset_root": str(root.resolve())}
