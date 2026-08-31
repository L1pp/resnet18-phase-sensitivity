"""Six primitive families. Center is the generative (x, y), never ink centroid."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from phase1_gap_rep.common import CURVE_SAMPLES, SUPER_SAMPLE
from phase1_gap_rep.generate_data import make_relative_shape

from . import RENDERER_VERSION

FAMILIES: Tuple[str, ...] = ("line", "circle", "arc", "quadratic", "polygon", "blob")


def _resampling():
    return getattr(Image, "Resampling", Image).LANCZOS


def normalize_xy(x_px: float, y_px: float, image_size: int) -> Tuple[float, float]:
    scale = float(image_size - 1)
    return float(x_px) / scale, float(y_px) / scale


def denormalize_xy(u: float, v: float, image_size: int) -> Tuple[float, float]:
    scale = float(image_size - 1)
    return float(u) * scale, float(v) * scale


def _blank(high: int) -> Image.Image:
    return Image.new("L", (high, high), color=0)


def _down(image: Image.Image, image_size: int) -> np.ndarray:
    out = np.asarray(image.resize((image_size, image_size), resample=_resampling()), dtype=np.uint8)
    if out.shape != (image_size, image_size):
        raise RuntimeError(f"renderer size {out.shape} != {(image_size, image_size)}")
    return out


def _stroke_polyline(points_px: np.ndarray, image_size: int, width_px: float) -> np.ndarray:
    high = image_size * SUPER_SAMPLE
    image = _blank(high)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * SUPER_SAMPLE, float(y) * SUPER_SAMPLE) for x, y in np.asarray(points_px, dtype=np.float64)]
    width = max(1, int(round(width_px * SUPER_SAMPLE)))
    try:
        draw.line(xy, fill=255, width=width, joint="curve")
    except TypeError:
        draw.line(xy, fill=255, width=width)
    return _down(image, image_size)


def render_line(cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    half = float(params.get("length", 28.0)) * 0.5
    theta = np.deg2rad(float(params.get("theta_deg", 0.0)))
    c, s = np.cos(theta), np.sin(theta)
    p0 = (cx - half * c, cy - half * s)
    p1 = (cx + half * c, cy + half * s)
    return _stroke_polyline(np.array([p0, p1]), image_size, float(params.get("width", 3.0)))


def render_circle(cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    radius = float(params.get("radius", 16.0))
    high = image_size * SUPER_SAMPLE
    image = _blank(high)
    draw = ImageDraw.Draw(image)
    box = [
        (cx - radius) * SUPER_SAMPLE,
        (cy - radius) * SUPER_SAMPLE,
        (cx + radius) * SUPER_SAMPLE,
        (cy + radius) * SUPER_SAMPLE,
    ]
    width = max(1, int(round(float(params.get("width", 3.0)) * SUPER_SAMPLE)))
    draw.ellipse(box, outline=255, width=width)
    return _down(image, image_size)


def render_arc(cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    radius = float(params.get("radius", 18.0))
    start = float(params.get("start_deg", 20.0))
    end = float(params.get("end_deg", 200.0))
    high = image_size * SUPER_SAMPLE
    image = _blank(high)
    draw = ImageDraw.Draw(image)
    box = [
        (cx - radius) * SUPER_SAMPLE,
        (cy - radius) * SUPER_SAMPLE,
        (cx + radius) * SUPER_SAMPLE,
        (cy + radius) * SUPER_SAMPLE,
    ]
    width = max(1, int(round(float(params.get("width", 3.0)) * SUPER_SAMPLE)))
    draw.arc(box, start=start, end=end, fill=255, width=width)
    return _down(image, image_size)


def render_quadratic(cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    q = make_relative_shape(
        float(params.get("theta_deg", 0.0)),
        float(params.get("chord_px", 48.0)),
        float(params.get("curve_sign", 1.0)),
        float(params.get("alpha", 0.0)),
    )
    scale = float(params.get("size_scale", 1.0))
    pts = q * scale + np.array([[cx, cy]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, CURVE_SAMPLES)[:, None]
    curve = (1.0 - t) ** 2 * pts[0] + 2.0 * (1.0 - t) * t * pts[1] + t**2 * pts[2]
    return _stroke_polyline(curve, image_size, float(params.get("width", 3.0)))


def render_polygon(cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    n = int(params.get("n_sides", 3))
    n = max(3, min(n, 6))
    radius = float(params.get("radius", 18.0))
    theta0 = np.deg2rad(float(params.get("theta_deg", 0.0)))
    angles = theta0 + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    verts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    high = image_size * SUPER_SAMPLE
    image = _blank(high)
    draw = ImageDraw.Draw(image)
    xy = [(float(x) * SUPER_SAMPLE, float(y) * SUPER_SAMPLE) for x, y in verts]
    draw.polygon(xy, fill=255)
    return _down(image, image_size)


def render_blob(cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    sigma = float(params.get("sigma", 6.0))
    high = image_size * SUPER_SAMPLE
    yy, xx = np.mgrid[0:high, 0:high].astype(np.float64)
    dx = xx - cx * SUPER_SAMPLE
    dy = yy - cy * SUPER_SAMPLE
    val = np.exp(-(dx * dx + dy * dy) / (2.0 * (sigma * SUPER_SAMPLE) ** 2))
    image = Image.fromarray(np.clip(val * 255.0, 0, 255).astype(np.uint8), mode="L")
    return _down(image, image_size)


RENDERERS = {
    "line": render_line,
    "circle": render_circle,
    "arc": render_arc,
    "quadratic": render_quadratic,
    "polygon": render_polygon,
    "blob": render_blob,
}


def render_family(family: str, cx: float, cy: float, params: Mapping[str, float], image_size: int) -> np.ndarray:
    if family not in RENDERERS:
        raise ValueError(family)
    return RENDERERS[family](cx, cy, params, image_size)


def random_params(family: str, rng: np.random.Generator, image_size: int, size_mode: str = "absolute") -> Dict[str, float]:
    """Internal geometry independent of center. size_mode: absolute | relative."""
    scale = (image_size / 224.0) if size_mode == "relative" else 1.0
    if family == "line":
        return {
            "length": float(rng.uniform(18.0, 40.0) * scale),
            "theta_deg": float(rng.uniform(0.0, 180.0)),
            "width": 3.0 * scale,
        }
    if family == "circle":
        return {"radius": float(rng.uniform(10.0, 22.0) * scale), "width": 3.0 * scale}
    if family == "arc":
        start = float(rng.uniform(0.0, 360.0))
        return {
            "radius": float(rng.uniform(12.0, 24.0) * scale),
            "start_deg": start,
            "end_deg": start + float(rng.uniform(90.0, 250.0)),
            "width": 3.0 * scale,
        }
    if family == "quadratic":
        return {
            "theta_deg": float(rng.choice([0.0, 45.0, 90.0, 135.0])),
            "chord_px": float(rng.uniform(36.0, 64.0)),
            "curve_sign": float(rng.choice([-1.0, 1.0])),
            "alpha": float(rng.uniform(-0.35, 0.35)),
            "size_scale": float(scale),
            "width": 3.0 * scale,
        }
    if family == "polygon":
        return {
            "n_sides": int(rng.integers(3, 7)),
            "radius": float(rng.uniform(12.0, 22.0) * scale),
            "theta_deg": float(rng.uniform(0.0, 180.0)),
        }
    if family == "blob":
        return {"sigma": float(rng.uniform(4.0, 9.0) * scale)}
    raise ValueError(family)


def extent_px(family: str, params: Mapping[str, float]) -> float:
    if family == "line":
        return 0.5 * float(params["length"]) + float(params.get("width", 3.0))
    if family in {"circle", "arc"}:
        return float(params["radius"]) + float(params.get("width", 3.0))
    if family == "quadratic":
        return 0.55 * float(params["chord_px"]) * float(params.get("size_scale", 1.0)) + 8.0
    if family == "polygon":
        return float(params["radius"]) + 2.0
    if family == "blob":
        return 3.0 * float(params["sigma"])
    raise ValueError(family)


def sample_center(
    rng: np.random.Generator,
    image_size: int,
    extent: float,
    margin: float = 8.0,
) -> Tuple[float, float]:
    lo = margin + extent
    hi = image_size - 1.0 - margin - extent
    if hi <= lo:
        return (image_size - 1.0) * 0.5, (image_size - 1.0) * 0.5
    return float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))


def make_sample(
    family: str,
    rng: np.random.Generator,
    image_size: int = 224,
    size_mode: str = "absolute",
    center: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    params = random_params(family, rng, image_size, size_mode=size_mode)
    ext = extent_px(family, params)
    if center is None:
        cx, cy = sample_center(rng, image_size, ext)
    else:
        cx, cy = float(center[0]), float(center[1])
    image = render_family(family, cx, cy, params, image_size)
    u, v = normalize_xy(cx, cy, image_size)
    return {
        "family": family,
        "image": image,
        "cx_px": cx,
        "cy_px": cy,
        "xy_norm": np.array([u, v], dtype=np.float64),
        "params": params,
        "image_size": image_size,
        "size_mode": size_mode,
        "renderer_version": RENDERER_VERSION,
    }


def make_batch(
    families: Sequence[str],
    n: int,
    seed: int,
    image_size: int = 224,
    size_mode: str = "absolute",
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    images = np.zeros((n, image_size, image_size), dtype=np.uint8)
    xy = np.zeros((n, 2), dtype=np.float64)
    fam = []
    for i in range(n):
        family = str(rng.choice(list(families)))
        sample = make_sample(family, rng, image_size=image_size, size_mode=size_mode)
        images[i] = sample["image"]
        xy[i] = sample["xy_norm"]
        fam.append(family)
    return {"images": images, "xy_norm": xy, "family": np.asarray(fam), "image_size": image_size}
