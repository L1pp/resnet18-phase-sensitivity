"""Generate the frozen Shape × Translation factorial dataset and preview figures."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .common import (
    ALPHAS,
    ANGLES_DEG,
    CHORDS_PX,
    COORD_SCALE,
    CURVE_FRAC,
    CURVE_SIGNS,
    IMAGE_SIZE,
    MARGIN_PX,
    MAX_Q_HALF_EXTENT_PX,
    MIN_CHORD_PX,
    MIN_CURVE_PX,
    Profile,
    SEED,
    dump_json,
    dirs,
    ensure_dirs,
    fingerprint,
    geometry_config,
    profile_spec,
    px_to_norm,
    quadratic_points,
    render_quadratic,
    seed_everything,
    setup_matplotlib_chinese,
    write_metadata,
)


def _angles(spec: Profile) -> Tuple[float, ...]:
    return ANGLES_DEG[: spec.n_angle]


def _chords(spec: Profile) -> Tuple[float, ...]:
    return CHORDS_PX[: spec.n_scale]


def _signs(spec: Profile) -> Tuple[float, ...]:
    return CURVE_SIGNS[: spec.n_curve]


def _alphas(spec: Profile) -> Tuple[float, ...]:
    return ALPHAS[: spec.n_alpha]


def make_relative_shape(theta_deg: float, chord_px: float, curve_sign: float, alpha: float) -> np.ndarray:
    length = float(chord_px)
    height = max(MIN_CURVE_PX, CURVE_FRAC * length) * float(curve_sign)
    q0 = np.array([-0.5 * length, 0.0], dtype=np.float64)
    q2 = np.array([0.5 * length, 0.0], dtype=np.float64)
    q1 = np.array([float(alpha) * 0.5 * length, height], dtype=np.float64)
    q = np.stack([q0, q1, q2], axis=0)
    q = q - q.mean(axis=0, keepdims=True)
    rad = np.deg2rad(float(theta_deg))
    cosine, sine = np.cos(rad), np.sin(rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    q = q @ rotation.T
    if (float(q[0, 0]), float(q[0, 1])) > (float(q[2, 0]), float(q[2, 1])):
        q[[0, 2]] = q[[2, 0]]
    q = q - q.mean(axis=0, keepdims=True)
    return q


def build_shapes(spec: Profile) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    shapes = np.zeros((spec.n_shapes, 3, 2), dtype=np.float64)
    index = 0
    for theta in _angles(spec):
        for chord in _chords(spec):
            for sign in _signs(spec):
                for alpha in _alphas(spec):
                    q = make_relative_shape(theta, chord, sign, alpha)
                    chord_vec = q[2] - q[0]
                    chord_len = float(np.linalg.norm(chord_vec))
                    if chord_len < MIN_CHORD_PX:
                        raise RuntimeError(f"degenerate chord {chord_len:.3f}px at shape {index}")
                    offset = q[1] - q[0]
                    curve = abs(chord_vec[0] * offset[1] - chord_vec[1] * offset[0]) / chord_len
                    if curve < MIN_CURVE_PX - 1e-6:
                        raise RuntimeError(f"near-linear curve {curve:.3f}px at shape {index}")
                    half = float(np.max(np.abs(q)))
                    if half > MAX_Q_HALF_EXTENT_PX + 1e-6:
                        raise RuntimeError(f"shape {index} half-extent {half:.3f}px exceeds {MAX_Q_HALF_EXTENT_PX}")
                    if float(np.max(np.abs(q.mean(axis=0)))) > 1e-8:
                        raise RuntimeError(f"shape {index} centroid not at origin")
                    shapes[index] = q
                    records.append(
                        {
                            "shape_id": index,
                            "theta_deg": theta,
                            "chord_px": chord,
                            "curve_sign": sign,
                            "alpha": alpha,
                            "realized_chord_px": chord_len,
                            "realized_curve_px": curve,
                            "half_extent_px": half,
                        }
                    )
                    index += 1
    if index != spec.n_shapes:
        raise RuntimeError(f"expected {spec.n_shapes} shapes, built {index}")
    return shapes, records


def translation_grid(shapes_px: np.ndarray, spec: Profile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_min = shapes_px.reshape(-1, 2).min(axis=0)
    q_max = shapes_px.reshape(-1, 2).max(axis=0)
    low = MARGIN_PX - q_min
    high = (COORD_SCALE - MARGIN_PX) - q_max
    if float(high[0] - low[0]) < spec.n_tx - 1 or float(high[1] - low[1]) < spec.n_ty - 1:
        raise RuntimeError(f"translation range too small: low={low}, high={high}")
    xs = np.linspace(float(low[0]), float(high[0]), spec.n_tx)
    ys = np.linspace(float(low[1]), float(high[1]), spec.n_ty)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    if grid.shape[0] != spec.n_translations:
        raise RuntimeError("translation grid size mismatch")
    return grid, low, high, q_min, q_max


def assign_splits(spec: Profile, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.n_shapes != spec.n_translations:
        raise RuntimeError("balanced recombination split requires n_shapes == n_translations")
    n = spec.n_shapes
    row_perm = rng.permutation(n)
    col_perm = rng.permutation(n)
    hold = n // 8
    if hold < 1:
        raise RuntimeError("profile too small for val/test hold-out")
    table = np.empty((n, n), dtype=object)
    for shape_id in range(n):
        for trans_id in range(n):
            symbol = (int(row_perm[shape_id]) + int(col_perm[trans_id])) % n
            if symbol < hold:
                table[shape_id, trans_id] = "val"
            elif symbol < 2 * hold:
                table[shape_id, trans_id] = "test"
            else:
                table[shape_id, trans_id] = "train"
    return table, row_perm, col_perm


def validate_split(table: np.ndarray) -> Dict[str, Any]:
    n = table.shape[0]
    counts = {name: int(np.sum(table == name)) for name in ("train", "val", "test")}
    per_shape = {
        name: [int(np.sum(table[i] == name)) for i in range(n)] for name in ("train", "val", "test")
    }
    per_trans = {
        name: [int(np.sum(table[:, j] == name)) for j in range(n)] for name in ("train", "val", "test")
    }
    hold = n // 8
    expected = {"train": n * (n - 2 * hold), "val": n * hold, "test": n * hold}
    if counts != expected:
        raise RuntimeError(f"split counts {counts} != {expected}")
    for name, want in (("train", n - 2 * hold), ("val", hold), ("test", hold)):
        if any(value != want for value in per_shape[name]) or any(value != want for value in per_trans[name]):
            raise RuntimeError(f"unbalanced {name} margins")
    if np.any(np.array(per_shape["train"]) == 0) or np.any(np.array(per_trans["train"]) == 0):
        raise RuntimeError("a shape or translation is missing from train")
    return {"counts": counts, "per_shape": per_shape, "per_trans": per_trans, "expected": expected}


def _assert_in_canvas(p_px: np.ndarray, shape_id: int, trans_id: int) -> None:
    if np.any(p_px < MARGIN_PX - 1e-6) or np.any(p_px > COORD_SCALE - MARGIN_PX + 1e-6):
        raise RuntimeError(f"control-point clip at shape={shape_id} translation={trans_id}: {p_px}")
    curve = quadratic_points(p_px / COORD_SCALE) * COORD_SCALE
    if np.any(curve < MARGIN_PX - 1e-6) or np.any(curve > COORD_SCALE - MARGIN_PX + 1e-6):
        raise RuntimeError(f"curve clip at shape={shape_id} translation={trans_id}")


def _write_csv(path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _overlay_rgb(image: np.ndarray, p_norm: np.ndarray) -> np.ndarray:
    rgb = np.repeat(image.astype(np.float32)[..., None] / 255.0, 3, axis=2)
    colors = ((1.0, 0.25, 0.2), (0.2, 0.9, 0.3), (0.25, 0.45, 1.0))
    pts = p_norm.reshape(3, 2)
    for (x, y), color in zip(pts, colors):
        cx = int(round(float(x) * COORD_SCALE))
        cy = int(round(float(y) * COORD_SCALE))
        rad = 3
        y0, y1 = max(0, cy - rad), min(IMAGE_SIZE, cy + rad + 1)
        x0, x1 = max(0, cx - rad), min(IMAGE_SIZE, cx + rad + 1)
        rgb[y0:y1, x0:x1] = color
    return np.clip(rgb, 0.0, 1.0)


def _imshow(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def write_preview_figures(
    spec: Profile,
    images: np.ndarray,
    p_norm: np.ndarray,
    shapes_px: np.ndarray,
    translations_px: np.ndarray,
    split_table: np.ndarray,
    t_low: np.ndarray,
    t_high: np.ndarray,
) -> List[str]:
    plt = setup_matplotlib_chinese()
    figure_dir = dirs(spec)["figures"]
    written: List[str] = []
    n_s, n_t = spec.n_shapes, spec.n_translations
    center_t = (spec.n_tx // 2) * spec.n_ty + (spec.n_ty // 2)
    center_t = min(center_t, n_t - 1)

    cols = int(np.ceil(np.sqrt(n_s)))
    rows = int(np.ceil(n_s / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n_s:
            rgb = _overlay_rgb(images[i, center_t], p_norm[i, center_t])
            ax.imshow(rgb, origin="upper")
            ax.set_title(f"s{i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n_s:
            ax.axis("off")
    fig.suptitle(f"全部 {n_s} 个 shape（固定 translation_id={center_t}；红P0 绿P1 蓝P2）")
    fig.tight_layout()
    path = figure_dir / "preview_all_shapes.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    pick = np.unique(np.linspace(0, n_s - 1, num=min(4, n_s), dtype=int))
    sub_n = min(4, spec.n_tx, spec.n_ty)
    tx_idx = np.linspace(0, spec.n_tx - 1, sub_n, dtype=int)
    ty_idx = np.linspace(0, spec.n_ty - 1, sub_n, dtype=int)
    fig, axes = plt.subplots(len(pick), sub_n * sub_n, figsize=(sub_n * sub_n * 1.35, len(pick) * 1.55))
    axes = np.atleast_2d(axes)
    for r, sid in enumerate(pick):
        k = 0
        for ix in tx_idx:
            for iy in ty_idx:
                tid = int(ix * spec.n_ty + iy)
                ax = axes[r, k]
                _imshow(ax, images[sid, tid], f"s{sid} t{tid}")
                k += 1
        axes[r, 0].set_ylabel(f"shape {sid}", fontsize=8)
    fig.suptitle("translation 扫描：同一 shape 只应平移，边缘不应裁切")
    fig.tight_layout()
    path = figure_dir / "preview_translation_scan.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    corner_t = 0
    show = min(16, n_s)
    cols = 4
    rows = int(np.ceil(show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < show:
            _imshow(ax, images[i, corner_t], f"s{i} t{corner_t}")
        else:
            ax.axis("off")
    fig.suptitle(f"同一 translation_id={corner_t} 下的不同 shape")
    fig.tight_layout()
    path = figure_dir / "preview_same_translation_shapes.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(3, 8, figsize=(14.5, 5.6))
    rng = np.random.default_rng(SEED + 7)
    for r, name in enumerate(("train", "val", "test")):
        pairs = [(s, t) for s in range(n_s) for t in range(n_t) if split_table[s, t] == name]
        chosen = [pairs[i] for i in rng.choice(len(pairs), size=min(8, len(pairs)), replace=False)]
        for c in range(8):
            ax = axes[r, c]
            if c < len(chosen):
                s, t = chosen[c]
                _imshow(ax, images[s, t], f"{name} ({s},{t})")
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("split 抽查：test 应是未见过的 (shape, translation) 组合，不是新 shape")
    fig.tight_layout()
    path = figure_dir / "preview_split_samples.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    labels = ("Q0", "Q1", "Q2")
    ax = axes[0]
    for k, color, label in zip(range(3), colors, labels):
        ax.scatter(shapes_px[:, k, 0], shapes_px[:, k, 1], s=18, c=color, label=label, alpha=0.85)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("相对控制点 Q（像素）")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(loc="best", fontsize=8)
    ax = axes[1]
    ax.scatter(translations_px[:, 0], translations_px[:, 1], s=22, c="#ff7f0e")
    ax.set_xlim(0, COORD_SCALE)
    ax.set_ylim(COORD_SCALE, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("translation 网格与可行矩形")
    ax.set_xlabel("tx (px)")
    ax.set_ylabel("ty (px)")
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (float(t_low[0]), float(t_low[1])),
            float(t_high[0] - t_low[0]),
            float(t_high[1] - t_low[1]),
            fill=False,
            edgecolor="0.2",
            lw=1.2,
        )
    )
    fig.tight_layout()
    path = figure_dir / "preview_q_and_translation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path.name)

    color_map = {"train": (0.20, 0.45, 0.85), "val": (0.20, 0.75, 0.35), "test": (0.85, 0.25, 0.20)}
    rgb = np.zeros((n_s, n_t, 3), dtype=np.float32)
    for name, color in color_map.items():
        rgb[split_table == name] = color
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(rgb, origin="upper")
    ax.set_xlabel("translation_id")
    ax.set_ylabel("shape_id")
    ax.set_title("split 矩阵（蓝 train / 绿 val / 红 test）")
    path = figure_dir / "preview_split_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path.name)
    return written


def prepare(profile: str = "factorial", force: bool = False) -> Dict[str, Any]:
    spec = profile_spec(profile)
    seed_everything(SEED)
    d = ensure_dirs(spec)
    marker = d["data"] / "train.npz"
    if marker.exists() and not force:
        raise RuntimeError(f"data already exist in {d['data']}; pass --force to overwrite")

    rng = np.random.default_rng(SEED)
    shapes_px, shape_records = build_shapes(spec)
    translations_px, t_low, t_high, q_min, q_max = translation_grid(shapes_px, spec)
    split_table, row_perm, col_perm = assign_splits(spec, rng)
    split_info = validate_split(split_table)

    images = np.zeros((spec.n_shapes, spec.n_translations, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    p_norm = np.zeros((spec.n_shapes, spec.n_translations, 6), dtype=np.float32)
    q_norm = np.zeros((spec.n_shapes, spec.n_translations, 6), dtype=np.float32)
    t_norm = np.zeros((spec.n_shapes, spec.n_translations, 2), dtype=np.float32)
    total = spec.n_total
    done = 0
    for s in range(spec.n_shapes):
        q = shapes_px[s]
        qn = px_to_norm(q).reshape(6).astype(np.float32)
        for t_id in range(spec.n_translations):
            t = translations_px[t_id]
            p = q + t[None, :]
            _assert_in_canvas(p, s, t_id)
            pn = px_to_norm(p).reshape(6).astype(np.float32)
            images[s, t_id] = render_quadratic(pn)
            if int(images[s, t_id].max()) < 32:
                raise RuntimeError(f"blank render at shape={s} translation={t_id}")
            p_norm[s, t_id] = pn
            q_norm[s, t_id] = qn
            t_norm[s, t_id] = px_to_norm(t).astype(np.float32)
            done += 1
            if done % 256 == 0 or done == total:
                print(f"[prepare:{spec.name}] rendered {done}/{total}")

    if not np.all(np.isfinite(p_norm)):
        raise RuntimeError("NaN/Inf in coordinates")

    fp = fingerprint(spec)
    np.savez_compressed(
        d["data"] / "shapes.npz",
        Q_px=shapes_px.astype(np.float32),
        Q_norm=px_to_norm(shapes_px).astype(np.float32),
        fingerprint=np.asarray(fp),
    )
    np.savez_compressed(
        d["data"] / "translations.npz",
        t_px=translations_px.astype(np.float32),
        t_norm=px_to_norm(translations_px).astype(np.float32),
        t_low_px=t_low.astype(np.float32),
        t_high_px=t_high.astype(np.float32),
        fingerprint=np.asarray(fp),
    )
    split_codes = np.zeros(split_table.shape, dtype=np.int8)
    split_codes[split_table == "train"] = 0
    split_codes[split_table == "val"] = 1
    split_codes[split_table == "test"] = 2
    np.savez_compressed(
        d["data"] / "split.npz",
        split_codes=split_codes,
        row_perm=row_perm.astype(np.int32),
        col_perm=col_perm.astype(np.int32),
        fingerprint=np.asarray(fp),
    )

    split_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
    for s in range(spec.n_shapes):
        for t_id in range(spec.n_translations):
            name = str(split_table[s, t_id])
            p = p_norm[s, t_id]
            q = q_norm[s, t_id]
            t = t_norm[s, t_id]
            split_rows[name].append(
                {
                    "shape_id": int(s),
                    "translation_id": int(t_id),
                    "split": name,
                    "p0x": float(p[0]),
                    "p0y": float(p[1]),
                    "p1x": float(p[2]),
                    "p1y": float(p[3]),
                    "p2x": float(p[4]),
                    "p2y": float(p[5]),
                    "q0x": float(q[0]),
                    "q0y": float(q[1]),
                    "q1x": float(q[2]),
                    "q1y": float(q[3]),
                    "q2x": float(q[4]),
                    "q2y": float(q[5]),
                    "tx": float(t[0]),
                    "ty": float(t[1]),
                }
            )

    for name in ("train", "val", "test"):
        rows = split_rows[name]
        sid = np.array([row["shape_id"] for row in rows], dtype=np.int32)
        tid = np.array([row["translation_id"] for row in rows], dtype=np.int32)
        np.savez_compressed(
            d["data"] / f"{name}.npz",
            images=images[sid, tid],
            P=p_norm[sid, tid],
            Q=q_norm[sid, tid],
            t=t_norm[sid, tid],
            shape_id=sid,
            translation_id=tid,
            fingerprint=np.asarray(fp),
        )
        dump_json(d["manifest"] / f"{name}.json", {"split": name, "count": len(rows), "fingerprint": fp})
        _write_csv(d["tables"] / f"{name}_index.csv", rows)

    dump_json(
        d["manifest"] / "manifest.json",
        {
            "fingerprint": fp,
            "counts": split_info["counts"],
            "n_shapes": spec.n_shapes,
            "n_translations": spec.n_translations,
        },
    )
    _write_csv(d["tables"] / "shapes.csv", shape_records)
    dump_json(d["config"] / "config.json", {**geometry_config(spec), "fingerprint": fp})

    checks = {
        "n_shapes": spec.n_shapes,
        "n_translations": spec.n_translations,
        "n_total": spec.n_total,
        "split": split_info["counts"],
        "q_min_px": q_min.tolist(),
        "q_max_px": q_max.tolist(),
        "t_low_px": t_low.tolist(),
        "t_high_px": t_high.tolist(),
        "min_P_px": float(p_norm.min() * COORD_SCALE),
        "max_P_px": float(p_norm.max() * COORD_SCALE),
        "margin_px": MARGIN_PX,
        "all_in_canvas": True,
        "centroid_ok": True,
        "no_nan": True,
        "fingerprint": fp,
    }
    dump_json(d["tables"] / "data_checks.json", checks)
    previews = write_preview_figures(
        spec, images, p_norm, shapes_px, translations_px, split_table, t_low, t_high
    )
    write_metadata(spec, "prepare", {"checks": checks, "previews": previews})
    print(f"[prepare:{spec.name}] split={split_info['counts']}")
    print(f"[prepare:{spec.name}] P range px=[{checks['min_P_px']:.2f}, {checks['max_P_px']:.2f}] margin={MARGIN_PX}")
    print(f"[prepare:{spec.name}] figures: {', '.join(previews)}")
    print(f"[prepare:{spec.name}] 请人工查看 {d['figures']} ，确认后再运行 train。")
    return checks


def load_split(profile: str | Profile, split: str) -> Dict[str, np.ndarray]:
    spec = profile_spec(profile)
    path = dirs(spec)["data"] / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare")
    with np.load(path, allow_pickle=False) as payload:
        cache_fp = str(np.asarray(payload["fingerprint"]).reshape(-1)[0])
        if cache_fp != fingerprint(spec):
            raise RuntimeError(f"fingerprint mismatch in {path}")
        return {key: np.asarray(payload[key]) for key in payload.files if key != "fingerprint"}
