"""中文可视化：概念图、固定输入网格和结果总览。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CONFIG


FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)


def chinese_font_path() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("没有找到可用的中文字体（尝试了微软雅黑、黑体和宋体）")


def chinese_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = chinese_font_path()
    if bold:
        bold_path = Path(r"C:\Windows\Fonts\msyhbd.ttc")
        if bold_path.exists():
            path = bold_path
    return ImageFont.truetype(str(path), int(size))


def font_can_render(text: str) -> bool:
    font = chinese_font(28)
    return font.getbbox(text) is not None and bool(font.getmask(text).getbbox())


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, *, fill: str = "#172033") -> None:
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=5)


def _center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.FreeTypeFont, *, fill: str = "#172033") -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=4)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=4)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, fill: str = "#e76f51", width: int = 5) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    left = (int(end[0] - 20 * ux + 9 * px), int(end[1] - 20 * uy + 9 * py))
    right = (int(end[0] - 20 * ux - 9 * px), int(end[1] - 20 * uy - 9 * py))
    draw.polygon([tip, left, right], fill=fill)


def _triangle_polygon(center_xy: tuple[float, float], origin: tuple[int, int], scale: float) -> list[tuple[int, int]]:
    vertices = np.asarray(CONFIG["triangle_vertices_xy"], dtype=np.float64) + np.asarray(center_xy, dtype=np.float64)[None, :]
    return [
        (int(round(origin[0] + float(x) * scale)), int(round(origin[1] + float(y) * scale)))
        for x, y in vertices
    ]


def write_concept_figure(path: Path) -> str:
    """Draw the finite-versus-torus idea before showing any metrics."""

    width, height = 1800, 860
    canvas = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    title = chinese_font(42, bold=True)
    heading = chinese_font(30, bold=True)
    body = chinese_font(24)
    small = chinese_font(20)
    _center_text(draw, (0, 22, width, 86), "先看懂：同一个三角形，边界条件改变了什么？", title)

    panels = [(70, 120, 830, 770), (970, 120, 1730, 770)]
    for box in panels:
        draw.rounded_rectangle(box, radius=24, fill="#ffffff", outline="#ccd5e3", width=3)
    _center_text(draw, (90, 135, 810, 205), "有限画布（finite）", heading, fill="#254e70")
    _center_text(draw, (990, 135, 1710, 205), "环面画布（torus）", heading, fill="#28745b")

    size = 430
    left_origin = (220, 255)
    right_origin = (1125, 255)
    for origin in (left_origin, right_origin):
        draw.rectangle((origin[0], origin[1], origin[0] + size, origin[1] + size), fill="#202938", outline="#4a6078", width=4)
        for tick in (0, 32, 64):
            x = int(origin[0] + tick * size / 64)
            y = int(origin[1] + tick * size / 64)
            draw.line((x, origin[1] + size, x, origin[1] + size + 8), fill="#4a6078", width=2)
            draw.line((origin[0] - 8, y, origin[0], y), fill="#4a6078", width=2)
    finite_poly = _triangle_polygon((3.0, 4.0), left_origin, size / 64.0)
    draw.polygon(finite_poly, fill="#ffffff", outline="#f4a261")
    draw.line((left_origin[0] + 3 * size / 64, left_origin[1] + 4 * size / 64) * 2, fill="#f4a261", width=3)
    _arrow(draw, (175, 230), (230, 275), fill="#e76f51")
    _text(draw, (92, 207), "中心靠近左上角", small, fill="#b84a35")
    _text(draw, (125, 700), "三角形有一部分在画布外\n→ 被裁掉；输入不再完整", body, fill="#8f3d2e")
    _text(draw, (205, 716), "坐标刻度：0、32、64", small, fill="#d8e2ee")

    torus_center = (62.0, 60.0)
    for shift_x in (-64.0, 0.0, 64.0):
        for shift_y in (-64.0, 0.0, 64.0):
            poly = _triangle_polygon((torus_center[0] + shift_x, torus_center[1] + shift_y), right_origin, size / 64.0)
            draw.polygon(poly, fill="#8de0c0", outline="#3fb68a")
    draw.line((right_origin[0] + torus_center[0] * size / 64, right_origin[1] + torus_center[1] * size / 64) * 2, fill="#f4a261", width=3)
    _arrow(draw, (1600, 310), (1690, 310), fill="#3fb68a")
    _arrow(draw, (1300, 690), (1210, 690), fill="#3fb68a")
    _text(draw, (1510, 235), "越过右边", small, fill="#28745b")
    _text(draw, (1180, 705), "从另一边绕回", small, fill="#28745b")
    _text(draw, (1035, 700), "边界是周期连接的\n→ 形状不会被画布边缘截断", body, fill="#24624e")

    draw.rounded_rectangle((160, 792, 1640, 835), radius=14, fill="#e8eef6")
    _center_text(draw, (180, 792, 1620, 835), "白色或绿色区域就是输入三角形；标出的坐标是三角形中心，而不是某个顶点。", small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _array_to_tile(array: np.ndarray, size: int) -> Image.Image:
    gray = np.asarray(array)[0]
    image = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def _touches_border(array: np.ndarray) -> bool:
    gray = np.asarray(array)[0]
    return bool(np.any(gray[0] > 0) or np.any(gray[-1] > 0) or np.any(gray[:, 0] > 0) or np.any(gray[:, -1] > 0))


def write_input_grid_figure(
    path: Path,
    *,
    dataset: str,
    images: np.ndarray,
    points: np.ndarray,
    fully_visible: np.ndarray | None,
) -> str:
    """Write exactly the ten configured positions in a readable 2x5 grid."""

    configured = np.asarray(CONFIG["visual_positions_xy"], dtype=np.float32)
    if configured.shape != (10, 2):
        raise ValueError(f"visual_positions_xy must be [10,2], got {configured.shape}")
    lookup = {tuple(np.rint(point).astype(int).tolist()): i for i, point in enumerate(np.asarray(points))}
    selected = []
    for point in configured:
        key = tuple(np.rint(point).astype(int).tolist())
        if key not in lookup:
            raise ValueError(f"configured visualization point {key} is missing from {dataset}")
        selected.append(lookup[key])
    if len(selected) != 10 or len(set(selected)) != 10:
        raise ValueError("input visualization must contain exactly ten unique configured positions")

    tile = 210
    cell_w, cell_h = 300, 300
    width, height = cell_w * 5, 110 + cell_h * 2
    canvas = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    title = chinese_font(34, bold=True)
    body = chinese_font(21)
    small = chinese_font(18)
    domain = "有限画布" if dataset == "finite" else "环面画布"
    _center_text(draw, (0, 8, width, 52), f"{domain}：固定十个中心位置（2×5）", title)
    _center_text(draw, (0, 55, width, 100), "白色区域=输入三角形；坐标标的是三角形中心", body, fill="#455468")
    for order, index in enumerate(selected):
        row, col = divmod(order, 5)
        left = col * cell_w
        top = 110 + row * cell_h
        tile_image = _array_to_tile(images[index], tile)
        image_left = left + (cell_w - tile) // 2
        image_top = top + 45
        canvas.paste(tile_image, (image_left, image_top))
        draw.rectangle((image_left, image_top, image_left + tile, image_top + tile), outline="#75869a", width=3)
        point = np.asarray(points[index], dtype=np.float64)
        x, y = int(round(point[0])), int(round(point[1]))
        _center_text(draw, (left + 4, top + 4, left + cell_w - 4, top + 39), f"中心 ({x},{y})", body)
        if dataset == "finite":
            visible = bool(np.asarray(fully_visible, dtype=bool)[index]) if fully_visible is not None else False
            status, color = ("完整", "#247a4d") if visible else ("裁切", "#b43f36")
        else:
            status, color = ("绕回", "#226d9f") if _touches_border(images[index]) else ("环面内", "#28745b")
        draw.rounded_rectangle((left + 82, top + 263, left + 218, top + 292), radius=9, fill="#ffffff", outline=color, width=2)
        _center_text(draw, (left + 82, top + 263, left + 218, top + 292), status, small, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def write_summary_figure(path: Path, probe: dict[str, Any]) -> str:
    """Write a three-panel Chinese overview from the machine-readable summary."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_path = chinese_font_path()
    font_prop = FontProperties(fname=str(font_path))
    names = ["a0_standard", "a1_zero_s1", "a2_torus_s32", "a3_torus_s1"]
    labels = ["A0\n有限/零填充/S32\n基线", "A1\n有限/零填充/S1", "A2\n环面/循环/S32", "A3\n环面/循环/S1"]
    colors = ["#3b6ea8", "#e08a35", "#3c9b78", "#9a5cb4"]
    rows = [probe["models"][name] for name in names]
    values = [
        [float(row["global_xy"]["test_mae_px"]) for row in rows],
        [float(row["modulo32_phase"]["test_circular_mae_px"]) for row in rows],
        [float(row["coarse_cell"]["test_average_accuracy"]) for row in rows],
    ]
    titles = ["绝对坐标误差（像素）", "32像素周期位置误差（像素）", "粗网格准确率"]
    directions = ["↓ 越低越好", "↓ 越低越好", "↑ 越高越好"]
    y_labels = ["平均绝对误差", "周期误差", "分类准确率"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=180)
    figure.patch.set_facecolor("#f5f7fb")
    for axis, series, title_text, direction, y_label in zip(axes, values, titles, directions, y_labels):
        bars = axis.bar(np.arange(4), series, color=colors, edgecolor="#243448", linewidth=0.7)
        axis.set_title(title_text, fontproperties=font_prop, fontsize=16, pad=14)
        axis.text(0.98, 0.95, direction, transform=axis.transAxes, ha="right", va="top", fontproperties=font_prop, fontsize=12, color="#4b596b")
        axis.set_ylabel(y_label, fontproperties=font_prop, fontsize=12)
        axis.set_xticks(np.arange(4), labels)
        for tick in axis.get_xticklabels():
            tick.set_fontproperties(font_prop)
            tick.set_fontsize(10)
        for tick in axis.get_yticklabels():
            tick.set_fontproperties(font_prop)
        axis.grid(axis="y", color="#dbe3ec", linewidth=0.8)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, series):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=10, fontproperties=font_prop)
    figure.suptitle("结果总览（只看 all-256 主口径；A0 是有限画布基线）", fontproperties=font_prop, fontsize=22, y=0.99)
    a3 = probe["models"].get("a3_torus_s1", {})
    if a3.get("constant_baseline") and int(a3.get("active_feature_count", -1)) == 0:
        footer = "A1 只改变总 stride；A2/A3 改用环面边界。A3 GAP 无有效变化特征，按常数 baseline；其余为未训练随机特征线性 probe。"
    else:
        footer = "A1 只改变总 stride；A2/A3 改用环面边界。图中只展示未训练的随机特征线性 probe。"
    figure.text(0.5, 0.02, footer, ha="center", fontproperties=font_prop, fontsize=12, color="#455468")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return str(path)


def write_training_curve_figure(path: Path, model_name: str, history: list[dict[str, Any]]) -> str:
    """Write a Chinese learning-curve figure from full-domain eval history."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_prop = FontProperties(fname=str(chinese_font_path()))
    steps = [int(row["step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    maes = [float(row["full_domain_mae_px"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    figure.patch.set_facecolor("#f5f7fb")
    axes[0].plot(steps, losses, marker="o", color="#3b6ea8", label="训练批次 MSE")
    axes[0].set_title("训练损失", fontproperties=font_prop, fontsize=15)
    axes[0].set_xlabel("步数", fontproperties=font_prop)
    axes[0].set_ylabel("均方误差", fontproperties=font_prop)
    axes[1].plot(steps, maes, marker="o", color="#d05a4e", label="全256位置 MAE")
    axes[1].axhline(2.0, color="#28745b", linestyle="--", linewidth=1.2, label="2像素早停线")
    axes[1].set_title("全域评估误差", fontproperties=font_prop, fontsize=15)
    axes[1].set_xlabel("步数", fontproperties=font_prop)
    axes[1].set_ylabel("平均绝对误差（像素）", fontproperties=font_prop)
    for axis in axes:
        axis.grid(axis="y", color="#dbe3ec", linewidth=0.8)
        axis.set_axisbelow(True)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontproperties(font_prop)
        axis.legend(prop=font_prop, loc="best")
    figure.suptitle(f"{model_name}：短训练学习曲线（全256位置）", fontproperties=font_prop, fontsize=18, y=1.02)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return str(path)


__all__ = [
    "chinese_font_path",
    "chinese_font",
    "font_can_render",
    "write_concept_figure",
    "write_input_grid_figure",
    "write_summary_figure",
    "write_training_curve_figure",
]
