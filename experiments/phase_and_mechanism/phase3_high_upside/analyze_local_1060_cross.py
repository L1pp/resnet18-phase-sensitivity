"""CPU: tabulate the already-computed 5x5 universal operator cross matrix and redraw figures."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from phase1_gap_rep.common import setup_matplotlib_chinese
from phase3_high_upside.io_util import dump_json, load_json
from phase3_high_upside.local_1060_cat1 import CONTENTS, OUT, plot_layerwise
from phase3_high_upside.protocol import PHASE3_ROOT


def _matrix(rows: List[dict]) -> tuple[np.ndarray, List[str]]:
    labels = list(CONTENTS)
    mat = np.full((len(labels), len(labels)), np.nan)
    for row in rows:
        i = labels.index(row["fit_on"])
        j = labels.index(row["apply_to"])
        mat[i, j] = row["mean_e_cross"]
    return mat, labels


def plot_annotated(mat: np.ndarray, labels: List[str], path: Path) -> None:
    plt = setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=float(np.nanmax(mat)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("应用到的内容 B")
    ax.set_ylabel("拟合 M 的内容 A")
    ax.set_title("Universal operator 交叉误差（对角=自拟合）")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.35 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean E_cross")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(mat: np.ndarray, labels: List[str], path: Path) -> None:
    lines = [
        "# Universal operator 5×5 交叉表",
        "",
        "已有 `cross.json` 的汇总。行 = 拟合 M^A 的内容，列 = 被移动的内容 z^B。对角是自拟合。",
        "",
        "| fit \\ apply | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---:" for _ in labels]) + "|",
    ]
    for i, a in enumerate(labels):
        cells = [f"{mat[i, j]:.3f}" if np.isfinite(mat[i, j]) else "—" for j in range(len(labels))]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    # best off-diagonal transfers
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j or not np.isfinite(mat[i, j]):
                continue
            pairs.append((float(mat[i, j]), a, b, float(mat[j, j])))
    pairs.sort()
    lines += ["", "## 最好的跨内容迁移（排除对角）", ""]
    for e, a, b, self_b in pairs[:6]:
        lines.append(f"- `{a} → {b}`: E_cross={e:.3f}（B 自拟合 {self_b:.3f}，比值 {e / self_b:.1f}×）")
    lines += ["", "## 最差的跨内容迁移", ""]
    for e, a, b, self_b in pairs[-4:]:
        lines.append(f"- `{a} → {b}`: E_cross={e:.3f}（B 自拟合 {self_b:.3f}，比值 {e / self_b:.1f}×）")
    lines += [
        "",
        "## 读法",
        "",
        "- 几何族之间可以部分共用算子：`line → quadratic` / `quadratic → line` 约 0.05–0.07，只比自拟合差几倍，不是噪声级失败。",
        "- `line → blob` 约 0.057 也好；反过来 `blob → line` 约 0.27，不对称。",
        "- noise 当**源**最差（→line 0.75）；noise 当**目标**也不好（别人的 M 搬 noise 仍 0.09–0.60）。",
        "- 结论比只看 noise→Bezier 更细：**不是完全 network-level 万能算子，也不是完全不能跨内容。** 线/二次曲线这一档几何更像同一族；noise patch 学到的 M 不能当通用平移。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_handoff(mat: np.ndarray, labels: List[str]) -> None:
    handoff = OUT / "HANDOFF.md"
    extra = [
        "",
        "## 补：完整 5×5（CPU 重读 cross.json）",
        "",
        "详见 `CROSS_MATRIX.md`。短结论：",
        f"- 最好跨内容：`quadratic→line`={mat[labels.index('quadratic'), labels.index('line')]:.3f}，`line→quadratic`={mat[labels.index('line'), labels.index('quadratic')]:.3f}。",
        f"- 最差：`noise→line`={mat[labels.index('noise'), labels.index('line')]:.3f}。",
        "- 几何内容之间的 M 有一定可迁移性；noise 拟合的 M 不能当通用 network-level 平移。",
        "",
    ]
    text = handoff.read_text(encoding="utf-8") if handoff.exists() else ""
    marker = "## 补：完整 5×5"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    handoff.write_text(text.rstrip() + "\n" + "\n".join(extra), encoding="utf-8")


def run() -> Dict[str, object]:
    rows = load_json(OUT / "universal" / "cross.json")["rows"]
    mat, labels = _matrix(rows)
    dump_json(OUT / "universal" / "cross_matrix.json", {"labels": labels, "matrix": mat.tolist()})
    plot_annotated(mat, labels, OUT / "universal" / "figures" / "cross_heatmap.png")
    write_markdown(mat, labels, OUT / "CROSS_MATRIX.md")
    panel = OUT / "layerwise" / "panel.json"
    if panel.exists():
        plot_layerwise(load_json(panel)["rows"], OUT / "layerwise" / "figures" / "e_by_stage.png")
    patch_handoff(mat, labels)
    print("wrote", OUT / "CROSS_MATRIX.md")
    print("matrix")
    print(np.array2string(mat, precision=3))
    return {"matrix": mat.tolist(), "labels": labels}


if __name__ == "__main__":
    run()
