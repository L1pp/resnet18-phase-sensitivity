from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image

from track_a_toy.config import FEATURE_ROOT, FIGURE_ROOT
from track_a_toy.visuals import (
    chinese_font_path,
    font_can_render,
    write_concept_figure,
    write_input_grid_figure,
    write_summary_figure,
)
from track_a_toy.pipeline import _load_dataset


class VisualTests(unittest.TestCase):
    def test_chinese_font_and_generated_figures_are_readable(self) -> None:
        font_path = chinese_font_path()
        self.assertTrue(font_path.exists())
        self.assertTrue(font_can_render("有限画布 环面画布 裁切 绕回"))
        finite = _load_dataset("finite")
        torus = _load_dataset("torus")
        # Avoid Python tempfile's Windows 0700 cleanup behavior in the
        # Codex sandbox; use a known workspace-local scratch directory.
        root = FIGURE_ROOT / "_visual_test_scratch"
        root.mkdir(parents=True, exist_ok=True)
        try:
            concept = Path(write_concept_figure(root / "概念.png"))
            finite_path = Path(
                write_input_grid_figure(
                    root / "有限.png",
                    dataset="finite",
                    images=finite["images"],
                    points=finite["points"],
                    fully_visible=finite["fully_visible"],
                )
            )
            torus_path = Path(
                write_input_grid_figure(
                    root / "环面.png",
                    dataset="torus",
                    images=torus["images"],
                    points=torus["points"],
                    fully_visible=None,
                )
            )
            summary = json.loads((FEATURE_ROOT.parent / "probe" / "summary.json").read_text(encoding="utf-8"))
            overview = Path(write_summary_figure(root / "总览.png", summary))
            for path in (concept, finite_path, torus_path, overview):
                self.assertTrue(path.exists(), path)
                with Image.open(path) as image:
                    self.assertGreater(image.width, 500, path)
                    self.assertGreater(image.height, 300, path)
                    image.verify()
        finally:
            for path in root.glob("*"):
                path.unlink(missing_ok=True)
            root.rmdir()

    def test_final_named_figures_are_openable_when_present(self) -> None:
        expected = (
            FIGURE_ROOT / "实验概念图.png",
            FIGURE_ROOT / "有限画布_固定输入_10张.png",
            FIGURE_ROOT / "环面画布_固定输入_10张.png",
            FIGURE_ROOT / "结果总览.png",
        )
        if not all(path.exists() for path in expected):
            self.skipTest("wallclock/report visual generation has not run yet")
        for path in expected:
            with Image.open(path) as image:
                image.verify()
        training_summary = FIGURE_ROOT.parent / "training" / "summary.json"
        if training_summary.exists():
            payload = json.loads(training_summary.read_text(encoding="utf-8"))
            for row in payload["models"].values():
                for key in ("training_curve", "prediction_visualization"):
                    path = Path(row[key])
                    self.assertTrue(path.exists(), path)
                    with Image.open(path) as image:
                        image.verify()
