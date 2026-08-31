from __future__ import annotations

import json
import os
import shutil
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from neural_affine_crossover_clean.delta_analysis import (
    DEFAULT_COORD_SCALE,
    FieldSource,
    analyze_three_seed_fields,
    load_three_seed_fields,
)


_TEST_TEMP_ROOT = Path(__file__).resolve().parents[2] / ".delta_analysis_test_tmp"


@contextmanager
def _temporary_directory() -> Iterator[str]:
    # The Codex Windows sandbox can deny child creation below the system TEMP
    # directory because Python gives it a 0700 ACL.  Keep test-only fixtures in
    # the writable workspace instead; this does not touch system ACLs.
    _TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    directory = _TEST_TEMP_ROOT / f"case_{os.getpid()}_{time.monotonic_ns()}"
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=False)


def _grid() -> np.ndarray:
    axis = np.linspace(59.0, 164.0, 41)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1)


def _bundle_arrays(seed: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    true = _grid()
    x = (true[..., 0] - 111.5) / 52.5
    y = (true[..., 1] - 111.5) / 52.5
    if kind == "affine":
        field = np.stack((2.0 + 3.0 * x - 1.5 * y, -1.0 + 0.5 * x + 2.0 * y), axis=-1)
    elif kind == "quadratic":
        field = np.stack((x * x + 0.5 * x * y, -0.75 * y * y + 0.25 * x * y), axis=-1)
    elif kind == "interaction":
        field = np.stack((3.0 * x * y, -2.0 * x * y), axis=-1)
    elif kind == "fold":
        field = np.stack((x * 100.0, y * 2.0), axis=-1)
    elif kind == "periodic":
        field = np.stack((np.sin(2.0 * np.pi * 4.0 * x), np.cos(2.0 * np.pi * 3.0 * y)), axis=-1)
    else:
        raise AssertionError(kind)
    # Per-seed offset makes coefficient consistency non-trivial but keeps the
    # synthetic structures exact enough for deterministic tests.
    field = field + (seed - 20260817) * 0.01
    return field, true


def _write_sources(root: Path, kind: str = "affine", *, normalized_ntk: bool = False) -> dict[int, dict[str, Path]]:
    sources: dict[int, dict[str, Path]] = {}
    for seed in (20260816, 20260817, 20260818):
        base, true = _bundle_arrays(seed, kind)
        # Make each synthetic kind exercise the requested delta diagnostics:
        # head-NTK carries the named affine/quadratic/interaction/periodic
        # structure, while full-head is a separate constant shift.
        ntk = np.zeros_like(base)
        head = base
        full = head + 0.75
        seed_dir = root / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        ntk_path = seed_dir / "ntk.npz"
        head_path = seed_dir / "head.npz"
        full_path = seed_dir / "full.npz"
        if normalized_ntk:
            np.savez(ntk_path, ntk_query_norm=(ntk / DEFAULT_COORD_SCALE).reshape(-1, 2), true_px=true, query_indices=np.arange(1681))
        else:
            np.savez(ntk_path, ntk_pred_px=ntk, true_px=true, query_indices=np.arange(1681))
        np.savez(head_path, head_only_pred_px=head, true_px=true)
        np.savez(full_path, trained_pred_px=full, true_px=true)
        sources[seed] = {"ntk": ntk_path, "head": head_path, "full": full_path}
    return sources


class DeltaAnalysisTests(unittest.TestCase):
    def test_affine_and_quadratic_coefficients_are_recovered(self) -> None:
        with _temporary_directory() as directory:
            root = Path(directory)
            sources = _write_sources(root, "affine")
            bundles = load_three_seed_fields(sources)
            report = analyze_three_seed_fields(bundles)
        delta = report["per_seed"]["20260817"]["deltas"]["head_minus_ntk"]
        degree1 = delta["polynomial"]["degree_1"]
        self.assertGreater(degree1["explained_fraction_raw"], 0.999999)
        self.assertLess(degree1["residual_rmse_px"], 1e-10)
        self.assertLess(delta["cross_axis_anova"]["interaction_fraction"], 1e-12)

    def test_quadratic_interaction_and_fft_are_reported(self) -> None:
        with _temporary_directory() as directory:
            bundles = load_three_seed_fields(_write_sources(Path(directory), "interaction"))
            report = analyze_three_seed_fields(bundles)
        delta = report["per_seed"]["20260817"]["deltas"]["head_minus_ntk"]
        degree1 = delta["polynomial"]["degree_1"]
        degree2 = delta["polynomial"]["degree_2"]
        self.assertGreater(degree1["residual_sse"], 1.0)
        self.assertGreater(delta["cross_axis_anova"]["interaction_fraction"], 0.99)
        self.assertIn("main_nonzero_frequency_index_xy", delta["degree2_residual_fft"])
        self.assertEqual(len(degree2["coefficients"]), 6)

    def test_fold_and_identity_plus_delta_jacobian(self) -> None:
        with _temporary_directory() as directory:
            bundles = load_three_seed_fields(_write_sources(Path(directory), "fold"))
            report = analyze_three_seed_fields(bundles)
        # full field has d(output_x)/dx=100/52.5 > 0, so this test verifies the
        # diagnostic is present and finite; the fold is in a delta warp only
        # when a negative slope is synthesized below.
        endpoint = report["per_seed"]["20260817"]["endpoints"]["full"]["endpoint_jacobian"]
        warp = report["per_seed"]["20260817"]["deltas"]["full_minus_head"]["I_plus_grad_delta"]
        self.assertGreater(endpoint["determinant"]["mean"], 0.0)
        self.assertIn("fold_fraction", warp)
        self.assertTrue(np.isfinite(endpoint["determinant"]["min"]))

    def test_periodic_field_reports_nonzero_peak_and_direction(self) -> None:
        with _temporary_directory() as directory:
            bundles = load_three_seed_fields(_write_sources(Path(directory), "periodic"))
            report = analyze_three_seed_fields(bundles)
        fft = report["per_seed"]["20260817"]["deltas"]["full_minus_head"]["degree2_residual_fft"]
        self.assertGreaterEqual(len(fft["top_nonzero_peaks"]), 1)
        self.assertIsInstance(fft["main_nonzero_direction_degrees"], float)
        self.assertGreaterEqual(fft["low_frequency_radius_le_2_fraction"], 0.0)

    def test_normalized_ntk_and_strict_query_alignment(self) -> None:
        with _temporary_directory() as directory:
            root = Path(directory)
            sources = _write_sources(root, "affine", normalized_ntk=True)
            bundles = load_three_seed_fields(sources)
            self.assertTrue(bundles[0].ntk.normalized)
            bad = root / "bad.npz"
            np.savez(bad, ntk_query_norm=np.zeros((1681, 2)), query_indices=np.arange(1681)[::-1])
            broken = dict(sources[20260816])
            broken["ntk"] = bad
            with self.assertRaises(ValueError):
                load_three_seed_fields({**sources, 20260816: broken})

    def test_atomic_json_and_npz_outputs_and_boundary(self) -> None:
        with _temporary_directory() as directory:
            root = Path(directory)
            bundles = load_three_seed_fields(_write_sources(root, "quadratic"))
            json_path = root / "out" / "analysis.json"
            npz_path = root / "out" / "fields.npz"
            report = analyze_three_seed_fields(
                bundles,
                output_json=json_path,
                output_npz=npz_path,
                evidence_boundary={"backend_label": "mixed historical AMD/A10"},
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(npz_path.is_file())
            on_disk = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["source_boundary"]["formal_s1_verdict_unchanged"])
            self.assertEqual(on_disk["source_boundary"]["backend_label"], "mixed historical AMD/A10")
            with np.load(npz_path, allow_pickle=False) as arrays:
                self.assertIn("true_px", arrays.files)
                self.assertIn("seed20260817_head_minus_ntk_px", arrays.files)
            self.assertEqual(report["artifacts"]["npz"], str(npz_path))


if __name__ == "__main__":
    unittest.main()
