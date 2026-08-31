from __future__ import annotations

import unittest

import numpy as np

from phase3_high_upside.group_law import (
    commutativity_error,
    composition_error,
    inverse_error,
    path_independence_error,
    repeated_composition_error,
    run_group_law_battery,
)
from phase3_high_upside.operators import apply_affine
from phase3_high_upside.protocol import delta_key
from phase3_high_upside.tests.synthetic_group import make_latent_field, true_operator


def _ops_from_true(deltas, dim=8):
    ops = {}
    for dx, dy in deltas:
        m, b = true_operator(dx, dy, dim)
        ops[delta_key((dx, dy))] = {"matrix": m, "bias": b}
    return ops


class GroupLawSyntheticTests(unittest.TestCase):
    def setUp(self):
        origins = np.stack(np.meshgrid(np.arange(0.0, 8.0), np.arange(0.0, 8.0), indexing="ij"), axis=-1).reshape(-1, 2)
        self.field = make_latent_field(6, origins, dim=8, seed=7)
        self.z = self.field["z"].reshape(-1, 8)
        self.needed = [
            (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0), (8.0, 0.0),
            (0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (0.0, 4.0), (0.0, 8.0),
            (1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (4.0, -4.0),
            (-0.5, 0.0), (-1.0, 0.0), (-2.0, 0.0), (-4.0, 0.0), (-8.0, 0.0),
            (0.0, -0.5), (0.0, -1.0), (0.0, -2.0), (0.0, -4.0), (0.0, -8.0),
        ]
        self.ops = _ops_from_true(self.needed)

    def test_true_group_passes_composition_inverse_commute(self):
        z = self.z[:40]
        target = apply_affine(z, *true_operator(2.0, 0.0, 8))
        comp = composition_error(self.ops, (1.0, 0.0), (1.0, 0.0), z, target)
        self.assertLess(comp["e_composed"], 1e-10)
        inv = inverse_error(self.ops, (2.0, 0.0), z)
        self.assertLess(inv["e_roundtrip"], 1e-10)
        comm = commutativity_error(self.ops, (1.0, 0.0), (0.0, 1.0), z)
        self.assertLess(comm["e_xy_vs_yx"], 1e-10)
        path = path_independence_error(self.ops, (2.0, 0.0), (0.0, 2.0), z)
        self.assertLess(path["e_path_disagree"], 1e-10)
        rep = repeated_composition_error(self.ops, (1.0, 0.0), 4, z, apply_affine(z, *true_operator(4.0, 0.0, 8)))
        self.assertLess(rep["e_rep_vs_direct"], 1e-10)

    def test_battery_summary_near_zero(self):
        report = run_group_law_battery(self.ops, self.z[:30])
        for key, value in report["summary"].items():
            if value is None:
                continue
            self.assertLess(value, 1e-8, msg=key)

    def test_random_operators_fail_group_law(self):
        rng = np.random.default_rng(9)
        fake = {}
        for dx, dy in self.needed:
            fake[delta_key((dx, dy))] = {
                "matrix": rng.standard_normal((8, 8)),
                "bias": rng.standard_normal(8),
            }
        z = self.z[:30]
        target = apply_affine(z, *true_operator(2.0, 0.0, 8))
        comp = composition_error(fake, (1.0, 0.0), (1.0, 0.0), z, target)
        self.assertGreater(comp["e_composed"], 0.3)
        inv = inverse_error(fake, (2.0, 0.0), z)
        self.assertGreater(inv["e_roundtrip"], 0.3)
        comm = commutativity_error(fake, (1.0, 0.0), (0.0, 1.0), z)
        self.assertGreater(comm["e_xy_vs_yx"], 0.05)
