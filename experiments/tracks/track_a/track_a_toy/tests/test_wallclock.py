from __future__ import annotations

import unittest

from track_a_toy.wallclock import PROFILE_MODELS, estimate_training_wallclock


class WallclockTests(unittest.TestCase):
    def test_estimate_counts_evaluations_and_has_conservative_upper_bound(self) -> None:
        profile = {
            "models": {
                name: {
                    "step_seconds_median": 0.1,
                    "step_seconds_mean": 0.11,
                    "step_seconds_max": 0.12,
                    "eval_seconds_full_256": 1.0,
                }
                for name in PROFILE_MODELS
            }
        }
        result = estimate_training_wallclock(profile)
        self.assertEqual(result["models"]["a0_standard"]["evaluations_including_step1"], 21)
        self.assertEqual(result["models"]["a3_torus_s1"]["evaluations_including_step1"], 7)
        self.assertGreater(
            result["training_only_conservative_seconds"],
            result["training_only_typical_seconds"],
        )
        for name in PROFILE_MODELS:
            self.assertTrue(result["models"][name]["reasonable_early_stop_scenarios"])
