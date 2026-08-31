from __future__ import annotations

import sys
import copy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossover.data import BalancedBatchStream, dense_points_px, grid_points_px, support_points_px, support_tids
from crossover.protocol import EXPECTED_SEEDS, SUPPORT_NAMES, load_protocol, protocol_hash, validate_protocol


def test_protocol_is_frozen() -> None:
    cfg = load_protocol()
    assert tuple(cfg["seeds"]) == EXPECTED_SEEDS
    assert tuple(cfg["regimes"]) == ("frozen_feature", "head_only", "full")
    assert tuple(SUPPORT_NAMES) == ("corners4", "G9", "G16", "G64")
    assert support_tids("corners4").tolist() == [0, 7, 56, 63]
    assert support_tids("G9").tolist() == [0, 3, 7, 24, 27, 31, 56, 59, 63]
    assert support_tids("G16").tolist() == [0, 2, 5, 7, 16, 18, 21, 23, 40, 42, 45, 47, 56, 58, 61, 63]
    assert support_tids("G64").tolist() == list(range(64))
    assert grid_points_px().shape == (64, 2)
    assert dense_points_px().shape == (1681, 2)
    np.testing.assert_array_equal(support_points_px("corners4"), grid_points_px()[[0, 7, 56, 63]])
    assert len(protocol_hash()) == 64


def test_protocol_rejects_semantic_drift() -> None:
    cfg = load_protocol()
    mutations = [
        ("schema_version", 2),
        ("protocol_id", "other"),
        ("task", "other"),
    ]
    for key, value in mutations:
        changed = copy.deepcopy(cfg)
        changed[key] = value
        try:
            validate_protocol(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"protocol drift must be rejected: {key}")
    nested = [
        ("renderer", "family", "other"),
        ("renderer", "mode", "RGB"),
        ("model", "head_dim", 3),
        ("training.scheduler", "name", "StepLR"),
        ("training", "anchor_eval_every", 10),
        ("evaluation", "batch_size", 32),
        ("gates", "crossover_gap_px", 1.0),
    ]
    for section, key, value in nested:
        changed = copy.deepcopy(cfg)
        target = changed
        for part in section.split("."):
            target = target[part]
        target[key] = value
        try:
            validate_protocol(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"protocol drift must be rejected: {section}.{key}")


def test_balanced_stream_is_reproducible_and_tiled() -> None:
    left = BalancedBatchStream(9, 64, 123)
    right = BalancedBatchStream(9, 64, 123)
    for _ in range(8):
        np.testing.assert_array_equal(left.next_indices(), right.next_indices())
    assert left.next_indices().shape == (64,)
    assert left.digest() == right.digest()
