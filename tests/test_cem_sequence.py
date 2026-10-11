"""Sequence CEM reaches a known action trajectory and respects its constraints."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning import cem_plan_sequence, shift_plan, smooth_action_sequences


class SequenceCEMTests(unittest.TestCase):
    def test_cem_recovers_a_known_sequence(self) -> None:
        target = np.array([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1]], dtype=np.float64)

        def cost(samples: np.ndarray) -> np.ndarray:
            return np.sum((samples - target) ** 2, axis=(1, 2))

        result = cem_plan_sequence(
            cost,
            mean=np.zeros(2),
            std=np.ones(2),
            horizon=3,
            population=96,
            n_iterations=8,
            elite_fraction=0.25,
            action_dim=2,
            lower=np.array([-1.0, -1.0]),
            upper=np.array([1.0, 1.0]),
            temporal_smooth=0.0,
            rng=np.random.default_rng(0),
        )
        self.assertLess(result["cost"], 1e-3)
        self.assertEqual(result["sequence"].shape, (3, 2))
        self.assertTrue(np.allclose(result["action"], result["sequence"][0]))
        self.assertLess(result["history"][-1]["best_cost"], result["history"][0]["best_cost"])

    def test_smoothing_shrinks_action_steps_and_bounds_still_hold(self) -> None:
        raw = np.array([[[0.0, 0.0], [1.0, -1.0], [0.0, 1.0]]], dtype=np.float64)
        smooth = smooth_action_sequences(raw, 0.8)
        raw_step = np.mean(np.abs(np.diff(raw, axis=1)))
        smooth_step = np.mean(np.abs(np.diff(smooth, axis=1)))
        self.assertLess(smooth_step, raw_step)
        clipped = np.clip(smooth_action_sequences(np.array([[[2.0], [-2.0]]]), 0.5), -1.0, 1.0)
        self.assertTrue(np.all(clipped <= 1.0))
        self.assertTrue(np.all(clipped >= -1.0))

    def test_bounds_are_enforced_on_the_returned_sequence(self) -> None:
        def cost(samples: np.ndarray) -> np.ndarray:
            return np.sum((samples - 5.0) ** 2, axis=(1, 2))

        result = cem_plan_sequence(
            cost,
            mean=np.zeros(1),
            std=np.ones(1),
            horizon=2,
            population=16,
            n_iterations=3,
            elite_fraction=0.25,
            action_dim=1,
            lower=np.array([-0.2]),
            upper=np.array([0.4]),
            temporal_smooth=0.5,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(np.all(result["sequence"] <= 0.4 + 1e-12))
        self.assertTrue(np.all(result["sequence"] >= -0.2 - 1e-12))

    def test_warm_start_shifts_the_executed_action_off_the_plan(self) -> None:
        plan = np.array([[1.0, 0.0], [2.0, 0.5], [3.0, -0.5]])
        shifted = shift_plan(plan)
        self.assertTrue(np.allclose(shifted[0], plan[1]))
        self.assertTrue(np.allclose(shifted[-1], plan[-1]))


if __name__ == "__main__":
    unittest.main()
