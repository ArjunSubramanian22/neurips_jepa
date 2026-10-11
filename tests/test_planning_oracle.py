"""Oracle checks for random shooting and CEM before any learned dynamics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning import (
    cem_minimize,
    linear_residual_next,
    oracle_linear_action_recovery,
    select_lowest_cost,
    squared_error_cost,
)


class PlanningOracleTests(unittest.TestCase):
    def test_identity_dynamics_recover_the_designated_action(self) -> None:
        result = oracle_linear_action_recovery(seed=0, n_candidates=8, latent_dim=4, correct_index=3)
        self.assertTrue(result["unique_minimum"])
        self.assertEqual(result["chosen_index"], 3)
        self.assertEqual(result["utility"]["top1"], 1.0)
        self.assertAlmostEqual(result["utility"]["chance"], 0.125, places=12)
        self.assertAlmostEqual(min(result["costs"]), 0.0, places=12)

    def test_wrong_action_is_not_a_zero_cost_under_identity_dynamics(self) -> None:
        state = np.zeros(3)
        actions = np.eye(3)
        predicted = linear_residual_next(state, actions, np.eye(3))
        costs = squared_error_cost(predicted, goal=np.array([1.0, 0.0, 0.0]))
        self.assertEqual(select_lowest_cost(costs), 0)
        self.assertGreater(costs[1], 0.0)
        self.assertGreater(costs[2], 0.0)

    def test_cem_reduces_a_known_quadratic(self) -> None:
        target = np.array([0.4, -0.2, 0.7, 0.1])
        initial_cost = float(np.sum(target ** 2))

        def cost(samples: np.ndarray) -> np.ndarray:
            return np.sum((samples - target) ** 2, axis=-1)

        result = cem_minimize(
            cost,
            mean=np.zeros(4),
            std=np.ones(4),
            population=128,
            n_iterations=8,
            elite_fraction=0.25,
            rng=np.random.default_rng(0),
        )
        self.assertLess(result["cost"], initial_cost)
        self.assertLess(result["cost"], 1e-3)
        self.assertEqual(len(result["history"]), 8)
        self.assertLess(result["history"][-1]["best_cost"], result["history"][0]["best_cost"])


if __name__ == "__main__":
    unittest.main()
