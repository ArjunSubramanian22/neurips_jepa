"""Planner invariants on analytic tasks. No learned dynamics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning import cem_minimize, oracle_linear_action_recovery


class PlanningInvariantTests(unittest.TestCase):
    def test_oracle_beats_a_random_choice_on_identity_dynamics(self) -> None:
        for seed in range(8):
            result = oracle_linear_action_recovery(
                seed=seed,
                n_candidates=8,
                latent_dim=4,
                correct_index=3,
            )
            costs = np.asarray(result["costs"], dtype=np.float64)
            chosen = int(result["chosen_index"])
            self.assertEqual(chosen, int(result["correct_index"]))
            self.assertGreater(result["utility"]["top1"], result["utility"]["chance"])
            self.assertLess(float(costs[chosen]), float(costs.mean()))
            self.assertTrue(np.all(costs[chosen] <= costs + 1e-12))

    def test_cem_lowers_a_known_quadratic(self) -> None:
        target = np.array([0.4, -0.2, 0.7, 0.1])
        initial = float(np.sum(target ** 2))

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
        self.assertLess(result["cost"], initial)
        best = [float(step["best_cost"]) for step in result["history"]]
        self.assertLess(best[-1], best[0])
        self.assertTrue(all(best[index] <= best[index - 1] + 1e-12 for index in range(1, len(best))))


if __name__ == "__main__":
    unittest.main()
