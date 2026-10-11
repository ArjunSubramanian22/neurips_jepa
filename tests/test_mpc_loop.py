"""The closed loop executes one action, then replans from the new state."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mpc import heuristic_push, mean_force_action, random_action, run_closed_loop


class ClosedLoopTests(unittest.TestCase):
    def test_replanning_reaches_a_one_dimensional_goal(self) -> None:
        state = {"x": 0.0}
        seen_warm: list[np.ndarray | None] = []

        def propose(warm: np.ndarray | None):
            seen_warm.append(None if warm is None else warm.copy())
            remaining = 0.5 - state["x"]
            plan = np.array([[0.2], [remaining]], dtype=np.float64)
            return plan[0], plan, remaining ** 2

        def step(action: np.ndarray) -> None:
            state["x"] += float(action[0])

        result = run_closed_loop(
            propose,
            step,
            lambda: abs(0.5 - state["x"]),
            initial_distance=0.5,
            max_steps=4,
            success_distance=0.125,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions_to_success"], 2)
        self.assertLess(result["final_distance"], 0.125)
        self.assertIsNone(seen_warm[0])
        self.assertAlmostEqual(float(seen_warm[1][0, 0]), 0.5)

    def test_zero_motion_times_out_without_counting_as_success(self) -> None:
        result = run_closed_loop(
            lambda _warm: (np.zeros(1), np.zeros((2, 1)), 0.0),
            lambda _action: None,
            lambda: 0.5,
            initial_distance=0.5,
            max_steps=3,
            success_distance=0.125,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertFalse(result["success"])
        self.assertIsNone(result["actions_to_success"])
        self.assertEqual(result["n_actions"], 3)
        self.assertAlmostEqual(result["cumulative_cost"], 1.5)

    def test_invalid_goals_are_not_scored_as_success(self) -> None:
        ambiguous = run_closed_loop(
            lambda _warm: (_ for _ in ()).throw(AssertionError("should not plan")),
            lambda _action: None,
            lambda: 0.0,
            initial_distance=0.2,
            max_steps=2,
            success_distance=0.05,
            ambiguous=True,
        )
        self.assertEqual(ambiguous["status"], "observationally_ambiguous")
        self.assertFalse(ambiguous["scored"])
        already = run_closed_loop(
            lambda _warm: (np.zeros(1), None, 0.0),
            lambda _action: None,
            lambda: 0.0,
            initial_distance=0.0,
            max_steps=2,
            success_distance=0.0,
        )
        self.assertEqual(already["status"], "goal_already_satisfied")
        self.assertFalse(already["scored"])

    def test_baselines_stay_inside_their_definitions(self) -> None:
        mean = mean_force_action()
        self.assertTrue(np.allclose(mean, np.array([0.5, 0.5, 0.0, 0.0])))
        positions = np.array([[0.0, 0.0], [0.2, 0.0]])
        goal = positions + np.array([0.4, 0.0])
        push = heuristic_push(positions, goal)
        self.assertGreater(push[2], 0.0)
        self.assertAlmostEqual(push[3], 0.0)
        draw = random_action(np.random.default_rng(0), [0.08, 0.08, -1.2, -1.2], [0.92, 0.92, 1.2, 1.2])
        self.assertEqual(draw.shape, (4,))
        self.assertTrue(np.all(draw[:2] >= 0.08) and np.all(draw[:2] <= 0.92))


if __name__ == "__main__":
    unittest.main()
