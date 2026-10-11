"""Oracle MPC rolls true dynamics and puts the live state back."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MPMConfig, apply_action_frames
from src.mpc import oracle_propose, oracle_sequence_cost, run_closed_loop
from src.planning_tasks import SUCCESS_FRACTION, make_reachable_goal, task_distance


class LinearWorld:
    def __init__(self, state: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()

    def snapshot(self) -> np.ndarray:
        return self.state.copy()

    def restore(self, snap: np.ndarray) -> None:
        self.state = np.asarray(snap, dtype=np.float64).copy()

    def step_action(self, action: np.ndarray) -> None:
        delta = np.asarray(action, dtype=np.float64).reshape(-1)
        self.state = self.state + delta[: self.state.size]


class OraclePlannerTests(unittest.TestCase):
    def test_scoring_restores_the_live_state(self) -> None:
        world = LinearWorld(np.zeros(2))
        sequences = np.array([[[0.5, 0.0]], [[0.0, -0.2]]], dtype=np.float64)
        costs = oracle_sequence_cost(world, sequences, lambda item: float(np.sum(item.state ** 2)))
        self.assertTrue(np.allclose(world.state, 0.0))
        self.assertGreater(costs[0], costs[1])

    def test_oracle_reaches_a_linear_goal_and_zero_does_not(self) -> None:
        goal = np.array([0.3, -0.1])
        world = LinearWorld(np.zeros(2))
        cem = {
            "mean": np.zeros(2),
            "std": np.full(2, 0.5),
            "horizon": 1,
            "population": 64,
            "n_iterations": 5,
            "elite_fraction": 0.25,
            "action_dim": 2,
            "lower": np.array([-1.0, -1.0]),
            "upper": np.array([1.0, 1.0]),
            "temporal_smooth": 0.0,
        }
        initial = float(np.linalg.norm(goal))
        result = run_closed_loop(
            lambda warm: oracle_propose(world, lambda item: float(np.linalg.norm(item.state - goal)), cem, np.random.default_rng(0), warm),
            world.step_action,
            lambda: float(np.linalg.norm(world.state - goal)),
            initial_distance=initial,
            max_steps=1,
            success_distance=SUCCESS_FRACTION * initial,
        )
        self.assertEqual(result["status"], "success")
        self.assertLess(result["final_distance"], SUCCESS_FRACTION * initial)
        idle = LinearWorld(np.zeros(2))
        failed = run_closed_loop(
            lambda _warm: (np.zeros(2), np.zeros((1, 2)), 0.0),
            idle.step_action,
            lambda: float(np.linalg.norm(idle.state - goal)),
            initial_distance=initial,
            max_steps=1,
            success_distance=SUCCESS_FRACTION * initial,
        )
        self.assertEqual(failed["status"], "timeout")

    def test_scripted_simulator_action_reaches_its_own_goal(self) -> None:
        mpm = MPMConfig(n_grid=12, substeps=1, img_size=16)
        for name in ("center_target", "region_target", "configuration", "bar_orientation"):
            spec = make_reachable_goal(name, seed=0, scripted_steps=2, mpm=mpm)
            sim = spec["sim"]
            for _ in range(spec["scripted_steps"]):
                apply_action_frames(sim, spec["scripted_action"], 1)
            distance = task_distance(name, sim.x, spec["goal_positions"], spec["region"])
            self.assertLess(distance, 1e-5)
            if name == "center_target":
                self.assertGreater(spec["initial_distance"], 1e-8)


if __name__ == "__main__":
    unittest.main()
