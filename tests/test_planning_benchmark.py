"""Planning benchmark metrics on toy dynamics, with checkpoints left unmeasured."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.planning_benchmark import aggregate_controller, record_episode, select_planning_config
from src.mpc import learned_propose, run_closed_loop
from src.planning_tasks import SUCCESS_FRACTION
from tests.test_oracle_planner import LinearWorld


def _identity_rollout(latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state = np.repeat(latent.reshape(1, -1), actions.shape[0], axis=0)
    predicted = []
    for step in range(actions.shape[1]):
        state = state + actions[:, step, : latent.size]
        predicted.append(state.copy())
    return np.stack(predicted, axis=1)


def _constant_rollout(latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.broadcast_to(latent.reshape(1, 1, -1), (actions.shape[0], actions.shape[1], latent.size)).copy()


class PlanningBenchmarkTests(unittest.TestCase):
    def _cem(self) -> dict:
        return {
            "mean": np.zeros(2),
            "std": np.full(2, 0.5),
            "horizon": 1,
            "population": 48,
            "n_iterations": 4,
            "elite_fraction": 0.25,
            "action_dim": 2,
            "lower": np.array([-1.0, -1.0]),
            "upper": np.array([1.0, 1.0]),
            "temporal_smooth": 0.0,
        }

    def test_action_following_rollout_succeeds_and_a_constant_rollout_does_not(self) -> None:
        goal = np.array([0.4, 0.1])
        initial = float(np.linalg.norm(goal))
        cem = self._cem()

        def run(rollout):
            world = LinearWorld(np.zeros(2))
            return run_closed_loop(
                lambda warm: learned_propose(
                    world.state,
                    goal,
                    rollout,
                    cem,
                    np.random.default_rng(0),
                    warm,
                    action_mean=np.zeros(2),
                    action_std=np.ones(2),
                ),
                world.step_action,
                lambda: float(np.linalg.norm(world.state - goal)),
                initial_distance=initial,
                max_steps=1,
                success_distance=SUCCESS_FRACTION * initial,
            )

        followed = run(_identity_rollout)
        ignored = run(_constant_rollout)
        self.assertEqual(followed["status"], "success")
        self.assertEqual(ignored["status"], "timeout")
        self.assertGreater(ignored["final_distance"], followed["final_distance"])

    def test_aggregation_keeps_failures_and_unscored_goals(self) -> None:
        records = [
            record_episode("oracle", "center_target", 0, {"status": "success", "scored": True, "success": True, "final_distance": 0.01, "initial_distance": 0.2, "cumulative_cost": 0.01, "actions_to_success": 2, "planning_runtime_s": 0.5, "error": None}),
            record_episode("oracle", "center_target", 1, {"status": "timeout", "scored": True, "success": False, "final_distance": 0.2, "initial_distance": 0.2, "cumulative_cost": 0.8, "actions_to_success": None, "planning_runtime_s": 0.4, "error": None}),
            record_episode("oracle", "center_target", 2, {"status": "observationally_ambiguous", "scored": False, "success": False, "final_distance": 0.0, "initial_distance": 0.0, "cumulative_cost": 0.0, "actions_to_success": None, "planning_runtime_s": 0.0, "error": None}),
        ]
        summary = aggregate_controller(records)
        self.assertEqual(summary["n_scored"], 2)
        self.assertEqual(summary["n_unscored"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["actions_to_success_mean"], 2.0)
        self.assertEqual(summary["failure_counts"]["timeout"], 1)
        self.assertEqual(summary["unscored_statuses"]["observationally_ambiguous"], 1)
        self.assertIsNone(records[1]["actions_to_success"])

    def test_config_selection_uses_validation_oracle_only(self) -> None:
        rows = [
            {"config": "short", "split_role": "val", "oracle_success": 0.0, "oracle_final_distance": 0.2},
            {"config": "long", "split_role": "val", "oracle_success": 1.0, "oracle_final_distance": 0.05},
            {"config": "short", "split_role": "test", "oracle_success": 1.0, "oracle_final_distance": 0.0},
        ]
        with self.assertRaises(ValueError):
            select_planning_config(rows)
        chosen = select_planning_config(rows[:2])
        self.assertEqual(chosen["selected_config"], "long")
        self.assertTrue(chosen["same_config_for_every_model"])
        self.assertIsNone(select_planning_config([])["selected_config"])

    def test_highlight_ignores_report_seeds(self) -> None:
        from src.metrics.planning_benchmark import choose_highlight

        def row(name: str, seed: int, success: bool, distance: float) -> dict:
            return {"controller": name, "seed": seed, "scored": True, "success": success, "final_distance": distance}

        records = [
            row("action", 0, False, 0.2),
            row("action", 1, False, 0.2),
            row("other", 0, True, 0.01),
            row("other", 1, True, 0.01),
            row("action", 2, True, 0.0),
        ]
        chosen = choose_highlight(records, [0, 1], ["action", "other"])
        self.assertEqual(chosen["highlight"], "other")

    def test_action_moments_ignore_the_held_out_tail(self) -> None:
        from src.metrics.planning_benchmark import population_action_moments

        actions = np.zeros((4, 2, 2))
        actions[3] = 100.0
        mean, std = population_action_moments(actions, 3)
        self.assertTrue(np.allclose(mean, 0.0))
        self.assertTrue(np.all(std >= 1e-6))


if __name__ == "__main__":
    unittest.main()
