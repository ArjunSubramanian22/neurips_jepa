"""The evaluator and planner run on a toy fork without a simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.adapters import assert_adapter_identity
from benchmark.environments import ForkableEnvironment, register_env, register_task_spec, resolve_split
from benchmark.evaluator import CounterfactualEvaluator
from benchmark.planner import Planner
from benchmark.schema import validate_result


class _Toy:
    environment_id = "toy"

    def __init__(self, task: str) -> None:
        self.task = task
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)

    def get_state(self) -> dict:
        vector = np.concatenate([self.pos, self.vel])
        return {"spec": 1, "vector": vector.copy(), "qpos": self.pos.copy(), "qvel": self.vel.copy()}

    def set_state(self, state: dict) -> None:
        vector = np.asarray(state["vector"], dtype=np.float64)
        self.pos = vector[:2].copy()
        self.vel = vector[2:4].copy()

    def step(self, action) -> None:
        delta = np.asarray(action, dtype=np.float64).reshape(-1)
        self.vel = delta[:2]
        self.pos = self.pos + delta[:2]

    def render(self) -> np.ndarray:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = np.uint8(np.clip(self.pos[0] * 20 + 30, 0, 255))
        image[:, :, 1] = np.uint8(np.clip(self.pos[1] * 20 + 30, 0, 255))
        return image

    def tracked_positions(self) -> np.ndarray:
        return self.pos.reshape(1, 2).copy()

    def tracked_velocities(self) -> np.ndarray:
        return self.vel.reshape(1, 2).copy()

    def sample_action(self) -> np.ndarray:
        return self._rng.normal(size=2)

    def physical_state_distance(self, other: dict) -> dict:
        return {"rms": float(np.linalg.norm(self.pos - np.asarray(other["qpos"])))}

    def close(self) -> None:
        return None


def _register() -> None:
    register_env("toy", lambda task: ForkableEnvironment(_Toy(task)), ("slide",), {"val": ("validation", False)})

    def spec(_task: str) -> dict:
        return {
            "actions": {
                "along": np.array([0.4, 0.0]),
                "hold": np.zeros(2),
                "miss": np.array([0.0, 0.4]),
            },
            "positive": "along",
            "baseline": "hold",
            "alternate": "miss",
            "horizon": 2,
            "goal_cost": lambda _name, future: abs(float(future["positions"][0, 0]) - 1.0),
            "learned_reason": "toy has no checkpoint",
        }

    register_task_spec("toy", spec)


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _register()

    def test_clone_and_restore_round_trip(self) -> None:
        env = ForkableEnvironment(_Toy("slide"))
        env.reset(1)
        env.step(np.array([0.2, 0.0]))
        state = env.clone_state()
        env.step(np.array([0.0, 0.3]))
        env.restore_state(state)
        self.assertTrue(np.array_equal(env.tracked_positions(), state["qpos"].reshape(1, 2)))

    def test_layers_match_the_cascade_functions(self) -> None:
        evaluator = CounterfactualEvaluator()
        physical = evaluator.physical_response([[0.0, 0.0]], [[3.0, 4.0]])
        self.assertAlmostEqual(physical["rms_pos"], 5.0)
        image = evaluator.observation_visibility(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(image["mse"], 0.0)
        direction = evaluator.direction_metrics([1.0, 0.0], [0.0, 1.0])
        self.assertEqual(direction["status"], "ok")
        self.assertAlmostEqual(direction["cosine"], 0.0)

    def test_a_toy_fork_fills_the_schema(self) -> None:
        record = CounterfactualEvaluator().run("toy", "slide", split="val", seed=0)
        validate_result(record)
        self.assertFalse(record["is_test_set"])
        self.assertEqual(record["split_role"], "validation")
        layer = record["results"][0]["layers"]
        self.assertGreater(layer["physical_response"]["rms_pos"], 0.0)
        self.assertGreater(layer["observation_visibility"]["mse"], 0.0)
        self.assertEqual(layer["planning"]["oracle"]["chosen"], "along")
        self.assertEqual(layer["planning"]["learned"]["status"], "not_measured")
        self.assertEqual(layer["planning"]["status"], "not_measured")
        self.assertEqual(layer["predictor_response"]["status"], "not_measured")
        self.assertEqual(layer["predictor_response"]["oracle_copies_target"]["status"], "ok")
        self.assertEqual(layer["multi_step_metrics"]["horizon"], 2)

    def test_an_undeclared_test_split_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            resolve_split("mujoco_arm", "test")

    def test_a_tie_chooses_nothing(self) -> None:
        choice = Planner().plan({"push": 1.0, "fold": 1.0})
        self.assertEqual(choice["status"], "ambiguous_tie")
        self.assertIsNone(choice["chosen"])

    def test_adapter_names_follow_loss_and_action_conditioning(self) -> None:
        assert_adapter_identity("jepa", "ACJEPA", "vicreg", True)
        assert_adapter_identity("mse", "ACJEPA", "mse", True)
        assert_adapter_identity("noaction", "ACJEPA", "vicreg", False)
        with self.assertRaises(ValueError):
            assert_adapter_identity("jepa", "ACJEPA", "mse", True)
        with self.assertRaises(ValueError):
            assert_adapter_identity("jepa", "ACJEPA", "vicreg", False)
        with self.assertRaises(ValueError):
            assert_adapter_identity("obs_pred", "ActionConditionedUNet", "obs_pred", True)


if __name__ == "__main__":
    unittest.main()
