"""MuJoCo box-push state restoration is exact, not approximate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.mujoco_box_push import MujocoBoxPush


class MujocoRestoreTests(unittest.TestCase):
    def test_same_seed_resets_to_the_same_state(self) -> None:
        first = MujocoBoxPush("slide_x")
        second = MujocoBoxPush("slide_x")
        first.reset(7)
        second.reset(7)
        self.assertTrue(np.array_equal(first.get_state()["vector"], second.get_state()["vector"]))
        self.assertTrue(np.array_equal(first.sample_action(), second.sample_action()))
        first.close()
        second.close()

    def test_set_state_reproduces_the_next_step_exactly(self) -> None:
        env = MujocoBoxPush("slide_x")
        env.reset(0)
        action = np.array([0.008, 0.0])
        other = np.array([0.0, 0.004])
        env.step(action)
        saved = env.get_state()
        env.step(other)
        continued = env.get_state()
        env.set_state(saved)
        env.step(other)
        restored = env.get_state()
        self.assertTrue(np.array_equal(restored["qpos"], continued["qpos"]))
        self.assertTrue(np.array_equal(restored["qvel"], continued["qvel"]))
        self.assertTrue(np.array_equal(restored["vector"], continued["vector"]))
        self.assertGreater(env.action_distance(action, other), 0.0)
        env.close()

    def test_a_push_moves_the_box_and_a_miss_does_not(self) -> None:
        env = MujocoBoxPush("slide_x")
        env.reset(1)
        start = env.get_state()
        env.step(np.array([0.008, 0.0]))
        for _ in range(11):
            env.step(np.array([0.008, 0.0]))
        pushed = env.box_xy()
        self.assertGreater(env.physical_state_distance(start)["box_rms_pos"], 0.02)
        env.set_state(start)
        for _ in range(12):
            env.step(np.array([0.0, 0.01]))
        missed = env.box_xy()
        env.set_state(start)
        for _ in range(12):
            env.step(np.array([0.0, 0.0]))
        held = env.box_xy()
        self.assertGreater(pushed[0], held[0] + 0.02)
        self.assertAlmostEqual(missed[0], held[0], places=6)
        env.close()


if __name__ == "__main__":
    unittest.main()
