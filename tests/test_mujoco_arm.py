"""The arm restores exactly, and the three tasks separate contact from appearance."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.arm_audit import audit_task
from src.envs.mujoco_arm import FOLD, PUSH, MujocoArm


class MujocoArmTests(unittest.TestCase):
    def test_set_state_reproduces_the_next_step_exactly(self) -> None:
        env = MujocoArm("push_object")
        env.reset(0)
        env.step(PUSH)
        saved = env.get_state()
        env.step(FOLD)
        continued = env.get_state()
        env.set_state(saved)
        env.step(FOLD)
        restored = env.get_state()
        self.assertTrue(np.array_equal(restored["qpos"], continued["qpos"]))
        self.assertTrue(np.array_equal(restored["qvel"], continued["qvel"]))
        self.assertTrue(np.array_equal(restored["vector"], continued["vector"]))
        env.close()

    def test_same_seed_and_actions_match(self) -> None:
        first = MujocoArm("push_object")
        second = MujocoArm("push_object")
        first.reset(3)
        second.reset(3)
        action = first.sample_action()
        self.assertTrue(np.array_equal(action, second.sample_action()))
        first.step(PUSH)
        second.step(PUSH)
        self.assertTrue(np.array_equal(first.get_state()["qpos"], second.get_state()["qpos"]))
        first.close()
        second.close()

    def test_push_moves_the_object_and_a_fold_does_not(self) -> None:
        push = audit_task("push_object", seed=0)
        fold = audit_task("fold_clear", seed=0)
        cover = audit_task("cover_post", seed=0)
        pushed = push["positive_vs_baseline"]
        self.assertGreater(pushed["physical"]["rms_pos"], 0.05)
        self.assertGreater(pushed["observation"]["mse"], 0.01)
        self.assertLess(pushed["observation"]["fraction_identical"], 0.95)
        self.assertEqual(len(pushed["observation"]["mse_per_channel"]), 3)
        self.assertEqual(pushed["direction"]["oracle_status"], "ok")
        self.assertAlmostEqual(pushed["direction"]["oracle_cosine"], 1.0)
        self.assertEqual(pushed["direction"]["constant_status"], "undefined_zero_vector")
        self.assertEqual(push["retrieval"]["oracle_copies_target"]["top1"], 1.0)
        self.assertLess(push["retrieval"]["constant_ignores_action"]["top1"], 1.0)
        self.assertEqual(push["planning"]["oracle"]["chosen"], "push")
        self.assertEqual(push["planning"]["learned"]["status"], "not_measured")
        folded = fold["positive_vs_baseline"]
        self.assertLess(folded["physical"]["rms_pos"], 1e-8)
        self.assertGreater(folded["observation"]["mse"], 0.01)
        self.assertEqual(fold["planning"]["oracle"]["chosen"], "fold")
        costs = cover["planning"]["oracle"]["costs"]
        self.assertLess(costs["fold"], costs["hold"] * 0.85)
        self.assertEqual(cover["planning"]["oracle"]["chosen"], "fold")
        self.assertGreater(push["positions"]["push"][0], push["positions"]["hold"][0] + 0.05)

    def test_a_copied_future_uses_the_shared_learned_metrics(self) -> None:
        def represent(action: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            del action
            last = np.asarray(frames[-1], dtype=np.float64).reshape(-1)
            return last, last

        report = audit_task("push_object", seed=0, represent=represent)
        learned = report["positive_vs_baseline"]["direction"]["learned"]
        self.assertEqual(learned["status"], "ok")
        self.assertAlmostEqual(learned["cosine"], 1.0)
        self.assertEqual(report["retrieval"]["learned"]["top1"], 1.0)
        self.assertEqual(report["planning"]["learned"]["chosen"], "push")

        def ignore(action: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            del action
            start = np.asarray(frames[0], dtype=np.float64).reshape(-1)
            last = np.asarray(frames[-1], dtype=np.float64).reshape(-1)
            return start, last

        blind = audit_task("push_object", seed=0, represent=ignore)
        self.assertEqual(blind["planning"]["learned"]["status"], "ambiguous_tie")
        self.assertIsNone(blind["planning"]["learned"]["chosen"])


if __name__ == "__main__":
    unittest.main()
