"""A tiny arm dataset must move, render, replay, and be plannable before training."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.arm_data import generate_arm_dataset, validate_arm_dataset


class ArmDatasetTests(unittest.TestCase):
    def test_three_kinds_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generate_arm_dataset(tmp, n_episodes=6, length=4)
            report = validate_arm_dataset(tmp)
        self.assertEqual(report["status"], "passed")
        self.assertGreater(report["push_object_displacement_min"], 0.05)
        self.assertLess(report["fold_object_displacement_max"], 1e-6)
        self.assertTrue(report["replay_exact"])
        self.assertFalse(report["is_test_set"])
        self.assertEqual(report["learned_model"], "not_measured")


if __name__ == "__main__":
    unittest.main()
