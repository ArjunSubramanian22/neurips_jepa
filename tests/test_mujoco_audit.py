"""The box-push forks separate the diagnostic layers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.mujoco_audit import audit_task


class MujocoAuditTests(unittest.TestCase):
    def test_each_task_separates_contact_from_a_miss(self) -> None:
        reports = {}
        for task in ("slide_x", "slide_y", "glance_x"):
            report = audit_task(task, seed=0)
            pair = report["along_vs_hold"]
            self.assertGreater(pair["physical"]["rms_pos"], 0.02, task)
            self.assertGreater(pair["observation"]["mse"], 0.0, task)
            self.assertLess(pair["observation"]["fraction_identical"], 1.0, task)
            self.assertGreater(pair["target_latent"]["delta_l2"], 0.02, task)
            self.assertEqual(pair["direction"]["oracle_status"], "ok")
            self.assertAlmostEqual(pair["direction"]["oracle_cosine"], 1.0)
            self.assertEqual(pair["direction"]["constant_status"], "undefined_zero_vector")
            self.assertIsNone(pair["direction"]["constant_cosine"])
            self.assertEqual(pair["predictor"]["learned"]["status"], "not_measured")
            self.assertEqual(report["retrieval"]["oracle_copies_target"]["top1"], 1.0)
            self.assertLess(report["retrieval"]["constant_ignores_action"]["top1"], 1.0)
            self.assertEqual(report["planning"]["oracle_chosen"], "along")
            self.assertEqual(report["planning"]["learned"]["status"], "not_measured")
            self.assertEqual(len(pair["observation"]["mse_per_channel"]), 3)
            reports[task] = report
        self.assertLess(
            reports["glance_x"]["box_xy"]["along"][0],
            reports["slide_x"]["box_xy"]["along"][0] - 0.2,
        )


if __name__ == "__main__":
    unittest.main()
