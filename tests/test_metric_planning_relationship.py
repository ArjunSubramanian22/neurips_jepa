"""Correlations stay undefined when a planning outcome does not vary."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis" / "metric_planning_relationship.py"
spec = importlib.util.spec_from_file_location("metric_planning_relationship", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["metric_planning_relationship"] = module
spec.loader.exec_module(module)


class MetricPlanningTests(unittest.TestCase):
    def test_tied_ranks_do_not_invent_a_correlation(self) -> None:
        result = module.spearman([1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["value"], 1.0)

    def test_constant_planning_outcome_is_undefined(self) -> None:
        result = module.spearman([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(result["value"])
        self.assertEqual(result["status"], "undefined_zero_variance")
        fit = module.loo_r2([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(fit["value"])

    def test_bootstrap_interval_stays_positive_for_a_monotone_line(self) -> None:
        values = np.arange(8, dtype=np.float64)
        interval = module.bootstrap_spearman_ci(values, values * 2.0 + 1.0, draws=200, seed=0)
        self.assertGreater(interval["low"], 0.9)

    def test_headline_does_not_claim_a_winner_when_intervals_cover_zero(self) -> None:
        rows = [
            {
                "metric": "one_step_latent_mse",
                "group": "conventional",
                "ci_excludes_zero_and_sign_matches": False,
            }
        ]
        sentence = module.headline_sentence(rows, "undefined_zero_variance")
        self.assertIn("no correlation", sentence)
        self.assertIn("No Spearman interval", sentence)
        self.assertIn("better-planning", sentence)


if __name__ == "__main__":
    unittest.main()
