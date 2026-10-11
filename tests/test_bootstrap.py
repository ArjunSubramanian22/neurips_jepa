"""Sanity checks for the equal-episode bootstrap."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats import episode_bootstrap_mean


class BootstrapTests(unittest.TestCase):
    def test_constant_episodes_have_a_degenerate_interval(self) -> None:
        values = np.full((12, 4), 3.0)
        result = episode_bootstrap_mean(values, n_resamples=200, rng=np.random.default_rng(0))
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["ci_low"], 3.0)
        self.assertEqual(result["ci_high"], 3.0)
        self.assertEqual(result["n_undefined_dropped"], 0)
        self.assertEqual(result["n_episodes"], 12)

    def test_interval_contains_the_point_estimate(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(loc=np.arange(8)[:, None], scale=0.1, size=(8, 5))
        result = episode_bootstrap_mean(values, n_resamples=500, rng=np.random.default_rng(2))
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertGreaterEqual(result["ci_high"], result["mean"])
        self.assertEqual(result["weighting"], "equal_episode")

    def test_non_finite_values_are_not_dropped(self) -> None:
        values = np.ones((4, 2))
        values[1, 0] = np.nan
        with self.assertRaises(ValueError):
            episode_bootstrap_mean(values, n_resamples=10, rng=np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
