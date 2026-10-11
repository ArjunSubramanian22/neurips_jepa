"""Hierarchical bootstrap weights seeds equally and withholds incomplete registries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats.hierarchical_bootstrap import hierarchical_bootstrap_mean

SEEDS = [42, 43, 44, 45, 46]


class HierarchicalBootstrapTests(unittest.TestCase):
    def test_equal_seed_weight_and_exact_resample(self) -> None:
        episodes = {
            42: [0.0, 0.0, 0.0, 0.0],
            43: [10.0],
            44: [5.0, 5.0],
            45: [5.0],
            46: [5.0, 5.0, 5.0],
        }
        status = {seed: "COMPLETE" for seed in SEEDS}
        result = hierarchical_bootstrap_mean(
            episodes,
            required_seeds=SEEDS,
            seed_status=status,
            n_resamples=30,
            rng=np.random.default_rng(0),
        )
        interval = result["training_seed_interval"]
        self.assertEqual(interval["status"], "ok")
        self.assertEqual(interval["interval_kind"], "training_seed")
        self.assertAlmostEqual(interval["mean"], 5.0)
        self.assertLessEqual(interval["ci_low"], interval["mean"])
        self.assertGreaterEqual(interval["ci_high"], interval["mean"])
        arrays = [np.asarray(episodes[seed], dtype=np.float64) for seed in SEEDS]
        rng = np.random.default_rng(0)
        replicates = []
        for _ in range(30):
            seed_index = rng.integers(0, 5, size=5)
            seed_means = []
            for index in seed_index.tolist():
                values = arrays[index]
                episode_index = rng.integers(0, values.size, size=values.size)
                seed_means.append(float(values[episode_index].mean()))
            replicates.append(float(np.mean(seed_means)))
        self.assertAlmostEqual(interval["ci_low"], float(np.quantile(replicates, 0.025)))
        self.assertAlmostEqual(interval["ci_high"], float(np.quantile(replicates, 0.975)))
        self.assertEqual(result["conditional_on_checkpoint"]["42"]["interval_kind"], "conditional_on_checkpoint")

    def test_constant_seeds_have_a_degenerate_training_interval(self) -> None:
        episodes = {seed: [3.0, 3.0] for seed in SEEDS}
        result = hierarchical_bootstrap_mean(
            episodes,
            required_seeds=SEEDS,
            seed_status={seed: "COMPLETE" for seed in SEEDS},
            n_resamples=20,
            rng=np.random.default_rng(1),
        )
        interval = result["training_seed_interval"]
        self.assertEqual(interval["ci_low"], 3.0)
        self.assertEqual(interval["ci_high"], 3.0)

    def test_missing_seed_blocks_the_training_interval_but_keeps_episode_intervals(self) -> None:
        episodes = {seed: [float(seed)] for seed in (42, 43, 44, 45)}
        status = {42: "COMPLETE", 43: "COMPLETE", 44: "COMPLETE", 45: "COMPLETE", 46: "NOT_RUN"}
        result = hierarchical_bootstrap_mean(
            episodes,
            required_seeds=SEEDS,
            seed_status=status,
            n_resamples=20,
            rng=np.random.default_rng(2),
        )
        interval = result["training_seed_interval"]
        self.assertEqual(interval["status"], "insufficient_training_seeds")
        self.assertIsNone(interval["ci_low"])
        self.assertIsNone(interval["ci_high"])
        self.assertEqual(interval["not_run_seeds"], [46])
        self.assertIn("42", result["conditional_on_checkpoint"])
        self.assertNotIn("46", result["conditional_on_checkpoint"])

    def test_non_finite_episode_is_not_dropped(self) -> None:
        episodes = {seed: [1.0] for seed in SEEDS}
        episodes[42] = [1.0, np.nan]
        with self.assertRaises(ValueError):
            hierarchical_bootstrap_mean(
                episodes,
                required_seeds=SEEDS,
                seed_status={seed: "COMPLETE" for seed in SEEDS},
                n_resamples=5,
                rng=np.random.default_rng(3),
            )


if __name__ == "__main__":
    unittest.main()
