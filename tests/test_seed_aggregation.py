"""Seed summaries stay descriptive until every registry seed is complete."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats.seed_aggregation import aggregate_seeds, paired_seed_comparison

SEEDS = [42, 43, 44, 45, 46]


def _complete(seed: int, value: float) -> dict:
    return {"seed": seed, "status": "COMPLETE", "value": value}


class SeedAggregationTests(unittest.TestCase):
    def test_two_complete_seeds_do_not_produce_a_confidence_interval(self) -> None:
        records = [_complete(42, 1.0), _complete(43, 3.0)]
        summary = aggregate_seeds(records, SEEDS)
        self.assertIsNone(summary["training_seed_ci"])
        self.assertFalse(summary["half_range_is_confidence_interval"])
        self.assertNotIn("half_range", summary)
        self.assertEqual(summary["n_complete"], 2)
        self.assertEqual(summary["n_not_run"], 3)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["median"], 2.0)
        self.assertAlmostEqual(summary["std"], 2.0 ** 0.5)

    def test_failed_seed_is_retained_and_excluded_from_the_mean(self) -> None:
        records = [_complete(seed, float(seed)) for seed in (42, 43, 44, 45)]
        records.append({"seed": 46, "status": "FAILED", "value": 1000.0})
        summary = aggregate_seeds(records, SEEDS)
        self.assertEqual(summary["failed_seeds"], [46])
        self.assertEqual(summary["n_failed"], 1)
        self.assertAlmostEqual(summary["mean"], (42 + 43 + 44 + 45) / 4)
        self.assertEqual(summary["seeds"][-1]["value"], 1000.0)
        self.assertIsNone(summary["training_seed_ci"])

    def test_extra_seed_outside_the_registry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_seeds([_complete(99, 1.0)], SEEDS)

    def test_paired_effect_uses_only_seeds_completed_on_both_sides(self) -> None:
        left = [_complete(42, 1.0), _complete(43, 2.0), _complete(44, 3.0), _complete(45, 4.0)]
        right = [_complete(42, 1.0), _complete(43, 1.0), _complete(44, 1.0), _complete(45, 1.0)]
        compared = paired_seed_comparison(left, right, SEEDS)
        self.assertEqual(compared["paired_seeds"], [42, 43, 44, 45])
        self.assertEqual(compared["unpaired_seeds"], [46])
        self.assertAlmostEqual(compared["effect_size"]["mean_difference"], 1.5)
        self.assertEqual(compared["effect_size"]["status"], "descriptive")
        self.assertIsNone(compared["training_seed_ci"])
        self.assertEqual(compared["training_seed_ci_status"], "insufficient_training_seeds")


if __name__ == "__main__":
    unittest.main()
