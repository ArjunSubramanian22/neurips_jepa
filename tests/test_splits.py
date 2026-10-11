"""Disjointness checks for historical and fresh-pool episode splits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import _episode_split_counts
from src.splits import fresh_pool_v2_split, historical_split, historical_val_counts


class SplitTests(unittest.TestCase):
    def test_historical_counts_match_the_loader(self) -> None:
        for n_episodes, fraction in ((2, 0.1), (12, 0.25), (200, 0.1), (2000, 0.1)):
            self.assertEqual(
                historical_val_counts(n_episodes, fraction),
                _episode_split_counts(n_episodes, fraction),
            )

    def test_historical_dump_has_no_test_and_no_overlap(self) -> None:
        split = historical_split(2000, 0.1)
        split.assert_disjoint()
        self.assertEqual(split.train, (0, 1800))
        self.assertEqual(split.val, (1800, 2000))
        self.assertFalse(split.test_defined)
        self.assertIsNone(split.test)
        self.assertEqual(len(split.ids("train") & split.ids("val")), 0)
        self.assertEqual(split.ids("test"), set())

    def test_fresh_pool_partitions_without_overlap(self) -> None:
        split = fresh_pool_v2_split(120)
        split.assert_disjoint()
        self.assertEqual(split.train, (0, 96))
        self.assertEqual(split.val, (96, 108))
        self.assertEqual(split.test, (108, 120))
        self.assertTrue(split.ids("train").isdisjoint(split.ids("val")))
        self.assertTrue(split.ids("train").isdisjoint(split.ids("test")))
        self.assertTrue(split.ids("val").isdisjoint(split.ids("test")))
        for name in ("train", "val", "test"):
            self.assertEqual(len(split.ids(name)) % 6, 0)

    def test_fresh_pool_refuses_the_historical_dump_size(self) -> None:
        with self.assertRaises(ValueError):
            fresh_pool_v2_split(2000)

    def test_fresh_pool_refuses_a_short_pool(self) -> None:
        with self.assertRaises(ValueError):
            fresh_pool_v2_split(30)


if __name__ == "__main__":
    unittest.main()
