"""Metric invariants on synthetic inputs, not on a training run."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import effective_rank
from src.eval_protocol import ProtocolError, fit_visibility_threshold
from src.metrics.cascade import cosine_alignment
from src.metrics.future_ranking import rank_candidates
from src.metrics.visibility_sensitivity import select_visibility
from src.splits import cvpr_v1_split


class CosineInvariantTests(unittest.TestCase):
    def test_known_angles(self) -> None:
        axis = np.array([3.0, -4.0])
        parallel = cosine_alignment(axis, 2.0 * axis)
        opposite = cosine_alignment(axis, -5.0 * axis)
        orthogonal = cosine_alignment(np.array([1.0, 0.0]), np.array([0.0, 2.0]))
        self.assertEqual(parallel["status"], "ok")
        self.assertEqual(opposite["status"], "ok")
        self.assertEqual(orthogonal["status"], "ok")
        self.assertAlmostEqual(parallel["cosine"], 1.0, places=12)
        self.assertAlmostEqual(opposite["cosine"], -1.0, places=12)
        self.assertAlmostEqual(orthogonal["cosine"], 0.0, places=12)

    def test_a_zero_vector_is_not_stored_as_zero_cosine(self) -> None:
        result = cosine_alignment(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        self.assertIsNone(result["cosine"])
        self.assertEqual(result["status"], "undefined_zero_vector")
        self.assertNotEqual(result["cosine"], 0.0)


class RankingInvariantTests(unittest.TestCase):
    def test_the_true_future_ranks_first_when_predictions_match_it(self) -> None:
        target = np.eye(3, dtype=np.float64)
        obs_equal = np.eye(3, dtype=bool)
        physical = np.ones((3, 3), dtype=np.float64) - np.eye(3)
        summary = rank_candidates(
            target,
            target,
            obs_equal,
            physical,
            distance="mse",
            topk=1,
        )["summary"]
        self.assertEqual(summary["n_scored"], 3)
        self.assertEqual(summary["top1"], 1.0)
        self.assertGreater(summary["top1"], summary["chance_top1"])

    def test_a_swapped_pair_is_not_ranked_as_a_hit(self) -> None:
        target = np.eye(2, dtype=np.float64)
        pred = target[::-1].copy()
        summary = rank_candidates(
            pred,
            target,
            np.eye(2, dtype=bool),
            np.ones((2, 2)) - np.eye(2),
            distance="mse",
            topk=1,
        )["summary"]
        self.assertEqual(summary["n_scored"], 2)
        self.assertEqual(summary["top1"], 0.0)
        self.assertLess(summary["top1"], summary["chance_top1"])


class EffectiveRankInvariantTests(unittest.TestCase):
    def test_a_line_has_participation_near_one_and_a_square_uses_both_axes(self) -> None:
        line = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
        square = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        collapsed = effective_rank(line)
        filled = effective_rank(square)
        self.assertAlmostEqual(collapsed["participation_ratio"], 1.0, places=5)
        self.assertGreater(filled["participation_ratio"], 1.5)
        self.assertLessEqual(filled["participation_ratio"], filled["latent_dim"] + 1e-5)
        self.assertLess(collapsed["participation_ratio"], filled["participation_ratio"])


class VisibilityThresholdTests(unittest.TestCase):
    def test_threshold_keeps_values_on_the_declared_side_and_drops_nonfinite(self) -> None:
        values = np.array([0.0, 0.5, 1.0, np.nan])
        at_least = select_visibility(values, threshold=0.5, comparison=">=")
        above = select_visibility(values, threshold=0.5, comparison=">")
        missing = select_visibility(values, threshold=None, comparison=">=")
        np.testing.assert_array_equal(at_least, np.array([False, True, True, False]))
        np.testing.assert_array_equal(above, np.array([False, False, True, False]))
        self.assertFalse(bool(missing.any()))

    def test_a_threshold_fit_on_validation_does_not_see_the_test_set(self) -> None:
        split = cvpr_v1_split()
        values = {episode: float(index) for index, episode in enumerate(split.val_ids)}
        fitted = fit_visibility_threshold(values, split, role="val", quantile=0.5)
        self.assertFalse(fitted["saw_test"])
        self.assertEqual(fitted["role"], "val")
        self.assertEqual(fitted["n_episodes"], len(split.val_ids))
        leaked = dict(values)
        leaked[split.test_ids[0]] = 1.0
        with self.assertRaises(ProtocolError):
            fit_visibility_threshold(leaked, split, role="val", quantile=0.5)


if __name__ == "__main__":
    unittest.main()
