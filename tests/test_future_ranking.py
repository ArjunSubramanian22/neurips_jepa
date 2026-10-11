"""Synthetic embeddings with a known future-ranking answer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.future_ranking import (
    STATUS_AMBIGUOUS,
    STATUS_TIE,
    STATUS_UNDEFINED,
    bootstrap_top1,
    candidate_actions,
    control_predictions,
    derangement,
    physical_bin,
    rank_candidates,
    visibility_bin,
)


def _distinct(k: int) -> tuple[np.ndarray, np.ndarray]:
    equal = np.eye(k, dtype=bool)
    rms = np.zeros((k, k), dtype=np.float64)
    return equal, rms


class FutureRankingTests(unittest.TestCase):
    def test_oracle_embeddings_retrieve_every_distinct_future(self) -> None:
        target = np.eye(4)
        equal, rms = _distinct(4)
        scored = rank_candidates(target, target, equal, rms, distance="mse", topk=2)
        self.assertEqual(scored["summary"]["n_total"], 4)
        self.assertEqual(scored["summary"]["n_scored"], 4)
        self.assertEqual(scored["summary"]["n_ambiguous"], 0)
        self.assertEqual(scored["summary"]["top1"], 1.0)
        self.assertEqual(scored["summary"]["topk"], 1.0)
        self.assertEqual(scored["summary"]["mean_reciprocal_rank"], 1.0)
        self.assertEqual(scored["summary"]["pairwise_probability"], 1.0)
        self.assertGreater(scored["summary"]["margin_median"], 0.0)

    def test_second_nearest_has_top2_and_negative_margin(self) -> None:
        target = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
        pred = target.copy()
        pred[0] = np.array([1.2, 0.0])
        equal, rms = _distinct(4)
        scored = rank_candidates(pred, target, equal, rms, distance="l2", topk=2)
        query = scored["queries"][0]
        self.assertEqual(query["status"], "scored")
        self.assertEqual(query["top1"], 0.0)
        self.assertEqual(query["topk"], 1.0)
        self.assertEqual(query["rank"], 2)
        self.assertAlmostEqual(query["reciprocal_rank"], 0.5, places=12)
        self.assertAlmostEqual(query["margin"], 0.2 - 1.2, places=12)
        self.assertAlmostEqual(query["pairwise_rate"], 2.0 / 3.0, places=12)

    def test_identical_observations_are_ambiguous_not_misses(self) -> None:
        target = np.eye(3)
        equal = np.ones((3, 3), dtype=bool)
        rms = np.zeros((3, 3))
        scored = rank_candidates(target, target, equal, rms, distance="mse")
        self.assertEqual(scored["summary"]["n_total"], 3)
        self.assertEqual(scored["summary"]["n_scored"], 0)
        self.assertEqual(scored["summary"]["n_ambiguous"], 3)
        self.assertIsNone(scored["summary"]["top1"])
        self.assertTrue(all(query["status"] == STATUS_AMBIGUOUS for query in scored["queries"]))
        self.assertFalse(any(query["physically_distinct_observationally_identical"] for query in scored["queries"]))

    def test_hidden_physical_difference_is_flagged_and_not_scored(self) -> None:
        target = np.eye(2)
        equal = np.ones((2, 2), dtype=bool)
        rms = np.array([[0.0, 0.02], [0.02, 0.0]])
        scored = rank_candidates(target, target, equal, rms, distance="mse", topk=2)
        self.assertEqual(scored["summary"]["n_scored"], 0)
        self.assertEqual(scored["summary"]["n_physically_distinct_observationally_identical"], 2)
        self.assertIsNone(scored["summary"]["top1"])
        self.assertIsNone(scored["summary"]["topk"])
        self.assertTrue(all(query["physically_distinct_observationally_identical"] for query in scored["queries"]))

    def test_constant_prediction_hits_one_of_k(self) -> None:
        target = np.eye(4)
        pred = np.repeat(target[:1], 4, axis=0)
        equal, rms = _distinct(4)
        scored = rank_candidates(pred, target, equal, rms, distance="mse")
        self.assertEqual(scored["summary"]["n_hit"], 1)
        self.assertAlmostEqual(scored["summary"]["top1"], 0.25, places=12)

    def test_shuffled_predictions_match_chance_on_a_perfect_assignment(self) -> None:
        rng = np.random.default_rng(0)
        target = np.eye(4)
        equal, rms = _distinct(4)
        rates = []
        for _ in range(4000):
            controls = control_predictions(target, target, rng)
            scored = rank_candidates(
                controls["shuffled_predictions"], target, equal, rms, distance="mse"
            )
            rates.append(scored["summary"]["top1"])
        self.assertAlmostEqual(float(np.mean(rates)), 0.25, delta=0.025)

    def test_zero_prediction_leaves_cosine_undefined_and_mse_scored(self) -> None:
        target = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        pred = target.copy()
        pred[0] = 0.0
        equal, rms = _distinct(3)
        cosine = rank_candidates(pred, target, equal, rms, distance="cosine")
        mse = rank_candidates(pred, target, equal, rms, distance="mse")
        self.assertEqual(cosine["queries"][0]["status"], STATUS_UNDEFINED)
        self.assertIsNone(cosine["queries"][0]["top1"])
        self.assertEqual(cosine["summary"]["n_undefined"], 1)
        self.assertEqual(cosine["summary"]["n_scored"], 2)
        self.assertEqual(mse["queries"][0]["status"], "scored")
        self.assertEqual(mse["queries"][0]["top1"], 0.0)
        self.assertEqual(mse["summary"]["n_total"], 3)

    def test_distance_tie_is_not_a_top1_hit(self) -> None:
        target = np.stack([np.ones(2), np.ones(2), np.array([0.0, 4.0])])
        equal, rms = _distinct(3)
        scored = rank_candidates(target, target, equal, rms, distance="mse")
        self.assertEqual(scored["queries"][0]["status"], STATUS_TIE)
        self.assertEqual(scored["queries"][0]["top1"], 0.0)
        self.assertEqual(scored["summary"]["n_tie"], 2)
        self.assertEqual(scored["summary"]["n_scored"], 3)

    def test_variance_normalized_margin_scales_mse(self) -> None:
        target = np.eye(3)
        equal, rms = _distinct(3)
        mse = rank_candidates(target, target, equal, rms, distance="mse")
        scaled = rank_candidates(target, target, equal, rms, distance="vn_mse", ref_variance=4.0)
        self.assertEqual(mse["summary"]["top1"], scaled["summary"]["top1"])
        self.assertAlmostEqual(scaled["summary"]["margin_median"], mse["summary"]["margin_median"] / 4.0, places=12)

    def test_zero_force_candidates_are_kept(self) -> None:
        actions, roles = candidate_actions(np.array([0.2, 0.4, 0.0, 0.0]), np.array([0.1, 0.2, 0.3, -0.2]))
        self.assertEqual(roles, ("recorded", "shuffled", "force_reversed", "force_half"))
        self.assertTrue(np.allclose(actions[0], actions[2]))
        self.assertTrue(np.allclose(actions[0], actions[3]))
        self.assertEqual(actions.shape[0], 4)

    def test_derangement_has_no_fixed_point(self) -> None:
        perm = derangement(50, np.random.default_rng(1))
        self.assertTrue(np.all(perm != np.arange(50)))

    def test_visibility_bin_keeps_ambiguity_out_of_the_low_bin(self) -> None:
        self.assertEqual(visibility_bin(False, 0.0, high_visibility=0.05), "ambiguous")
        self.assertEqual(visibility_bin(True, 0.04, high_visibility=0.05), "low_visibility")
        self.assertEqual(visibility_bin(True, 0.05, high_visibility=0.05), "high_visibility")
        self.assertEqual(physical_bin(0.0, small_rms=1e-3), "zero_displacement")
        self.assertEqual(physical_bin(1e-4, small_rms=1e-3), "small_displacement")
        self.assertEqual(physical_bin(0.01, small_rms=1e-3), "large_displacement")

    def test_bootstrap_lists_episodes_with_no_scored_query(self) -> None:
        result = bootstrap_top1(
            [0, 0, 1],
            [1.0, 0.0, None],
            ["scored", "distance_tie", "observationally_ambiguous"],
            n_resamples=200,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(result["n_episodes_with_scored"], 1)
        self.assertEqual(result["excluded_episode_ids"], [1])
        self.assertAlmostEqual(result["bootstrap"]["mean"], 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
