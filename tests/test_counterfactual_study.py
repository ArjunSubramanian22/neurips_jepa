"""Guards for the pre-test counterfactual study. No test-set reads."""

from __future__ import annotations

import unittest

import numpy as np

from src.counterfactual_study import (
    assert_high_validation_episodes,
    bootstrap_median,
    classify_bottleneck,
    episode_aggregate,
    pair_metrics,
    visibility_threshold,
)


class CounterfactualStudyTest(unittest.TestCase):
    def test_high_validation_rejects_the_test_suffix(self) -> None:
        self.assertEqual(assert_high_validation_episodes([1440, 1619]), [1440, 1619])
        with self.assertRaises(ValueError):
            assert_high_validation_episodes([1619, 1620])
        with self.assertRaises(ValueError):
            assert_high_validation_episodes([1439])

    def test_undefined_cosine_stays_null(self) -> None:
        metrics = pair_metrics(np.zeros(4), np.ones(4))
        self.assertIsNone(metrics["cosine"])
        self.assertEqual(metrics["cosine_status"], "undefined_zero_vector")
        self.assertAlmostEqual(metrics["response_ratio"], 0.0)

    def test_episode_bootstrap_does_not_count_branch_times(self) -> None:
        rows = []
        for episode in (1, 2):
            for branch, cosine in ((0, 0.0), (1, 1.0)):
                rows.append({"episode": episode, "cosine": cosine, "cosine_status": "ok", "pred_norm": 1.0})
        aggregated = episode_aggregate(rows[:2], ("cosine", "pred_norm"))
        self.assertEqual(aggregated["n_branch"], 2)
        self.assertAlmostEqual(aggregated["cosine"], 0.5)
        summary = bootstrap_median([0.0, 1.0], np.random.default_rng(0), n_resamples=100)
        self.assertEqual(summary["n_episodes"], 2)

    def test_bottleneck_order(self) -> None:
        self.assertEqual(classify_bottleneck({"rms_pos": 0.0, "identical_raster": 0.0, "target_norm": 1.0, "cosine": 1.0, "response_ratio": 1.0}), "PHYSICAL_FAILURE")
        self.assertEqual(classify_bottleneck({"rms_pos": 0.01, "identical_raster": 1.0, "target_norm": 1.0, "cosine": 1.0, "response_ratio": 1.0}), "OBSERVATION_BOTTLENECK")
        self.assertEqual(classify_bottleneck({"rms_pos": 0.01, "identical_raster": 0.0, "target_norm": 0.0, "cosine": None, "response_ratio": None}), "REPRESENTATION_BOTTLENECK")
        self.assertEqual(classify_bottleneck({"rms_pos": 0.01, "identical_raster": 0.0, "target_norm": 1.0, "cosine": 0.1, "response_ratio": 0.1}), "PREDICTOR_BOTTLENECK")
        self.assertEqual(classify_bottleneck({"rms_pos": 0.01, "identical_raster": 0.0, "target_norm": 1.0, "cosine": 0.8, "response_ratio": 0.9}), "SUCCESSFUL_COUNTERFACTUAL_TRACKING")

    def test_visibility_threshold_is_shared(self) -> None:
        threshold = visibility_threshold(np.asarray([0.0, 1.0, 2.0, 3.0]))
        self.assertEqual(threshold["n_nonzero"], 3)
        self.assertGreater(threshold["top_third_mse"], 1.0)


if __name__ == "__main__":
    unittest.main()
