"""Synthetic checks for directional counterfactual metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.counterfactual_direction import (
    CONTRAST_NEAR_ZERO,
    CONTRAST_VALID,
    bootstrap_valid_cosine,
    directional_contrast,
    summarize_metric,
)


class DirectionalContrastTests(unittest.TestCase):
    def test_identical_and_scaled_deltas_have_cosine_one(self) -> None:
        target = np.array([1.0, 2.0, 0.0])
        identical = directional_contrast(target, target, fraction_affected=[1.0])
        scaled = directional_contrast(3.0 * target, target, fraction_affected=[1.0])
        self.assertEqual(identical["contrast_status"][0], CONTRAST_VALID)
        self.assertEqual(identical["cosine_status"][0], "ok")
        self.assertAlmostEqual(float(identical["cosine"][0]), 1.0, places=12)
        self.assertAlmostEqual(float(scaled["cosine"][0]), 1.0, places=12)
        self.assertAlmostEqual(float(identical["normalized_vector_error"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(identical["angular_error_rad"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(scaled["magnitude_ratio"][0]), 3.0, places=6)
        self.assertAlmostEqual(float(identical["signed_projection"][0]), float(np.linalg.norm(target)), places=12)

    def test_opposite_deltas_have_cosine_minus_one(self) -> None:
        target = np.array([0.5, -1.0, 0.25])
        result = directional_contrast(-target, target, fraction_affected=[0.4])
        self.assertAlmostEqual(float(result["cosine"][0]), -1.0, places=12)
        self.assertAlmostEqual(float(result["angular_error_rad"][0]), np.pi, places=12)
        self.assertAlmostEqual(float(result["magnitude_ratio"][0]), 1.0, places=6)
        self.assertLess(float(result["signed_projection"][0]), 0.0)
        self.assertAlmostEqual(float(result["normalized_vector_error"][0]), 4.0, places=6)

    def test_orthogonal_deltas_have_cosine_zero(self) -> None:
        result = directional_contrast(np.array([1.0, 0.0]), np.array([0.0, 2.0]), fraction_affected=[0.2])
        self.assertAlmostEqual(float(result["cosine"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(result["angular_error_rad"][0]), 0.5 * np.pi, places=12)
        self.assertAlmostEqual(float(result["signed_projection"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(result["magnitude_ratio"][0]), 0.5, places=6)

    def test_zero_target_is_excluded_and_still_counted(self) -> None:
        result = directional_contrast(np.array([1.0, 0.0]), np.zeros(2), fraction_affected=[1.0])
        self.assertEqual(result["contrast_status"][0], CONTRAST_NEAR_ZERO)
        self.assertTrue(bool(result["target_near_zero"][0]))
        self.assertTrue(np.isnan(result["cosine"][0]))
        self.assertEqual(result["cosine_status"][0], "undefined_zero_vector")
        self.assertTrue(np.isnan(result["angular_error_rad"][0]))
        self.assertTrue(np.isnan(result["signed_projection"][0]))
        summary = summarize_metric(result["cosine"], result["cosine_status"], result["contrast_status"])
        self.assertEqual(summary["n_total"], 1)
        self.assertEqual(summary["n_defined"], 0)
        self.assertEqual(summary["n_undefined"], 1)
        self.assertEqual(summary["contrast_counts"][CONTRAST_NEAR_ZERO], 1)
        self.assertIsNone(summary["mean_defined"])

    def test_zero_prediction_keeps_magnitude_and_vector_error(self) -> None:
        target = np.array([3.0, 0.0])
        result = directional_contrast(np.zeros(2), target, fraction_affected=[0.5])
        self.assertEqual(result["contrast_status"][0], CONTRAST_VALID)
        self.assertEqual(result["cosine_status"][0], "undefined_zero_vector")
        self.assertTrue(np.isnan(result["cosine"][0]))
        self.assertAlmostEqual(float(result["magnitude_ratio"][0]), 0.0, places=8)
        self.assertAlmostEqual(float(result["normalized_vector_error"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(result["signed_projection"][0]), 0.0, places=12)

    def test_invisible_target_is_flagged_even_when_the_latent_moves(self) -> None:
        result = directional_contrast(np.array([1.0, 0.0]), np.array([1.0, 0.0]), fraction_affected=[0.0])
        self.assertEqual(result["contrast_status"][0], "target_invisible")
        self.assertTrue(bool(result["target_invisible"][0]))
        self.assertAlmostEqual(float(result["cosine"][0]), 1.0, places=12)

    def test_summary_does_not_drop_mixed_rows(self) -> None:
        pred = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        target = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        affected = [1.0, 1.0, 1.0]
        result = directional_contrast(pred, target, fraction_affected=affected)
        summary = summarize_metric(result["cosine"], result["cosine_status"], result["contrast_status"])
        self.assertEqual(summary["n_total"], 3)
        self.assertEqual(summary["n_defined"], 1)
        self.assertEqual(summary["n_undefined"], 2)
        self.assertAlmostEqual(summary["mean_defined"], 1.0, places=12)

    def test_bootstrap_reports_episodes_without_a_valid_cosine(self) -> None:
        result = bootstrap_valid_cosine(
            episode_ids=[0, 0, 1],
            cosine=[1.0, 1.0, None],
            cosine_status=["ok", "ok", "undefined_zero_vector"],
            contrast_status=[CONTRAST_VALID, CONTRAST_VALID, CONTRAST_NEAR_ZERO],
            n_resamples=20,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(result["n_episodes"], 2)
        self.assertEqual(result["n_episodes_with_valid"], 1)
        self.assertEqual(result["n_episodes_without_valid"], 1)
        self.assertEqual(result["excluded_episode_ids"], [1])
        self.assertAlmostEqual(result["bootstrap"]["mean"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
