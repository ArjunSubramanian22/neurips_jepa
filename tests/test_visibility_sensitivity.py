"""Visibility cuts keep excluded rows counted and do not follow the outcome."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.visibility_sensitivity import (
    equal_frequency_bins,
    integrate_top_fraction_curve,
    percentile_cutoff,
    select_visibility,
    spearman_visibility,
    summarize_selected,
)


class VisibilitySensitivityTests(unittest.TestCase):
    def test_top_third_matches_the_historical_nonzero_quantile(self) -> None:
        visibility = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        cut = percentile_cutoff(visibility, 0.33)
        self.assertEqual(cut["quantile"], 0.67)
        self.assertAlmostEqual(cut["threshold"], float(np.quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.67)))
        selected = select_visibility(visibility, threshold=cut["threshold"], comparison=cut["comparison"])
        self.assertTrue(np.array_equal(selected, visibility >= cut["threshold"]))
        self.assertFalse(selected[0])

    def test_all_nonzero_keeps_ties_and_drops_only_zeros(self) -> None:
        visibility = np.array([0.0, 1.0, 1.0, 1.0])
        cut = percentile_cutoff(visibility, 0.10)
        selected = select_visibility(visibility, threshold=cut["threshold"], comparison=">=")
        self.assertEqual(int(selected.sum()), 3)
        everyone = select_visibility(visibility, threshold=0.0, comparison=">")
        self.assertEqual(int(everyone.sum()), 3)
        summary = summarize_selected(
            [{"raster_mse": float(value), "rms_pos": 1.0, "pred_delta_l2": 0.0, "target_delta_l2": 0.0, "cosine": None, "cosine_status": "undefined_zero_vector", "counterfactual_ranking_status": "observationally_ambiguous", "counterfactual_ranking_top1": None, "observationally_ambiguous": True} for value in visibility],
            everyone,
            label="all_nonzero",
            visibility_metric="raster_mse",
            threshold=0.0,
            comparison=">",
            family="percentile",
        )
        self.assertEqual(summary["n_total"], 4)
        self.assertEqual(summary["n_selected"], 3)
        self.assertEqual(summary["n_excluded"], 1)
        self.assertEqual(summary["n_ranking_ambiguous"], 3)
        self.assertIsNone(summary["ranking_top1"])
        self.assertFalse(summary["selected_by_outcome"])
        self.assertEqual(summary["state_action_ratio_status"], "not_available")

    def test_state_action_ratio_is_reported_when_the_column_exists(self) -> None:
        rows = [
            {"raster_mse": 0.2, "rms_pos": 0.1, "pred_delta_l2": 2.0, "target_delta_l2": 4.0, "cosine": 1.0, "cosine_status": "ok", "counterfactual_ranking_status": "scored", "counterfactual_ranking_top1": 1.0, "state_action_ratio": 3.0},
            {"raster_mse": 0.4, "rms_pos": 0.2, "pred_delta_l2": 1.0, "target_delta_l2": 4.0, "cosine": 0.0, "cosine_status": "ok", "counterfactual_ranking_status": "scored", "counterfactual_ranking_top1": 0.0, "state_action_ratio": 5.0},
        ]
        summary = summarize_selected(rows, [True, True], label="all_nonzero", visibility_metric="raster_mse", threshold=0.0, comparison=">", family="percentile")
        self.assertEqual(summary["state_action_ratio_status"], "measured")
        self.assertAlmostEqual(summary["state_action_ratio_median"], 4.0)
        self.assertAlmostEqual(summary["response_ratio_median"], 0.375)
        self.assertAlmostEqual(summary["ranking_top1"], 0.5)
        self.assertEqual(summary["n_cosine_defined"], 2)

    def test_spearman_and_auc_do_not_invent_missing_points(self) -> None:
        aligned = spearman_visibility([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], nonzero_only=False)
        flipped = spearman_visibility([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], nonzero_only=False)
        self.assertAlmostEqual(aligned["spearman"], 1.0)
        self.assertAlmostEqual(flipped["spearman"], -1.0)
        flat = spearman_visibility([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], nonzero_only=False)
        self.assertEqual(flat["status"], "undefined_zero_variance")
        self.assertIsNone(flat["spearman"])
        curve = integrate_top_fraction_curve([0.1, 0.5, 1.0], [0.4, 0.4, 0.4])
        self.assertAlmostEqual(curve["mean_over_fractions"], 0.4)
        incomplete = integrate_top_fraction_curve([0.1, 1.0], [0.4, None])
        self.assertIsNone(incomplete["integral"])
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertEqual(incomplete["missing_fractions"], [1.0])

    def test_bins_cover_finite_values_and_leave_nonfinite_unlabeled(self) -> None:
        labels, edges = equal_frequency_bins([0.0, 1.0, 2.0, 3.0, np.nan], n_bins=2)
        self.assertEqual(labels[-1], 0)
        self.assertEqual(set(labels[:-1].tolist()), {1, 2})
        self.assertEqual(len(edges), 1)


if __name__ == "__main__":
    unittest.main()
