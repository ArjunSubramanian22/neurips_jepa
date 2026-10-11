"""Known-value checks for layer-separated intervention metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.cascade import (
    LAYERS,
    action_delta,
    assemble_cascade,
    cosine_alignment,
    counterfactual_alignment,
    latent_contrast,
    not_measured,
    observation_visibility,
    physical_response,
    summarize_defined,
    variance_normalized_mse,
)
from src.metrics.ranking import chance_top1, random_ranking_top1


class CascadeMetricTests(unittest.TestCase):
    def test_identical_action_has_zero_delta(self) -> None:
        action = np.array([0.2, 0.4, -0.5, 0.7], dtype=np.float32)
        delta = action_delta(action, action.copy())
        self.assertEqual(delta["l2"], 0.0)
        self.assertEqual(delta["linf"], 0.0)
        self.assertEqual(delta["status"], "measured")

    def test_identical_target_futures_have_zero_contrast(self) -> None:
        latent = np.array([0.5, -1.0, 2.0, 0.25])
        contrast = latent_contrast(latent, latent.copy())
        self.assertEqual(contrast["mse"], 0.0)
        self.assertEqual(contrast["l2"], 0.0)

    def test_aligned_vectors_have_cosine_one(self) -> None:
        vector = np.array([0.3, -0.4, 0.5])
        aligned = cosine_alignment(vector, 2.5 * vector)
        self.assertEqual(aligned["status"], "ok")
        self.assertAlmostEqual(aligned["cosine"], 1.0, places=12)

    def test_opposite_vectors_have_cosine_minus_one(self) -> None:
        vector = np.array([0.3, -0.4, 0.5])
        opposite = cosine_alignment(vector, -4.0 * vector)
        self.assertEqual(opposite["status"], "ok")
        self.assertAlmostEqual(opposite["cosine"], -1.0, places=12)

    def test_zero_vector_cosine_is_undefined(self) -> None:
        result = cosine_alignment(np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertIsNone(result["cosine"])
        self.assertEqual(result["status"], "undefined_zero_vector")

    def test_counterfactual_alignment_matches_difference_cosine(self) -> None:
        factual = np.array([1.0, 0.0, 0.0])
        target_cf = np.array([1.0, 1.0, 0.0])
        pred_cf = np.array([1.0, -1.0, 0.0])
        packed = counterfactual_alignment(factual, pred_cf, factual, target_cf)
        self.assertAlmostEqual(packed["predictor"]["mse"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(packed["target"]["mse"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(packed["alignment"]["cosine"], -1.0, places=12)

    def test_identical_particles_and_rasters_are_zero(self) -> None:
        positions = np.array([[0.2, 0.3], [0.4, 0.5]])
        velocities = np.zeros_like(positions)
        physical = physical_response(positions, positions, velocities, velocities)
        self.assertEqual(physical["rms_pos"], 0.0)
        self.assertEqual(physical["rms_vel"], 0.0)
        raster = np.zeros((3, 4, 4), dtype=np.uint8)
        raster[0, 1, 1] = 255
        visible = observation_visibility(raster, raster.copy())
        self.assertEqual(visible["fraction_identical"], 1.0)
        self.assertEqual(visible["mse"], 0.0)

    def test_uint8_difference_is_scaled_to_unit_interval(self) -> None:
        left = np.zeros((1, 2, 2), dtype=np.uint8)
        right = np.full((1, 2, 2), 255, dtype=np.uint8)
        visible = observation_visibility(left, right)
        self.assertEqual(visible["fraction_identical"], 0.0)
        self.assertAlmostEqual(visible["mse"], 1.0, places=12)

    def test_nonpositive_reference_variance_is_undefined(self) -> None:
        result = variance_normalized_mse(0.25, 0.0)
        self.assertIsNone(result["value"])
        self.assertEqual(result["status"], "undefined_nonpositive_variance")
        scaled = variance_normalized_mse(0.25, 0.5)
        self.assertAlmostEqual(scaled["value"], 0.5, places=12)

    def test_summary_counts_undefined_instead_of_dropping(self) -> None:
        summary = summarize_defined(
            [1.0, None, 3.0],
            ["ok", "undefined_zero_vector", "measured"],
        )
        self.assertEqual(summary["n_total"], 3)
        self.assertEqual(summary["n_defined"], 2)
        self.assertEqual(summary["n_undefined"], 1)
        self.assertAlmostEqual(summary["mean"], 2.0, places=12)

    def test_cascade_keeps_layers_separate(self) -> None:
        payload = assemble_cascade(
            physical=physical_response(np.zeros((2, 2)), np.zeros((2, 2))),
            observation=observation_visibility(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),
            representation=not_measured("no encoder in this check"),
            predictor=not_measured("no predictor in this check"),
            decision=not_measured("no planner in this check"),
            action=action_delta(np.zeros(4), np.ones(4)),
        )
        self.assertEqual(payload["schema"], "cascade_v1")
        self.assertEqual(tuple(payload["layers"]), LAYERS)
        self.assertNotIn("score", payload)
        self.assertNotIn("overall", payload)
        self.assertEqual(payload["layers"]["representation"]["status"], "not_measured")

    def test_random_ranking_is_near_chance(self) -> None:
        rng = np.random.default_rng(0)
        n_candidates = 4
        result = random_ranking_top1(4000, n_candidates, rng)
        self.assertAlmostEqual(result["chance"], chance_top1(n_candidates), places=12)
        self.assertLess(abs(result["top1"] - result["chance"]), 0.03)
        self.assertEqual(result["n_hits"] + (result["n_trials"] - result["n_hits"]), result["n_trials"])


if __name__ == "__main__":
    unittest.main()
