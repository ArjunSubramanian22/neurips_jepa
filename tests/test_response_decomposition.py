"""The median ratio can rank a rotated response ahead of an aligned one."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.response_decomposition import (
    decompose_response,
    ratio_rank_disagrees_with_error,
)


class ResponseDecompositionTests(unittest.TestCase):
    def test_parallel_overshoot_keeps_cosine_and_sets_the_gain(self) -> None:
        target = np.array([3.0, 0.0, 0.0])
        pred = np.array([6.0, 0.0, 0.0])
        scored = decompose_response(pred, target)
        self.assertEqual(scored["cosine_status"][0], "ok")
        self.assertAlmostEqual(float(scored["cosine"][0]), 1.0, places=12)
        self.assertAlmostEqual(float(scored["response_ratio"][0]), 2.0, places=12)
        self.assertAlmostEqual(float(scored["parallel_gain"][0]), 2.0, places=12)
        self.assertAlmostEqual(float(scored["orthogonal_norm"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(scored["normalized_vector_error"][0]), 1.0, places=12)

    def test_orthogonal_residual_keeps_the_ratio_near_one(self) -> None:
        target = np.array([4.0, 0.0])
        pred = np.array([1.2, 3.8])
        scored = decompose_response(pred, target)
        self.assertAlmostEqual(float(scored["response_ratio"][0]), np.linalg.norm(pred) / 4.0, places=12)
        self.assertLess(abs(float(scored["response_ratio"][0]) - 1.0), 0.02)
        self.assertAlmostEqual(float(scored["parallel_gain"][0]), 1.2 / 4.0, places=12)
        self.assertAlmostEqual(float(scored["orthogonal_norm"][0]), 3.8, places=12)
        self.assertGreater(float(scored["normalized_vector_error"][0]), 0.5)
        self.assertLess(float(scored["cosine"][0]), 0.4)

    def test_opposite_direction_is_a_gain_of_minus_one_when_norms_match(self) -> None:
        target = np.array([2.0, -1.0])
        scored = decompose_response(-target, target)
        self.assertAlmostEqual(float(scored["cosine"][0]), -1.0, places=12)
        self.assertAlmostEqual(float(scored["parallel_gain"][0]), -1.0, places=12)
        self.assertAlmostEqual(float(scored["response_ratio"][0]), 1.0, places=12)
        self.assertAlmostEqual(float(scored["orthogonal_norm"][0]), 0.0, places=12)

    def test_near_zero_target_leaves_cosine_and_gain_undefined(self) -> None:
        scored = decompose_response(np.array([1.0, 0.0]), np.zeros(2))
        self.assertEqual(scored["cosine_status"][0], "undefined_zero_vector")
        self.assertTrue(np.isnan(scored["cosine"][0]))
        self.assertTrue(np.isnan(scored["parallel_gain"][0]))
        self.assertTrue(np.isnan(scored["response_ratio"][0]))

    def test_ratio_near_one_can_lose_to_a_biased_but_aligned_response(self) -> None:
        # Open-loop-like: ratio near 1, large orthogonal error.
        rotated = decompose_response(np.array([1.15, 3.9]), np.array([4.0, 0.0]))
        # Teacher-forced-like: ratio 1.27, aligned.
        aligned = decompose_response(np.array([5.08, 0.0]), np.array([4.0, 0.0]))
        decision = ratio_rank_disagrees_with_error(
            float(rotated["response_ratio"][0]),
            float(rotated["normalized_vector_error"][0]),
            float(aligned["response_ratio"][0]),
            float(aligned["normalized_vector_error"][0]),
        )
        self.assertTrue(decision["disagrees"])
        self.assertEqual(decision["ratio_prefers"], "a")
        self.assertEqual(decision["error_prefers"], "b")


if __name__ == "__main__":
    unittest.main()
