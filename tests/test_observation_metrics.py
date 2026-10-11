"""Pixel metrics stay defined without a trained observation model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.obs_pred import ActionConditionedUNet
from src.observation_metrics import (
    counterfactual_observation_accuracy,
    observation_plan_cost,
    pixel_mse,
    psnr,
    ssim,
)
from tests.test_obs_pred import _cfg


class _Echo(torch.nn.Module):
    def rollout(self, frame: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        value = actions[:, -1, :1].reshape(-1, 1, 1, 1)
        return (frame + value).unsqueeze(1)


class ObservationMetricTests(unittest.TestCase):
    def test_matching_images_have_zero_error_and_unit_ssim(self) -> None:
        image = torch.rand(2, 3, 16, 16)
        self.assertEqual(pixel_mse(image, image), 0.0)
        self.assertEqual(psnr(image, image)["status"], "perfect_match")
        self.assertIsNone(psnr(image, image)["value"])
        self.assertAlmostEqual(ssim(image, image), 1.0, places=4)

    def test_a_different_image_has_finite_psnr_and_lower_ssim(self) -> None:
        image = torch.rand(2, 3, 16, 16)
        other = (image + 0.4).clamp(0, 1)
        score = psnr(image, other)
        self.assertEqual(score["status"], "ok")
        self.assertGreater(score["value"], 0.0)
        self.assertLess(ssim(image, other), 0.99)

    def test_counterfactual_hit_requires_the_matching_future(self) -> None:
        true = torch.zeros(2, 3, 8, 8)
        other = torch.ones(2, 3, 8, 8)
        result = counterfactual_observation_accuracy(true, true, other)
        self.assertEqual(result["n_hits"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["action_retrieval_top1"], result["accuracy"])
        tied = counterfactual_observation_accuracy(true, true, true)
        self.assertEqual(tied["n_hits"], 0)
        self.assertEqual(tied["n_ties"], 2)
        self.assertEqual(tied["accuracy"], 0.0)

    def test_planning_cost_uses_the_predicted_last_frame(self) -> None:
        frame = torch.zeros(1, 3, 8, 8)
        goal = torch.ones(1, 3, 8, 8)
        sequences = torch.zeros(3, 2, 4)
        sequences[:, -1, 0] = torch.tensor([0.0, 0.5, 1.0])
        costs = observation_plan_cost(_Echo(), frame, sequences, goal)
        self.assertEqual(tuple(costs.shape), (3,))
        self.assertLess(float(costs[2]), float(costs[0]))

    def test_unet_rollout_can_be_scored_without_a_trained_checkpoint(self) -> None:
        model = ActionConditionedUNet(_cfg()).eval()
        frame = torch.rand(1, 3, 16, 16)
        sequences = torch.randn(2, 2, 4)
        goal = torch.rand(1, 3, 16, 16)
        costs = observation_plan_cost(model, frame, sequences, goal)
        self.assertEqual(tuple(costs.shape), (2,))
        self.assertTrue(torch.isfinite(costs).all())


if __name__ == "__main__":
    unittest.main()
