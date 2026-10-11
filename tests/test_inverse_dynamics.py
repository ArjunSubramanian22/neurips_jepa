"""Inverse dynamics keeps center and force errors separate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TrainingConfig
from src.models.inverse_dynamics import (
    InverseDynamicsHead,
    combine_prediction_and_inverse,
    inverse_dynamics_loss,
    normalize_action,
)


class InverseDynamicsTests(unittest.TestCase):
    def test_force_error_does_not_enter_the_center_term(self) -> None:
        prediction = torch.zeros(6, 4)
        target = torch.zeros(6, 4)
        target[:, 2] = 2.0
        terms = inverse_dynamics_loss(prediction, target, "mse")
        self.assertAlmostEqual(float(terms["center"]), 0.0)
        self.assertAlmostEqual(float(terms["per_coordinate"][2]), 4.0)
        self.assertAlmostEqual(float(terms["force"]), 2.0)
        self.assertAlmostEqual(float(terms["loss"]), 1.0)

    def test_smooth_l1_is_zero_on_a_perfect_prediction(self) -> None:
        target = torch.randn(5, 4)
        terms = inverse_dynamics_loss(target.clone(), target, "smooth_l1")
        self.assertAlmostEqual(float(terms["loss"]), 0.0)

    def test_zero_weight_does_not_change_the_prediction_loss(self) -> None:
        prediction = torch.tensor(2.5)
        inverse = torch.tensor(9.0)
        combined = combine_prediction_and_inverse(prediction, inverse, 0.0)
        self.assertIs(combined, prediction)
        weighted = combine_prediction_and_inverse(prediction, inverse, 0.1)
        self.assertAlmostEqual(float(weighted), 3.4, places=5)

    def test_normalized_coordinates_remove_train_scale(self) -> None:
        raw = torch.tensor([[2.0, 4.0, 10.0, -2.0]])
        mean = torch.tensor([2.0, 4.0, 0.0, 0.0])
        std = torch.tensor([1.0, 2.0, 5.0, 1.0])
        normalized = normalize_action(raw, mean, std)
        self.assertTrue(torch.allclose(normalized, torch.tensor([[0.0, 0.0, 2.0, -2.0]])))

    def test_joint_inverse_loss_reaches_the_encoder(self) -> None:
        torch.manual_seed(0)
        encoder = nn.Linear(8, 8)
        head = InverseDynamicsHead(8, 4, 16)
        observation = torch.randn(7, 8)
        future = torch.randn(7, 8)
        action = torch.randn(7, 4)
        loss = inverse_dynamics_loss(head(encoder(observation), encoder(future)), action)["loss"]
        loss.backward()
        self.assertIsNotNone(encoder.weight.grad)
        self.assertGreater(float(encoder.weight.grad.abs().sum()), 0.0)

    def test_probe_does_not_update_the_encoder(self) -> None:
        torch.manual_seed(1)
        encoder = nn.Linear(8, 8)
        head = InverseDynamicsHead(8, 4, 16)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        observation = torch.randn(7, 8)
        future = torch.randn(7, 8)
        action = torch.randn(7, 4)
        with torch.no_grad():
            latent = encoder(observation)
            future_latent = encoder(future)
        loss = inverse_dynamics_loss(head(latent, future_latent), action)["loss"]
        loss.backward()
        self.assertIsNone(encoder.weight.grad)
        self.assertGreater(float(head.net[-1].weight.grad.abs().sum()), 0.0)

    def test_changing_the_transition_changes_the_predicted_action(self) -> None:
        torch.manual_seed(2)
        head = InverseDynamicsHead(8, 4, 16)
        start = torch.randn(3, 8)
        future = torch.randn(3, 8)
        other = future.clone()
        other[:, 0] += 1.0
        self.assertFalse(torch.allclose(head(start, future), head(start, other)))

    def test_probe_mode_rejects_a_joint_weight(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(inverse_mode="probe", inverse_aux_weight=0.1, inverse_probe_epochs=2)
        TrainingConfig(inverse_mode="probe", inverse_aux_weight=0.0, inverse_probe_epochs=2)


if __name__ == "__main__":
    unittest.main()
