"""Inverse dynamics: predict the applied action from a latent transition.

The head reads ``(z_t, z_{t+1})`` and regresses the normalized action.
Center coordinates are the first half of the action vector and force
coordinates the second half, matching the 2D dump layout
``[contact_x, contact_y, force_x, force_y]``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InverseDynamicsHead(nn.Module):
    """MLP from the concatenated transition to the action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if latent_dim < 1 or action_dim < 1 or hidden_dim < 1:
            raise ValueError("latent_dim, action_dim, and hidden_dim must be positive.")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t: Tensor, z_tp1: Tensor) -> Tensor:
        if z_t.shape != z_tp1.shape or z_t.ndim != 2:
            raise ValueError("z_t and z_tp1 must share shape (B, D).")
        if z_t.shape[-1] != self.latent_dim:
            raise ValueError(f"expected latent dim {self.latent_dim}, got {z_t.shape[-1]}.")
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


def normalize_action(action: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Map an action into the training coordinate system."""
    return (action - mean) / std.clamp_min(1e-6)


def inverse_dynamics_loss(
    prediction: Tensor,
    target: Tensor,
    loss_type: Literal["mse", "smooth_l1"] = "mse",
) -> dict[str, Tensor]:
    """Regression on normalized actions, with per-coordinate terms.

    ``target`` is already normalized. The scalar loss is the mean over
    coordinates. Center and force groups split the vector in half.
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape (B, A).")
    if prediction.shape[1] < 2:
        raise ValueError("action dim must be at least 2 to separate center and force.")
    if loss_type == "mse":
        per_coordinate = (prediction - target).pow(2).mean(dim=0)
    elif loss_type == "smooth_l1":
        per_coordinate = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=0)
    else:
        raise ValueError(f"Unknown inverse loss_type '{loss_type}'.")
    split = prediction.shape[1] // 2
    return {
        "loss": per_coordinate.mean(),
        "per_coordinate": per_coordinate,
        "center": per_coordinate[:split].mean(),
        "force": per_coordinate[split:].mean(),
    }


def combine_prediction_and_inverse(prediction: Tensor, inverse: Tensor, weight: float) -> Tensor:
    """Add the auxiliary term. A zero weight leaves the prediction loss unchanged."""
    scale = float(weight)
    if scale < 0.0:
        raise ValueError("inverse_aux_weight must be >= 0.")
    if scale == 0.0:
        return prediction
    return prediction + prediction.new_tensor(scale) * inverse
