"""Pixel metrics for the observation-space predictor.

Images are float tensors in ``[0, 1]``. Counterfactual accuracy and action
retrieval are the same two-future comparison: the prediction for an action
is a hit when it is closer to that action's real future than to the
alternative. Ties are not hits. Planning cost is pixel error of the last
predicted frame against a goal image.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

PRIMARY_INTERVENTION = "force_reversal"


def pixel_mse(predicted: Tensor, target: Tensor) -> float:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target images must share a shape.")
    return float(F.mse_loss(predicted, target).item())


def psnr(predicted: Tensor, target: Tensor, *, data_range: float = 1.0) -> dict[str, Any]:
    mse = pixel_mse(predicted, target)
    if mse == 0.0:
        return {"value": None, "mse": 0.0, "status": "perfect_match"}
    value = 10.0 * math.log10((float(data_range) ** 2) / mse)
    return {"value": value, "mse": mse, "status": "ok"}


def _window(size: int, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size // 2)
    kernel = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    weight = (kernel[:, None] * kernel[None, :]).view(1, 1, size, size)
    return weight.expand(channels, 1, size, size).contiguous()


def ssim(predicted: Tensor, target: Tensor, *, data_range: float = 1.0) -> float:
    """Mean SSIM. A matching pair is 1."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target images must share a shape.")
    if predicted.ndim == 3:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    if predicted.ndim != 4:
        raise ValueError("images must be (C, H, W) or (B, C, H, W).")
    window = 7 if min(predicted.shape[-2:]) >= 7 else 3
    channels = int(predicted.shape[1])
    kernel = _window(window, channels, predicted.device, predicted.dtype)
    pad = window // 2
    mu_x = F.conv2d(predicted, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)
    sigma_x = F.conv2d(predicted * predicted, kernel, padding=pad, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(predicted * target, kernel, padding=pad, groups=channels) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator).mean().item())


def _image_mse(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("expected paired image batches (B, C, H, W).")
    return (left - right).pow(2).flatten(1).mean(dim=-1)


def counterfactual_observation_accuracy(
    prediction: Tensor,
    true_future: Tensor,
    alternative_future: Tensor,
) -> dict[str, Any]:
    """Two-way pixel retrieval. Ties are counted and are not successes."""
    correct = _image_mse(prediction, true_future)
    wrong = _image_mse(prediction, alternative_future)
    ties = torch.isclose(correct, wrong)
    hits = (correct < wrong) & ~ties
    n = int(correct.numel())
    return {
        "accuracy": float(hits.float().mean().item()) if n else None,
        "action_retrieval_top1": float(hits.float().mean().item()) if n else None,
        "same_two_way_comparison": True,
        "n": n,
        "n_hits": int(hits.sum().item()),
        "n_ties": int(ties.sum().item()),
        "intervention": PRIMARY_INTERVENTION,
    }


def observation_plan_cost(model: Any, frame: Tensor, sequences: Tensor, goal: Tensor) -> Tensor:
    """Pixel MSE of each sequence's last predicted frame against one goal image."""
    if sequences.ndim != 3:
        raise ValueError("sequences must be (population, horizon, action_dim).")
    population = sequences.shape[0]
    if frame.ndim == 3:
        frame = frame.unsqueeze(0)
    if goal.ndim == 3:
        goal = goal.unsqueeze(0)
    predicted = model.rollout(frame.expand(population, -1, -1, -1), sequences)
    return _image_mse(predicted[:, -1], goal.expand(population, -1, -1, -1))
