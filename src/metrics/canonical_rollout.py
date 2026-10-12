"""Rollout reductions used by the canonical legacy evaluator.

``scripts/evaluate_final.py`` reports these three tensors for each horizon:

* ``rollout_mse``: mean over episodes of the per-dimension squared error
* ``target_variance``: mean over latent dimensions of the population variance
  of the target (``unbiased=False``, divide by the episode count)
* ``vn_mse``: ``rollout_mse / target_variance``

A zero target variance leaves ``vn_mse`` non-finite. That matches the
evaluator; callers decide whether the horizon is usable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def horizon_rollout_metrics(predicted: Any, target: Any) -> dict[str, Tensor]:
    """Reduce ``(episodes, horizon, latent_dim)`` predictions and targets."""
    pred = torch.as_tensor(predicted)
    tgt = torch.as_tensor(target)
    if pred.shape != tgt.shape or pred.ndim != 3 or pred.shape[-1] < 1:
        raise ValueError(
            "predicted and target must share shape (episodes, horizon, latent_dim); "
            f"got {tuple(pred.shape)} and {tuple(tgt.shape)}."
        )
    if pred.shape[0] < 1 or pred.shape[1] < 1:
        raise ValueError("rollout metrics need at least one episode and one horizon.")
    err = (pred - tgt).square().mean(dim=-1)
    rollout_mse = err.mean(dim=0)
    target_variance = tgt.var(dim=0, unbiased=False).mean(dim=-1)
    return {
        "per_episode_mse": err,
        "rollout_mse": rollout_mse,
        "target_variance": target_variance,
        "vn_mse": rollout_mse / target_variance,
    }
