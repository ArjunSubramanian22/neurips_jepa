"""Within-checkpoint metrics for the RSSM.

Open-loop features are prior states. Targets are filtered posterior states
of the true frames. Both use this checkpoint's ``[h, z]`` coordinates.
Raw latent MSE is not compared with a JEPA checkpoint. Variance-normalized
MSE divides by the target variance in the same space.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def within_model_rollout_error(predicted: Tensor, target: Tensor) -> dict[str, Any]:
    """Per-horizon MSE and variance-normalized MSE. ``predicted`` is ``(N, K, D)``."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must share shape (N, K, D).")
    raw = (predicted - target).pow(2).mean(dim=(0, 2))
    variance = target.var(dim=0, unbiased=False).mean(dim=-1)
    if torch.any(variance <= 0):
        normalized = torch.full_like(raw, float("nan"))
        status = "undefined_nonpositive_variance"
    else:
        normalized = raw / variance
        status = "ok"
    return {
        "rollout_mse": [float(value) for value in raw.detach().cpu()],
        "vn_mse": [None if value != value else float(value) for value in normalized.detach().cpu()],
        "target_variance": [float(value) for value in variance.detach().cpu()],
        "raw_mse_comparable_across_architectures": False,
        "normalized_metric": "vn_mse",
        "status": status,
    }


@torch.no_grad()
def rssm_rollout_predictions(
    model: Any,
    observations: Tensor,
    actions: Tensor,
    max_horizon: int,
    device: torch.device,
    batch_size: int = 8,
    start_frame: int = 0,
) -> tuple[Tensor, Tensor]:
    """``(z_hat, z_tgt)`` with the recurrent state carried inside the RSSM."""
    if getattr(model, "architecture", None) != "rssm":
        raise TypeError("rssm_rollout_predictions requires an RSSM.")
    model.eval()
    n_episodes, n_frames = observations.shape[0], observations.shape[1]
    if start_frame < 0 or start_frame >= n_frames - 1:
        raise ValueError("start_frame does not leave a future frame.")
    k_max = min(int(max_horizon), n_frames - 1 - start_frame)
    if k_max < 1:
        raise ValueError("horizon must be at least 1.")
    predicted = torch.empty(n_episodes, k_max, model.latent_dim, device=device)
    target = torch.empty(n_episodes, k_max, model.latent_dim, device=device)
    for start in range(0, n_episodes, batch_size):
        end = min(start + batch_size, n_episodes)
        pred, tgt = model.paired_open_loop(
            observations[start:end].to(device),
            actions[start:end].to(device),
            int(start_frame),
            k_max,
        )
        predicted[start:end] = pred
        target[start:end] = tgt
    return predicted, target


@torch.no_grad()
def rssm_counterfactual_deltas(
    model: Any,
    frame: Tensor,
    action: Tensor,
    action_prime: Tensor,
    future: Tensor,
    future_prime: Tensor,
) -> dict[str, Tensor]:
    """One-step action deltas from a shared initial RSSM state.

    The predicted delta is the prior. The target delta is the stop-gradient
    posterior of the two real next frames. No particle state is read.
    """
    if getattr(model, "architecture", None) != "rssm":
        raise TypeError("rssm_counterfactual_deltas requires an RSSM.")
    model.eval()
    deter, stochastic = model.initial_state(frame.shape[0], frame.device)
    deter, stochastic, _feature = model.observe(deter, stochastic, None, frame)
    _h, _z, pred_a = model.imagine(deter, stochastic, action)
    _h, _z, pred_b = model.imagine(deter, stochastic, action_prime)
    _h, _z, tgt_a = model.observe(deter, stochastic, action, future)
    _h, _z, tgt_b = model.observe(deter, stochastic, action_prime, future_prime)
    return {
        "delta_pred": pred_b - pred_a,
        "delta_target": (tgt_b - tgt_a).detach(),
        "raw_space": "rssm_h_z",
        "comparable_to_jepa_mse": False,
    }
