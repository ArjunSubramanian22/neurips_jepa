"""Observation-space masks from a counterfactual future.

The mask is ``abs(frame(a) - frame(a'))`` pooled onto encoder patches.
It is training supervision. Inference does not read it, and it does not
read particle state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

TOPK_FRACTION = 0.25
TEMPERATURE = 0.05


def observation_difference(factual: Tensor, counterfactual: Tensor) -> Tensor:
    """Mean absolute channel difference, shape ``(B, H, W)``."""
    if factual.shape != counterfactual.shape or factual.ndim != 4:
        raise ValueError("factual and counterfactual observations must share shape (B, C, H, W).")
    return (factual - counterfactual).abs().mean(dim=1)


def patch_scores(difference: Tensor, patch_size: int) -> Tensor:
    """Average-pool a pixel map onto the non-overlapping patch grid."""
    if difference.ndim != 3:
        raise ValueError(f"difference must be (B, H, W), got {tuple(difference.shape)}.")
    if patch_size < 1:
        raise ValueError("patch_size must be positive.")
    height, width = int(difference.shape[-2]), int(difference.shape[-1])
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"frame {height}x{width} is not divisible by patch_size {patch_size}.")
    pooled = F.avg_pool2d(difference.unsqueeze(1), kernel_size=patch_size, stride=patch_size)
    return pooled.flatten(1)


def _normalize(raw: Tensor) -> tuple[Tensor, Tensor]:
    total = raw.sum(dim=-1, keepdim=True)
    degenerate = total.squeeze(-1) <= 0
    uniform = raw.new_full(raw.shape, 1.0 / raw.shape[-1])
    weights = torch.where(degenerate.unsqueeze(-1), uniform, raw / total.clamp_min(1e-12))
    return weights, degenerate


def uniform_weights(batch: int, n_patches: int, like: Tensor) -> tuple[Tensor, Tensor]:
    weights = like.new_full((batch, n_patches), 1.0 / float(n_patches))
    degenerate = torch.zeros(batch, dtype=torch.bool, device=like.device)
    return weights, degenerate


def soft_weights(scores: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize non-negative scores. A blank map becomes uniform."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be (B, N), got {tuple(scores.shape)}.")
    return _normalize(scores.clamp_min(0))


def soft_topk_weights(
    scores: Tensor,
    *,
    fraction: float = TOPK_FRACTION,
    temperature: float = TEMPERATURE,
) -> tuple[Tensor, Tensor]:
    """Down-weight patches below the k-th score with a sigmoid, not a hard cut.

    A hard top-k would zero the complement and jump when two patches swap
    rank. The gate is ``sigmoid((score - score_(k)) / temperature)``. A blank
    or tied map normalizes back to uniform.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be (B, N), got {tuple(scores.shape)}.")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("topk fraction must lie in (0, 1].")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    n_patches = int(scores.shape[-1])
    k = max(1, int(round(float(fraction) * n_patches)))
    k = min(k, n_patches)
    threshold = scores.topk(k, dim=-1).values[:, -1]
    gate = torch.sigmoid((scores - threshold.unsqueeze(-1)) / float(temperature))
    return _normalize(scores.clamp_min(0) * gate)


def weighted_patch_pool(tokens: Tensor, weights: Tensor) -> Tensor:
    """Combine patch tokens with weights that already sum to one."""
    if tokens.ndim != 3 or weights.ndim != 2:
        raise ValueError("tokens must be (B, N, D) and weights (B, N).")
    if tokens.shape[:2] != weights.shape:
        raise ValueError(f"token grid {tuple(tokens.shape[:2])} does not match weights {tuple(weights.shape)}.")
    return (tokens * weights.unsqueeze(-1)).sum(dim=1)


def mask_weights(
    factual: Tensor,
    counterfactual: Tensor | None,
    *,
    mode: str,
    patch_size: int,
    n_patches: int,
    topk_fraction: float = TOPK_FRACTION,
    temperature: float = TEMPERATURE,
) -> tuple[Tensor, Tensor]:
    """Return ``(weights, degenerate)``. ``uniform`` ignores the fork."""
    batch = int(factual.shape[0])
    if mode == "uniform" or counterfactual is None:
        return uniform_weights(batch, n_patches, factual)
    scores = patch_scores(observation_difference(factual, counterfactual), patch_size)
    if int(scores.shape[-1]) != int(n_patches):
        raise ValueError(f"patch grid {int(scores.shape[-1])} does not match encoder patches {n_patches}.")
    if mode == "soft":
        return soft_weights(scores)
    if mode == "topk":
        return soft_topk_weights(scores, fraction=topk_fraction, temperature=temperature)
    raise ValueError(f"unknown target mask mode {mode!r}.")
