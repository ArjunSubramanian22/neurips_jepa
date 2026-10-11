"""Optional spatial targets. The canonical model does not construct these."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalPatchHead(nn.Module):
    """Predict the next patch tokens from the current ones and an action.

    This is an auxiliary head. It is not part of the CLS baseline.
    """

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action = nn.Linear(action_dim, latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, patches: Tensor, action: Tensor) -> Tensor:
        if patches.ndim != 3:
            raise ValueError(f"patches must be (B, N, D), got {tuple(patches.shape)}.")
        return self.proj(patches + self.action(action).unsqueeze(1))


class PatchTokenPredictor(nn.Module):
    """Lift the global forecast to one vector per patch.

    The global predictor is unchanged. Position embeddings make the patch
    outputs distinct. Inference through ``ACJEPA.forward`` does not call this.
    """

    def __init__(self, latent_dim: int, action_dim: int, n_patches: int) -> None:
        super().__init__()
        if n_patches < 1:
            raise ValueError("n_patches must be positive.")
        self.action = nn.Linear(action_dim, latent_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, latent_dim))
        self.out = nn.Linear(latent_dim, latent_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, z_hat: Tensor, action: Tensor) -> Tensor:
        if z_hat.ndim != 2:
            raise ValueError(f"z_hat must be (B, D), got {tuple(z_hat.shape)}.")
        tokens = z_hat.unsqueeze(1) + self.action(action).unsqueeze(1) + self.pos
        return self.out(tokens)
