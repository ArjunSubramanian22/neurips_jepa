"""Action-conditioning modules for the dynamics predictor.

``concat`` remains the baseline and is implemented in ``ActionPredictor`` so
existing checkpoints keep their parameter names. New mechanisms live here and
are selected with ``ModelConfig.fuse_mode``.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class GatedResidualFuser(nn.Module):
    """Latent projection plus a sigmoid-gated action residual.

    The shared predictor mixer is applied by ``ActionPredictor`` after this
    module, matching the depth used by concatenation.
    """

    def __init__(self, latent_dim: int, action_embed_dim: int, pred_dim: int) -> None:
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, pred_dim)
        self.delta = nn.Linear(action_embed_dim, pred_dim)
        self.gate = nn.Linear(action_embed_dim, pred_dim)

    def forward(self, z_t: Tensor, action_embed: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(action_embed))
        return self.z_proj(z_t) + gate * self.delta(action_embed)

    def action_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            if name.startswith("delta") or name.startswith("gate"):
                yield name, param


class CrossAttentionFuser(nn.Module):
    """One latent query attending to several tokens built from the action.

    A single action token would make the softmax identically one. Several
    tokens keep the attention weights dependent on the action.
    """

    def __init__(
        self,
        latent_dim: int,
        action_embed_dim: int,
        pred_dim: int,
        num_heads: int,
        n_tokens: int,
    ) -> None:
        super().__init__()
        if pred_dim % num_heads != 0:
            raise ValueError("pred_dim must be divisible by num_heads.")
        if n_tokens < 2:
            raise ValueError("cross-attention uses at least two action tokens.")
        self.num_heads = int(num_heads)
        self.head_dim = pred_dim // int(num_heads)
        self.n_tokens = int(n_tokens)
        self.q_proj = nn.Linear(latent_dim, pred_dim)
        self.token_proj = nn.Linear(action_embed_dim, self.n_tokens * pred_dim)
        self.out_proj = nn.Linear(pred_dim, pred_dim)

    def forward(self, z_t: Tensor, action_embed: Tensor) -> Tensor:
        batch = z_t.shape[0]
        query = self.q_proj(z_t).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        tokens = self.token_proj(action_embed).view(batch, self.n_tokens, self.num_heads, self.head_dim)
        tokens = tokens.permute(0, 2, 1, 3)
        attended = F.scaled_dot_product_attention(query, tokens, tokens)
        attended = attended.transpose(1, 2).reshape(batch, -1)
        return self.out_proj(attended)

    def action_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            if name.startswith("token_proj") or name.startswith("out_proj"):
                yield name, param
