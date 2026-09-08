"""GNS / DPI-Net style particle-graph world model.

Message passing uses the mass-spring lattice connectivity already built by
:func:`src.dataset._edge_list` (structural + shear springs). The model
predicts the next particle positions (and velocities) rather than a ViT
latent, matching the established deformable-object baseline family used by
DPI-Net, GNS, and PhysWorld-style particle simulators.

This is a *within-simulator* baseline: it sees the same synthetic lattice
trajectories as AC-JEPA, not captured video or MPM-optimized digital twins.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from ..config import DataConfig, ModelConfig
from ..dataset import _edge_list, _grid_positions


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    dim = in_dim
    for i in range(layers - 1):
        mods.extend([nn.Linear(dim, hidden), nn.GELU()])
        dim = hidden
    mods.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*mods)


class ParticleGraphWorldModel(nn.Module):
    """Encode-process-decode GNN over the ``n_grid`` mass-spring lattice.

    Node features are ``[position, velocity, rest_position]``. External
    contact forces enter as per-node features using the same Gaussian kernel
    as :class:`~src.dataset.MassSpringSimulator`. Edges are the undirected
    spring list from :func:`src.dataset._edge_list`, duplicated both ways
    for message passing.
    """

    requires_particles: bool = True

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        data_cfg: DataConfig | None = None,
        *,
        hidden_dim: int = 64,
        message_steps: int = 3,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg if model_cfg is not None else ModelConfig()
        self.data_cfg = data_cfg if data_cfg is not None else DataConfig()
        self.spatial_dims = int(self.data_cfg.spatial_dims)
        self.n_grid = int(self.data_cfg.n_grid)
        self.n_particles = int(self.data_cfg.n_particles)
        self.hidden_dim = int(hidden_dim)
        self.message_steps = int(message_steps)
        self.latent_dim = int(self.model_cfg.effective_latent_dim)
        rest = torch.from_numpy(_grid_positions(self.n_grid, self.spatial_dims).copy())
        edges = torch.from_numpy(_edge_list(self.n_grid, self.spatial_dims).copy()).long()
        src = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
        dst = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
        rest_len = torch.norm(rest[edges[:, 1]] - rest[edges[:, 0]], dim=-1)
        rest_len = torch.cat([rest_len, rest_len], dim=0)
        self.register_buffer("rest_pos", rest.float())
        self.register_buffer("edge_src", src)
        self.register_buffer("edge_dst", dst)
        self.register_buffer("rest_length", rest_len.float())
        node_in = self.spatial_dims * 3 + self.spatial_dims + 2  # pos, vel, rest, force, weight, dist
        edge_in = self.spatial_dims + 2  # rel pos, dist, rest length
        self.node_encoder = _mlp(node_in, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(edge_in, hidden_dim, hidden_dim)
        self.message_mlp = _mlp(hidden_dim * 3, hidden_dim, hidden_dim)
        self.update_mlp = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.pos_head = _mlp(hidden_dim, hidden_dim, self.spatial_dims)
        self.vel_head = _mlp(hidden_dim, hidden_dim, self.spatial_dims)
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.tau = float(self.model_cfg.ema_momentum)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def update_target_encoder(self, tau: float | None = None) -> None:
        """No EMA target; particle GNN is trained with next-state MSE."""
        del tau

    def _action_node_features(self, positions: Tensor, actions: Tensor) -> Tensor:
        dims = self.spatial_dims
        contact = actions[:, :dims].unsqueeze(1)
        force = actions[:, dims : 2 * dims].unsqueeze(1) * float(self.data_cfg.action_gain)
        offset = positions - contact
        dist2 = offset.pow(2).sum(dim=-1, keepdim=True)
        radius2 = float(self.data_cfg.contact_radius) ** 2
        weights = torch.exp(-dist2 / (2.0 * radius2 + 1e-8))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        applied = weights * force
        return torch.cat([applied, weights, torch.sqrt(dist2 + 1e-8)], dim=-1)

    def _edge_attr(self, positions: Tensor) -> Tensor:
        src = positions[:, self.edge_src]
        dst = positions[:, self.edge_dst]
        rel = dst - src
        dist = torch.norm(rel, dim=-1, keepdim=True)
        rest = self.rest_length.view(1, -1, 1).expand(positions.shape[0], -1, 1)
        return torch.cat([rel, dist, rest], dim=-1)

    def _scatter_sum(self, messages: Tensor) -> Tensor:
        batch, n_edges, hidden = messages.shape
        n_nodes = self.n_particles
        flat = messages.reshape(batch * n_edges, hidden)
        dst = self.edge_dst.view(1, n_edges).expand(batch, n_edges)
        dst = dst + torch.arange(batch, device=messages.device).view(batch, 1) * n_nodes
        agg = messages.new_zeros(batch * n_nodes, hidden)
        agg.index_add_(0, dst.reshape(-1), flat)
        return agg.view(batch, n_nodes, hidden)

    def encode_nodes(self, positions: Tensor, velocities: Tensor, actions: Tensor) -> Tensor:
        rest = self.rest_pos.unsqueeze(0).expand(positions.shape[0], -1, -1)
        action_feat = self._action_node_features(positions, actions)
        nodes = torch.cat([positions, velocities, rest, action_feat], dim=-1)
        hidden = self.node_encoder(nodes)
        edge_h = self.edge_encoder(self._edge_attr(positions))
        for _ in range(self.message_steps):
            src_h = hidden[:, self.edge_src]
            dst_h = hidden[:, self.edge_dst]
            messages = self.message_mlp(torch.cat([src_h, dst_h, edge_h], dim=-1))
            aggregated = self._scatter_sum(messages)
            hidden = hidden + self.update_mlp(torch.cat([hidden, aggregated], dim=-1))
        return hidden

    def encode_state(
        self, positions: Tensor, velocities: Tensor, actions: Tensor | None = None
    ) -> Tensor:
        """Pool GNN node embeddings into a latent ``(B, D)`` for archival probes."""
        if actions is None:
            actions = positions.new_zeros(positions.shape[0], self.model_cfg.action_dim)
        hidden = self.encode_nodes(positions, velocities, actions)
        return self.readout(hidden.mean(dim=1))

    def encode(self, x: Tensor, *, target: bool = False) -> Tensor:
        """Particle GNN does not encode pixels; use :meth:`encode_state`."""
        del target
        raise TypeError(
            "ParticleGraphWorldModel.encode() does not take image frames. "
            "Call encode_state(positions, velocities, actions) or "
            "rollout_from_particles(...)."
        )

    def step(
        self, positions: Tensor, velocities: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encode_nodes(positions, velocities, actions)
        pred_pos = positions + self.pos_head(hidden)
        pred_vel = velocities + self.vel_head(hidden)
        latent = self.readout(hidden.mean(dim=1))
        return pred_pos, pred_vel, latent

    def forward(
        self, positions: Tensor, velocities: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.step(positions, velocities, actions)

    def rollout(self, z_t: Tensor, actions: Tensor) -> Tensor:
        del z_t, actions
        raise TypeError(
            "ParticleGraphWorldModel.rollout expects particle state; use "
            "rollout_from_particles(positions, velocities, actions, k_max)."
        )

    @torch.no_grad()
    def rollout_from_particles(
        self,
        positions: Tensor,
        velocities: Tensor,
        actions: Tensor,
        k_max: int,
        batch_size: int = 8,
    ) -> tuple[Tensor, Tensor]:
        """Autoregressive particle rollout scored in pooled-latent space.

        ``ẑ_{t+k}`` is the GNN readout of the *predicted* lattice after ``k``
        steps; ``z_{t+k}`` is the readout of the ground-truth lattice.
        """
        self.eval()
        n_episodes = positions.shape[0]
        k_max = min(int(k_max), int(actions.shape[1]), int(positions.shape[1] - 1))
        z_hat = positions.new_empty(n_episodes, k_max, self.latent_dim)
        z_tgt = positions.new_empty(n_episodes, k_max, self.latent_dim)
        for start in range(0, n_episodes, batch_size):
            end = min(start + batch_size, n_episodes)
            pos = positions[start:end, 0]
            vel = velocities[start:end, 0]
            acts = actions[start:end, :k_max]
            for horizon in range(k_max):
                a_t = acts[:, horizon]
                pos, vel, z_pred = self.step(pos, vel, a_t)
                z_hat[start:end, horizon] = z_pred
                gt_pos = positions[start:end, horizon + 1]
                gt_vel = velocities[start:end, horizon + 1]
                z_tgt[start:end, horizon] = self.encode_state(gt_pos, gt_vel, a_t)
        return z_hat, z_tgt
