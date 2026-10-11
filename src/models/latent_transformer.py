"""Causal transformer dynamics over frame latents and action tokens.

The predictor is not an MLP and not a recurrent state. Each next-latent
query can attend to a bounded history of frame latents and explicit action
tokens. Attention mass on those action tokens is a diagnostic. It is not a
measure of causal importance.

One-step mode is teacher-forced. Autoregressive mode feeds predicted latents
back in. The canonical multi-step rollout is autoregressive.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import ModelConfig


class _FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, img_size: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width // 2, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.net(frames)


class _CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must be divisible by heads.")
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        hidden = width * 2
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.dropout = dropout

    def forward(self, tokens: Tensor, mask: Tensor, *, return_attention: bool) -> tuple[Tensor, Tensor | None]:
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(batch, length, width)
        if self.training and self.dropout > 0:
            mixed = F.dropout(mixed, self.dropout)
        tokens = tokens + self.proj(mixed)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens, attention if return_attention else None


def history_mask(length: int, history: int, device: torch.device) -> Tensor:
    """True where a key is hidden. Causal, and no older than ``history`` steps."""
    if history < 1:
        raise ValueError("history must be >= 1.")
    index = torch.arange(length, device=device)
    query_time = index // 2
    key_time = index // 2
    key_after_query = index[None, :] > index[:, None]
    too_old = key_time[None, :] < query_time[:, None] - (history - 1)
    return (key_after_query | too_old).view(1, 1, length, length)


class CausalLatentDynamics(nn.Module):
    """Predict the next frame latent from a causal latent/action sequence."""

    architecture = "latent_transformer"

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.width = int(cfg.latent_tf_width)
        self.history = int(cfg.latent_tf_history)
        self.train_mode = cfg.latent_tf_train_mode
        self.latent_dim = self.width
        self.condition_on_action = bool(cfg.condition_on_action)
        self.encoder = _FrameEncoder(cfg.in_channels, cfg.img_size, self.width)
        self.target_encoder = _FrameEncoder(cfg.in_channels, cfg.img_size, self.width)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.action_embed = nn.Sequential(
            nn.Linear(cfg.action_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.kind_embed = nn.Embedding(2, self.width)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.history, self.width))
        self.blocks = nn.ModuleList(
            _CausalBlock(self.width, int(cfg.latent_tf_heads), float(cfg.dropout))
            for _ in range(int(cfg.latent_tf_layers))
        )
        self.pred_head = nn.Linear(self.width, self.width)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.tau = float(cfg.ema_momentum)
        self.last_attention_action_mass: float | None = None
        self.last_attention_latent_mass: float | None = None

    def trainable_parameters(self):
        yield from self.encoder.parameters()
        yield from self.action_embed.parameters()
        yield from self.kind_embed.parameters()
        yield from self.blocks.parameters()
        yield from self.pred_head.parameters()
        yield self.pos_embed

    def train(self, mode: bool = True) -> CausalLatentDynamics:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def action_embedding_parameters(self):
        yield from self.action_embed.parameters()

    @torch.no_grad()
    def update_target_encoder(self, tau: float | None = None) -> None:
        momentum = self.tau if tau is None else float(tau)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.data.lerp_(online.data, 1.0 - momentum)

    def encode(self, frames: Tensor, *, target: bool = False) -> Tensor:
        if frames.ndim != 4:
            raise ValueError("encode expects one frame batch (B, C, H, W).")
        if target:
            with torch.no_grad():
                return self.target_encoder(frames)
        return self.encoder(frames)

    def encode_sequence(self, frames: Tensor, *, target: bool = False) -> Tensor:
        """Encode ``(B, T, C, H, W)`` into ``(B, T, D)``."""
        if frames.ndim == 4:
            return self.encode(frames, target=target)
        batch, steps = frames.shape[:2]
        flat = frames.reshape(batch * steps, *frames.shape[2:])
        return self.encode(flat, target=target).reshape(batch, steps, self.width)
        if target:
            with torch.no_grad():
                return self.target_encoder(frames)
        return self.encoder(frames)

    def _action_tokens(self, actions: Tensor) -> Tensor:
        if not self.condition_on_action:
            actions = torch.zeros_like(actions)
        return self.action_embed(actions)

    def _mask_for(self, length: int, device: torch.device) -> Tensor:
        return history_mask(length, self.history, device)

    def predict_latents(self, latents: Tensor, actions: Tensor, *, record_attention: bool = False) -> Tensor:
        """Teacher-forced next-latent predictions. ``latents`` is ``(B, T, D)``."""
        if latents.ndim != 3 or actions.ndim != 3:
            raise ValueError("latents must be (B, T, D) and actions (B, T, A).")
        if latents.shape[1] != actions.shape[1]:
            raise ValueError("one action is required for every conditioning latent.")
        steps = int(latents.shape[1])
        if steps <= self.history:
            return self._predict_window(latents, actions, record_attention=record_attention)
        predicted = []
        for time in range(steps):
            start = max(0, time + 1 - self.history)
            window = self._predict_window(
                latents[:, start : time + 1],
                actions[:, start : time + 1],
                record_attention=record_attention and time == steps - 1,
            )
            predicted.append(window[:, -1])
        return torch.stack(predicted, dim=1)

    def _predict_window(self, latents: Tensor, actions: Tensor, *, record_attention: bool) -> Tensor:
        steps = latents.shape[1]
        latent_tokens = latents + self.kind_embed.weight[0]
        action_tokens = self._action_tokens(actions) + self.kind_embed.weight[1]
        sequence = torch.stack([latent_tokens, action_tokens], dim=2).reshape(latents.shape[0], 2 * steps, self.width)
        sequence = sequence + self.pos_embed[:, : 2 * steps]
        mask = self._mask_for(2 * steps, sequence.device)
        attention = None
        for block in self.blocks:
            sequence, attention = block(sequence, mask, return_attention=record_attention and block is self.blocks[-1])
        if record_attention and attention is not None:
            self._store_attention(attention)
        return self.pred_head(sequence[:, 1::2])

    def _store_attention(self, attention: Tensor) -> None:
        """Mean attention from action-token queries onto action and latent keys.

        ``attention`` is ``(B, heads, query, key)``. This is not causal importance.
        """
        queries = attention[:, :, 1::2, :]
        key_index = torch.arange(attention.shape[-1], device=attention.device)
        action_keys = (key_index % 2 == 1).view(1, 1, 1, -1)
        action_mass = (queries * action_keys).sum(dim=-1).mean()
        self.last_attention_action_mass = float(action_mass.detach().item())
        self.last_attention_latent_mass = float(1.0 - self.last_attention_action_mass)

    def rollout(self, latent: Tensor, actions: Tensor, *, record_attention: bool = False) -> Tensor:
        """Autoregressive predictions ``(B, K, D)`` from ``latent`` ``(B, D)`` and ``(B, K, A)``."""
        if latent.ndim != 2 or actions.ndim != 3:
            raise ValueError("rollout expects latent (B, D) and actions (B, K, A).")
        history_latents = [latent]
        history_actions: list[Tensor] = []
        predicted = []
        for step in range(actions.shape[1]):
            history_actions.append(actions[:, step])
            window = self.history
            latent_window = torch.stack(history_latents[-window:], dim=1)
            action_window = torch.stack(history_actions[-window:], dim=1)
            nxt = self.predict_latents(
                latent_window,
                action_window,
                record_attention=record_attention and step == actions.shape[1] - 1,
            )[:, -1]
            predicted.append(nxt)
            history_latents.append(nxt)
        return torch.stack(predicted, dim=1)

    def loss(self, frames: Tensor, actions: Tensor, *, record_attention: bool = False) -> dict[str, Tensor]:
        if frames.shape[1] != actions.shape[1] + 1:
            raise ValueError("frames must be one longer than actions.")
        if self.train_mode == "one_step":
            predicted = self.predict_latents(
                self.encode_sequence(frames[:, :-1]), actions, record_attention=record_attention
            )
        elif self.train_mode == "autoregressive":
            predicted = self.rollout(self.encode(frames[:, 0]), actions, record_attention=record_attention)
        else:
            raise ValueError(f"unknown latent_tf_train_mode {self.train_mode!r}.")
        with torch.no_grad():
            target = self.encode_sequence(frames[:, 1:], target=True)
        reconstruction = F.mse_loss(predicted, target)
        return {"loss": reconstruction, "pred": predicted, "target": target}

    def action_embedding_grad_norm(self) -> float:
        total = None
        for parameter in self.action_embed.parameters():
            if parameter.grad is None:
                continue
            square = parameter.grad.detach().float().pow(2).sum()
            total = square if total is None else total + square
        if total is None:
            return 0.0
        return float(torch.sqrt(total).item())

    @torch.no_grad()
    def action_replacement_sensitivity(self, frames: Tensor, actions: Tensor, replacement: Tensor) -> float:
        """Mean squared change in the one-step prediction when actions are replaced."""
        was_training = self.training
        self.eval()
        original = self.predict_latents(self.encode_sequence(frames[:, :-1]), actions)
        replaced = self.predict_latents(self.encode_sequence(frames[:, :-1]), replacement)
        if was_training:
            self.train()
        return float((replaced - original).pow(2).mean().item())
