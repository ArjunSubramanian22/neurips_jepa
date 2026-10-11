"""Compact recurrent state-space model.

This is an original implementation of the RSSM transition published in
PlaNet and Dreamer. It is not derived from the dreamerv3 source tree.

Provenance:
  Hafner et al., Learning Latent Dynamics for Planning from Pixels, ICML 2019.
  Hafner et al., Dream to Control, ICLR 2020.
  Hafner et al., Mastering Atari with Discrete World Models, ICLR 2021.
    The KL balance and free-nats floor follow DreamerV2. The stochastic
    state here is a diagonal Gaussian, as in PlaNet and DreamerV1, rather
    than the categorical latent of DreamerV2.

The evaluated latent is the concatenation of the deterministic recurrent
state and the stochastic mean. Raw squared error in that space is not
comparable to a JEPA latent. Variance-normalized error is the cross-model
summary.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import ModelConfig


def diagonal_kl(mean_q: Tensor, std_q: Tensor, mean_p: Tensor, std_p: Tensor) -> Tensor:
    """KL(q || p) for diagonal Gaussians, summed over the latent axis."""
    var_q = std_q.square()
    var_p = std_p.square()
    return 0.5 * (var_q / var_p + (mean_p - mean_q).square() / var_p - 1.0 + var_p.log() - var_q.log()).sum(dim=-1)


class _ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, img_size: int, width: int, embed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = int(self.net(torch.zeros(1, in_channels, img_size, img_size)).shape[-1])
        self.proj = nn.Linear(flat, embed)

    def forward(self, frames: Tensor) -> Tensor:
        return self.proj(self.net(frames))


class _ConvDecoder(nn.Module):
    def __init__(self, feature_dim: int, out_channels: int, img_size: int, width: int) -> None:
        super().__init__()
        if img_size % 8 != 0:
            raise ValueError("RSSM images must be divisible by 8.")
        self.grid = img_size // 8
        self.width = width * 4
        self.fc = nn.Linear(feature_dim, self.width * self.grid * self.grid)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.width, width * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        hidden = self.fc(feature).reshape(-1, self.width, self.grid, self.grid)
        return self.net(hidden)


class CompactRSSM(nn.Module):
    """Encoder, GRU state, stochastic posterior and prior, and a pixel decoder."""

    architecture = "rssm"

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.deter = int(cfg.rssm_deter)
        self.stoch = int(cfg.rssm_stoch)
        self.latent_dim = self.deter + self.stoch
        self.condition_on_action = bool(cfg.condition_on_action)
        self.kl_balance = float(cfg.rssm_kl_balance)
        self.free_nats = float(cfg.rssm_free_nats)
        self.action_dim = int(cfg.action_dim)
        self.encoder = _ConvEncoder(cfg.in_channels, cfg.img_size, cfg.rssm_channels, cfg.rssm_embed)
        self.action_proj = nn.Linear(cfg.action_dim, self.stoch)
        self.rnn = nn.GRUCell(self.stoch + self.stoch, self.deter)
        self.prior_head = nn.Linear(self.deter, 2 * self.stoch)
        self.posterior_head = nn.Linear(self.deter + cfg.rssm_embed, 2 * self.stoch)
        self.decoder = _ConvDecoder(self.latent_dim, cfg.in_channels, cfg.img_size, cfg.rssm_channels)
        self.h0 = nn.Parameter(torch.zeros(1, self.deter))

    def trainable_parameters(self):
        yield from self.parameters()

    def _distribution(self, layer: nn.Linear, inputs: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_std = layer(inputs).chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + 1e-4

    def _transition_input(self, stochastic: Tensor, action: Tensor) -> Tensor:
        if not self.condition_on_action:
            action = torch.zeros_like(action)
        return torch.cat([stochastic, self.action_proj(action)], dim=-1)

    def initial_state(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        deter = self.h0.expand(batch, -1)
        stochastic = deter.new_zeros(batch, self.stoch)
        return deter, stochastic

    def split_feature(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        if feature.shape[-1] != self.latent_dim:
            raise ValueError(f"expected feature dim {self.latent_dim}, got {feature.shape[-1]}.")
        return feature[..., : self.deter], feature[..., self.deter :]

    def observe(self, deter: Tensor, stochastic: Tensor, action: Tensor | None, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One filtering step. ``action`` is None only for the first frame."""
        if action is not None:
            deter = self.rnn(self._transition_input(stochastic, action), deter)
        mean, std = self._distribution(self.posterior_head, torch.cat([deter, self.encoder(frame)], dim=-1))
        if self.training:
            stochastic = mean + std * torch.randn_like(mean)
        else:
            stochastic = mean
        return deter, stochastic, torch.cat([deter, mean], dim=-1)

    def imagine(self, deter: Tensor, stochastic: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One prior step. Uses the mean stochastic state."""
        deter = self.rnn(self._transition_input(stochastic, action), deter)
        mean, _std = self._distribution(self.prior_head, deter)
        return deter, mean, torch.cat([deter, mean], dim=-1)

    def loss(self, frames: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """ELBO on a window. ``frames`` is ``(B, T, C, H, W)`` and ``actions`` is ``(B, T-1, A)``."""
        if frames.ndim != 5 or actions.ndim != 3:
            raise ValueError("RSSM loss expects frames (B, T, C, H, W) and actions (B, T-1, A).")
        if actions.shape[1] != frames.shape[1] - 1:
            raise ValueError("actions must cover every step after the first frame.")
        batch = frames.shape[0]
        deter, stochastic = self.initial_state(batch, frames.device)
        reconstructions = []
        kl_terms = []
        for time in range(frames.shape[1]):
            action = None if time == 0 else actions[:, time - 1]
            if time > 0:
                deter = self.rnn(self._transition_input(stochastic, action), deter)
            prior_mean, prior_std = self._distribution(self.prior_head, deter)
            post_mean, post_std = self._distribution(
                self.posterior_head, torch.cat([deter, self.encoder(frames[:, time])], dim=-1)
            )
            stochastic = post_mean + post_std * torch.randn_like(post_mean)
            reconstructions.append(self.decoder(torch.cat([deter, stochastic], dim=-1)))
            dynamics = diagonal_kl(post_mean.detach(), post_std.detach(), prior_mean, prior_std)
            representation = diagonal_kl(post_mean, post_std, prior_mean.detach(), prior_std.detach())
            kl = self.kl_balance * dynamics + (1.0 - self.kl_balance) * representation
            kl_terms.append((kl - self.free_nats).clamp_min(0.0))
        recon = torch.stack(reconstructions, dim=1)
        reconstruction = F.mse_loss(recon, frames)
        kl_loss = torch.stack(kl_terms, dim=1).mean()
        return {"loss": reconstruction + kl_loss, "recon": reconstruction, "kl": kl_loss}

    def encode(self, frames: Tensor, *, target: bool = False) -> Tensor:
        """Posterior feature of one frame from the learned initial state.

        This is the t=0 representation. Multi-step targets use
        :meth:`paired_open_loop`, which keeps the recurrent state.
        """
        if target:
            with torch.no_grad():
                return self.encode(frames, target=False)
        deter, stochastic = self.initial_state(frames.shape[0], frames.device)
        _deter, _stochastic, feature = self.observe(deter, stochastic, None, frames)
        return feature

    def rollout(self, feature: Tensor, actions: Tensor) -> Tensor:
        """Open-loop prior features ``(B, K, D)`` from a state and ``(B, K, A)`` actions."""
        deter, stochastic = self.split_feature(feature)
        predicted = []
        for step in range(actions.shape[1]):
            deter, stochastic, nxt = self.imagine(deter, stochastic, actions[:, step])
            predicted.append(nxt)
        return torch.stack(predicted, dim=1)

    @torch.no_grad()
    def paired_open_loop(self, frames: Tensor, actions: Tensor, start: int, horizon: int) -> tuple[Tensor, Tensor]:
        """Open-loop prior features and stop-gradient posterior features.

        Both sequences live in this checkpoint's ``[h, z]`` space. The target
        is the filtered posterior of the true frames, not an independent encoder.
        """
        if frames.ndim != 5 or actions.ndim != 3:
            raise ValueError("paired_open_loop expects a batch of episodes.")
        if start < 0 or start + horizon >= frames.shape[1]:
            raise ValueError("horizon does not fit in the episode.")
        batch = frames.shape[0]
        deter, stochastic = self.initial_state(batch, frames.device)
        for time in range(start + 1):
            action = None if time == 0 else actions[:, time - 1]
            deter, stochastic, _feature = self.observe(deter, stochastic, action, frames[:, time])
        predicted = []
        target_deter, target_stochastic = deter, stochastic
        targets = []
        for step in range(horizon):
            action = actions[:, start + step]
            deter, stochastic, pred = self.imagine(deter, stochastic, action)
            predicted.append(pred)
            target_deter, target_stochastic, tgt = self.observe(
                target_deter, target_stochastic, action, frames[:, start + 1 + step]
            )
            targets.append(tgt)
        return torch.stack(predicted, dim=1), torch.stack(targets, dim=1).detach()


def window_index(n_episodes: int, n_frames: int, window: int, stride: int = 1) -> list[tuple[int, int]]:
    if window < 2 or window > n_frames:
        raise ValueError("RSSM window must satisfy 2 <= window <= episode length.")
    index = []
    for episode in range(n_episodes):
        last = n_frames - window
        for start in range(0, last + 1, stride):
            index.append((episode, start))
    if not index:
        raise ValueError("RSSM window produced no sequences.")
    return index


class RSSMWindowDataset(torch.utils.data.Dataset):
    """Contiguous observation windows. Actions use the dataset normalizer."""

    def __init__(self, base: Any, episodes: range | list[int], window: int) -> None:
        self.base = base
        self.window = int(window)
        n_frames = int(base.observations.shape[1])
        if window < 2 or window > n_frames:
            raise ValueError("RSSM window must satisfy 2 <= window <= episode length.")
        self.index = [
            (int(episode), start)
            for episode in episodes
            for start in range(0, n_frames - window + 1)
        ]
        if not self.index:
            raise ValueError("RSSM window produced no sequences.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        episode, start = self.index[item]
        frames = self.base.observations[episode, start : start + self.window].clone()
        actions = self.base.actions[episode, start : start + self.window - 1].clone()
        if getattr(self.base, "_should_augment", None) and self.base._should_augment(episode):
            frames, actions = self.base.augmentor(frames, actions)
        if getattr(self.base, "_normalize_action", None):
            actions = self.base._normalize_action(actions)
        return frames, actions
