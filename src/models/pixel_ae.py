"""Pixel-space action-conditioned autoencoder baseline (ablation Variant C).

Predicts ``x_{t+1}`` in image space via ``Decoder(Predictor(Encoder(x_t), a_t))``
while exposing the same latent rollout API as :class:`~src.models.ac_jepa.ACJEPA`
so representation-drift probes stay comparable.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from ..config import ModelConfig
from .ac_jepa import ACJEPAOutput, ActionPredictor, build_encoder


class PixelDecoder(nn.Module):
    """Maps a latent ``(B, D)`` back to a frame ``(B, C, H, W)``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.img_size = cfg.img_size
        self.in_channels = cfg.in_channels
        latent_dim = cfg.effective_latent_dim
        stem = 4
        if cfg.img_size % stem != 0:
            raise ValueError(f"img_size={cfg.img_size} must be divisible by {stem} for the decoder.")
        n_up = int(round(math.log2(cfg.img_size / stem)))
        if stem * (2**n_up) != cfg.img_size:
            raise ValueError(
                f"img_size={cfg.img_size} is not 4 * 2^k; cannot use the conv-transpose decoder."
            )
        width = max(32, cfg.embed_dim // 2)
        self.fc = nn.Linear(latent_dim, width * stem * stem)
        self.stem = stem
        self.width = width
        blocks: list[nn.Module] = []
        channels = width
        for i in range(n_up):
            out_ch = cfg.in_channels if i == n_up - 1 else max(channels // 2, cfg.in_channels)
            blocks.append(nn.ConvTranspose2d(channels, out_ch, kernel_size=4, stride=2, padding=1))
            if i < n_up - 1:
                blocks.append(nn.GELU())
            channels = out_ch
        self.up = nn.Sequential(*blocks)

    def forward(self, z: Tensor) -> Tensor:
        hidden = self.fc(z).view(z.shape[0], self.width, self.stem, self.stem)
        return self.up(hidden)


class PixelSpaceAutoencoder(nn.Module):
    """Encoder / action predictor / decoder with an EMA target encoder for probes."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        self.latent_dim = self.cfg.effective_latent_dim
        self.encoder = build_encoder(self.cfg)
        self.predictor = ActionPredictor(self.cfg)
        self.decoder = PixelDecoder(self.cfg)
        self.target_encoder = build_encoder(self.cfg)
        self.tau = float(self.cfg.ema_momentum)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_encoder.eval()
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> PixelSpaceAutoencoder:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.encoder.parameters()
        yield from self.predictor.parameters()
        yield from self.decoder.parameters()

    def encode(self, x: Tensor, *, target: bool = False) -> Tensor:
        if target:
            with torch.no_grad():
                return self.target_encoder(x)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    @torch.no_grad()
    def update_target_encoder(self, tau: float | None = None) -> None:
        momentum = self.tau if tau is None else float(tau)
        for online_param, target_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target_param.data.lerp_(online_param.data, 1.0 - momentum)
        for online_buf, target_buf in zip(
            self.encoder.buffers(), self.target_encoder.buffers(), strict=True
        ):
            target_buf.data.copy_(online_buf.data)

    def forward(self, x_t: Tensor, a_t: Tensor, x_tp1: Tensor | None = None) -> ACJEPAOutput:
        z_t = self.encoder(x_t)
        z_hat_tp1 = self.predictor(z_t, a_t)
        z_tp1: Tensor | None = None
        if x_tp1 is not None:
            with torch.no_grad():
                z_tp1 = self.target_encoder(x_tp1).detach()
        return ACJEPAOutput(z_t=z_t, z_hat_tp1=z_hat_tp1, z_tp1=z_tp1)

    def rollout(self, z_t: Tensor, actions: Tensor) -> Tensor:
        latent = z_t
        preds: list[Tensor] = []
        for step in range(actions.shape[1]):
            latent = self.predictor(latent, actions[:, step])
            preds.append(latent)
        return torch.stack(preds, dim=1)
