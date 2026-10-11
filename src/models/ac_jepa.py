"""Action-Conditioned JEPA world model.

The online encoder ``f_θ`` maps an observation ``x_t`` to a latent ``z_t``.
An action-conditioned predictor ``g_φ`` maps ``[z_t, a_t]`` to a forecast
``ẑ_{t+1}``. A stop-gradient target encoder ``f_θ̄`` (EMA of ``f_θ``) encodes
``x_{t+1}`` into the regression target ``z_{t+1}``.

Gradient flow is restricted to ``f_θ`` and ``g_φ``. ``f_θ̄`` is updated only
by the momentum rule ``θ̄ ← τ θ̄ + (1 − τ) θ`` with default ``τ = 0.996``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig
from .action_conditioning import CrossAttentionFuser, GatedResidualFuser


class DropPath(nn.Module):
    """Stochastic depth (per-sample residual dropout)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class FeedForward(nn.Module):
    """Two-layer GELU MLP used inside transformer blocks."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN multi-head self-attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)

    def _attention(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        drop_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        attn = attn.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(attn))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self._attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block for concat/FiLM predictors."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc2(self.drop(F.gelu(self.fc1(self.norm(x)))))
        return x + self.drop(h)


class PatchEmbed(nn.Module):
    """Non-overlapping convolutional patch embedding."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if height != self.img_size or width != self.img_size:
            raise ValueError(
                f"Expected square frames of size {self.img_size}, got {height}x{width}."
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def _zero_residual_mlp_heads(module: nn.Module) -> None:
    """Restore identity-at-init after generic Linear initialization."""
    if isinstance(module, ResidualMLPBlock):
        nn.init.zeros_(module.fc2.weight)
        nn.init.zeros_(module.fc2.bias)


def _init_transformer_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class VisionTransformerEncoder(nn.Module):
    """ViT encoder: ``(B, C, H, W) -> (B, D)`` via a CLS token."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.latent_dim = cfg.effective_latent_dim
        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, cfg.embed_dim)
        )
        self.pos_drop = nn.Dropout(cfg.dropout)
        drop_rates = torch.linspace(0.0, cfg.drop_path, cfg.encoder_depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.encoder_num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    drop_path=drop_rates[i],
                )
                for i in range(cfg.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.latent_proj = (
            nn.Identity()
            if self.latent_dim == cfg.embed_dim
            else nn.Linear(cfg.embed_dim, self.latent_dim)
        )
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_transformer_weights)

    def _tokens(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, x: Tensor) -> Tensor:
        """CLS latent. This is the canonical encoder output."""
        pooled = self._tokens(x)[:, 0]
        return self.latent_norm(self.latent_proj(pooled))

    def forward_patches(self, x: Tensor) -> Tensor:
        """Patch tokens, excluding CLS. Optional. ``forward`` does not call this."""
        patches = self._tokens(x)[:, 1:]
        return self.latent_norm(self.latent_proj(patches))


class _BasicBlock(nn.Module):
    """CIFAR-style residual block (3x3 / 3x3)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ResNetEncoder(nn.Module):
    """Lightweight ResNet encoder: ``(B, C, H, W) -> (B, D)``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.effective_latent_dim
        stem_width = 64
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(stem_width, 64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, self.latent_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self._init_resnet()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [_BasicBlock(in_channels, out_channels, stride=stride)]
        blocks.extend(_BasicBlock(out_channels, out_channels, stride=1) for _ in range(n_blocks - 1))
        return nn.Sequential(*blocks)

    def _init_resnet(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.latent_norm(self.fc(x))


def build_encoder(cfg: ModelConfig) -> nn.Module:
    """Construct the observation encoder named in ``cfg.backbone``."""
    if cfg.backbone == "vit":
        return VisionTransformerEncoder(cfg)
    if cfg.backbone == "cnn":
        return ResNetEncoder(cfg)
    raise ValueError(f"Unknown backbone '{cfg.backbone}'. Use 'vit' or 'cnn'.")


class ActionPredictor(nn.Module):
    """Action-conditioned dynamics head: ``[z_t, a_t] -> ẑ_{t+1}``.

    ``fuse_mode="concat"`` (default) embeds the action, concatenates it with
    ``z_t``, and mixes the pair with residual MLP blocks. ``"film"`` modulates
    ``z_t`` with a feature-wise affine transform. ``"gate"`` adds a
    sigmoid-gated action residual. ``"cross_attn"`` attends from the latent
    to action tokens. ``"token"`` feeds ``(z, a)`` as a two-token sequence.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fuse_mode = cfg.fuse_mode
        self.condition_on_action = bool(cfg.condition_on_action)
        latent_dim = cfg.effective_latent_dim
        pred_dim = cfg.predictor_embed_dim
        if not self.condition_on_action:
            self.uncond_proj = nn.Linear(latent_dim, pred_dim)
            self.mixer = nn.Sequential(
                *[
                    ResidualMLPBlock(pred_dim, cfg.mlp_ratio, cfg.dropout)
                    for _ in range(cfg.predictor_depth)
                ]
            )
            self.head = nn.Sequential(
                nn.LayerNorm(pred_dim),
                nn.Linear(pred_dim, latent_dim),
            )
            self.apply(_init_transformer_weights)
            self.apply(_zero_residual_mlp_heads)
            return
        self.action_embed = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.action_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.action_embed_dim, cfg.action_embed_dim),
            nn.LayerNorm(cfg.action_embed_dim),
        )
        if cfg.fuse_mode == "concat":
            self.fuse_proj = nn.Linear(latent_dim + cfg.action_embed_dim, pred_dim)
            self.mixer = nn.Sequential(
                *[
                    ResidualMLPBlock(pred_dim, cfg.mlp_ratio, cfg.dropout)
                    for _ in range(cfg.predictor_depth)
                ]
            )
        elif cfg.fuse_mode == "film":
            self.film = nn.Linear(cfg.action_embed_dim, 2 * latent_dim)
            self.fuse_proj = nn.Linear(latent_dim, pred_dim)
            self.mixer = nn.Sequential(
                *[
                    ResidualMLPBlock(pred_dim, cfg.mlp_ratio, cfg.dropout)
                    for _ in range(cfg.predictor_depth)
                ]
            )
        elif cfg.fuse_mode == "gate":
            self.fuser = GatedResidualFuser(latent_dim, cfg.action_embed_dim, pred_dim)
            self.mixer = nn.Sequential(
                *[
                    ResidualMLPBlock(pred_dim, cfg.mlp_ratio, cfg.dropout)
                    for _ in range(cfg.predictor_depth)
                ]
            )
        elif cfg.fuse_mode == "cross_attn":
            self.fuser = CrossAttentionFuser(
                latent_dim,
                cfg.action_embed_dim,
                pred_dim,
                cfg.predictor_num_heads,
                cfg.n_action_tokens,
            )
            self.mixer = nn.Sequential(
                *[
                    ResidualMLPBlock(pred_dim, cfg.mlp_ratio, cfg.dropout)
                    for _ in range(cfg.predictor_depth)
                ]
            )
        elif cfg.fuse_mode == "token":
            self.z_proj = nn.Linear(latent_dim, pred_dim)
            self.a_proj = nn.Linear(cfg.action_embed_dim, pred_dim)
            drop_rates = torch.linspace(0.0, cfg.drop_path, max(cfg.predictor_depth, 1)).tolist()
            self.mixer = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=pred_dim,
                        num_heads=cfg.predictor_num_heads,
                        mlp_ratio=cfg.mlp_ratio,
                        dropout=cfg.dropout,
                        drop_path=drop_rates[i],
                    )
                    for i in range(cfg.predictor_depth)
                ]
            )
            self.mixer_norm = nn.LayerNorm(pred_dim)
        else:
            raise ValueError(f"Unknown fuse_mode '{cfg.fuse_mode}'.")
        self.head = nn.Sequential(
            nn.LayerNorm(pred_dim),
            nn.Linear(pred_dim, latent_dim),
        )
        self.apply(_init_transformer_weights)
        self.apply(_zero_residual_mlp_heads)

    def forward(self, z_t: Tensor, a_t: Tensor) -> Tensor:
        if z_t.ndim != 2 or a_t.ndim != 2:
            raise ValueError(
                f"Predictor expects z_t (B, D) and a_t (B, A); got {tuple(z_t.shape)}, {tuple(a_t.shape)}."
            )
        if z_t.shape[0] != a_t.shape[0]:
            raise ValueError(
                f"Batch mismatch between z_t ({z_t.shape[0]}) and a_t ({a_t.shape[0]})."
            )
        if not self.condition_on_action:
            return self.head(self.mixer(self.uncond_proj(z_t)))
        a_embed = self.action_embed(a_t)
        if self.fuse_mode == "concat":
            hidden = self.mixer(self.fuse_proj(torch.cat([z_t, a_embed], dim=-1)))
        elif self.fuse_mode == "film":
            gamma, beta = self.film(a_embed).chunk(2, dim=-1)
            hidden = self.mixer(self.fuse_proj((1.0 + gamma) * z_t + beta))
        elif self.fuse_mode == "gate":
            hidden = self.mixer(self.fuser(z_t, a_embed))
        elif self.fuse_mode == "cross_attn":
            hidden = self.mixer(self.fuser(z_t, a_embed))
        else:
            tokens = torch.stack([self.z_proj(z_t), self.a_proj(a_embed)], dim=1)
            for block in self.mixer:
                tokens = block(tokens)
            hidden = self.mixer_norm(tokens)[:, 0]
        return self.head(hidden)

    def action_path_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Weights that read the action. Shared latent mixers are excluded."""
        if not self.condition_on_action:
            return
        for name, param in self.action_embed.named_parameters():
            yield f"action_embed.{name}", param
        if self.fuse_mode == "concat":
            for name, param in self.fuse_proj.named_parameters():
                yield f"fuse_proj.{name}", param
        elif self.fuse_mode == "film":
            for name, param in self.film.named_parameters():
                yield f"film.{name}", param
        elif self.fuse_mode == "token":
            for name, param in self.a_proj.named_parameters():
                yield f"a_proj.{name}", param
        elif self.fuse_mode == "gate":
            for name, param in self.fuser.action_parameters():
                yield f"fuser.{name}", param
        elif self.fuse_mode == "cross_attn":
            for name, param in self.fuser.action_parameters():
                yield f"fuser.{name}", param


def action_conditioning_grad_norm(predictor: ActionPredictor) -> float:
    """L2 norm of gradients in the action path. Zero when the path is absent."""
    total = None
    for _name, param in predictor.action_path_parameters():
        if param.grad is None:
            continue
        square = param.grad.detach().float().pow(2).sum()
        total = square if total is None else total + square
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


@dataclass
class ACJEPAOutput:
    """Latents produced by a single AC-JEPA forward pass."""

    z_t: Tensor
    z_hat_tp1: Tensor
    z_tp1: Tensor | None = None


class ACJEPA(nn.Module):
    """Full AC-JEPA module: online encoder, predictor, and EMA target encoder.

    Args:
        cfg: Architecture hyperparameters. ``ema_momentum`` is the target
            update rate ``τ`` (0.996 by default).
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        self.latent_dim = self.cfg.effective_latent_dim
        self.encoder = build_encoder(self.cfg)
        self.predictor = ActionPredictor(self.cfg)
        self.target_encoder = build_encoder(self.cfg)
        self.patch_predictor = None
        self.local_head = None
        if self.cfg.target_readout == "patch_tokens":
            from .patch_targets import PatchTokenPredictor

            self.patch_predictor = PatchTokenPredictor(
                self.latent_dim, self.cfg.action_dim, self.cfg.num_patches
            )
        elif self.cfg.target_readout == "local_head":
            from .patch_targets import LocalPatchHead

            self.local_head = LocalPatchHead(self.latent_dim, self.cfg.action_dim)
        elif self.cfg.target_readout not in ("cls", "weighted_pool"):
            raise ValueError(f"unknown target_readout {self.cfg.target_readout!r}.")
        self.tau = float(self.cfg.ema_momentum)
        self._copy_online_to_target()
        self._freeze_target_encoder()

    def _copy_online_to_target(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def _freeze_target_encoder(self) -> None:
        self.target_encoder.eval()
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> ACJEPA:
        """Keep the target encoder in eval mode so BN/dropout stay frozen."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters that should receive gradients (encoder + predictor)."""
        yield from self.encoder.parameters()
        yield from self.predictor.parameters()
        if self.patch_predictor is not None:
            yield from self.patch_predictor.parameters()
        if self.local_head is not None:
            yield from self.local_head.parameters()

    @torch.no_grad()
    def update_target_encoder(self, tau: float | None = None) -> None:
        """EMA update ``θ̄ ← τ θ̄ + (1 − τ) θ``.

        Buffers (e.g. BatchNorm running stats) are copied from the online
        encoder rather than interpolated, matching BYOL/JEPA practice.
        """
        momentum = self.tau if tau is None else float(tau)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"tau must lie in [0, 1], got {momentum}.")
        for online_param, target_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target_param.data.lerp_(online_param.data, 1.0 - momentum)
        for online_buf, target_buf in zip(
            self.encoder.buffers(), self.target_encoder.buffers(), strict=True
        ):
            target_buf.data.copy_(online_buf.data)

    def encode(self, x: Tensor, *, target: bool = False) -> Tensor:
        """Encode frames with the online (``target=False``) or EMA encoder."""
        if target:
            with torch.no_grad():
                return self.target_encoder(x)
        return self.encoder(x)

    def encode_patches(self, x: Tensor, *, target: bool = False) -> Tensor:
        """Patch tokens. The target branch is stop-gradient. CLS ``encode`` is unchanged."""
        encoder = self.target_encoder if target else self.encoder
        forward_patches = getattr(encoder, "forward_patches", None)
        if forward_patches is None:
            raise TypeError("patch tokens require a ViT encoder.")
        if target:
            with torch.no_grad():
                return forward_patches(x)
        return forward_patches(x)

    def forward(self, x_t: Tensor, a_t: Tensor, x_tp1: Tensor | None = None) -> ACJEPAOutput:
        """Predict ``ẑ_{t+1}`` and optionally encode the EMA target.

        Args:
            x_t: Observation batch ``(B, C, H, W)``.
            a_t: Action batch ``(B, A_dim)``.
            x_tp1: Future observation ``(B, C, H, W)``. If omitted, ``z_tp1``
                is ``None`` (inference / rollout).

        Returns:
            :class:`ACJEPAOutput` with ``z_t``, ``z_hat_tp1``, and detached
            ``z_tp1`` when ``x_tp1`` is provided.
        """
        z_t = self.encoder(x_t)
        z_hat_tp1 = self.predictor(z_t, a_t)
        z_tp1: Tensor | None = None
        if x_tp1 is not None:
            with torch.no_grad():
                z_tp1 = self.target_encoder(x_tp1)
                z_tp1 = z_tp1.detach()
        return ACJEPAOutput(z_t=z_t, z_hat_tp1=z_hat_tp1, z_tp1=z_tp1)

    def rollout(self, z_t: Tensor, actions: Tensor) -> Tensor:
        """Open-loop latent rollout ``ẑ_{t+1:t+K}`` from ``(z_t, a_{t:t+K-1})``.

        Args:
            z_t: Current latent, shape ``(B, D)``.
            actions: Action sequence, shape ``(B, K, A_dim)``.

        Returns:
            Predicted latents of shape ``(B, K, D)``, where index ``k`` is
            ``ẑ_{t+k+1}`` (``k=0`` is the one-step prediction).
        """
        if actions.ndim != 3:
            raise ValueError(f"rollout expects actions (B, K, A), got {tuple(actions.shape)}.")
        latent = z_t
        preds: list[Tensor] = []
        for step in range(actions.shape[1]):
            latent = self.predictor(latent, actions[:, step])
            preds.append(latent)
        return torch.stack(preds, dim=1)
