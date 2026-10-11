"""Compact U-Net that predicts the next observation from the current frame and action.

The action is concatenated as extra channels and also modulates the bottleneck.
It is not dropped by construction. Outputs are pixels in ``[0, 1]``, either the
physical raster or an RGB appearance image supplied by the caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import ModelConfig


class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.net(frames)


class ActionConditionedUNet(nn.Module):
    """Next-frame predictor. ``rollout`` feeds predictions back for a short horizon."""

    architecture = "obs_pred"

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.img_size % 4 != 0:
            raise ValueError("observation U-Net images must be divisible by 4.")
        width = int(cfg.obs_pred_width)
        self.action_dim = int(cfg.action_dim)
        self.condition_on_action = bool(cfg.condition_on_action)
        self.enc1 = _Block(cfg.in_channels + cfg.action_dim, width)
        self.down1 = nn.Conv2d(width, width, 4, stride=2, padding=1)
        self.enc2 = _Block(width, width * 2)
        self.down2 = nn.Conv2d(width * 2, width * 2, 4, stride=2, padding=1)
        self.bottleneck = _Block(width * 2, width * 4)
        self.film = nn.Linear(cfg.action_dim, 2 * width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
        self.dec2 = _Block(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.dec1 = _Block(width * 2, width)
        self.head = nn.Conv2d(width, cfg.in_channels, 1)

    def trainable_parameters(self):
        yield from self.parameters()

    def action_path_grad_norm(self) -> float:
        """Gradient norm through the FiLM and the action-channel input weights."""
        total = None
        for parameter in self.film.parameters():
            if parameter.grad is None:
                continue
            square = parameter.grad.detach().float().pow(2).sum()
            total = square if total is None else total + square
        first = self.enc1.net[0].weight
        if first.grad is not None:
            square = first.grad[:, -self.action_dim :].detach().float().pow(2).sum()
            total = square if total is None else total + square
        if total is None:
            return 0.0
        return float(torch.sqrt(total).item())

    def _prepare_action(self, action: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
        if not self.condition_on_action:
            action = torch.zeros_like(action)
        action_map = action.reshape(action.shape[0], self.action_dim, 1, 1).expand(
            -1, -1, frame.shape[-2], frame.shape[-1]
        )
        return action, torch.cat([frame, action_map], dim=1)

    def forward(self, frame: Tensor, action: Tensor) -> Tensor:
        if frame.ndim != 4:
            raise ValueError("frame must be (B, C, H, W).")
        action, inputs = self._prepare_action(action, frame)
        skip1 = self.enc1(inputs)
        skip2 = self.enc2(F.relu(self.down1(skip1)))
        hidden = self.bottleneck(F.relu(self.down2(skip2)))
        scale, shift = self.film(action).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        decoded = self.up2(hidden)
        decoded = self.dec2(torch.cat([decoded, skip2], dim=1))
        decoded = self.up1(decoded)
        decoded = self.dec1(torch.cat([decoded, skip1], dim=1))
        return torch.sigmoid(self.head(decoded))

    def rollout(self, frame: Tensor, actions: Tensor, *, teacher_frames: Tensor | None = None) -> Tensor:
        """Predict ``(B, K, C, H, W)``. Teacher frames replace the feedback when given."""
        if actions.ndim != 3:
            raise ValueError("actions must be (B, K, A).")
        predicted = []
        current = frame
        for step in range(actions.shape[1]):
            nxt = self.forward(current, actions[:, step])
            predicted.append(nxt)
            if teacher_frames is not None and step < teacher_frames.shape[1]:
                current = teacher_frames[:, step]
            else:
                current = nxt
        return torch.stack(predicted, dim=1)

    def prediction_loss(
        self,
        frames: Tensor,
        actions: Tensor,
        *,
        horizon: int,
        teacher_force: bool,
    ) -> dict[str, Tensor]:
        """Pixel MSE over the next ``horizon`` frames."""
        if frames.shape[1] < horizon + 1 or actions.shape[1] < horizon:
            raise ValueError("the window is shorter than the prediction horizon.")
        if teacher_force:
            predicted = self.rollout(
                frames[:, 0],
                actions[:, :horizon],
                teacher_frames=frames[:, 1:horizon],
            )
        else:
            predicted = self.rollout(frames[:, 0], actions[:, :horizon])
        teacher = frames[:, 1 : horizon + 1]
        loss = F.mse_loss(predicted, teacher)
        return {"loss": loss, "pred": predicted, "target": teacher}
