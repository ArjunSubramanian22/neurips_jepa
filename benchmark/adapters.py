"""Checkpoint adapters. A new model implements four methods; RSSM also overrides the fork target.

The fork target for JEPA and the latent transformer is the encoder of the final
frame. The RSSM target is the filtered posterior along the frame sequence,
because encoding the final frame alone restarts the recurrent state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

ARCHITECTURES = {
    "jepa": "ACJEPA",
    "mse": "ACJEPA",
    "noaction": "ACJEPA",
    "rssm": "CompactRSSM",
    "latent_tf": "CausalLatentDynamics",
}

# loss_type, and whether the checkpoint must condition on the action.
# None skips the conditioning check. obs_pred is refused below: it predicts pixels.
_LOSS = {
    "jepa": ("vicreg", True),
    "mse": ("mse", True),
    "noaction": ("vicreg", False),
    "rssm": ("rssm", None),
    "latent_tf": ("latent_tf", None),
}


def assert_adapter_identity(name: str, class_name: str, loss_type: str, condition_on_action: bool) -> None:
    """Refuse a checkpoint whose class, loss, or action switch does not match ``name``."""
    if name == "obs_pred" or loss_type == "obs_pred":
        raise ValueError(
            "obs_pred predicts pixels. The latent fork adapter does not score it."
        )
    if name not in ARCHITECTURES:
        raise KeyError(f"unknown model {name!r}. Known models: {sorted(ARCHITECTURES)}.")
    expected = ARCHITECTURES[name]
    if class_name != expected:
        raise ValueError(f"checkpoint architecture is {class_name}, not {expected} ({name}).")
    loss, action = _LOSS[name]
    if loss_type != loss:
        raise ValueError(f"{name} requires loss_type {loss}, got {loss_type}.")
    if action is not None and bool(condition_on_action) != action:
        raise ValueError(
            f"{name} requires condition_on_action={action}, got {bool(condition_on_action)}."
        )


def _as_frame(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    frame = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4) and frame.shape[0] not in (1, 3, 4):
        frame = frame.permute(2, 0, 1)
    if frame.ndim == 3:
        frame = frame.unsqueeze(0)
    if float(frame.detach().max()) > 1.5:
        frame = frame / 255.0
    return frame


class CheckpointAdapter:
    """A loaded checkpoint plus the training-split action normalization."""

    def __init__(self, model: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor, name: str, checkpoint: str) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        self.name = name
        self.checkpoint = checkpoint
        self.device = mean.device

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "architecture": type(self.model).__name__,
            "checkpoint": self.checkpoint,
            "action_dim": int(self.mean.numel()),
        }

    def _normalize(self, action: np.ndarray) -> torch.Tensor:
        raw = torch.as_tensor(np.asarray(action, dtype=np.float64), dtype=torch.float32, device=self.device).view(1, -1)
        return (raw - self.mean) / self.std

    def encode(self, obs: np.ndarray) -> np.ndarray:
        frame = _as_frame(obs, self.device)
        with torch.no_grad():
            latent = self.model.encode(frame)
        return latent[0].detach().cpu().numpy()

    def predict(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        sequence = self.rollout(latent, np.asarray(action, dtype=np.float64).reshape(1, -1))
        return np.asarray(sequence)[-1]

    def rollout(self, latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(np.asarray(latent), dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        raw = np.asarray(actions, dtype=np.float64).reshape(-1, self.mean.numel())
        norm = torch.stack([self._normalize(row)[0] for row in raw], dim=0).unsqueeze(0)
        with torch.no_grad():
            predicted = self.model.rollout(state, norm)
        return predicted[0].detach().cpu().numpy()

    def represent_fork(self, action: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        video = torch.as_tensor(np.asarray(frames), dtype=torch.float32, device=self.device).unsqueeze(0)
        if float(video.detach().max()) > 1.5:
            video = video / 255.0
        horizon = int(video.shape[1] - 1)
        acts = self._normalize(action).view(1, 1, -1).expand(1, horizon, -1).contiguous()
        with torch.no_grad():
            if type(self.model).__name__ == "CompactRSSM":
                predicted, target = self.model.paired_open_loop(video, acts, 0, horizon)
                pred = predicted[0, -1]
                tgt = target[0, -1]
            else:
                latent = self.model.encode(video[:, 0])
                pred = self.model.rollout(latent, acts)[0, -1]
                tgt = self.model.encode(video[:, -1], target=True)[0]
        return pred.detach().cpu().numpy(), tgt.detach().cpu().numpy()


def load_adapter(name: str, checkpoint: str, device: str = "cpu") -> CheckpointAdapter:
    """Load a checkpoint and refuse a model name that does not match its class."""
    from src.dataset import build_dataset
    from src.train import load_model_from_checkpoint

    if name == "obs_pred":
        raise ValueError("obs_pred predicts pixels. The latent fork adapter does not score it.")
    if name not in ARCHITECTURES:
        raise KeyError(f"unknown model {name!r}. Known models: {sorted(ARCHITECTURES)}.")
    torch_device = torch.device(device)
    model, cfg = load_model_from_checkpoint(checkpoint, torch_device)
    assert_adapter_identity(
        name,
        type(model).__name__,
        str(cfg.training.loss_type),
        bool(cfg.model.condition_on_action),
    )
    dataset = build_dataset(cfg.data, split="train")
    mean = dataset.action_mean.to(torch_device).float()
    std = dataset.action_std.to(torch_device).float()
    model.eval()
    return CheckpointAdapter(model, mean, std, name, checkpoint)
