"""The observation U-Net uses the action and can predict a short pixel future."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.obs_pred import ActionConditionedUNet
from src.rgb_render import RGBRenderConfig, render_rgb_image


def _cfg(**overrides: object) -> ModelConfig:
    payload: dict[str, object] = {
        "img_size": 16,
        "in_channels": 3,
        "action_dim": 4,
        "obs_pred_width": 8,
    }
    payload.update(overrides)
    return ModelConfig(**payload)  # type: ignore[arg-type]


class ObservationPredictorTests(unittest.TestCase):
    def test_changing_the_action_changes_the_predicted_frame(self) -> None:
        torch.manual_seed(0)
        model = ActionConditionedUNet(_cfg()).eval()
        frame = torch.rand(2, 3, 16, 16)
        quiet = torch.zeros(2, 4)
        pushed = quiet.clone()
        pushed[:, 2] = 1.0
        self.assertFalse(torch.allclose(model(frame, quiet), model(frame, pushed)))

    def test_disabled_action_input_ignores_the_command(self) -> None:
        torch.manual_seed(1)
        model = ActionConditionedUNet(_cfg(condition_on_action=False)).eval()
        frame = torch.rand(2, 3, 16, 16)
        self.assertTrue(torch.allclose(model(frame, torch.zeros(2, 4)), model(frame, torch.ones(2, 4))))

    def test_teacher_forced_horizon_matches_the_true_prefix(self) -> None:
        torch.manual_seed(2)
        model = ActionConditionedUNet(_cfg())
        frames = torch.rand(2, 4, 3, 16, 16)
        actions = torch.randn(2, 3, 4)
        terms = model.prediction_loss(frames, actions, horizon=2, teacher_force=True)
        terms["loss"].backward()
        self.assertEqual(tuple(terms["pred"].shape), (2, 2, 3, 16, 16))
        self.assertGreater(model.action_path_grad_norm(), 0.0)
        autoregressive = model.rollout(frames[:, 0], actions[:, :2])
        self.assertEqual(tuple(autoregressive.shape), tuple(terms["pred"].shape))

    def test_rgb_target_is_not_the_physical_raster(self) -> None:
        positions = torch.tensor([[0.45, 0.50], [0.55, 0.50]]).numpy()
        velocities = torch.zeros(2, 2).numpy()
        image = render_rgb_image(positions, velocities, RGBRenderConfig(resolution=64))
        self.assertEqual(image.shape, (3, 64, 64))
        self.assertGreater(float(image.max()), 0.2)
        self.assertLess(float(image.min()), 0.95)


if __name__ == "__main__":
    unittest.main()
