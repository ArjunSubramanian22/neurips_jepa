"""The RSSM transition uses the action, and its ELBO is finite."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.rssm import CompactRSSM, RSSMWindowDataset, diagonal_kl


def _cfg(**overrides: object) -> ModelConfig:
    payload: dict[str, object] = {
        "img_size": 16,
        "in_channels": 3,
        "action_dim": 4,
        "rssm_deter": 16,
        "rssm_stoch": 8,
        "rssm_embed": 16,
        "rssm_channels": 8,
    }
    payload.update(overrides)
    return ModelConfig(**payload)  # type: ignore[arg-type]


class _Frames:
    def __init__(self) -> None:
        self.observations = torch.rand(2, 6, 3, 16, 16)
        self.actions = torch.ones(2, 5, 4)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * 3

    def _should_augment(self, _episode: int) -> bool:
        return False


class RSSMTests(unittest.TestCase):
    def test_identical_gaussians_have_zero_kl(self) -> None:
        mean = torch.randn(4, 8)
        std = torch.rand(4, 8) + 0.1
        self.assertAlmostEqual(float(diagonal_kl(mean, std, mean, std).abs().max()), 0.0, places=5)

    def test_action_changes_the_transition_and_the_image_changes_the_posterior(self) -> None:
        torch.manual_seed(0)
        model = CompactRSSM(_cfg()).eval()
        deter, stochastic = model.initial_state(2, torch.device("cpu"))
        quiet = torch.zeros(2, 4)
        pushed = quiet.clone()
        pushed[:, 2] = 1.0
        _, _, calm = model.imagine(deter, stochastic, quiet)
        _, _, moved = model.imagine(deter, stochastic, pushed)
        self.assertFalse(torch.allclose(calm, moved))
        image = torch.rand(2, 3, 16, 16)
        other = image.clone()
        other[:, :, 0, 0] += 0.4
        _, _, first = model.observe(deter, stochastic, None, image)
        _, _, second = model.observe(deter, stochastic, None, other)
        self.assertFalse(torch.allclose(first, second))

    def test_disabled_action_input_ignores_the_command(self) -> None:
        torch.manual_seed(1)
        model = CompactRSSM(_cfg(condition_on_action=False)).eval()
        deter, stochastic = model.initial_state(2, torch.device("cpu"))
        quiet = torch.zeros(2, 4)
        pushed = torch.ones(2, 4)
        _, _, calm = model.imagine(deter, stochastic, quiet)
        _, _, moved = model.imagine(deter, stochastic, pushed)
        self.assertTrue(torch.allclose(calm, moved))

    def test_open_loop_matches_the_filtered_target_shape(self) -> None:
        model = CompactRSSM(_cfg()).eval()
        frames = torch.rand(2, 5, 3, 16, 16)
        actions = torch.randn(2, 4, 4)
        predicted, target = model.paired_open_loop(frames, actions, start=1, horizon=2)
        self.assertEqual(tuple(predicted.shape), (2, 2, model.latent_dim))
        self.assertEqual(tuple(target.shape), tuple(predicted.shape))
        self.assertFalse(target.requires_grad)

    def test_loss_reaches_the_recurrent_action_path(self) -> None:
        torch.manual_seed(2)
        model = CompactRSSM(_cfg())
        frames = torch.rand(2, 4, 3, 16, 16)
        actions = torch.randn(2, 3, 4)
        terms = model.loss(frames, actions)
        terms["loss"].backward()
        self.assertGreater(float(model.rnn.weight_ih.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.action_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.decoder.fc.weight.grad.abs().sum()), 0.0)

    def test_windows_use_the_dataset_action_normalizer(self) -> None:
        dataset = RSSMWindowDataset(_Frames(), [1], 4)
        _frames, actions = dataset[0]
        self.assertEqual(tuple(actions.shape), (3, 4))
        self.assertTrue(torch.allclose(actions, torch.full_like(actions, 3.0)))


if __name__ == "__main__":
    unittest.main()
