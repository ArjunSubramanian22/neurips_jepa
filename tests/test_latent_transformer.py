"""The latent transformer reads action tokens and respects its history."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.latent_transformer import CausalLatentDynamics, history_mask


def _cfg(**overrides: object) -> ModelConfig:
    payload: dict[str, object] = {
        "img_size": 16,
        "in_channels": 3,
        "action_dim": 4,
        "latent_tf_width": 32,
        "latent_tf_layers": 2,
        "latent_tf_heads": 4,
        "latent_tf_history": 2,
        "dropout": 0.0,
    }
    payload.update(overrides)
    return ModelConfig(**payload)  # type: ignore[arg-type]


class LatentTransformerTests(unittest.TestCase):
    def test_history_mask_hides_tokens_older_than_the_window(self) -> None:
        mask = history_mask(4, history=1, device=torch.device("cpu"))
        self.assertTrue(bool(mask[0, 0, 3, 0]))
        self.assertFalse(bool(mask[0, 0, 3, 2]))
        self.assertFalse(bool(mask[0, 0, 3, 3]))
        self.assertTrue(bool(mask[0, 0, 1, 2]))

    def test_a_latent_outside_the_history_does_not_change_the_prediction(self) -> None:
        torch.manual_seed(0)
        model = CausalLatentDynamics(_cfg(latent_tf_history=1)).eval()
        latents = torch.randn(2, 3, model.width)
        actions = torch.randn(2, 3, 4)
        original = model.predict_latents(latents, actions)
        changed = latents.clone()
        changed[:, 0, 0] += 4.0
        shifted = model.predict_latents(changed, actions)
        self.assertTrue(torch.allclose(original[:, 1:], shifted[:, 1:]))
        self.assertFalse(torch.allclose(original[:, 0], shifted[:, 0]))

    def test_replacing_the_action_changes_the_prediction_and_the_gradient(self) -> None:
        torch.manual_seed(1)
        model = CausalLatentDynamics(_cfg())
        frames = torch.rand(2, 3, 3, 16, 16)
        actions = torch.randn(2, 2, 4)
        terms = model.loss(frames, actions, record_attention=True)
        terms["loss"].backward()
        self.assertGreater(model.action_embedding_grad_norm(), 0.0)
        self.assertIsNotNone(model.last_attention_action_mass)
        self.assertAlmostEqual(
            float(model.last_attention_action_mass) + float(model.last_attention_latent_mass),
            1.0,
            places=5,
        )
        replacement = actions.clone()
        replacement[:, :, 2] *= -1
        self.assertGreater(model.action_replacement_sensitivity(frames, actions, replacement), 0.0)

    def test_autoregressive_rollout_has_one_latent_per_action(self) -> None:
        model = CausalLatentDynamics(_cfg(latent_tf_train_mode="autoregressive")).eval()
        frames = torch.rand(2, 4, 3, 16, 16)
        actions = torch.randn(2, 3, 4)
        predicted = model.rollout(model.encode(frames[:, 0]), actions)
        self.assertEqual(tuple(predicted.shape), (2, 3, model.latent_dim))
        terms = model.loss(frames, actions)
        self.assertEqual(tuple(terms["pred"].shape), tuple(predicted.shape))

    def test_canonical_rollout_matches_autoregressive_latents(self) -> None:
        from src.eval import rollout_predictions

        model = CausalLatentDynamics(_cfg()).eval()
        frames = torch.rand(2, 4, 3, 16, 16)
        actions = torch.randn(2, 3, 4)
        predicted, target = rollout_predictions(model, frames, actions, 2, torch.device("cpu"), batch_size=2)
        direct = model.rollout(model.encode(frames[:, 0]), actions[:, :2])
        self.assertEqual(tuple(predicted.shape), (2, 2, model.latent_dim))
        self.assertTrue(torch.allclose(predicted, direct))
        self.assertEqual(tuple(target.shape), tuple(predicted.shape))

    def test_disabled_actions_are_not_the_default_construction(self) -> None:
        self.assertTrue(CausalLatentDynamics(_cfg()).condition_on_action)
        model = CausalLatentDynamics(_cfg(condition_on_action=False)).eval()
        latents = torch.randn(2, 2, model.width)
        quiet = torch.zeros(2, 2, 4)
        pushed = torch.ones(2, 2, 4)
        self.assertTrue(torch.allclose(model.predict_latents(latents, quiet), model.predict_latents(latents, pushed)))


if __name__ == "__main__":
    unittest.main()
