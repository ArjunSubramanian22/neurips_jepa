"""Forward and action-conditioning invariants at initialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.ac_jepa import ACJEPA


def _model() -> ACJEPA:
    return ACJEPA(
        ModelConfig(
            in_channels=3,
            img_size=32,
            patch_size=8,
            embed_dim=32,
            encoder_depth=2,
            encoder_num_heads=4,
            mlp_ratio=2.0,
            action_dim=4,
            action_embed_dim=16,
            predictor_embed_dim=32,
            predictor_depth=2,
            predictor_num_heads=4,
            latent_dim=0,
            ema_momentum=0.996,
            dropout=0.0,
            drop_path=0.0,
            backbone="vit",
            fuse_mode="concat",
            condition_on_action=True,
        )
    )


class ModelInvariantTests(unittest.TestCase):
    def test_forward_shapes_match_the_batch_and_latent_width(self) -> None:
        torch.manual_seed(0)
        model = _model()
        frames = torch.randn(2, 3, 32, 32)
        actions = torch.randn(2, 4)
        future = torch.randn(2, 3, 32, 32)
        out = model(frames, actions, future)
        self.assertEqual(tuple(out.z_t.shape), (2, model.latent_dim))
        self.assertEqual(tuple(out.z_hat_tp1.shape), (2, model.latent_dim))
        self.assertEqual(tuple(out.z_tp1.shape), (2, model.latent_dim))
        self.assertTrue(torch.isfinite(out.z_hat_tp1).all())

    def test_an_action_change_reaches_the_action_embedding(self) -> None:
        torch.manual_seed(0)
        model = _model()
        seen: list[torch.Tensor] = []

        def _record(_module, inputs, _output) -> None:
            seen.append(inputs[0].detach().clone())

        handle = model.predictor.action_embed.register_forward_hook(_record)
        try:
            latent = torch.randn(2, model.latent_dim)
            quiet = torch.zeros(2, 4)
            loud = quiet.clone()
            loud[:, 2] = 1.0
            model.predictor(latent, quiet)
            model.predictor(latent, loud)
        finally:
            handle.remove()
        self.assertEqual(len(seen), 2)
        self.assertFalse(torch.allclose(seen[0], seen[1]))

    def test_prediction_loss_reaches_action_embedding_parameters(self) -> None:
        torch.manual_seed(0)
        model = _model()
        frames = torch.randn(2, 3, 32, 32)
        actions = torch.randn(2, 4)
        future = torch.randn(2, 3, 32, 32)
        out = model(frames, actions, future)
        out.z_hat_tp1.square().mean().backward()
        grads = [
            parameter.grad
            for name, parameter in model.predictor.named_parameters()
            if "action_embed" in name and parameter.requires_grad
        ]
        self.assertTrue(grads)
        self.assertTrue(all(grad is not None and float(grad.abs().sum()) > 0.0 for grad in grads))


if __name__ == "__main__":
    unittest.main()
