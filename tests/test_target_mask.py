"""Spatial target masks stay off the canonical CLS path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.models.ac_jepa import ACJEPA
from src.target_mask import (
    mask_weights,
    observation_difference,
    patch_scores,
    soft_topk_weights,
    soft_weights,
    weighted_patch_pool,
)
from tests.test_model import _dummy_batch, _tiny_cfg


class TargetMaskTests(unittest.TestCase):
    def test_difference_is_the_mean_absolute_image_gap(self) -> None:
        factual = torch.zeros(1, 3, 2, 2)
        counter = torch.zeros(1, 3, 2, 2)
        counter[0, :, 0, 0] = torch.tensor([1.0, 3.0, 5.0])
        gap = observation_difference(factual, counter)
        self.assertAlmostEqual(float(gap[0, 0, 0]), 3.0)
        self.assertAlmostEqual(float(gap[0, 1, 1]), 0.0)

    def test_blank_map_is_uniform_and_a_corner_takes_more_mass(self) -> None:
        blank = torch.zeros(2, 3, 8, 8)
        weights, degenerate = soft_weights(patch_scores(observation_difference(blank, blank), 4))
        self.assertTrue(bool(degenerate.all()))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 0.25)))
        counter = blank.clone()
        counter[:, :, :4, :4] = 1.0
        weights, degenerate = soft_weights(patch_scores(observation_difference(blank, counter), 4))
        self.assertFalse(bool(degenerate.any()))
        self.assertAlmostEqual(float(weights.sum()), 2.0)
        self.assertGreater(float(weights[0, 0]), float(weights[0, 1]))

    def test_soft_topk_keeps_a_tail_instead_of_a_hard_zero(self) -> None:
        scores = torch.tensor([[1.0, 1.0, 0.2, 0.0]])
        weights, degenerate = soft_topk_weights(scores, fraction=0.5, temperature=0.05)
        self.assertFalse(bool(degenerate.any()))
        self.assertGreater(float(weights[0, 2]), 0.0)
        self.assertGreater(float(weights[0, 0]), float(weights[0, 2]))
        tied, _ = soft_topk_weights(torch.ones(1, 4), fraction=0.25, temperature=0.05)
        self.assertTrue(torch.allclose(tied, torch.full_like(tied, 0.25)))

    def test_uniform_mode_ignores_the_fork(self) -> None:
        factual = torch.zeros(1, 3, 8, 8)
        counter = torch.ones(1, 3, 8, 8)
        weights, degenerate = mask_weights(
            factual, counter, mode="uniform", patch_size=4, n_patches=4
        )
        self.assertFalse(bool(degenerate.any()))
        self.assertTrue(torch.allclose(weights, torch.full((1, 4), 0.25)))

    def test_cls_forward_is_the_first_token_and_adds_no_parameters(self) -> None:
        torch.manual_seed(0)
        cfg = _tiny_cfg()
        model = ACJEPA(cfg)
        model.eval()
        frames, _actions, _future = _dummy_batch(2, cfg)
        tokens = model.encoder._tokens(frames)
        cls = model.encoder.latent_norm(model.encoder.latent_proj(tokens[:, 0]))
        patches = model.encoder.latent_norm(model.encoder.latent_proj(tokens[:, 1:]))
        self.assertTrue(torch.allclose(model.encoder(frames), cls))
        self.assertTrue(torch.allclose(model.encoder.forward_patches(frames), patches))
        self.assertEqual(patches.shape[1], cfg.num_patches)
        keys = set(model.state_dict())
        pooled = ACJEPA(_tiny_cfg(target_readout="weighted_pool"))
        self.assertEqual(keys, set(pooled.state_dict()))
        self.assertTrue(all("local_head" not in key and "patch_predictor" not in key for key in keys))

    def test_inference_does_not_read_an_optional_patch_head(self) -> None:
        cfg = _tiny_cfg(target_readout="patch_tokens")
        model = ACJEPA(cfg)
        model.eval()
        frames, actions, future = _dummy_batch(2, cfg)
        before = model(frames, actions, future)
        with torch.no_grad():
            model.patch_predictor.out.weight.add_(1.0)
        after = model(frames, actions, future)
        self.assertTrue(torch.equal(before.z_hat_tp1, after.z_hat_tp1))
        self.assertTrue(torch.equal(before.z_t, after.z_t))

    def test_pooled_target_trains_the_predictor_and_not_the_target_encoder(self) -> None:
        torch.manual_seed(1)
        model = ACJEPA(_tiny_cfg(target_readout="weighted_pool"))
        frames, actions, future = _dummy_batch(2, _tiny_cfg())
        future = frames + 0.25
        output = model(frames, actions, future)
        tokens = model.encode_patches(future, target=True)
        weights, _degenerate = soft_weights(patch_scores(observation_difference(future, frames), 8))
        loss = (output.z_hat_tp1 - weighted_patch_pool(tokens, weights)).pow(2).mean()
        loss.backward()
        self.assertGreater(float(model.predictor.action_embed[0].weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.target_encoder.parameters()))

    def test_local_head_trains_the_online_encoder_only(self) -> None:
        torch.manual_seed(2)
        model = ACJEPA(_tiny_cfg(target_readout="local_head"))
        frames, actions, future = _dummy_batch(2, _tiny_cfg())
        predicted = model.local_head(model.encode_patches(frames), actions)
        target = model.encode_patches(future, target=True)
        weights, _degenerate = mask_weights(future, frames, mode="soft", patch_size=8, n_patches=target.shape[1])
        loss = ((predicted - target).pow(2).mean(dim=-1) * weights).sum(dim=-1).mean()
        loss.backward()
        self.assertGreater(float(next(model.encoder.parameters()).grad.abs().sum()), 0.0)
        self.assertGreater(float(model.local_head.proj.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.target_encoder.parameters()))

    def test_cnn_has_no_patch_tokens(self) -> None:
        model = ACJEPA(_tiny_cfg(backbone="cnn"))
        frames, _actions, _future = _dummy_batch(2, _tiny_cfg(backbone="cnn"))
        with self.assertRaises(TypeError):
            model.encode_patches(frames)

    def test_soft_mask_cannot_replace_the_cls_baseline(self) -> None:
        self.assertEqual(ModelConfig().target_readout, "cls")
        self.assertEqual(TrainingConfig().target_mask, "uniform")
        with self.assertRaises(ValueError):
            ExperimentConfig(
                model=ModelConfig(),
                training=TrainingConfig(target_mask="soft", cf_cache_path="outputs/pairs.npz"),
                data=DataConfig(),
            )


if __name__ == "__main__":
    unittest.main()
