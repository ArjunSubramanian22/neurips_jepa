"""RSSM rollout error stays inside one checkpoint's latent scale."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import rollout_predictions
from src.models.ac_jepa import ACJEPA
from src.models.rssm import CompactRSSM
from src.rssm_eval import rssm_counterfactual_deltas, within_model_rollout_error
from tests.test_model import _dummy_batch, _tiny_cfg
from tests.test_rssm import _cfg


class RSSMEvalTests(unittest.TestCase):
    def test_variance_normalization_removes_a_shared_scale(self) -> None:
        target = torch.randn(6, 3, 8)
        predicted = target + 0.2
        base = within_model_rollout_error(predicted, target)
        scaled = within_model_rollout_error(predicted * 4, target * 4)
        self.assertFalse(base["raw_mse_comparable_across_architectures"])
        self.assertAlmostEqual(scaled["rollout_mse"][0], base["rollout_mse"][0] * 16, places=4)
        self.assertAlmostEqual(scaled["vn_mse"][0], base["vn_mse"][0], places=4)

    def test_canonical_rollout_uses_the_recurrent_target(self) -> None:
        model = CompactRSSM(_cfg()).eval()
        frames = torch.rand(2, 5, 3, 16, 16)
        actions = torch.randn(2, 4, 4)
        predicted, target = rollout_predictions(model, frames, actions, 2, torch.device("cpu"), batch_size=2)
        direct_pred, direct_tgt = model.paired_open_loop(frames, actions, 0, 2)
        self.assertEqual(tuple(predicted.shape), (2, 2, model.latent_dim))
        self.assertTrue(torch.allclose(predicted, direct_pred))
        self.assertTrue(torch.allclose(target, direct_tgt))

    def test_jepa_rollout_is_unchanged(self) -> None:
        model = ACJEPA(_tiny_cfg())
        frames, actions, _future = _dummy_batch(2, _tiny_cfg())
        observations = torch.stack([frames, _future, frames, _future], dim=1)
        action_seq = actions.unsqueeze(1).repeat(1, 3, 1)
        predicted, target = rollout_predictions(model, observations, action_seq, 2, torch.device("cpu"), batch_size=2)
        self.assertEqual(predicted.shape[-1], model.latent_dim)
        self.assertEqual(tuple(target.shape), tuple(predicted.shape))

    def test_counterfactual_deltas_depend_on_the_action(self) -> None:
        torch.manual_seed(0)
        model = CompactRSSM(_cfg()).eval()
        frame = torch.rand(2, 3, 16, 16)
        future = torch.rand(2, 3, 16, 16)
        other = future + 0.3
        action = torch.zeros(2, 4)
        prime = action.clone()
        prime[:, 2] = 1.0
        deltas = rssm_counterfactual_deltas(model, frame, action, prime, future, other)
        self.assertEqual(tuple(deltas["delta_pred"].shape), tuple(deltas["delta_target"].shape))
        self.assertGreater(float(deltas["delta_pred"].abs().sum()), 0.0)
        self.assertFalse(deltas["comparable_to_jepa_mse"])


if __name__ == "__main__":
    unittest.main()
