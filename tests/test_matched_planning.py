"""Matched planning gives RSSM the final goal image and no goal actions."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.run_matched_planning import _score_rssm_goal_image
from scripts.run_rssm_target_controls import ablate
from src.config import ModelConfig
from src.models.rssm import CompactRSSM


class MatchedPlanningTest(unittest.TestCase):
    def test_rssm_target_does_not_observe_goal_actions(self) -> None:
        cfg = ModelConfig(img_size=16, rssm_deter=8, rssm_stoch=4, rssm_embed=8, rssm_channels=4, action_dim=4)
        model = CompactRSSM(cfg).eval()
        seen = []
        original = model.observe

        def observe(deter, stochastic, action, frame):
            seen.append(None if action is None else tuple(action.shape))
            return original(deter, stochastic, action, frame)

        model.observe = observe
        generator = np.random.default_rng(0)
        frame0 = generator.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)
        goal = generator.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)
        actions = generator.normal(size=(3, 4)).astype(np.float32)
        mean = np.zeros(4, dtype=np.float32)
        std = np.ones(4, dtype=np.float32)
        with torch.no_grad():
            distance = _score_rssm_goal_image(model, frame0, goal, actions, mean, std, torch.device("cpu"))
        self.assertTrue(np.isfinite(distance))
        self.assertEqual(seen, [None, None])
        self.assertEqual(len(seen), 2)

    def test_target_ablations_change_only_the_named_input(self) -> None:
        frame = torch.ones(1, 3, 2, 2)
        other = torch.full((1, 3, 2, 2), 2.0)
        action = torch.tensor([[1.0, -1.0]])
        other_action = torch.tensor([[0.25, 0.5]])
        blank, kept = ablate(frame, action, other, other_action, "blank_image")
        self.assertEqual(float(blank.sum()), 0.0)
        self.assertTrue(torch.equal(kept, action))
        swapped_frame, swapped_action = ablate(frame, action, other, other_action, "swapped_action")
        self.assertTrue(torch.equal(swapped_frame, frame))
        self.assertTrue(torch.equal(swapped_action, other_action))
        zero_frame, zero_action = ablate(frame, action, other, other_action, "zero_action")
        self.assertTrue(torch.equal(zero_frame, frame))
        self.assertEqual(float(zero_action.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
