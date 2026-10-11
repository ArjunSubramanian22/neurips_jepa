"""Counterfactual alignment matches a forked target move and stops its gradient."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MPMConfig
from src.action_alignment import (
    PRIMARY_FAMILY,
    CounterfactualCache,
    PairedCounterfactualDataset,
    action_alignment_losses,
    apply_action_alignment,
    apply_intervention,
    assert_one_family,
    fork_episode,
    included_in_fraction,
    intervene_normalized,
    load_counterfactual_cache,
    same_time_donor,
)
from src.config import TrainingConfig
from src.train import _AverageMeter


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.pred = nn.Linear(4, 4)
        self.tgt = nn.Linear(4, 4)

    def predictor(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.pred(torch.tanh(z) * action + z)


class ActionAlignmentTests(unittest.TestCase):
    def test_parallel_equal_vectors_have_zero_component_losses(self) -> None:
        target = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        parts = action_alignment_losses(target.clone(), target.clone())
        self.assertAlmostEqual(float(parts["dir"]), 0.0, places=5)
        self.assertAlmostEqual(float(parts["mag"]), 0.0, places=5)
        self.assertAlmostEqual(float(parts["vec"]), 0.0, places=5)
        self.assertEqual(float(parts["n_valid"]), 2.0)

    def test_opposite_direction_is_not_a_magnitude_error(self) -> None:
        target = torch.tensor([[2.0, 0.0]])
        parts = action_alignment_losses(-target, target)
        self.assertAlmostEqual(float(parts["dir"]), 2.0, places=5)
        self.assertAlmostEqual(float(parts["mag"]), 0.0, places=5)
        self.assertAlmostEqual(float(parts["vec"]), 2.0, places=5)

    def test_log_magnitude_is_one_when_the_prediction_is_e_times_larger(self) -> None:
        target = torch.tensor([[1.0, 0.0]])
        parts = action_alignment_losses(target * np.e, target)
        self.assertAlmostEqual(float(parts["mag"]), 1.0, places=4)

    def test_near_zero_target_is_excluded_from_the_mean(self) -> None:
        good = torch.tensor([[1.0, 0.0]])
        tiny = torch.zeros(1, 2)
        pred = torch.cat([good, tiny], dim=0)
        target = torch.cat([good, tiny], dim=0)
        parts = action_alignment_losses(pred, target, norm_tau=1e-8)
        self.assertAlmostEqual(float(parts["dir"]), 0.0, places=5)
        self.assertEqual(float(parts["n_valid"]), 1.0)
        self.assertEqual(float(parts["n_excluded"]), 1.0)

    def test_target_delta_is_stop_gradient(self) -> None:
        target_map = nn.Linear(3, 3)
        delta_target = target_map(torch.ones(2, 3)) - target_map(torch.zeros(2, 3))
        delta_pred = torch.nn.Parameter(torch.ones(2, 3))
        loss = action_alignment_losses(delta_pred, delta_target)["dir"]
        loss.backward()
        self.assertIsNone(target_map.weight.grad)
        self.assertIsNotNone(delta_pred.grad)
        self.assertGreater(float(delta_pred.grad.abs().sum()), 0.0)

    def test_batch_gradient_reaches_predictor_and_not_target_encoder(self) -> None:
        torch.manual_seed(0)
        model = _Toy()
        with torch.no_grad():
            model.tgt.weight.copy_(torch.eye(4))
            model.tgt.bias.zero_()
        observation = torch.randn(6, 4)
        action = torch.randn(6, 4)
        action_prime = action.clone()
        action_prime[:, 0] += 1.0
        future = torch.randn(6, 4)
        future_prime = future + 0.4
        z_t = model.enc(observation)
        z_hat = model.predictor(z_t, action)
        with torch.no_grad():
            z_target = model.tgt(future)
        base = z_hat.new_zeros(())
        loss, _logs = apply_action_alignment(
            base,
            model.predictor,
            z_t,
            z_hat,
            z_target,
            action_prime,
            future_prime,
            model.tgt,
            torch.ones(6),
            lambda_dir=1.0,
            lambda_mag=1.0,
            lambda_vec=0.0,
        )
        loss.backward()
        self.assertIsNotNone(model.pred.weight.grad)
        self.assertGreater(float(model.pred.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.enc.weight.grad)
        self.assertGreater(float(model.enc.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(model.tgt.weight.grad)

    def test_force_families_do_not_edit_each_others_channels(self) -> None:
        action = np.array([0.2, 0.4, 0.5, -0.3], dtype=np.float32)
        reversed_action = apply_intervention(action, "force_reversal")
        self.assertTrue(np.allclose(reversed_action[:2], action[:2]))
        self.assertTrue(np.allclose(reversed_action[2:], -action[2:]))
        rotated = apply_intervention(action, "force_rotation")
        self.assertTrue(np.allclose(rotated[:2], action[:2]))
        self.assertTrue(np.allclose(rotated[2:], np.array([0.3, 0.5], dtype=np.float32)))
        scaled = apply_intervention(action, "force_scaling")
        self.assertTrue(np.allclose(scaled[2:], action[2:] * 0.5))
        shifted = apply_intervention(action, "center_shift")
        self.assertTrue(np.allclose(shifted[2:], action[2:]))
        self.assertAlmostEqual(float(shifted[0]), 0.3)

    def test_normalized_reversal_uses_raw_coordinates(self) -> None:
        mean = torch.tensor([0.0, 0.0, 1.0, 0.0])
        std = torch.ones(4)
        raw = torch.tensor([0.2, 0.3, 2.0, 0.5])
        normalized = (raw - mean) / std
        prime = intervene_normalized(normalized, mean, std, "force_reversal")
        naive = normalized.clone()
        naive[2] = -naive[2]
        self.assertFalse(torch.allclose(prime, naive))
        self.assertTrue(torch.allclose(prime * std + mean, torch.tensor([0.2, 0.3, -2.0, -0.5])))

    def test_same_time_shuffle_is_a_derangement(self) -> None:
        self.assertEqual(same_time_donor([10, 3, 7], 3), 7)
        self.assertEqual(same_time_donor([10, 3, 7], 10), 3)
        self.assertNotEqual(same_time_donor([10, 3, 7], 7), 7)

    def test_families_are_not_pooled(self) -> None:
        with self.assertRaises(ValueError):
            assert_one_family(["force_reversal", "force_scaling"])
        self.assertEqual(assert_one_family(["force_reversal", "force_reversal"]), "force_reversal")

    def test_alignment_config_requires_a_cache_and_keeps_the_primary_family(self) -> None:
        self.assertEqual(TrainingConfig().cf_intervention, PRIMARY_FAMILY)
        with self.assertRaises(ValueError):
            TrainingConfig(cf_lambda_dir=1.0, cf_cache_path=None)

    def test_skipped_steps_do_not_dilute_the_alignment_average(self) -> None:
        meter = _AverageMeter()
        meter.update({"loss": 1.0, "cf_dir": 2.0}, batch_size=2)
        meter.update({"loss": 3.0}, batch_size=2)
        means = meter.mean()
        self.assertAlmostEqual(means["loss"], 2.0)
        self.assertAlmostEqual(means["cf_dir"], 2.0)

    def test_fraction_membership_is_deterministic(self) -> None:
        first = [included_in_fraction(episode, 1, 0.25, 42) for episode in range(20)]
        second = [included_in_fraction(episode, 1, 0.25, 42) for episode in range(20)]
        self.assertEqual(first, second)
        self.assertTrue(all(included_in_fraction(episode, 0, 1.0, 1) for episode in range(5)))
        self.assertFalse(any(included_in_fraction(episode, 0, 0.0, 1) for episode in range(5)))

    def test_cache_roundtrip_and_paired_subset(self) -> None:
        cache = CounterfactualCache(
            "force_reversal",
            np.array([5, 5], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([[0.1, 0.2, -0.3, 0.4], [0.2, 0.2, -0.2, 0.2]], dtype=np.float32),
            np.zeros((2, 3, 4, 4), dtype=np.uint8),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            cache.save(path)
            loaded = load_counterfactual_cache(path)
        self.assertEqual(loaded.family, "force_reversal")
        base = _FakeBase()
        paired = PairedCounterfactualDataset(base, loaded, [0, 1], fraction=0.0, seed=0)
        _x_t, action, _x_tp1, _prime, _future, valid = paired[0]
        self.assertEqual(float(valid), 0.0)
        self.assertTrue(torch.equal(action, base[0][1]))
        always = PairedCounterfactualDataset(base, loaded, [0], fraction=1.0, seed=0)
        _x_t, action, _x_tp1, prime, future, valid = always[0]
        self.assertEqual(float(valid), 1.0)
        self.assertEqual(tuple(future.shape), (3, 4, 4))
        self.assertFalse(torch.allclose(prime, action))

    def test_fork_matches_the_replay_and_is_deterministic(self) -> None:
        from scripts.generate_trajectories import replay_episode

        mpm = MPMConfig(n_grid=12, substeps=1, img_size=16)
        first = fork_episode(0, [0], PRIMARY_FAMILY, episode_length=4, mpm=mpm)
        second = fork_episode(0, [0], PRIMARY_FAMILY, episode_length=4, mpm=mpm)
        replay = replay_episode(0, episode_length=4, mpm=mpm)
        self.assertEqual(first.family, PRIMARY_FAMILY)
        self.assertTrue(np.array_equal(first.frame_prime, second.frame_prime))
        self.assertTrue(np.allclose(first.action_prime[:, 2:], -replay["actions"][0:1, 2:]))
        self.assertTrue(np.allclose(first.action_prime[:, :2], replay["actions"][0:1, :2]))


class _FakeBase:
    def __init__(self) -> None:
        self.observations = torch.zeros(1, 3, 3, 4, 4)
        self.actions = torch.tensor([[[0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5]]])
        self._pair_index = [(0, 0), (0, 1)]
        self.global_episode_ids = (5,)

    def _should_augment(self, _episode: int) -> bool:
        return False

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * 2

    def __getitem__(self, index: int):
        local, time = self._pair_index[index]
        return self.observations[local, time], self._normalize_action(self.actions[local, time]), self.observations[local, time + 1]


if __name__ == "__main__":
    unittest.main()
