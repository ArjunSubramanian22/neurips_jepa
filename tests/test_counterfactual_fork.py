"""Exact MLS-MPM snapshot / clone / restore tests (Phase 1A)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import (
    MLSMPM2D,
    apply_action_frames,
    generate_episode,
    replay_episode,
    sample_action,
    sample_geometry,
)


def _assert_state_close(a: MLSMPM2D, b: MLSMPM2D, places: int = 7) -> None:
    np.testing.assert_allclose(a.x, b.x, rtol=0.0, atol=10 ** (-places))
    np.testing.assert_allclose(a.v, b.v, rtol=0.0, atol=10 ** (-places))
    np.testing.assert_allclose(a.C, b.C, rtol=0.0, atol=10 ** (-places))
    np.testing.assert_allclose(a.F, b.F, rtol=0.0, atol=10 ** (-places))
    unittest.TestCase().assertAlmostEqual(a.mu, b.mu, places=places)
    unittest.TestCase().assertAlmostEqual(a.lam, b.lam, places=places)


class SnapshotForkTests(unittest.TestCase):
    def _fresh(self, seed: int = 0) -> tuple[MLSMPM2D, np.ndarray]:
        rng = np.random.default_rng(seed)
        sim = MLSMPM2D()
        pts = sample_geometry("disk", sim.cfg.particle_spacing, rng)
        sim.reset(pts, 70.0)
        action = sample_action(rng, sim.x)
        apply_action_frames(sim, action, 2)
        return sim, action

    def test_snapshot_restore_same_action_identical(self) -> None:
        sim, action = self._fresh(1)
        snap = sim.snapshot()
        apply_action_frames(sim, action, 1)
        after = sim.snapshot()
        sim.restore(snap)
        apply_action_frames(sim, action, 1)
        np.testing.assert_array_equal(sim.x, after["x"])
        np.testing.assert_array_equal(sim.v, after["v"])
        np.testing.assert_array_equal(sim.C, after["C"])
        np.testing.assert_array_equal(sim.F, after["F"])

    def test_cloned_branches_same_action_identical(self) -> None:
        sim, action = self._fresh(2)
        left = sim.clone()
        right = sim.clone()
        apply_action_frames(left, action, 3)
        apply_action_frames(right, action, 3)
        _assert_state_close(left, right)

    def test_alternate_action_does_not_contaminate_original(self) -> None:
        sim, action = self._fresh(3)
        rng = np.random.default_rng(99)
        alt = sample_action(rng, sim.x)
        orig = sim.clone()
        branch = sim.clone()
        apply_action_frames(branch, alt, 2)
        np.testing.assert_array_equal(sim.x, orig.x)
        np.testing.assert_array_equal(sim.v, orig.v)
        self.assertFalse(np.allclose(branch.x, orig.x) and np.allclose(branch.v, orig.v))

    def test_geometry_material_unchanged_on_restore(self) -> None:
        sim, action = self._fresh(4)
        mu, lam = sim.mu, sim.lam
        n = sim.x.shape[0]
        snap = sim.snapshot()
        apply_action_frames(sim, action, 4)
        sim.restore(snap)
        self.assertEqual(sim.x.shape[0], n)
        self.assertEqual(sim.mu, mu)
        self.assertEqual(sim.lam, lam)

    def test_substep_has_no_hidden_rng(self) -> None:
        sim, action = self._fresh(5)
        a = sim.clone()
        b = sim.clone()
        np.random.seed(123)
        apply_action_frames(a, action, 2)
        np.random.seed(999)
        apply_action_frames(b, action, 2)
        _assert_state_close(a, b)

    def test_replay_matches_generate_episode_frames(self) -> None:
        ep = 7
        replay = replay_episode(ep, dump_seed=0, episode_length=8)
        rng = np.random.default_rng(0 + 10007 * ep)
        sim = MLSMPM2D()
        class_id = ep % 3
        geom_id = (ep // 3) % 2
        frames, acts = generate_episode(
            sim,
            geometry=("disk", "bar")[geom_id],
            youngs=(18.0, 70.0, 220.0)[class_id],
            episode_length=8,
            rng=rng,
        )
        np.testing.assert_array_equal(replay["frames"], frames)
        np.testing.assert_array_equal(replay["actions"], acts)

    def test_dump_replay_matches_stored_episode_if_present(self) -> None:
        dump = ROOT / "outputs" / "physics_mpm"
        obs_path = dump / "observations.npy"
        act_path = dump / "actions.npy"
        if not obs_path.is_file() or not act_path.is_file():
            self.skipTest("physics dump not on disk")
        obs = np.load(obs_path, mmap_mode="r")
        acts = np.load(act_path, mmap_mode="r")
        ep = 1800
        replay = replay_episode(ep, dump_seed=0, episode_length=int(obs.shape[1]))
        np.testing.assert_array_equal(replay["frames"], obs[ep])
        np.testing.assert_array_equal(replay["actions"], acts[ep])


if __name__ == "__main__":
    unittest.main()
