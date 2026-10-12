"""Simulator invariants. These do not pin a training loss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import (
    MLSMPM2D,
    MPMConfig,
    apply_action_frames,
    sample_geometry,
)


def _tiny() -> MLSMPM2D:
    cfg = MPMConfig(n_grid=12, substeps=1, img_size=16, particle_spacing=0.08)
    sim = MLSMPM2D(cfg)
    sim.reset(sample_geometry("disk", cfg.particle_spacing, np.random.default_rng(0)), 70.0)
    return sim


def _recorded_actions() -> list[np.ndarray]:
    return [
        np.array([0.50, 0.46, 0.40, 0.10], dtype=np.float32),
        np.array([0.42, 0.55, -0.25, 0.30], dtype=np.float32),
    ]


class SimulatorInvariantTests(unittest.TestCase):
    def test_clone_matches_source_and_does_not_share_memory(self) -> None:
        sim = _tiny()
        cloned = sim.clone()
        np.testing.assert_array_equal(cloned.x, sim.x)
        np.testing.assert_array_equal(cloned.v, sim.v)
        np.testing.assert_array_equal(cloned.C, sim.C)
        np.testing.assert_array_equal(cloned.F, sim.F)
        self.assertEqual(cloned.mu, sim.mu)
        self.assertEqual(cloned.lam, sim.lam)
        cloned.x = cloned.x.copy()
        cloned.x[0, 0] += np.float32(0.01)
        self.assertFalse(np.array_equal(cloned.x, sim.x))

    def test_recorded_branch_matches_the_unforked_simulation(self) -> None:
        sim = _tiny()
        actions = _recorded_actions()
        unforked = sim.clone()
        for action in actions:
            apply_action_frames(unforked, action, 1)
        forked = sim.clone()
        start = forked.snapshot()
        for action in actions:
            apply_action_frames(forked, action, 1)
        forked.restore(start)
        for action in actions:
            apply_action_frames(forked, action, 1)
        np.testing.assert_array_equal(forked.x, unforked.x)
        np.testing.assert_array_equal(forked.v, unforked.v)
        np.testing.assert_array_equal(forked.C, unforked.C)
        np.testing.assert_array_equal(forked.F, unforked.F)

    def test_identical_actions_produce_identical_futures(self) -> None:
        sim = _tiny()
        action = _recorded_actions()[0]
        left = sim.clone()
        right = sim.clone()
        apply_action_frames(left, action, 2)
        apply_action_frames(right, action.copy(), 2)
        np.testing.assert_array_equal(left.x, right.x)
        np.testing.assert_array_equal(left.v, right.v)
        np.testing.assert_array_equal(left.F, right.F)

    def test_a_changed_action_can_move_particles(self) -> None:
        cfg = MPMConfig(n_grid=12, substeps=4, img_size=16, particle_spacing=0.08)
        sim = MLSMPM2D(cfg)
        sim.reset(sample_geometry("disk", cfg.particle_spacing, np.random.default_rng(0)), 70.0)
        hold = sim.clone()
        push = sim.clone()
        apply_action_frames(hold, np.zeros(4, dtype=np.float32), 4)
        apply_action_frames(push, np.array([0.50, 0.46, 1.0, 0.0], dtype=np.float32), 4)
        self.assertGreater(float(np.max(np.abs(push.x - hold.x))), 1e-4)
        self.assertGreater(float(np.max(np.abs(push.v - hold.v))), 1e-3)


if __name__ == "__main__":
    unittest.main()
