"""Legacy actuator defaults stay bit-identical to MPMConfig()."""

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
    MPMConfig,
    apply_action_frames,
    sample_action,
    sample_geometry,
)
from src.environments import environment_from_dict, mpm_legacy


class LegacyEnvironmentTests(unittest.TestCase):
    def test_default_config_is_the_historical_constants(self) -> None:
        cfg = MPMConfig()
        self.assertEqual(cfg.n_grid, 32)
        self.assertEqual(cfg.substeps, 8)
        self.assertEqual(cfg.dt, 1.0e-3)
        self.assertEqual(cfg.action_gain, 55.0)
        self.assertEqual(cfg.contact_radius, 0.10)
        self.assertEqual(cfg.damping, 0.08)
        self.assertEqual(cfg.contact_weight_mode, "normalized")
        self.assertEqual(cfg.force_influence, 1.0)
        self.assertEqual(cfg.center_influence, 1.0)

    def test_mpm_legacy_matches_a_default_rollout(self) -> None:
        points = sample_geometry("disk", MPMConfig().particle_spacing, np.random.default_rng(1))
        action = None
        left = MLSMPM2D()
        right = MLSMPM2D(mpm_legacy().mpm)
        left.reset(points, 70.0)
        right.reset(points, 70.0)
        action = sample_action(np.random.default_rng(2), left.x)
        apply_action_frames(left, action, 2)
        apply_action_frames(right, action, 2)
        np.testing.assert_array_equal(left.x, right.x)
        np.testing.assert_array_equal(left.v, right.v)
        np.testing.assert_array_equal(left.C, right.C)
        np.testing.assert_array_equal(left.F, right.F)
        self.assertEqual(mpm_legacy().command_hold, 3)
        self.assertTrue(left._uses_legacy_actuator())

    def test_nondefault_actuator_is_not_the_legacy_path(self) -> None:
        cfg = MPMConfig(center_influence=0.5, force_influence=1.5, contact_weight_mode="local_peak")
        sim = MLSMPM2D(cfg)
        points = sample_geometry("bar", cfg.particle_spacing, np.random.default_rng(3))
        sim.reset(points, 18.0)
        self.assertFalse(sim._uses_legacy_actuator())
        action = sample_action(np.random.default_rng(4), sim.x)
        force = sim._action_force(action)
        self.assertEqual(force.shape, sim.x.shape)
        self.assertTrue(np.all(np.isfinite(force)))

    def test_round_trip_legacy_dict(self) -> None:
        spec = mpm_legacy()
        rebuilt = environment_from_dict(spec.to_dict())
        self.assertEqual(rebuilt.mpm, spec.mpm)
        self.assertEqual(rebuilt.command_hold, 3)
        self.assertEqual(rebuilt.environment_id, "mpm_legacy")


if __name__ == "__main__":
    unittest.main()
