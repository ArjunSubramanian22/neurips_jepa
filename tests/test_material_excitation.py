"""Open-loop material schedules do not read the stiffness label."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environments import mpm_legacy
from src.material_excitation import open_loop_action, simulate_episode


class MaterialScheduleTests(unittest.TestCase):
    def test_orbit_changes_sign_and_ignores_stiffness(self) -> None:
        centroid = np.array([0.5, 0.46], dtype=np.float32)
        first = open_loop_action("orbit", 0, centroid, amplitude=0.4, period=8, bend_offset=0.12)
        later = open_loop_action("orbit", 4, centroid, amplitude=0.4, period=8, bend_offset=0.12)
        self.assertLess(float(first[2]), 1e-6)
        self.assertGreater(float(first[3]), 0.0)
        self.assertLess(float(later[3]), 0.0)
        self.assertAlmostEqual(float(first[0]), 0.5)

    def test_matched_geometry_actions_do_not_depend_on_youngs(self) -> None:
        spec = mpm_legacy()
        kwargs = dict(
            episode_length=5,
            save_times=(0, 4),
            dump_seed=0,
            amplitude=0.25,
            period=4,
            bend_offset=0.1,
            max_particles=64,
        )
        soft = simulate_episode(spec, "orbit", 0, youngs_override=18.0, **kwargs)
        stiff = simulate_episode(spec, "orbit", 0, youngs_override=220.0, **kwargs)
        np.testing.assert_array_equal(soft["actions"], stiff["actions"])
        self.assertNotEqual(soft["mu"], stiff["mu"])
        self.assertTrue(np.all(np.isfinite(soft["mech"])))
        self.assertTrue(np.all(np.isfinite(stiff["mech"])))


if __name__ == "__main__":
    unittest.main()
