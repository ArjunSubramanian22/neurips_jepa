"""Known-value checks for excitation metrics and regime selection."""

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
from src.excitation import (
    STABILITY_LIMITS,
    aggregate_environment,
    channel_change_rates,
    correlation,
    excitation_index,
    intervene,
    select_regimes,
    stability_report,
)
from src.environments import mpm_legacy

CRITERIA = {
    "low": {"raster_change_max": 0.10, "pixel_rms_max": 0.05, "target_raster_change": 0.05},
    "medium": {
        "raster_change_min": 0.15,
        "raster_change_max": 0.45,
        "pixel_rms_min": 0.05,
        "pixel_rms_max": 2.0,
        "target_raster_change": 0.30,
    },
    "high": {
        "raster_change_min": 0.55,
        "raster_change_max": 0.90,
        "pixel_rms_min": 0.5,
        "pixel_rms_max": 8.0,
        "target_raster_change": 0.72,
    },
    "persistent": {
        "min_command_hold": 8,
        "max_damping": 0.03,
        "min_raster_change": 0.15,
        "max_pixel_rms": 8.0,
        "target_raster_change": 0.40,
    },
}


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "environment_id": "cand",
        "role": "candidate",
        "damping": 0.03,
        "action_gain": 55.0,
        "contact_radius": 0.10,
        "command_hold": 3,
        "contact_weight_mode": "normalized",
        "center_influence": 1.0,
        "force_influence": 1.0,
        "excitation_index": 1.0,
        "selection_fork": "force_reversed",
        "n_states": 6,
        "stable": True,
        "median_clip_fraction": 0.0,
        "median_svd_clip_fraction": 0.0,
        "max_speed": 1.0,
        "median_rms_pos": 0.01,
        "median_rms_vel": 0.1,
        "median_pixel_rms": 0.63,
        "median_raster_mse": 0.01,
        "median_fraction_identical": 0.70,
        "median_raster_change": 0.30,
        "median_density_change_rate": 0.1,
        "median_velocity_change_rate": 0.3,
        "median_action_l2": 1.0,
        "action_rms_pearson": 0.5,
        "action_rms_spearman": 0.5,
        "action_rms_correlation_status": "ok",
        "action_rms_n": 12,
    }
    base.update(overrides)
    return base


class ExcitationMetricTests(unittest.TestCase):
    def test_channel_change_rates(self) -> None:
        left = np.zeros((3, 4, 4), dtype=np.uint8)
        right = left.copy()
        right[0, 0, 0] = 255
        right[1, 1, 1] = 10
        rates = channel_change_rates(left, right)
        self.assertAlmostEqual(rates["density_change_rate"], 1.0 / 16.0)
        self.assertAlmostEqual(rates["velocity_change_rate"], 1.0 / 16.0)

    def test_correlation_extremes_and_undefined(self) -> None:
        aligned = correlation([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
        self.assertAlmostEqual(aligned["pearson"], 1.0, places=12)
        opposite = correlation([0.0, 1.0, 2.0], [5.0, 3.0, 1.0])
        self.assertAlmostEqual(opposite["pearson"], -1.0, places=12)
        flat = correlation([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
        self.assertIsNone(flat["pearson"])
        self.assertEqual(flat["status"], "undefined_zero_variance")

    def test_excitation_index_is_one_at_legacy(self) -> None:
        legacy = mpm_legacy()
        self.assertAlmostEqual(
            excitation_index(legacy.mpm.action_gain, legacy.mpm.damping, legacy.command_hold, legacy.mpm.contact_radius),
            1.0,
        )

    def test_intervene_identical_is_a_copy(self) -> None:
        schedule = [np.array([0.2, 0.3, 0.4, -0.5], dtype=np.float32)]
        out = intervene(schedule, "identical", 3, 0.1)
        np.testing.assert_array_equal(out[0], schedule[0])
        out[0][0] = 9.0
        self.assertAlmostEqual(float(schedule[0][0]), 0.2)

    def test_legacy_short_roll_is_stable_and_clipped_state_is_not(self) -> None:
        cfg = MPMConfig()
        sim = MLSMPM2D(cfg)
        points = sample_geometry("disk", cfg.particle_spacing, np.random.default_rng(0))
        sim.reset(points, 70.0)
        apply_action_frames(sim, sample_action(np.random.default_rng(1), sim.x), 2)
        report = stability_report(sim, STABILITY_LIMITS)
        self.assertTrue(report["stable"])
        sim.x[:, :] = np.float32(0.02)
        clipped = stability_report(sim, STABILITY_LIMITS)
        self.assertFalse(clipped["stable"])
        self.assertGreater(clipped["clip_fraction"], STABILITY_LIMITS["max_clip_fraction"])


class RegimeSelectionTests(unittest.TestCase):
    def test_selection_uses_bins_and_skips_unstable_and_huge(self) -> None:
        rows = [
            _row(environment_id="mpm_legacy", role="baseline", median_raster_change=0.04, median_pixel_rms=0.01, median_rms_pos=1e-5, damping=0.08, action_gain=55, command_hold=3),
            _row(environment_id="too_big", median_rms_pos=0.5, median_raster_change=0.30, median_pixel_rms=0.6),
            _row(environment_id="unstable", stable=False, median_raster_change=0.30, median_pixel_rms=0.6),
            _row(environment_id="medium_far", median_raster_change=0.16, median_pixel_rms=0.4, action_gain=800),
            _row(environment_id="medium_near", median_raster_change=0.31, median_pixel_rms=0.5, action_gain=200),
            _row(environment_id="high_near", median_raster_change=0.70, median_pixel_rms=1.5, action_gain=800, command_hold=3),
            _row(environment_id="persist", median_raster_change=0.42, median_pixel_rms=1.2, action_gain=200, command_hold=8, damping=0.01),
            _row(environment_id="persist_far", median_raster_change=0.80, median_pixel_rms=2.0, action_gain=55, command_hold=16, damping=0.01),
        ]
        chosen = select_regimes(rows, CRITERIA)
        self.assertEqual(chosen["low"]["environment_id"], "mpm_legacy")
        self.assertEqual(chosen["medium"]["environment_id"], "medium_near")
        self.assertEqual(chosen["high"]["environment_id"], "high_near")
        self.assertEqual(chosen["persistent"]["environment_id"], "persist")
        self.assertNotEqual(chosen["medium"]["environment_id"], chosen["persistent"]["environment_id"])

    def test_empty_band_stays_unfilled(self) -> None:
        chosen = select_regimes([_row(environment_id="only", median_raster_change=0.99, median_pixel_rms=20.0, median_rms_pos=0.5)], CRITERIA)
        self.assertEqual(chosen["medium"]["status"], "UNFILLED")
        self.assertIsNone(chosen["medium"]["environment_id"])

    def test_aggregate_correlation_excludes_identical_fork(self) -> None:
        rows = []
        for state in range(4):
            rows.append(
                {
                    "environment_id": "agg",
                    "role": "candidate",
                    "damping": 0.08,
                    "action_gain": 55.0,
                    "contact_radius": 0.1,
                    "command_hold": 3,
                    "contact_weight_mode": "normalized",
                    "center_influence": 1.0,
                    "force_influence": 1.0,
                    "excitation_index": 1.0,
                    "fork": "identical",
                    "stable": True,
                    "clip_fraction": 0.0,
                    "svd_clip_fraction": 0.0,
                    "max_speed": 0.1,
                    "action_l2": 0.0,
                    "rms_pos": 0.0,
                    "rms_vel": 0.0,
                    "pixel_rms": 0.0,
                    "raster_mse": 0.0,
                    "fraction_identical": 1.0,
                    "raster_change": 0.0,
                    "density_change_rate": 0.0,
                    "velocity_change_rate": 0.0,
                }
            )
            rows.append(
                {
                    **rows[-1],
                    "fork": "force_reversed",
                    "action_l2": float(state + 1),
                    "rms_pos": float(state + 1) * 0.01,
                    "rms_vel": 0.2,
                    "pixel_rms": 0.1,
                    "raster_mse": 0.01,
                    "fraction_identical": 0.8,
                    "raster_change": 0.2,
                    "density_change_rate": 0.05,
                    "velocity_change_rate": 0.2,
                }
            )
        summary = aggregate_environment(rows)
        self.assertEqual(summary["action_rms_n"], 4)
        self.assertAlmostEqual(summary["action_rms_pearson"], 1.0, places=12)
        self.assertAlmostEqual(summary["median_raster_change"], 0.2)


if __name__ == "__main__":
    unittest.main()
