"""The excitation grid is predeclared and includes the legacy anchor once."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environments import calibration_candidates, environment_from_dict, mpm_legacy
from src.excitation_plots import plot_excitation

CONFIG = ROOT / "configs" / "experiments" / "excitation_calibration_v1.yaml"


class ExcitationGridTests(unittest.TestCase):
    def test_legacy_yaml_matches_the_code_baseline(self) -> None:
        payload = yaml.safe_load((ROOT / "configs" / "environments" / "mpm_legacy.yaml").read_text(encoding="utf-8"))
        rebuilt = environment_from_dict(payload)
        self.assertEqual(rebuilt.mpm, mpm_legacy().mpm)
        self.assertEqual(rebuilt.command_hold, 3)

    def test_grid_contains_legacy_once_and_declares_bins(self) -> None:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        specs = calibration_candidates(config)
        self.assertEqual(specs[0].environment_id, "mpm_legacy")
        ids = [spec.environment_id for spec in specs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(specs), 30)
        self.assertEqual(config["split_version"], "not_applicable")
        self.assertEqual(config["selection_fork"], "force_reversed")
        for name in ("low", "medium", "high", "persistent"):
            self.assertIn(name, config["selection"])
        legacy_copies = [
            spec
            for spec in specs[1:]
            if spec.mpm == mpm_legacy().mpm and spec.command_hold == mpm_legacy().command_hold
        ]
        self.assertEqual(legacy_copies, [])

    def test_selected_regime_files_load(self) -> None:
        legacy = mpm_legacy().mpm
        for name in ("mpm_medium_excitation", "mpm_high_excitation", "mpm_persistent_excitation"):
            path = ROOT / "configs" / "environments" / f"{name}.yaml"
            self.assertTrue(path.is_file(), name)
            spec = environment_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
            self.assertEqual(spec.environment_id, name)
            self.assertNotEqual(spec.mpm, legacy)
            self.assertGreater(spec.mpm.action_gain, legacy.action_gain)
            self.assertLess(spec.mpm.damping, legacy.damping)

    def test_plot_writes_png(self) -> None:
        rows = [
            {
                "environment_id": "mpm_legacy",
                "role": "baseline",
                "excitation_index": 1.0,
                "median_pixel_rms": 0.01,
                "median_raster_change": 0.04,
                "median_rms_pos": 1e-5,
                "median_fraction_identical": 0.96,
                "median_density_change_rate": 0.01,
                "median_velocity_change_rate": 0.04,
            },
            {
                "environment_id": "cand",
                "role": "candidate",
                "excitation_index": 8.0,
                "median_pixel_rms": 0.5,
                "median_raster_change": 0.3,
                "median_rms_pos": 0.01,
                "median_fraction_identical": 0.7,
                "median_density_change_rate": 0.1,
                "median_velocity_change_rate": 0.3,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.png"
            plot_excitation(rows, path, {"medium": "cand"})
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
