"""Known-value checks for reduced-simulator forks and the cascade figure."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MPMConfig
from src.cascade_protocol import measure_forks
from src.metrics.plots import plot_cascade_pairs


def _tiny_config() -> MPMConfig:
    return MPMConfig(
        n_grid=12,
        img_size=24,
        particle_spacing=0.05,
        substeps=2,
        dt=1.0e-3,
        damping=0.08,
        action_gain=55.0,
        contact_radius=0.10,
        render_sigma=1.2,
        gravity=0.0,
    )


class CascadeProtocolTests(unittest.TestCase):
    def test_identical_action_is_zero_on_every_measured_layer(self) -> None:
        measured = measure_forks(
            seed=0,
            n_frames=1,
            youngs=70.0,
            geometry="disk",
            interventions=["identical", "force_reversed"],
            mpm=_tiny_config(),
        )
        identical = measured["pairs"][0]
        self.assertEqual(identical["intervention"], "identical")
        self.assertEqual(identical["cascade"]["action"]["l2"], 0.0)
        self.assertEqual(identical["cascade"]["layers"]["physical"]["rms_pos"], 0.0)
        self.assertEqual(identical["cascade"]["layers"]["physical"]["rms_vel"], 0.0)
        self.assertEqual(identical["cascade"]["layers"]["observation"]["fraction_identical"], 1.0)
        self.assertEqual(identical["cascade"]["layers"]["observation"]["mse"], 0.0)
        self.assertEqual(identical["controls"]["privileged_particle_coordinates"]["contrast"]["mse"], 0.0)
        self.assertEqual(identical["controls"]["constant_predictor_standin"]["contrast"]["mse"], 0.0)
        self.assertEqual(identical["cascade"]["layers"]["representation"]["status"], "not_measured")
        self.assertEqual(identical["cascade"]["layers"]["predictor"]["status"], "not_measured")
        self.assertEqual(identical["cascade"]["layers"]["decision"]["status"], "not_measured")
        self.assertNotIn("score", identical["cascade"])
        controls = measured["synthetic_controls"]
        self.assertEqual(controls["identical_action_l2"], 0.0)
        self.assertEqual(controls["identical_latent_mse"], 0.0)
        self.assertEqual(controls["aligned_cosine"], 1.0)
        self.assertEqual(controls["opposite_cosine"], -1.0)

    def test_force_reversal_changes_the_action_vector(self) -> None:
        measured = measure_forks(
            seed=1,
            n_frames=1,
            youngs=70.0,
            geometry="disk",
            interventions=["force_reversed"],
            mpm=_tiny_config(),
        )
        pair = measured["pairs"][0]
        self.assertGreater(pair["cascade"]["action"]["l2"], 0.0)
        self.assertFalse(measured["simulator"]["comparable_to_paper_dump"])

    def test_plot_writes_a_png(self) -> None:
        measured = measure_forks(
            seed=2,
            n_frames=1,
            youngs=18.0,
            geometry="bar",
            interventions=["identical", "zero_force"],
            mpm=_tiny_config(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cascade.png"
            plot_cascade_pairs(measured["pairs"], path)
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
