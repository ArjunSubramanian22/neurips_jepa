"""Deterministic RGB renderer tests. The physical raster function is untouched."""

from __future__ import annotations

import sys
import tempfile
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
    render_rgb,
    sample_action,
    sample_geometry,
)
from src.rgb_render import (
    RENDERER_VERSION,
    preset,
    render_rgb_cached,
    render_rgb_image,
    splat_sigma_px,
    to_uint8,
)


class RGBRenderTests(unittest.TestCase):
    def test_repeat_calls_match_and_velocity_is_not_painted(self) -> None:
        cfg = preset(64)
        positions = sample_geometry("disk", 0.04, np.random.default_rng(0))
        velocities = np.zeros_like(positions)
        first = render_rgb_image(positions, velocities, cfg)
        second = render_rgb_image(positions, velocities, cfg)
        moved = velocities.copy()
        moved[:, 0] = 1.5
        third = render_rgb_image(positions, moved, cfg)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, third)
        self.assertEqual(first.shape, (3, 64, 64))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(cfg.version, RENDERER_VERSION)

    def test_sigma_scales_with_resolution(self) -> None:
        self.assertAlmostEqual(splat_sigma_px(preset(64)), 1.35)
        self.assertAlmostEqual(splat_sigma_px(preset(128)), 2.7)
        self.assertAlmostEqual(splat_sigma_px(preset(256)), 5.4)
        for resolution in (64, 128, 256):
            image = render_rgb_image(
                np.array([[0.5, 0.5]], dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
                preset(resolution),
            )
            self.assertEqual(image.shape, (3, resolution, resolution))

    def test_texture_is_deterministic_and_distractors_are_seeded(self) -> None:
        positions = np.array([[0.45, 0.50], [0.55, 0.48], [0.50, 0.55]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        plain = render_rgb_image(positions, velocities, preset(64))
        textured = render_rgb_image(positions, velocities, preset(64, texture=True))
        again = render_rgb_image(positions, velocities, preset(64, texture=True))
        other = render_rgb_image(positions, velocities, preset(64, distractors=True, distractor_seed=1))
        same = render_rgb_image(positions, velocities, preset(64, distractors=True, distractor_seed=1))
        np.testing.assert_array_equal(textured, again)
        np.testing.assert_array_equal(other, same)
        self.assertFalse(np.array_equal(plain, textured))
        self.assertFalse(np.array_equal(plain, other))

    def test_paired_forks_are_reproducible(self) -> None:
        sim = MLSMPM2D(MPMConfig(action_gain=3200.0, damping=0.01))
        points = sample_geometry("disk", sim.cfg.particle_spacing, np.random.default_rng(1))
        sim.reset(points, 70.0)
        apply_action_frames(sim, sample_action(np.random.default_rng(2), sim.x), 1)
        recorded = sim.clone()
        alternate = sim.clone()
        action = np.array([0.50, 0.46, 0.4, 0.0], dtype=np.float32)
        other = action.copy()
        other[2:] *= -1
        apply_action_frames(recorded, action, 3)
        apply_action_frames(alternate, other, 3)
        cfg = preset(64)
        recorded_a = render_rgb_image(recorded.x, recorded.v, cfg)
        recorded_b = render_rgb_image(recorded.x, recorded.v, cfg)
        alternate_image = render_rgb_image(alternate.x, alternate.v, cfg)
        np.testing.assert_array_equal(recorded_a, recorded_b)
        self.assertFalse(np.array_equal(to_uint8(recorded_a), to_uint8(alternate_image)))
        physical_recorded = render_rgb(recorded.x, recorded.v, 64, sim.cfg.render_sigma)
        physical_again = render_rgb(recorded.x, recorded.v, 64, sim.cfg.render_sigma)
        np.testing.assert_array_equal(physical_recorded, physical_again)
        self.assertEqual(physical_recorded.dtype, np.uint8)
        self.assertEqual(physical_recorded.shape, (3, 64, 64))

    def test_cache_reuses_the_file(self) -> None:
        cfg = preset(64)
        positions = np.array([[0.4, 0.4], [0.6, 0.55]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = render_rgb_cached(positions, velocities, cfg, cache)
            files = list(cache.glob("*.npy"))
            self.assertEqual(len(files), 1)
            self.assertTrue((cache / "renderer_metadata.json").is_file())
            second = render_rgb_cached(positions, velocities, cfg, cache)
            self.assertEqual(len(list(cache.glob("*.npy"))), 1)
            np.testing.assert_array_equal(first, second)
            meta = (cache / "renderer_metadata.json").read_text(encoding="utf-8")
            self.assertIn("rgb_v1", meta)


if __name__ == "__main__":
    unittest.main()
