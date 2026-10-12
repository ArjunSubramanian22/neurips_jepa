"""Renderer invariants for the physical training raster."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import render_rgb


def _particles() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([[0.30, 0.40], [0.62, 0.55], [0.48, 0.70]], dtype=np.float32)
    velocities = np.array([[0.20, -0.10], [0.00, 0.30], [-0.40, 0.05]], dtype=np.float32)
    return positions, velocities


class RenderingInvariantTests(unittest.TestCase):
    def test_rendering_is_deterministic(self) -> None:
        positions, velocities = _particles()
        first = render_rgb(positions, velocities, 16, 1.35)
        second = render_rgb(positions.copy(), velocities.copy(), 16, 1.35)
        np.testing.assert_array_equal(first, second)

    def test_shape_dtype_and_range(self) -> None:
        positions, velocities = _particles()
        image = render_rgb(positions, velocities, 16, 1.35)
        self.assertEqual(image.shape, (3, 16, 16))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreaterEqual(int(image.min()), 0)
        self.assertLessEqual(int(image.max()), 255)
        self.assertGreater(int(image.max()), 0)

    def test_the_same_state_renders_the_same_image(self) -> None:
        positions, velocities = _particles()
        original = render_rgb(positions, velocities, 16, 1.35)
        shifted = positions + np.array([0.18, -0.12], dtype=np.float32)
        moved = render_rgb(shifted, velocities, 16, 1.35)
        repeated = render_rgb(positions, velocities, 16, 1.35)
        np.testing.assert_array_equal(repeated, original)
        self.assertFalse(np.array_equal(moved, original))


if __name__ == "__main__":
    unittest.main()
