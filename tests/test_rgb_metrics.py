"""Known-value checks for RGB visibility metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rgb_metrics import difference_maps, frame_visibility, ssim


class RGBMetricTests(unittest.TestCase):
    def test_identical_frames_have_zero_error_and_undefined_psnr(self) -> None:
        image = np.zeros((3, 32, 32), dtype=np.float32)
        image[:, 8:20, 8:20] = 0.7
        metrics = frame_visibility(image, image.copy())
        self.assertEqual(metrics["pixel_mse"], 0.0)
        self.assertEqual(metrics["fraction_identical_uint8"], 1.0)
        self.assertEqual(metrics["fraction_affected_pixels"], 0.0)
        self.assertIsNone(metrics["psnr"])
        self.assertEqual(metrics["psnr_status"], "identical")
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=5)
        self.assertEqual(metrics["lpips"]["status"], "unavailable")
        self.assertIsNone(metrics["lpips"]["value"])

    def test_known_mse_sets_psnr(self) -> None:
        left = np.zeros((3, 16, 16), dtype=np.float32)
        right = np.full((3, 16, 16), 0.1, dtype=np.float32)
        metrics = frame_visibility(left, right)
        self.assertAlmostEqual(metrics["pixel_mse"], 0.01, places=6)
        self.assertAlmostEqual(metrics["psnr"], 20.0, places=4)
        self.assertEqual(metrics["psnr_status"], "ok")
        self.assertEqual(metrics["fraction_affected_pixels"], 1.0)
        self.assertLess(ssim(left, right), 1.0)

    def test_difference_maps_amplify_without_leaving_unit_interval(self) -> None:
        left = np.zeros((3, 8, 8), dtype=np.float32)
        right = np.full_like(left, 0.2)
        absolute, amplified = difference_maps(left, right, amplify=8.0)
        np.testing.assert_allclose(absolute, 0.2)
        np.testing.assert_allclose(amplified, 1.0)


if __name__ == "__main__":
    unittest.main()
