"""Planning figures write a GIF and a distance curve from frame stacks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import os

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning_figures import as_time_channel_frames, plot_distance_curves, write_rollout_gif


class PlanningFigureTests(unittest.TestCase):
    def test_channel_order_and_gif_round_trip(self) -> None:
        frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        frames[0, :, :, 0] = 40
        frames[1, :, :, 0] = 120
        frames[2, :, :, 0] = 220
        converted = as_time_channel_frames(frames)
        self.assertEqual(converted.shape, (3, 3, 8, 8))
        self.assertEqual(int(converted[0, 0, 0, 0]), 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roll.gif"
            curve = Path(tmp) / "curve.png"
            write_rollout_gif({"oracle": converted, "random": converted}, path)
            plot_distance_curves({"oracle": [0.2, 0.1], "random": [0.2, 0.3]}, curve, title="toy")
            self.assertGreater(path.stat().st_size, 0)
            self.assertGreater(curve.stat().st_size, 0)
            from PIL import Image

            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 3)


if __name__ == "__main__":
    unittest.main()
