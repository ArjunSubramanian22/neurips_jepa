"""Teacher-forced physical rows must use the horizon frame, not the latent slot."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.correct_physical_index import correct_rows
from scripts.run_mpm_high_pretest import _raster_metrics, _state_metrics


def _blob(root: Path) -> None:
    folder = root / "mpm_high"
    folder.mkdir(parents=True)
    pos = np.zeros((6, 10, 2, 2), dtype=np.float32)
    vel = np.zeros_like(pos)
    frames_rec = np.zeros((6, 10, 3, 4, 4), dtype=np.uint8)
    frames_rev = np.zeros_like(frames_rec)
    for frame in range(10):
        pos[0, frame] = np.float32(frame)
        vel[0, frame] = np.float32(frame) / np.float32(10)
        frames_rec[0, frame] = np.uint8(frame)
    np.savez_compressed(
        folder / "ep7.npz",
        pos_rec=pos,
        pos_rev=np.zeros_like(pos),
        vel_rec=vel,
        vel_rev=np.zeros_like(vel),
        frames_rec=frames_rec,
        frames_rev=frames_rev,
    )


class PhysicalIndexCorrectionTest(unittest.TestCase):
    def test_horizon_five_uses_frame_four(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _blob(root)
            stored = _state_metrics(
                np.full((2, 2), 1, dtype=np.float32),
                np.full((2, 2), 0.1, dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
            )
            raster = _raster_metrics(np.full((3, 4, 4), 1, dtype=np.uint8), np.zeros((3, 4, 4), dtype=np.uint8))
            wrong = {
                "fork_id": "mpm_high:7:0:force_reversal",
                "regime": "mpm_high",
                "model": "vicreg_action",
                "protocol": "teacher_forced",
                "horizon": "5",
                "episode": "7",
                "branch_time": "0",
                "cosine": "0.5",
                "response_ratio": "1",
                "target_norm": "1",
                "pred_norm": "1",
                **{key: stored[key] for key in ("rms_pos", "rms_vel", "com_displacement")},
                "raster_mse": raster["raster_mse"],
                "raster_l1": raster["raster_l1"],
                "identical_raster": raster["identical_raster"],
                "changed_pixel_fraction": raster["changed_pixel_fraction"],
            }
            expected = _state_metrics(
                np.full((2, 2), 4, dtype=np.float32),
                np.full((2, 2), 0.4, dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
            )
            open_loop = dict(wrong)
            open_loop["protocol"] = "open_loop"
            open_loop["horizon"] = "5"
            for key in ("rms_pos", "rms_vel", "com_displacement"):
                open_loop[key] = expected[key]
            open_raster = _raster_metrics(np.full((3, 4, 4), 4, dtype=np.uint8), np.zeros((3, 4, 4), dtype=np.uint8))
            open_loop["raster_mse"] = open_raster["raster_mse"]
            open_loop["raster_l1"] = open_raster["raster_l1"]
            open_loop["identical_raster"] = open_raster["identical_raster"]
            open_loop["changed_pixel_fraction"] = open_raster["changed_pixel_fraction"]
            rows = [wrong, open_loop]
            summary = correct_rows(rows, root)
            self.assertEqual(summary["n_recomputed"], 1)
            self.assertEqual(summary["n_checked_unchanged"], 1)
            self.assertAlmostEqual(float(rows[0]["rms_pos"]), expected["rms_pos"])
            self.assertEqual(rows[0]["physical_step"], 4)
            self.assertEqual(rows[0]["cosine"], "0.5")
            self.assertEqual(rows[1]["physical_metric_version"], "v1_unchanged")

    def test_open_loop_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _blob(root)
            row = {
                "fork_id": "mpm_high:7:0:force_reversal",
                "regime": "mpm_high",
                "model": "mse_action",
                "protocol": "open_loop",
                "horizon": "1",
                "episode": "7",
                "branch_time": "0",
                "rms_pos": "99",
                "rms_vel": "99",
                "com_displacement": "99",
                "raster_mse": "99",
                "raster_l1": "99",
                "identical_raster": "0",
                "changed_pixel_fraction": "0",
                "cosine": "",
                "response_ratio": "",
                "target_norm": "1",
                "pred_norm": "1",
            }
            with self.assertRaises(RuntimeError):
                correct_rows([row], root)


if __name__ == "__main__":
    unittest.main()
