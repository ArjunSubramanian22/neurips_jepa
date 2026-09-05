"""Tests for synthetic and physics-dump dataset loaders."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.dataset import (
    ConsistentFrameAugment,
    DeformablePhysicsDataset,
    canonicalize_observations,
    load_physics_trajectories,
)
from scripts.generate_trajectories import GEOMETRIES, STIFFNESS_E, generate_dataset
from scripts.validate_dataset import validate_physics_dataset


def _write_npy_dump(root: Path, n: int = 6, t: int = 20, h: int = 40, a_dim: int = 4) -> Path:
    obs = np.random.rand(n, t, h, h, 3).astype(np.float32)  # channel-last RGB
    act = np.random.randn(n, t - 1, a_dim).astype(np.float32)
    labels = np.arange(n) % 3
    np.save(root / "observations.npy", obs)
    np.save(root / "actions.npy", act)
    np.save(root / "stiffness_labels.npy", labels)
    return root


class CanonicalizeTests(unittest.TestCase):
    def test_channel_last_and_resize(self) -> None:
        obs = torch.rand(2, 5, 48, 48, 3)
        out = canonicalize_observations(obs, img_size=32)
        self.assertEqual(tuple(out.shape), (2, 5, 3, 32, 32))


class ConsistentAugmentTests(unittest.TestCase):
    def test_shared_params_keep_identical_frames_equal(self) -> None:
        cfg = DataConfig(img_size=32, augment=True, dataset_path=None)
        aug = ConsistentFrameAugment(cfg)
        frame = torch.rand(1, 3, 40, 40)
        stacked = torch.cat([frame, frame.clone()], dim=0)
        torch.manual_seed(0)
        out = aug(stacked)
        self.assertEqual(tuple(out.shape), (2, 3, 32, 32))
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-6))


class PhysicsDatasetTests(unittest.TestCase):
    def test_npy_directory_chunking_and_action_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump = _write_npy_dump(Path(tmp))
            cfg = DataConfig(
                dataset_path=str(dump),
                img_size=32,
                in_channels=3,
                action_dim=4,
                chunk_length=16,
                chunk_stride=8,
                val_fraction=0.25,
                augment=False,
                normalize_actions=True,
                n_stiffness_classes=3,
            )
            ds = DeformablePhysicsDataset(cfg, split="all")
            self.assertEqual(ds.n_episodes, 6)
            self.assertGreater(len(ds), 0)
            x_t, a_t, x_tp1 = ds[0]
            self.assertEqual(tuple(x_t.shape), (3, 32, 32))
            self.assertEqual(tuple(a_t.shape), (4,))
            self.assertEqual(tuple(x_tp1.shape), (3, 32, 32))
            train_idx = ds.indices_for_episodes(0, ds.split_sizes()[0])
            self.assertTrue(train_idx)
            frames, acts = ds.get_chunk(0, 0)
            self.assertEqual(tuple(frames.shape), (16, 3, 32, 32))
            self.assertEqual(tuple(acts.shape), (15, 4))
            # z-scored train actions should be roughly zero-mean
            z = ds.model_actions[: ds.split_sizes()[0]].reshape(-1, 4)
            self.assertLess(float(z.mean().abs()), 0.25)

    def test_hdf5_flat_layout(self) -> None:
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.h5"
            obs = np.random.rand(4, 12, 3, 24, 24).astype(np.float32)
            act = np.random.randn(4, 11, 6).astype(np.float32)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("observations", data=obs)
                handle.create_dataset("actions", data=act)
            loaded_obs, loaded_act, labels = load_physics_trajectories(path)
            self.assertEqual(tuple(loaded_obs.shape), (4, 12, 3, 24, 24))
            self.assertEqual(tuple(loaded_act.shape), (4, 11, 6))
            self.assertEqual(int(labels.shape[0]), 4)
            cfg = DataConfig(
                dataset_path=str(path),
                img_size=32,
                in_channels=3,
                action_dim=6,
                chunk_length=8,
                augment=True,
                normalize_actions=True,
            )
            ds = DeformablePhysicsDataset(cfg, split="train")
            x_t, a_t, x_tp1 = ds[0]
            self.assertEqual(tuple(x_t.shape), (3, 32, 32))
            self.assertEqual(int(a_t.numel()), 6)


class GeneratedPhysicsPipelineTests(unittest.TestCase):
    """CI fixture: tiny MLS-MPM dump through validate_dataset (no extra sim install)."""

    def test_validate_small_generated_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "fixture"
            generate_dataset(
                dump,
                n_episodes=6,
                episode_length=8,
                seed=0,
                img_size=64,
            )
            self.assertTrue((dump / "observations.npy").is_file())
            self.assertTrue((dump / "actions.npy").is_file())
            self.assertTrue((dump / "stiffness_labels.npy").is_file())
            self.assertTrue((dump / "stiffness_values.npy").is_file())
            values = np.load(dump / "stiffness_values.npy")
            labels = np.load(dump / "stiffness_labels.npy")
            self.assertEqual(values.shape, labels.shape)
            self.assertTrue(np.allclose(values, np.asarray(STIFFNESS_E)[labels]))
            self.assertEqual(len(GEOMETRIES), 2)
            self.assertEqual(len(STIFFNESS_E), 3)
            report = validate_physics_dataset(
                dump,
                min_episodes=6,
                n_stiffness_classes=3,
                min_geometries=2,
                balance_min_frac=0.15,
            )
            self.assertEqual(report["n_episodes"], 6)
            self.assertEqual(report["episode_length"], 8)
            self.assertEqual(report["img_size"], 64)
            self.assertEqual(report["in_channels"], 3)
            self.assertEqual(report["action_dim"], 4)
            self.assertEqual(report["n_geometries"], 2)
            self.assertTrue(report["finite_observations"])
            self.assertTrue(report["finite_actions"])
            obs = np.load(dump / "observations.npy")
            self.assertEqual(obs.dtype, np.uint8)
            self.assertEqual(tuple(obs.shape), (6, 8, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
