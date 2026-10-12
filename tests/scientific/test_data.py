"""Split and normalization invariants."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.dataset import DeformablePhysicsDataset
from src.splits import cvpr_v1_split, legacy_v1_split


class SplitInvariantTests(unittest.TestCase):
    def test_cvpr_roles_are_disjoint_and_cover_the_dump(self) -> None:
        split = cvpr_v1_split()
        split.assert_disjoint()
        roles = (
            set(split.train_ids),
            set(split.val_ids),
            set(split.test_ids),
            set(split.excluded_ids),
        )
        self.assertEqual(len(roles[0] & roles[1] & roles[2] & roles[3]), 0)
        self.assertTrue(roles[0].isdisjoint(roles[2]))
        self.assertTrue(roles[1].isdisjoint(roles[2]))
        self.assertTrue(roles[2].isdisjoint(roles[3]))
        self.assertEqual(split.normalization_role, "train")
        self.assertEqual(len(split.train_ids), 1440)
        self.assertEqual(len(split.val_ids), 180)
        self.assertEqual(len(split.test_ids), 180)

    def test_legacy_validation_is_not_a_test_split(self) -> None:
        split = legacy_v1_split()
        split.assert_disjoint()
        self.assertEqual(split.test_ids, ())
        self.assertEqual(split.headline_role, "legacy_validation")
        self.assertTrue(set(split.train_ids).isdisjoint(split.val_ids))
        self.assertEqual(split.normalization_role, "train")


class NormalizationProvenanceTests(unittest.TestCase):
    def test_action_moments_use_only_the_training_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = np.zeros((4, 4, 8, 8, 3), dtype=np.float32)
            actions = np.zeros((4, 3, 4), dtype=np.float32)
            actions[2:] = 1.0
            np.save(root / "observations.npy", observations)
            np.save(root / "actions.npy", actions)
            np.save(root / "stiffness_labels.npy", np.zeros(4, dtype=np.int64))
            cfg = DataConfig(
                dataset_path=str(root),
                img_size=8,
                in_channels=3,
                action_dim=4,
                chunk_length=4,
                chunk_stride=4,
                val_fraction=0.5,
                augment=False,
                normalize_actions=True,
                n_stiffness_classes=1,
            )
            train = DeformablePhysicsDataset(cfg, split="train")
            val = DeformablePhysicsDataset(cfg, split="val")
            for dataset in (train, val):
                provenance = dataset.normalization_provenance
                self.assertEqual(provenance["source_role"], "train")
                self.assertEqual(provenance["n_episodes"], 2)
                self.assertFalse(provenance["uses_validation"])
                self.assertFalse(provenance["uses_test"])
                self.assertTrue(torch.allclose(dataset.action_mean, torch.zeros(4)))
            pooled = torch.tensor(actions.reshape(-1, 4)).mean(dim=0)
            self.assertGreater(float(pooled[0]), 0.0)
            self.assertFalse(torch.allclose(train.action_mean, pooled))


if __name__ == "__main__":
    unittest.main()
