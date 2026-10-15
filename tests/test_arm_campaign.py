"""Campaign arm episodes keep the sealed test out of the training loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import DataConfig
from src.dataset import DeformablePhysicsDataset
from src.envs.arm_campaign import audit_campaign_dataset, generate_campaign_dataset
from src.eval_protocol import ProtocolError


class ArmCampaignTests(unittest.TestCase):
    def test_all_split_excludes_the_test_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generate_campaign_dataset(tmp, n_train=2, n_val=2, n_test=2, length=4)
            report = audit_campaign_dataset(tmp, intervention_sample=2)
            self.assertTrue(report["replay_exact"])
            self.assertTrue(report["splits_disjoint"])
            cfg = DataConfig(
                dataset_path=tmp,
                img_size=64,
                augment=False,
                split_version="arm_campaign_v1",
                normalize_actions=True,
                action_dim=2,
                episode_length=4,
                n_episodes=6,
                val_fraction=0.25,
            )
            pooled = DeformablePhysicsDataset(cfg, split="all")
            held = DeformablePhysicsDataset(cfg, split="test")
            self.assertEqual(pooled.split_sizes(), (2, 2))
            self.assertTrue(set(pooled.global_episode_ids).isdisjoint(held.global_episode_ids))
            self.assertEqual(held.global_episode_ids, (4, 5))
            self.assertFalse(pooled.normalization_provenance["uses_test"])
            with self.assertRaises(ProtocolError):
                held.split_sizes()
            self.assertTrue((Path(tmp) / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
