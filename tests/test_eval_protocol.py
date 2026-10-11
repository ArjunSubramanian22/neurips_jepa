"""Normalization, thresholds, and checkpoint pairing follow the split roles."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig, load_experiment_config, workshop_config_path
from src.dataset import DeformablePhysicsDataset
from src.eval_protocol import (
    ProtocolError,
    assert_checkpoint_eval_pairing,
    assert_legacy_reproduction_config,
    fit_visibility_threshold,
    reporting_slice,
    train_action_statistics,
)
from src.splits import cvpr_v1_split, legacy_v1_split


class NormalizationProvenanceTests(unittest.TestCase):
    def test_moments_ignore_val_test_and_excluded(self) -> None:
        split = cvpr_v1_split()
        actions = np.zeros((split.n_episodes, 2, 4), dtype=np.float64)
        actions[list(split.val_ids)] = 100.0
        actions[list(split.test_ids)] = -100.0
        actions[list(split.excluded_ids)] = 50.0
        stats = train_action_statistics(actions, split)
        np.testing.assert_allclose(stats["mean"], 0.0)
        self.assertEqual(stats["source_role"], "train")
        self.assertEqual(stats["n_episodes"], 1440)
        self.assertFalse(stats["uses_validation"])
        self.assertFalse(stats["uses_test"])
        self.assertFalse(stats["uses_excluded"])
        actions[split.train_ids[0]] = 12.0
        shifted = train_action_statistics(actions, split)
        self.assertGreater(float(np.abs(shifted["mean"]).sum()), 0.0)

    def test_legacy_moments_match_the_train_prefix(self) -> None:
        split = legacy_v1_split()
        rng = np.random.default_rng(0)
        actions = rng.normal(size=(2000, 3, 4))
        stats = train_action_statistics(actions, split)
        prefix = actions[:1800].reshape(-1, 4)
        np.testing.assert_allclose(stats["mean"], prefix.mean(axis=0))
        np.testing.assert_allclose(stats["std"], np.maximum(prefix.std(axis=0), 1e-6))
        report = reporting_slice(split)
        self.assertFalse(report["is_test_set"])
        self.assertEqual(report["role"], "legacy_validation")
        self.assertEqual(report["episode_ids"], tuple(range(1800, 2000)))


class ThresholdAndPairingTests(unittest.TestCase):
    def test_threshold_rejects_test_episodes(self) -> None:
        split = cvpr_v1_split()
        values = {episode: 1.0 for episode in split.train_ids}
        fitted = fit_visibility_threshold(values, split, role="train", quantile=0.67)
        self.assertEqual(fitted["n_episodes"], 1440)
        self.assertFalse(fitted["saw_test"])
        leaked = dict(values)
        leaked[split.test_ids[0]] = 5.0
        with self.assertRaises(ProtocolError):
            fit_visibility_threshold(leaked, split, role="train")
        with self.assertRaises(ProtocolError):
            fit_visibility_threshold({episode: 1.0 for episode in split.test_ids}, split, role="test")
        with self.assertRaises(ProtocolError):
            fit_visibility_threshold(
                {episode: 1.0 for episode in split.excluded_ids},
                split,
                role="train",
            )

    def test_historical_checkpoint_cannot_use_cvpr_test(self) -> None:
        assert_checkpoint_eval_pairing("legacy_v1", "legacy_v1")
        assert_checkpoint_eval_pairing("hist_v1_last_fraction_val", "legacy_v1")
        assert_checkpoint_eval_pairing("cvpr_v1", "cvpr_v1")
        with self.assertRaises(ProtocolError):
            assert_checkpoint_eval_pairing("legacy_v1", "cvpr_v1")
        with self.assertRaises(ProtocolError):
            assert_checkpoint_eval_pairing("cvpr_v1", "legacy_v1")

    def test_configs_declare_split_version(self) -> None:
        paths = list((ROOT / "configs" / "experiments").glob("*.yaml"))
        paths.append(workshop_config_path())
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            declared = payload.get("split_version") or data.get("split_version")
            self.assertTrue(declared, path.name)
        heldout = yaml.safe_load(
            (ROOT / "configs" / "experiments" / "heldout_reproduction_v1.yaml").read_text(encoding="utf-8")
        )
        assert_legacy_reproduction_config(heldout)
        workshop = load_experiment_config(workshop_config_path())
        self.assertEqual(workshop.data.split_version, "legacy_v1")


class DatasetProvenanceTests(unittest.TestCase):
    def _dump(self, root: Path) -> None:
        observations = np.zeros((2000, 2, 8, 8, 3), dtype=np.float32)
        actions = np.zeros((2000, 1, 4), dtype=np.float32)
        split = cvpr_v1_split()
        actions[list(split.val_ids)] = 100.0
        actions[list(split.test_ids)] = -100.0
        actions[list(split.excluded_ids)] = 50.0
        np.save(root / "observations.npy", observations)
        np.save(root / "actions.npy", actions)

    def test_cvpr_loader_normalizes_from_train_ids_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dump(root)
            cfg = DataConfig(
                dataset_path=str(root),
                img_size=8,
                augment=False,
                split_version="cvpr_v1",
                normalize_actions=True,
            )
            train = DeformablePhysicsDataset(cfg, split="train")
            val = DeformablePhysicsDataset(cfg, split="val")
            test = DeformablePhysicsDataset(cfg, split="test")
            self.assertEqual(train.global_episode_ids, cvpr_v1_split().train_ids)
            self.assertEqual(val.global_episode_ids, cvpr_v1_split().val_ids)
            self.assertEqual(test.global_episode_ids, cvpr_v1_split().test_ids)
            self.assertTrue(set(train.global_episode_ids).isdisjoint(val.global_episode_ids))
            self.assertTrue(set(train.global_episode_ids).isdisjoint(test.global_episode_ids))
            for dataset in (train, val, test):
                self.assertEqual(dataset.normalization_provenance["source_role"], "train")
                self.assertEqual(dataset.normalization_provenance["n_episodes"], 1440)
                self.assertFalse(dataset.normalization_provenance["uses_test"])
                np.testing.assert_allclose(dataset.action_mean.numpy(), 0.0, atol=1e-6)
            with self.assertRaises(ProtocolError):
                DeformablePhysicsDataset(cfg, split="all")

    def test_legacy_loader_matches_the_prefix_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = np.zeros((2000, 2, 8, 8, 3), dtype=np.float32)
            actions = np.arange(2000, dtype=np.float32)[:, None, None]
            actions = np.broadcast_to(actions, (2000, 1, 4)).copy()
            np.save(root / "observations.npy", observations)
            np.save(root / "actions.npy", actions)
            base = DataConfig(dataset_path=str(root), img_size=8, augment=False, val_fraction=0.1)
            labeled = DataConfig(
                dataset_path=str(root),
                img_size=8,
                augment=False,
                val_fraction=0.1,
                split_version="legacy_v1",
            )
            plain = DeformablePhysicsDataset(base, split="val")
            legacy = DeformablePhysicsDataset(labeled, split="val")
            np.testing.assert_allclose(plain.action_mean.numpy(), legacy.action_mean.numpy())
            self.assertEqual(legacy.global_episode_ids, tuple(range(1800, 2000)))
            self.assertEqual(legacy.normalization_provenance["n_episodes"], 1800)
            with self.assertRaises(ProtocolError):
                DeformablePhysicsDataset(labeled, split="test")


if __name__ == "__main__":
    unittest.main()
