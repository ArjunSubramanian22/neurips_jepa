"""Probe diagnostics: label coverage, train acc, latent health, stiffness regression."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.dataset import DeformableObjectDataset
from src.eval import (
    effective_rank,
    evaluate_model,
    label_coverage,
    latent_health,
    linear_probe_over_horizon,
    write_probe_diagnostics,
)
from src.models.ac_jepa import ACJEPA
from src.train import build_smoke_config


class LabelCoverageTests(unittest.TestCase):
    def test_counts_distinct_and_missing(self) -> None:
        labels = torch.tensor([0, 0, 2, 2, 2])
        cov = label_coverage(labels, n_classes=3)
        self.assertEqual(cov["n_examples"], 5)
        self.assertEqual(cov["n_distinct"], 2)
        self.assertEqual(cov["class_ids"], [0, 2])
        self.assertEqual(cov["counts"], [2, 0, 3])
        self.assertEqual(cov["missing_class_ids"], [1])


class LatentHealthTests(unittest.TestCase):
    def test_collapsed_dims_bind_variance_hinge(self) -> None:
        collapsed = torch.zeros(32, 8)
        stats = latent_health(collapsed, gamma=1.0)
        self.assertEqual(stats["n_dims_std_below_gamma"], 8)
        self.assertGreater(stats["variance_hinge_mean"], 0.9)
        self.assertTrue(stats["variance_hinge_binding"])

    def test_unit_std_does_not_bind(self) -> None:
        torch.manual_seed(0)
        spread = torch.randn(256, 8)
        stats = latent_health(spread, gamma=1.0)
        self.assertLess(stats["frac_dims_std_below_gamma"], 0.5)
        self.assertGreater(stats["per_dim_variance_mean"], 0.5)


class EffectiveRankTests(unittest.TestCase):
    def test_rank_one_is_near_one(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(256, 1) @ torch.ones(1, 8)
        stats = effective_rank(z)
        self.assertLess(stats["participation_ratio"], 1.5)
        self.assertEqual(stats["latent_dim"], 8)

    def test_isotropic_near_dimension(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(4000, 16)
        stats = effective_rank(z)
        self.assertGreater(stats["participation_ratio"], 10.0)
        self.assertLess(stats["participation_ratio"], 16.5)


class EffectiveRankProvenanceTests(unittest.TestCase):
    def test_in_progress_rank_is_not_final(self) -> None:
        from src.eval import effective_rank_record_is_final

        self.assertFalse(effective_rank_record_is_final(None))
        self.assertFalse(
            effective_rank_record_is_final(
                {
                    "z_target": {"participation_ratio": 3.0},
                    "checkpoint_epoch": 162,
                    "metrics_jsonl_train_epochs": 162,
                    "final": False,
                }
            )
        )

    def test_final_requires_epoch_and_flag(self) -> None:
        from src.eval import effective_rank_record_is_final

        self.assertTrue(
            effective_rank_record_is_final(
                {
                    "z_target": {"participation_ratio": 12.0},
                    "checkpoint_epoch": 200,
                    "metrics_jsonl_train_epochs": 200,
                    "final": True,
                }
            )
        )
        self.assertFalse(
            effective_rank_record_is_final(
                {
                    "z_target": {"participation_ratio": 12.0},
                    "checkpoint_epoch": 200,
                    "metrics_jsonl_train_epochs": 200,
                    "final": False,
                }
            )
        )


class LinearProbeTrainAccTests(unittest.TestCase):
    def test_reports_train_and_val_accuracy(self) -> None:
        torch.manual_seed(0)
        z_tr = torch.zeros(20, 2, 4)
        z_tr[:10, :, 0] = 1.0
        y_tr = torch.tensor([0] * 10 + [1] * 10)
        z_va = torch.zeros(6, 2, 4)
        z_va[:3, :, 0] = 1.0
        y_va = torch.tensor([0, 0, 0, 1, 1, 1])
        out = linear_probe_over_horizon(
            z_tr, z_va, y_tr, y_va, n_classes=2,
            epochs=40, lr=0.05, device=torch.device("cpu"),
        )
        self.assertEqual(len(out["val_acc"]), 2)
        self.assertEqual(len(out["train_acc"]), 2)
        self.assertGreater(out["train_acc"][0], 0.9)
        self.assertEqual(out["val_coverage"]["n_examples"], 6)
        self.assertEqual(out["val_coverage"]["n_distinct"], 2)


class EvaluateModelDiagnosticsTests(unittest.TestCase):
    def test_smoke_eval_writes_coverage_train_acc_and_regression(self) -> None:
        cfg = build_smoke_config(epochs=1)
        cfg.data = DataConfig(
            n_episodes=16,
            episode_length=6,
            img_size=32,
            n_grid=4,
            action_dim=4,
            in_channels=3,
            seed=0,
            val_fraction=0.25,
            n_stiffness_classes=3,
        )
        dataset = DeformableObjectDataset(cfg.data, split="all")
        n_train, n_val = dataset.split_sizes()
        self.assertEqual(n_val, 4)
        self.assertEqual(tuple(dataset.stiffness_values.shape), (dataset.n_episodes,))
        model = ACJEPA(cfg.model)
        model.eval()
        payload = evaluate_model(
            model,
            dataset,
            torch.device("cpu"),
            max_horizon=3,
            probe_epochs=8,
            batch_size=4,
        )
        self.assertEqual(payload["n_val"], n_val)
        self.assertEqual(len(payload["stiffness_probe_acc"]), payload["k_max"])
        self.assertEqual(len(payload["stiffness_probe_train_acc"]), payload["k_max"])
        cov = payload["diagnostics"]["stiffness_val_coverage"]
        self.assertEqual(cov["n_examples"], n_val)
        self.assertLessEqual(cov["n_distinct"], 3)
        self.assertIn("norm_mean", payload["diagnostics"]["z_target"])
        self.assertIn("per_dim_variance_mean", payload["diagnostics"]["z_target"])
        self.assertIsNotNone(payload["stiffness_reg_val_r2"])
        self.assertEqual(len(payload["stiffness_reg_val_r2"]), payload["k_max"])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_probe_diagnostics({"vicreg": payload}, Path(tmp) / "probe_diagnostics.json")
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("vicreg", doc["models"])
            self.assertEqual(doc["models"]["vicreg"]["n_val"], n_val)
            self.assertTrue(doc["limitation"])


if __name__ == "__main__":
    unittest.main()
