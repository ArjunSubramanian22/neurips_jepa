"""Within-simulator baseline comparison: GNN edge reuse, no-action, CSV schema."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import (
    BASELINES,
    build_baseline_config,
    write_comparison_csv,
    write_readme,
)
from src.config import DataConfig
from src.dataset import DeformableObjectDataset, _edge_list
from src.eval import evaluate_model
from src.models.particle_gnn import ParticleGraphWorldModel
from src.train import ACJEPATrainer, build_world_model


class BaselineConfigTests(unittest.TestCase):
    def test_shared_split_covers_k10_and_includes_external_methods(self) -> None:
        cfg = build_baseline_config(epochs=2, n_episodes=32)
        self.assertGreaterEqual(cfg.data.episode_length, 11)
        self.assertGreaterEqual(cfg.data.n_episodes, 32)
        keys = tuple(spec.key for spec in BASELINES)
        self.assertIn("particle_gnn", keys)
        self.assertIn("no_action", keys)
        self.assertIn("ac_jepa_vicreg", keys)
        gnn_cfg = cfg
        gnn_cfg.training.loss_type = "gnn"
        self.assertIsInstance(build_world_model(gnn_cfg), ParticleGraphWorldModel)


class EdgeListReuseTests(unittest.TestCase):
    def test_gnn_buffers_match_dataset_edge_list(self) -> None:
        data = DataConfig(n_grid=5, spatial_dims=2, n_episodes=2, episode_length=3, img_size=32)
        model = ParticleGraphWorldModel(data_cfg=data, hidden_dim=16, message_steps=1)
        edges = _edge_list(5, 2)
        self.assertEqual(int(model.edge_src.numel()), 2 * int(edges.shape[0]))
        undirected = {tuple(sorted((int(s), int(d)))) for s, d in zip(model.edge_src, model.edge_dst)}
        expected = {tuple(sorted((int(a), int(b)))) for a, b in edges}
        self.assertEqual(undirected, expected)


class ComparisonExportTests(unittest.TestCase):
    def test_csv_and_readme_domain_gap(self) -> None:
        rows = [
            {
                "method": "particle_gnn",
                "method_label": "Particle GNN",
                "horizon": k,
                "rollout_mse": 0.1 * k,
                "stiffness_probe_acc": 0.3,
                "contact_probe_acc": 0.4,
                "stiffness_probe_train_acc": 0.5,
                "contact_probe_train_acc": 0.6,
                "n_train": 120,
                "n_val": 40,
                "chance_stiffness": 1.0 / 3.0,
                "chance_contact": 0.25,
            }
            for k in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_comparison_csv(rows, Path(tmp) / "baseline_comparison.csv")
            readme = write_readme(Path(tmp) / "baseline_comparison.md")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 10)
            self.assertEqual(table[0]["method"], "particle_gnn")
            self.assertEqual(int(table[-1]["horizon"]), 10)
            text = readme.read_text(encoding="utf-8")
            self.assertIn("within-simulator", text.lower())
            self.assertIn("PhysWorld", text)
            self.assertIn("limitation", text.lower())


class TinyJointEvalTests(unittest.TestCase):
    def test_gnn_and_no_action_share_eval_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_baseline_config(epochs=1, n_episodes=16, device="cpu", output_dir=tmp)
            cfg.data.n_episodes = 16
            cfg.data.episode_length = 6
            cfg.data.n_grid = 4
            cfg.data.val_fraction = 0.25
            cfg.data.normalize_actions = False
            cfg.data.augment = False
            dataset = DeformableObjectDataset(cfg.data, split="all")
            self.assertIsNotNone(dataset.positions)
            n_train, n_val = dataset.split_sizes()
            self.assertGreaterEqual(n_val, 4)
            cfg.training.loss_type = "gnn"
            cfg.training.batch_size = 8
            cfg.training.amp = False
            trainer = ACJEPATrainer(cfg, run_name="gnn_test", dataset=dataset)
            trainer.fit()
            gnn_metrics = evaluate_model(
                trainer.model, dataset, torch.device("cpu"), max_horizon=3, probe_epochs=4, batch_size=4
            )
            self.assertEqual(gnn_metrics["n_val"], n_val)
            self.assertEqual(len(gnn_metrics["rollout_mse"]), 3)
            self.assertEqual(len(gnn_metrics["stiffness_probe_acc"]), 3)
            self.assertEqual(len(gnn_metrics["contact_probe_acc"]), 3)

            cfg.training.loss_type = "vicreg"
            cfg.model.condition_on_action = False
            trainer2 = ACJEPATrainer(cfg, run_name="no_action_test", dataset=dataset)
            trainer2.fit()
            no_act = evaluate_model(
                trainer2.model, dataset, torch.device("cpu"), max_horizon=3, probe_epochs=4, batch_size=4
            )
            self.assertEqual(no_act["horizons"], gnn_metrics["horizons"])
            self.assertEqual(no_act["n_val"], gnn_metrics["n_val"])


if __name__ == "__main__":
    unittest.main()
