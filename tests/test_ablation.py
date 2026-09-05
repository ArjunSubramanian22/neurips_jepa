"""Tests for the automated ablation suite and long-horizon drift engine."""

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

from src.ablation import (
    HORIZONS,
    VARIANTS,
    build_ablation_config,
    plot_representation_drift,
    representation_drift,
    write_ablation_csv,
)
from src.config import DataConfig
from src.dataset import DeformableObjectDataset
from src.models.ac_jepa import ACJEPA
from src.models.pixel_ae import PixelSpaceAutoencoder
from src.train import ACJEPATrainer, build_world_model


class AblationConfigTests(unittest.TestCase):
    def test_episode_length_covers_k20(self) -> None:
        cfg = build_ablation_config(epochs=1)
        self.assertGreaterEqual(cfg.data.episode_length, max(HORIZONS) + 1)
        self.assertEqual(tuple(v.loss_type for v in VARIANTS), ("vicreg", "vicreg_no_cov", "mse", "pixel_ae"))

    def test_world_model_factory(self) -> None:
        cfg = build_ablation_config(epochs=1)
        cfg.training.loss_type = "pixel_ae"
        self.assertIsInstance(build_world_model(cfg), PixelSpaceAutoencoder)
        cfg.training.loss_type = "vicreg"
        self.assertIsInstance(build_world_model(cfg), ACJEPA)


class RepresentationDriftTests(unittest.TestCase):
    def test_autoregressive_horizons_and_metrics(self) -> None:
        cfg = build_ablation_config(epochs=1)
        data_cfg = DataConfig(
            n_episodes=8,
            episode_length=21,
            img_size=32,
            n_grid=4,
            action_dim=4,
            in_channels=3,
            seed=0,
            val_fraction=0.25,
        )
        dataset = DeformableObjectDataset(data_cfg, split="all")
        model = ACJEPA(cfg.model)
        model.eval()
        metrics = representation_drift(
            model, dataset, torch.device("cpu"), horizons=HORIZONS, batch_size=4
        )
        self.assertEqual(metrics["horizons"], list(HORIZONS))
        self.assertEqual(len(metrics["mse"]), len(HORIZONS))
        self.assertEqual(len(metrics["cosine"]), len(HORIZONS))
        for mse, cosine in zip(metrics["mse"], metrics["cosine"], strict=True):
            self.assertGreaterEqual(mse, 0.0)
            self.assertTrue(-1.0 <= cosine <= 1.0)
        n_train, n_val = dataset.split_sizes()
        self.assertEqual(metrics["n_val"], n_val)
        self.assertGreater(n_val, 0)


class AblationExportTests(unittest.TestCase):
    def test_csv_and_highres_plot(self) -> None:
        rows = []
        fake_results = {}
        for variant in VARIANTS:
            mse = [0.1 * (i + 1) for i, _ in enumerate(HORIZONS)]
            cosine = [0.9 - 0.02 * i for i, _ in enumerate(HORIZONS)]
            fake_results[variant.key] = {"horizons": list(HORIZONS), "mse": mse, "cosine": cosine}
            for horizon, m, c in zip(HORIZONS, mse, cosine, strict=True):
                rows.append(
                    {
                        "variant": variant.key,
                        "variant_label": variant.label,
                        "loss_type": variant.loss_type,
                        "horizon": horizon,
                        "mse": m,
                        "cosine_similarity": c,
                        "n_val": 4,
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ablation_results.csv"
            plot_path = Path(tmp) / "ablation_drift.png"
            write_ablation_csv(rows, csv_path)
            plot_representation_drift(fake_results, VARIANTS, plot_path, dpi=120)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), len(VARIANTS) * len(HORIZONS))
            self.assertEqual(
                list(table[0].keys()),
                [
                    "variant",
                    "variant_label",
                    "loss_type",
                    "horizon",
                    "mse",
                    "cosine_similarity",
                    "n_val",
                ],
            )
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 1000)


class PixelAETrainerTests(unittest.TestCase):
    def test_pixel_ae_training_step_runs(self) -> None:
        cfg = build_ablation_config(epochs=1)
        cfg.training.loss_type = "pixel_ae"
        cfg.training.device = "cpu"
        cfg.training.amp = False
        cfg.data.n_episodes = 4
        cfg.data.episode_length = 3
        cfg.data.val_fraction = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            cfg.training.output_dir = tmp
            trainer = ACJEPATrainer(cfg, run_name="pixel_ae_smoke")
            batch = next(iter(trainer.train_loader))
            metrics = trainer.training_step(*batch)
        self.assertTrue(metrics["loss"] >= 0.0)
        self.assertIn("variance", metrics)
        self.assertIn("covariance", metrics)


if __name__ == "__main__":
    unittest.main()
