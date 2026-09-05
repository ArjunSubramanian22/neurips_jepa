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

from dataclasses import asdict

from src.ablation import (
    ABLATION_SEEDS,
    EMA_MOMENTA,
    HORIZONS,
    REPORT_HORIZONS,
    VARIANTS,
    aggregate_seed_metrics,
    apply_variant,
    build_ablation_config,
    build_workshop_ablation_config,
    ema_sweep_variants,
    mean_std,
    plot_representation_drift,
    representation_drift,
    study_variants,
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
        self.assertEqual(
            tuple(v.loss_type for v in VARIANTS),
            ("vicreg", "vicreg_no_cov", "mse", "pixel_ae", "vicreg"),
        )
        no_action = next(v for v in VARIANTS if v.key == "no_action")
        self.assertFalse(no_action.condition_on_action)
        self.assertEqual(no_action.loss_type, "vicreg")

    def test_world_model_factory(self) -> None:
        cfg = build_ablation_config(epochs=1)
        cfg.training.loss_type = "pixel_ae"
        self.assertIsInstance(build_world_model(cfg), PixelSpaceAutoencoder)
        cfg.training.loss_type = "vicreg"
        self.assertIsInstance(build_world_model(cfg), ACJEPA)

    def test_no_action_variant_builds_unconditional_predictor(self) -> None:
        base = build_ablation_config(epochs=1)
        variant = next(v for v in VARIANTS if v.key == "no_action")
        cfg = apply_variant(base, variant, seed=42)
        self.assertFalse(cfg.model.condition_on_action)
        self.assertEqual(cfg.training.loss_type, "vicreg")
        self.assertEqual(cfg.training.lr, base.training.lr)
        self.assertEqual(cfg.training.epochs, base.training.epochs)
        self.assertEqual(cfg.training.seed, 42)
        model = build_world_model(cfg)
        self.assertIsInstance(model, ACJEPA)
        self.assertFalse(model.predictor.condition_on_action)
        z = torch.randn(2, cfg.model.effective_latent_dim)
        a = torch.randn(2, cfg.model.action_dim)
        a_alt = torch.randn(2, cfg.model.action_dim)
        model.eval()
        with torch.no_grad():
            y = model.predictor(z, a)
            y_alt = model.predictor(z, a_alt)
        self.assertEqual(tuple(y.shape), (2, cfg.model.effective_latent_dim))
        self.assertTrue(torch.allclose(y, y_alt))

    def test_ema_sweep_changes_only_momentum(self) -> None:
        base = build_workshop_ablation_config()
        items = ema_sweep_variants(EMA_MOMENTA)
        self.assertEqual(len(items), 3)
        self.assertEqual(tuple(v.ema_momentum for v in items), EMA_MOMENTA)
        self.assertEqual(ABLATION_SEEDS, (42, 43, 44))
        self.assertEqual(REPORT_HORIZONS, (1, 5, 10))
        cfgs = [apply_variant(base, item, seed=42) for item in items]
        for cfg, item in zip(cfgs, items, strict=True):
            self.assertEqual(cfg.model.ema_momentum, item.ema_momentum)
            self.assertEqual(cfg.training.loss_type, "vicreg")
            self.assertTrue(cfg.model.condition_on_action)
            self.assertEqual(cfg.training.epochs, base.training.epochs)
            self.assertEqual(cfg.training.lr, base.training.lr)
            self.assertEqual(cfg.training.batch_size, base.training.batch_size)
            self.assertEqual(cfg.training.weight_decay, base.training.weight_decay)
            self.assertEqual(cfg.training.seed, 42)
        for left, right in zip(cfgs, cfgs[1:], strict=False):
            left_d, right_d = asdict(left), asdict(right)
            self.assertEqual(left_d["training"], right_d["training"])
            left_model = dict(left_d["model"])
            right_model = dict(right_d["model"])
            left_model.pop("ema_momentum")
            right_model.pop("ema_momentum")
            self.assertEqual(left_model, right_model)
            self.assertNotEqual(left.model.ema_momentum, right.model.ema_momentum)
        self.assertIn("no_action", {v.key for v in study_variants()})
        self.assertGreaterEqual(len(study_variants()), len(VARIANTS) + 2)

    def test_mean_std_and_multiseed_aggregate(self) -> None:
        mean, std = mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, 1.0)
        single_mean, single_std = mean_std([4.0])
        self.assertAlmostEqual(single_mean, 4.0)
        self.assertAlmostEqual(single_std, 0.0)
        rows = []
        for seed, mse, stiff, contact in (
            (42, 0.10, 0.40, 0.50),
            (43, 0.20, 0.50, 0.60),
            (44, 0.30, 0.60, 0.70),
        ):
            rows.append(
                {
                    "variant": "variant_a_no_cov",
                    "variant_label": "Variant A (No Covariance)",
                    "loss_type": "vicreg_no_cov",
                    "condition_on_action": True,
                    "ema_momentum": 0.996,
                    "seed": seed,
                    "horizon": 5,
                    "rollout_mse": mse,
                    "stiffness_probe_acc": stiff,
                    "contact_probe_acc": contact,
                    "n_val": 200,
                }
            )
        summary = aggregate_seed_metrics(rows, seeds=ABLATION_SEEDS, horizons=(5,))
        stats = summary["variants"]["variant_a_no_cov"]["by_horizon"]["5"]
        self.assertAlmostEqual(stats["rollout_mse_mean"], 0.20)
        self.assertAlmostEqual(stats["rollout_mse_std"], 0.10)
        self.assertAlmostEqual(stats["stiffness_probe_acc_mean"], 0.50)
        self.assertAlmostEqual(stats["stiffness_probe_acc_std"], 0.10)
        self.assertAlmostEqual(stats["contact_probe_acc_mean"], 0.60)
        self.assertAlmostEqual(stats["contact_probe_acc_std"], 0.10)
        self.assertEqual(summary["n_seeds"], 3)
        self.assertEqual(summary["n_val"], 200)


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
