"""Workshop schedule, warmup scaling, and pixel-AE routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    TrainingConfig,
    build_workshop_config,
    load_experiment_config,
    resolve_warmup_epochs,
    workshop_config_path,
)
from src.models.pixel_ae import PixelSpaceAutoencoder
from src.train import build_world_model


class WarmupScheduleTests(unittest.TestCase):
    def test_default_100_epoch_keeps_10(self) -> None:
        self.assertEqual(resolve_warmup_epochs(100, 10), 10)
        self.assertEqual(TrainingConfig().warmup_epochs, 10)
        self.assertEqual(TrainingConfig().epochs, 100)

    def test_long_run_scales_stock_warmup(self) -> None:
        self.assertEqual(resolve_warmup_epochs(200, 10), 20)
        self.assertEqual(resolve_warmup_epochs(250, 10), 25)
        cfg = TrainingConfig(epochs=200)
        self.assertEqual(cfg.warmup_epochs, 20)

    def test_explicit_warmup_is_preserved(self) -> None:
        self.assertEqual(resolve_warmup_epochs(200, 20), 20)
        cfg = TrainingConfig(epochs=200, warmup_epochs=15)
        self.assertEqual(cfg.warmup_epochs, 15)

    def test_smoke_epochs_unchanged(self) -> None:
        cfg = TrainingConfig(epochs=5, warmup_epochs=1)
        self.assertEqual(cfg.warmup_epochs, 1)


class WorkshopConfigTests(unittest.TestCase):
    def test_yaml_loads_workshop_schedule(self) -> None:
        cfg = load_experiment_config(workshop_config_path())
        self.assertGreaterEqual(cfg.training.epochs, 200)
        self.assertEqual(cfg.training.batch_size, 128)
        self.assertTrue(cfg.training.amp)
        self.assertEqual(cfg.training.device, "auto")
        self.assertEqual(cfg.training.warmup_epochs, 20)
        self.assertEqual(cfg.model.img_size, 64)
        self.assertEqual(cfg.data.img_size, 64)
        self.assertEqual(cfg.data.action_dim, 4)
        self.assertEqual(cfg.data.in_channels, 3)
        self.assertIsNotNone(cfg.data.dataset_path)

    def test_factory_can_override_loss(self) -> None:
        cfg = build_workshop_config(loss_type="pixel_ae")
        self.assertEqual(cfg.training.loss_type, "pixel_ae")
        self.assertIsInstance(build_world_model(cfg), PixelSpaceAutoencoder)

    def test_vicreg_still_uses_acjepa(self) -> None:
        from src.models.ac_jepa import ACJEPA

        cfg = build_workshop_config(loss_type="vicreg")
        self.assertIsInstance(build_world_model(cfg), ACJEPA)

    def test_metrics_jsonl_is_truncated_on_new_run(self) -> None:
        import tempfile

        from src.train import ExperimentLogger

        cfg = build_workshop_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stale = run_dir / "metrics.jsonl"
            stale.write_text('{"run":"stale"}\n', encoding="utf-8")
            logger = ExperimentLogger("vicreg", run_dir, cfg)
            self.assertEqual(stale.read_text(encoding="utf-8"), "")
            logger.log(
                {
                    "run": "vicreg",
                    "split": "train_epoch",
                    "epoch": 1,
                    "step": 1,
                    "loss": 1.0,
                    "invariance": 0.1,
                    "variance": 0.2,
                    "covariance": 0.3,
                    "ema_param_l2": 0.4,
                    "grad_norm": 1.5,
                },
                step=1,
            )
            logger.close()
            lines = stale.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertNotIn("stale", lines[0])
            payload = __import__("json").loads(lines[0])
            self.assertEqual(payload["run"], "vicreg")
            self.assertIn("invariance", payload)
            self.assertIn("ema_param_l2", payload)


if __name__ == "__main__":
    unittest.main()
