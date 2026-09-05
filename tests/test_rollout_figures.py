"""Tests for qualitative rollout grids, post-hoc decoder, and multi-seed drift bands."""

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

from scripts.render_rollout_comparison import (
    REPORT_HORIZONS,
    frame_to_hwc,
    render_episode_grid,
    select_showcase_episodes,
    train_frozen_latent_decoder,
    worst_failure_episode,
    write_failure_caption,
)
from src.ablation import plot_multiseed_drift
from src.config import DataConfig, ModelConfig
from src.dataset import DeformableObjectDataset
from src.models.ac_jepa import ACJEPA
from src.utils.export_paper_artifacts import export_paper_artifacts


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        encoder_depth=1,
        encoder_num_heads=4,
        mlp_ratio=2.0,
        action_dim=4,
        action_embed_dim=32,
        predictor_embed_dim=64,
        predictor_depth=1,
        predictor_num_heads=4,
        backbone="vit",
    )


class RolloutFigureTests(unittest.TestCase):
    def test_horizons_match_eval_report_set(self) -> None:
        self.assertEqual(REPORT_HORIZONS, (1, 5, 10))

    def test_select_episodes_span_stiffness_and_failure_argmax(self) -> None:
        data_cfg = DataConfig(
            n_episodes=12,
            episode_length=12,
            img_size=32,
            n_grid=4,
            action_dim=4,
            seed=0,
            val_fraction=0.5,
            n_stiffness_classes=3,
        )
        dataset = DeformableObjectDataset(data_cfg, split="all")
        n_train, n_val = dataset.split_sizes()
        mse = torch.zeros(n_train + n_val)
        mse[n_train + 2] = 9.0
        chosen = select_showcase_episodes(dataset, n_episodes=4, mse_k10=mse)
        self.assertEqual(len(chosen), 4)
        self.assertTrue(all(n_train <= i < n_train + n_val for i in chosen))
        classes = {int(dataset.stiffness_labels[i].item()) for i in chosen}
        self.assertGreaterEqual(len(classes), min(3, int(dataset.stiffness_labels[n_train:].unique().numel())))
        fail_id, fail_mse = worst_failure_episode(mse, n_train, n_val)
        self.assertEqual(fail_id, n_train + 2)
        self.assertAlmostEqual(fail_mse, 9.0)

    def test_posthoc_decoder_and_grid_png(self) -> None:
        data_cfg = DataConfig(
            n_episodes=8,
            episode_length=12,
            img_size=32,
            n_grid=4,
            seed=1,
            val_fraction=0.25,
        )
        dataset = DeformableObjectDataset(data_cfg, split="all")
        model = ACJEPA(_tiny_model_cfg())
        model.eval()
        decoder = train_frozen_latent_decoder(
            model, dataset, torch.device("cpu"), epochs=1, batch_size=4, max_steps=3
        )
        z = torch.randn(2, 64)
        recon = decoder(z)
        self.assertEqual(tuple(recon.shape), (2, 3, 32, 32))
        hwc = frame_to_hwc(recon[0])
        self.assertEqual(hwc.shape[-1], 3)
        gt = torch.rand(3, 3, 32, 32)
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "rollout_comparison_0.png"
            render_episode_grid(gt, gt, gt, REPORT_HORIZONS, png, title="unit test")
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)

    def test_failure_caption_uses_probe_diagnostics(self) -> None:
        diagnostics = {
            "models": {
                "vicreg_physics_mpm": {
                    "findings": [
                        "stiffness class: after scaling to n_val=200, mean val acc=0.339 is still consistent with chance=0.333.",
                        "contact quadrant: mean val acc=0.574 is above chance=0.250.",
                    ],
                    "z_target": {
                        "norm_std": 1e-4,
                        "n_dims_std_below_gamma": 128,
                        "latent_dim": 128,
                        "variance_hinge_binding": True,
                    },
                    "contact_val_coverage": {"counts": [853, 829, 151, 167]},
                }
            },
            "paper_limitation": "Caveat: the physics VICReg checkpoint is from early workshop training, not the finished 200-epoch run.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_case_caption.txt"
            write_failure_caption(
                path,
                episode_id=1812,
                stiffness_class=2,
                stiffness_e=220.0,
                mse_k10=0.42,
                n_val=200,
                contact_class=0,
                diagnostics=diagnostics,
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("1812", text)
            self.assertIn("k=10", text)
            self.assertIn("chance", text)
            self.assertIn("not linearly readable", text)
            self.assertNotIn("contact-point extrapolation", text.lower().split("do not")[0])
            self.assertIn("do not attribute the failure to contact-point extrapolation", text)


class MultiseedDriftPlotTests(unittest.TestCase):
    def test_error_bands_written_from_summary_json(self) -> None:
        summary = {
            "seeds": [42, 43, 44],
            "n_seeds": 3,
            "horizons": [1, 5, 10],
            "dataset_path": "outputs/physics_mpm",
            "variants": {
                "full_vicreg": {
                    "label": "Full AC-JEPA (VICReg)",
                    "ema_momentum": 0.996,
                    "by_horizon": {
                        "1": {
                            "rollout_mse_mean": 0.12,
                            "rollout_mse_std": 0.02,
                            "contact_probe_acc_mean": 0.55,
                            "contact_probe_acc_std": 0.03,
                        },
                        "5": {
                            "rollout_mse_mean": 0.18,
                            "rollout_mse_std": 0.03,
                            "contact_probe_acc_mean": 0.54,
                            "contact_probe_acc_std": 0.02,
                        },
                        "10": {
                            "rollout_mse_mean": 0.25,
                            "rollout_mse_std": 0.04,
                            "contact_probe_acc_mean": 0.53,
                            "contact_probe_acc_std": 0.02,
                        },
                    },
                },
                "variant_b_mse": {
                    "label": "Variant B (MSE-only)",
                    "by_horizon": {
                        "1": {
                            "rollout_mse_mean": 0.20,
                            "rollout_mse_std": 0.01,
                            "contact_probe_acc_mean": 0.40,
                            "contact_probe_acc_std": 0.02,
                        },
                        "5": {
                            "rollout_mse_mean": 0.30,
                            "rollout_mse_std": 0.02,
                            "contact_probe_acc_mean": 0.38,
                            "contact_probe_acc_std": 0.02,
                        },
                        "10": {
                            "rollout_mse_mean": 0.40,
                            "rollout_mse_std": 0.03,
                            "contact_probe_acc_mean": 0.36,
                            "contact_probe_acc_std": 0.02,
                        },
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "ablation_drift.png"
            plot_multiseed_drift(summary, png, dpi=80)
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 2000)
            json_path = Path(tmp) / "ablation_multiseed.json"
            json_path.write_text(json.dumps(summary), encoding="utf-8")
            csv_path = Path(tmp) / "ablation_results.csv"
            csv_path.write_text(
                "variant,variant_label,loss_type,horizon,mse,cosine_similarity,n_val\n"
                "full_vicreg,Full AC-JEPA (VICReg),vicreg,1,0.12,0.9,200\n",
                encoding="utf-8",
            )
            report = export_paper_artifacts(
                csv_path=csv_path,
                output_dir=tmp,
                live_eval=False,
                multiseed_json=json_path,
            )
            self.assertIsNotNone(report["ablation_drift"])
            self.assertTrue(Path(report["ablation_drift"]).is_file())


if __name__ == "__main__":
    unittest.main()
