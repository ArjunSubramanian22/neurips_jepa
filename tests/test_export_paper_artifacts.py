"""Tests for camera-ready paper artifact export."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ac_jepa import ACJEPA
from src.utils.export_paper_artifacts import (
    TABLE_HORIZONS,
    VariantPaperRow,
    collect_latent_trajectories,
    export_paper_artifacts,
    load_ablation_csv,
    load_multiseed_summary,
    measure_inference_latency,
    pca_embed,
    plot_latent_trajectories,
    render_ablation_table,
    render_multiseed_table,
    tsne_embed,
    write_paper_tables_tex,
)


def _tiny_csv(path: Path) -> None:
    rows = [
        {
            "variant": "full_vicreg",
            "variant_label": "Full AC-JEPA (VICReg)",
            "loss_type": "vicreg",
            "horizon": k,
            "mse": 0.2 + 0.01 * k,
            "cosine_similarity": 0.9,
            "n_val": 4,
        }
        for k in (1, 5, 10, 20)
    ]
    rows.extend(
        {
            "variant": "variant_b_mse",
            "variant_label": "Variant B (MSE-only)",
            "loss_type": "mse",
            "horizon": k,
            "mse": 0.05 + 0.002 * k,
            "cosine_similarity": 0.98,
            "n_val": 4,
        }
        for k in (1, 5, 10, 20)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class EmbedTests(unittest.TestCase):
    def test_pca_shape_and_centering(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(32, 8))
        y = pca_embed(x, 2)
        self.assertEqual(y.shape, (32, 2))
        self.assertTrue(np.isfinite(y).all())
        self.assertLess(abs(y.mean()), 1e-6)

    def test_tsne_shape_numpy_fallback(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20, 6))
        y = tsne_embed(x, n_components=2, perplexity=5.0, n_iter=40, seed=1)
        self.assertEqual(y.shape, (20, 2))
        self.assertTrue(np.isfinite(y).all())


class LatexTableTests(unittest.TestCase):
    def test_table_contains_required_columns_and_bolds_best_mse(self) -> None:
        rows = [
            VariantPaperRow(
                key="full_vicreg",
                label="Full AC-JEPA (VICReg)",
                mse={1: 0.20, 5: 0.22, 10: 0.25, 20: 0.30},
                stiffness_acc=0.5,
                friction_acc=0.4,
                fps=120.0,
                ms_per_frame=8.3,
            ),
            VariantPaperRow(
                key="variant_b_mse",
                label="Variant B (MSE-only)",
                mse={1: 0.05, 5: 0.08, 10: 0.09, 20: 0.10},
                stiffness_acc=0.2,
                friction_acc=0.25,
                fps=150.0,
                ms_per_frame=6.6,
            ),
        ]
        tex = render_ablation_table(rows)
        self.assertIn(r"\begin{table}", tex)
        self.assertIn(r"\end{table}", tex)
        for k in TABLE_HORIZONS:
            self.assertIn(rf"$k{{=}}{k}$", tex)
        self.assertIn("Stiffness", tex)
        self.assertIn("Friction", tex)
        self.assertIn("FPS", tex)
        self.assertIn("ms/frame", tex)
        self.assertIn(r"\textbf{0.050}", tex)
        self.assertIn(r"\textbf{50.0}", tex)
        self.assertIn(r"\textbf{150.0}", tex)
        self.assertIn(r"\textbf{6.60}", tex)
        self.assertIn("Full AC-JEPA (VICReg)", tex)

    def test_csv_roundtrip_and_tex_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ablation_results.csv"
            tex_path = Path(tmp) / "paper_tables.tex"
            _tiny_csv(csv_path)
            rows = load_ablation_csv(csv_path)
            self.assertEqual([r.key for r in rows], ["full_vicreg", "variant_b_mse"])
            self.assertIn(20, rows[0].mse)
            write_paper_tables_tex(rows, tex_path, figure_name="paper_latent_trajectories.pdf")
            text = tex_path.read_text(encoding="utf-8")
            self.assertIn(r"\begin{table}", text)
            self.assertIn(r"\begin{figure}", text)
            self.assertIn("paper_latent_trajectories.pdf", text)


class MultiseedLatexTests(unittest.TestCase):
    def test_mean_std_table_and_export_ingest(self) -> None:
        payload = {
            "seeds": [42, 43, 44],
            "n_seeds": 3,
            "horizons": [1, 5, 10],
            "n_val": 200,
            "variants": {
                "full_vicreg": {
                    "label": "Full AC-JEPA (VICReg)",
                    "loss_type": "vicreg",
                    "condition_on_action": True,
                    "ema_momentum": 0.996,
                    "by_horizon": {
                        "1": {
                            "rollout_mse_mean": 0.123,
                            "rollout_mse_std": 0.045,
                            "stiffness_probe_acc_mean": 0.40,
                            "stiffness_probe_acc_std": 0.02,
                            "contact_probe_acc_mean": 0.55,
                            "contact_probe_acc_std": 0.03,
                        },
                        "5": {
                            "rollout_mse_mean": 0.200,
                            "rollout_mse_std": 0.010,
                            "stiffness_probe_acc_mean": 0.41,
                            "stiffness_probe_acc_std": 0.01,
                            "contact_probe_acc_mean": 0.56,
                            "contact_probe_acc_std": 0.02,
                        },
                        "10": {
                            "rollout_mse_mean": 0.310,
                            "rollout_mse_std": 0.020,
                            "stiffness_probe_acc_mean": 0.42,
                            "stiffness_probe_acc_std": 0.01,
                            "contact_probe_acc_mean": 0.57,
                            "contact_probe_acc_std": 0.02,
                        },
                    },
                },
                "no_action": {
                    "label": "No action conditioning",
                    "loss_type": "vicreg",
                    "condition_on_action": False,
                    "ema_momentum": 0.996,
                    "by_horizon": {
                        "1": {
                            "rollout_mse_mean": 0.400,
                            "rollout_mse_std": 0.050,
                            "stiffness_probe_acc_mean": 0.33,
                            "stiffness_probe_acc_std": 0.01,
                            "contact_probe_acc_mean": 0.30,
                            "contact_probe_acc_std": 0.02,
                        },
                        "5": {
                            "rollout_mse_mean": 0.500,
                            "rollout_mse_std": 0.040,
                            "stiffness_probe_acc_mean": 0.34,
                            "stiffness_probe_acc_std": 0.01,
                            "contact_probe_acc_mean": 0.31,
                            "contact_probe_acc_std": 0.02,
                        },
                        "10": {
                            "rollout_mse_mean": 0.600,
                            "rollout_mse_std": 0.030,
                            "stiffness_probe_acc_mean": 0.35,
                            "stiffness_probe_acc_std": 0.01,
                            "contact_probe_acc_mean": 0.32,
                            "contact_probe_acc_std": 0.02,
                        },
                    },
                },
                "ema_0p99": {
                    "label": "EMA tau=0.99",
                    "loss_type": "vicreg",
                    "condition_on_action": True,
                    "ema_momentum": 0.99,
                    "by_horizon": {
                        "1": {
                            "rollout_mse_mean": 0.150,
                            "rollout_mse_std": 0.020,
                            "stiffness_probe_acc_mean": 0.38,
                            "stiffness_probe_acc_std": 0.02,
                            "contact_probe_acc_mean": 0.50,
                            "contact_probe_acc_std": 0.02,
                        },
                        "5": {
                            "rollout_mse_mean": 0.220,
                            "rollout_mse_std": 0.015,
                            "stiffness_probe_acc_mean": 0.39,
                            "stiffness_probe_acc_std": 0.02,
                            "contact_probe_acc_mean": 0.51,
                            "contact_probe_acc_std": 0.02,
                        },
                        "10": {
                            "rollout_mse_mean": 0.330,
                            "rollout_mse_std": 0.018,
                            "stiffness_probe_acc_mean": 0.40,
                            "stiffness_probe_acc_std": 0.02,
                            "contact_probe_acc_mean": 0.52,
                            "contact_probe_acc_std": 0.02,
                        },
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "ablation_multiseed.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_multiseed_summary(json_path)
            self.assertEqual([row.key for row in loaded], ["full_vicreg", "no_action", "ema_0p99"])
            tex = render_multiseed_table(loaded)
            self.assertIn(r"{\pm}", tex)
            self.assertIn(r"$0.123{\pm}0.045$", tex)
            self.assertIn(r"$40.0{\pm}2.0$", tex)
            self.assertIn("No action conditioning", tex)
            self.assertIn("EMA tau=0.99", tex)
            self.assertIn(r"$k{=}1$", tex)
            self.assertIn(r"$k{=}5$", tex)
            self.assertIn(r"$k{=}10$", tex)
            self.assertIn(r"\textbf{$0.123{\pm}0.045$}", tex)
            csv_path = Path(tmp) / "ablation_results.csv"
            _tiny_csv(csv_path)
            report = export_paper_artifacts(
                csv_path=csv_path,
                output_dir=tmp,
                live_eval=False,
                multiseed_json=json_path,
            )
            text = Path(report["tex"]).read_text(encoding="utf-8")
            self.assertIn(r"\begin{table}[t]", text)
            self.assertIn(r"{\pm}", text)
            self.assertIn("tab:ac-jepa-ablation-multiseed", text)
            self.assertIn("Variant B (MSE-only)", text)


class TrajectoryPlotTests(unittest.TestCase):
    def test_plot_writes_png_and_pdf(self) -> None:
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 1.0, 8)
        z_gt = np.stack(
            [np.stack([t, 0.2 * ep + t**2, np.zeros_like(t)], axis=-1) for ep in range(3)],
            axis=0,
        )
        z_pred = z_gt + 0.05 * rng.normal(size=z_gt.shape)
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "paper_latent_trajectories.png"
            plot_latent_trajectories(z_gt, z_pred, png, dpi=80, tsne_iter=30, perplexity=5.0)
            pdf = png.with_suffix(".pdf")
            self.assertTrue(png.is_file())
            self.assertTrue(pdf.is_file())
            self.assertGreater(png.stat().st_size, 1000)


class ExportPipelineTests(unittest.TestCase):
    def test_export_from_csv_without_live_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ablation_results.csv"
            _tiny_csv(csv_path)
            report = export_paper_artifacts(
                csv_path=csv_path,
                output_dir=tmp,
                live_eval=False,
            )
            tex = Path(report["tex"])
            self.assertTrue(tex.is_file())
            text = tex.read_text(encoding="utf-8")
            self.assertIn(r"\begin{table}[t]", text)
            self.assertIn("Variant B (MSE-only)", text)
            self.assertIsNone(report["figure"])


class LatencyAndRolloutTests(unittest.TestCase):
    def test_latency_and_collect_shapes(self) -> None:
        from src.config import DataConfig, ModelConfig
        from src.dataset import DeformableObjectDataset

        model_cfg = ModelConfig(
            img_size=32,
            patch_size=8,
            embed_dim=64,
            encoder_depth=2,
            encoder_num_heads=4,
            mlp_ratio=2.0,
            action_dim=4,
            action_embed_dim=32,
            predictor_embed_dim=64,
            predictor_depth=2,
            predictor_num_heads=4,
            backbone="vit",
        )
        data_cfg = DataConfig(
            n_episodes=8,
            episode_length=6,
            img_size=32,
            n_grid=4,
            action_dim=4,
            seed=0,
            val_fraction=0.25,
        )
        dataset = DeformableObjectDataset(data_cfg, split="all")
        model = ACJEPA(model_cfg)
        model.eval()
        device = torch.device("cpu")
        latency = measure_inference_latency(
            model,
            dataset.observations[:, 0],
            dataset.actions[:, 0],
            device,
            warmup=1,
            steps=2,
        )
        self.assertGreater(latency["fps"], 0.0)
        self.assertGreater(latency["ms_per_frame"], 0.0)
        z_gt, z_pred = collect_latent_trajectories(
            model, dataset, device, n_episodes=2, max_horizon=4
        )
        self.assertEqual(z_gt.shape, z_pred.shape)
        self.assertEqual(z_gt.ndim, 3)
        self.assertEqual(z_gt.shape[0], 2)
        self.assertEqual(z_gt.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
