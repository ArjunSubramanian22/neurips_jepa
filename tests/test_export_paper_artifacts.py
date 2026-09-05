"""Tests for camera-ready paper artifact export."""

from __future__ import annotations

import csv
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
    measure_inference_latency,
    pca_embed,
    plot_latent_trajectories,
    render_ablation_table,
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
