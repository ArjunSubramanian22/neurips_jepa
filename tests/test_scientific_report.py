"""NaN rows stay in the report, and undefined cosines are not written as zero."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scientific_report import build_report, write_report


def _fixture(root: Path) -> None:
    models = root / "analysis"
    models.mkdir(parents=True)
    (models / "metric_planning_models.csv").write_text(
        "model,family,training_seed,one_step_latent_mse,multi_step_latent_mse,directional_cosine,retrieval_accuracy,n_scored_planning\n"
        "kept,vicreg_action,42,nan,0.2,,0.5,5\n"
        "other,vicreg_action,43,0.1,0.2,0.2,0.4,5\n",
        encoding="utf-8",
    )
    (root / "experiments/runs/rgb_visibility_v1").mkdir(parents=True)
    (root / "experiments/runs/rgb_visibility_v1/visibility_summary.json").write_text(
        json.dumps({"rows": [{"environment_id": "legacy", "pipeline": "rgb_64", "n": 2, "median_fraction_affected_pixels": 0.1}]}),
        encoding="utf-8",
    )
    (root / "analysis/metric_planning_headline.json").write_text(
        json.dumps(
            {
                "is_test_set": False,
                "evaluations": [
                    {
                        "metric": "directional_cosine",
                        "n": 2,
                        "spearman": 0.1,
                        "spearman_status": "ok",
                        "spearman_ci_low": -0.2,
                        "spearman_ci_high": 0.4,
                        "ci_status": "ok",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "experiments/runs/planning_benchmark_v1").mkdir(parents=True)
    (root / "experiments/runs/planning_benchmark_v1/aggregation.json").write_text(
        json.dumps(
            {
                "is_test_set": False,
                "split": "test",
                "report_seeds": {
                    "a": {"success_rate": 0.0, "final_distance_mean": 0.1, "n_scored": 5, "n_episodes": 8},
                    "b": {"success_rate": 0.0, "final_distance_mean": 0.1, "n_scored": 5, "n_episodes": 8},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "experiments/runs/headline_seed_uncertainty_v1").mkdir(parents=True)
    (root / "experiments/runs/headline_seed_uncertainty_v1/seed_summaries.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "metric": "ranking_top1",
                        "rollout_mode": "open_loop",
                        "configurations": [
                            {
                                "id": "vicreg_action",
                                "aggregate": {
                                    "mean": 0.5,
                                    "n_complete": 4,
                                    "n_required": 5,
                                    "not_run_seeds": [46],
                                    "failed_seeds": [],
                                    "std": 0.1,
                                },
                                "hierarchical": {"training_seed_interval": {"status": "insufficient_training_seeds"}},
                            },
                            {
                                "id": "mse_action",
                                "aggregate": {
                                    "mean": 0.4,
                                    "n_complete": 2,
                                    "n_required": 5,
                                    "not_run_seeds": [44, 45, 46],
                                    "failed_seeds": [],
                                    "std": 0.1,
                                },
                                "hierarchical": {"training_seed_interval": {"status": "insufficient_training_seeds"}},
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "analysis/capacity_sweep_headline.json").write_text(
        json.dumps(
            {
                "is_test_set": False,
                "scales": [
                    {"scale": "small", "status": "not_trained", "metrics": {"directional_alignment": {"mean": None, "n": 0}}},
                    {"scale": "base", "status": "trained", "metrics": {"directional_alignment": {"mean": 0.01, "n": 4}}},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "experiments/runs/inverse_aux_v1").mkdir(parents=True)
    (root / "experiments/runs/inverse_aux_v1/aggregation.json").write_text(
        json.dumps({"select_weight_on_test": False, "rows": [{"key": "w1", "status": "NOT_RUN"}]}),
        encoding="utf-8",
    )
    (root / "experiments/runs/action_alignment_v1").mkdir(parents=True)
    (root / "experiments/runs/action_alignment_v1/aggregation.json").write_text(
        json.dumps({"select_variant_on_test": False, "rows": [{"name": "direction", "status": "FAILED"}]}),
        encoding="utf-8",
    )
    (root / "experiments/ablations/v1").mkdir(parents=True)
    (root / "experiments/ablations/v1/ledger.jsonl").write_text(
        json.dumps({"jobs": [{"template": "model_rssm", "seed": 42, "status": "NOT_RUN"}]}) + "\n",
        encoding="utf-8",
    )


class ReportTests(unittest.TestCase):
    def test_nan_and_undefined_cosine_rows_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            report = build_report(root)
            rollout = report["tables"]["model_rollout"]
            self.assertEqual(len(rollout), 4)
            nan_rows = [row for row in rollout if row["value_status"] == "nan"]
            self.assertEqual(len(nan_rows), 1)
            undefined = [row for row in report["tables"]["directional_counterfactual"] if row["value_status"] == "undefined_cosine"]
            self.assertEqual(len(undefined), 1)
            self.assertIsNone(undefined[0]["value"])
            flags = {flag["flag"] for flag in report["flags"]}
            self.assertIn("nan", flags)
            self.assertIn("undefined_cosine", flags)
            self.assertIn("missing_seed", flags)
            self.assertIn("failed_run", flags)
            self.assertIn("unequal_sample_count", flags)
            self.assertIn("suspicious_duplicate", flags)
            self.assertIn("train_test_contamination", flags)
            write_report(root, root / "reports" / "latest")
            text = (root / "reports" / "latest" / "RESULTS.md").read_text(encoding="utf-8")
            self.assertIn("OBSERVED RESULT", text)
            self.assertIn("POSSIBLE INTERPRETATION", text)
            self.assertNotIn("statistically significant", text)
            with (root / "reports" / "latest" / "paper_tables.csv").open(encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
            self.assertTrue(any(row["value"] == "nan" for row in saved))
            self.assertTrue((root / "reports" / "latest" / "tables" / "model_rollout.tex").is_file())


if __name__ == "__main__":
    unittest.main()
