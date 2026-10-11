"""Target-mask aggregation keeps the CLS baseline and ignores sensitivity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.target_mask_sweep import NOT_MEASURED, aggregate_target_masks


def _done(
    rollout: float,
    sensitivity: float,
    cosine: float,
    retrieval: float,
    planning: float | None = None,
    test_cosine: float | None = None,
) -> dict:
    record = {
        "status": "COMPLETE",
        "metrics": {
            "rollout_error": rollout,
            "action_sensitivity": sensitivity,
            "directional_correctness": cosine,
            "retrieval": retrieval,
            "planning_success": planning,
        },
    }
    if test_cosine is not None:
        record["test_directional_correctness"] = test_cosine
    return record


class TargetMaskSweepTests(unittest.TestCase):
    def test_missing_runs_stay_unmeasured(self) -> None:
        summary = aggregate_target_masks({}, select_on_test=False)
        self.assertIsNone(summary["selected_variant"])
        self.assertEqual(summary["canonical_baseline"], "cls_uniform")
        self.assertFalse(summary["replaces_canonical_baseline"])
        self.assertFalse(summary["action_sensitivity_is_a_selector"])
        self.assertEqual(summary["rows"][0]["metrics"]["planning_success"], NOT_MEASURED)

    def test_sensitivity_and_a_worse_rollout_do_not_win(self) -> None:
        records = {
            "cls_uniform": _done(0.20, 0.01, 0.05, 0.50),
            "pool_uniform": _done(0.20, 0.01, 0.05, 0.50),
            "pool_soft": _done(0.40, 9.0, 0.90, 0.90),
            "pool_topk": _done(0.20, 8.0, 0.04, 0.40),
        }
        summary = aggregate_target_masks(records, select_on_test=False)
        self.assertEqual(summary["selected_variant"], "pool_uniform")
        self.assertFalse(summary["counterfactual_correctness_improved"])
        self.assertFalse(summary["replaces_canonical_baseline"])

    def test_validation_correctness_ignores_a_better_test_score(self) -> None:
        records = {
            "cls_uniform": _done(0.20, 0.01, 0.05, 0.50, planning=0.2, test_cosine=0.99),
            "pool_uniform": _done(0.20, 0.01, 0.10, 0.50, planning=0.2, test_cosine=0.99),
            "pool_soft": _done(0.18, 0.02, 0.30, 0.70, planning=0.4, test_cosine=0.01),
            "pool_topk": _done(0.19, 0.04, 0.20, 0.60, planning=0.3, test_cosine=0.98),
        }
        summary = aggregate_target_masks(records, select_on_test=False)
        self.assertEqual(summary["selected_variant"], "pool_soft")
        self.assertTrue(summary["counterfactual_correctness_improved"])
        self.assertTrue(summary["control_improved"])
        self.assertFalse(summary["replaces_canonical_baseline"])

    def test_planning_stays_unmeasured_without_a_control_score(self) -> None:
        records = {
            "cls_uniform": _done(0.2, 0.01, 0.05, 0.5),
            "pool_uniform": _done(0.2, 0.01, 0.10, 0.50),
            "pool_soft": _done(0.2, 0.02, 0.20, 0.60),
            "pool_topk": _done(0.3, 0.02, 0.11, 0.51),
        }
        summary = aggregate_target_masks(records, select_on_test=False)
        self.assertEqual(summary["selected_variant"], "pool_soft")
        self.assertEqual(summary["control_improved"], NOT_MEASURED)

    def test_incomplete_sweep_does_not_select(self) -> None:
        summary = aggregate_target_masks({"cls_uniform": _done(0.2, 0.0, 0.1, 0.5)}, select_on_test=False)
        self.assertIsNone(summary["selected_variant"])
        self.assertEqual(summary["selection_status"], "withheld_incomplete_sweep")

    def test_test_selection_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_target_masks({}, select_on_test=True)


if __name__ == "__main__":
    unittest.main()
