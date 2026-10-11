"""Alignment aggregation does not treat a larger action response as success."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.action_alignment_sweep import NOT_MEASURED, aggregate_action_alignment


def _complete(cosine: float, retrieval: float, magnitude: float, planning: float | None = None, test_cosine: float | None = None) -> dict:
    metrics = {
        "rollout_error": 0.2,
        "effective_rank": 10.0,
        "magnitude_response": magnitude,
        "cosine_alignment": cosine,
        "retrieval": retrieval,
        "multi_step_fidelity": 0.1,
        "planning_success": planning,
    }
    record = {"status": "COMPLETE", "metrics": metrics}
    if test_cosine is not None:
        record["test_cosine_alignment"] = test_cosine
    return record


class ActionAlignmentSweepTests(unittest.TestCase):
    def test_missing_runs_stay_unmeasured(self) -> None:
        summary = aggregate_action_alignment({}, select_on_test=False)
        self.assertIsNone(summary["selected_variant"])
        self.assertEqual(summary["selection_status"], "not_measured")
        self.assertFalse(summary["sensitivity_is_success"])
        self.assertFalse(summary["magnitude_response_is_a_selector"])
        self.assertTrue(all(row["metrics"]["planning_success"] == NOT_MEASURED for row in summary["rows"][:4]))

    def test_sensitivity_alone_does_not_win(self) -> None:
        records = {
            "base": _complete(0.05, 0.50, 0.02),
            "direction": _complete(0.04, 0.49, 5.0),
            "magnitude": _complete(0.05, 0.50, 9.0),
            "direction_magnitude": _complete(0.04, 0.48, 8.0),
        }
        summary = aggregate_action_alignment(records, select_on_test=False)
        self.assertEqual(summary["selected_variant"], "base")
        self.assertFalse(summary["counterfactual_correctness_improved"])
        self.assertFalse(summary["control_improved"])

    def test_validation_correctness_ignores_a_better_test_score(self) -> None:
        records = {
            "base": _complete(0.10, 0.50, 0.02, planning=0.2, test_cosine=0.9),
            "direction": _complete(0.20, 0.60, 0.03, planning=0.4, test_cosine=0.01),
            "magnitude": _complete(0.12, 0.55, 1.0, planning=0.1, test_cosine=0.99),
            "direction_magnitude": _complete(0.30, 0.70, 0.04, planning=0.5, test_cosine=0.02),
        }
        summary = aggregate_action_alignment(records, select_on_test=False)
        self.assertEqual(summary["selected_variant"], "direction_magnitude")
        self.assertTrue(summary["counterfactual_correctness_improved"])
        self.assertTrue(summary["control_improved"])
        self.assertTrue(all(row.get("test_metrics_present_but_unused") for row in summary["rows"] if row["eligible_for_selection"]))

    def test_planning_stays_unmeasured_when_the_run_did_not_score_it(self) -> None:
        records = {
            "base": _complete(0.10, 0.50, 0.02),
            "direction": _complete(0.20, 0.60, 0.03),
            "magnitude": _complete(0.01, 0.40, 2.0),
            "direction_magnitude": _complete(0.11, 0.51, 0.02),
        }
        summary = aggregate_action_alignment(records, select_on_test=False)
        self.assertEqual(summary["selected_variant"], "direction")
        self.assertEqual(summary["control_improved"], NOT_MEASURED)

    def test_incomplete_sweep_does_not_select(self) -> None:
        summary = aggregate_action_alignment(
            {"base": _complete(0.1, 0.5, 0.0)},
            select_on_test=False,
        )
        self.assertIsNone(summary["selected_variant"])
        self.assertEqual(summary["selection_status"], "withheld_incomplete_sweep")

    def test_test_selection_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_action_alignment({}, select_on_test=True)


if __name__ == "__main__":
    unittest.main()
