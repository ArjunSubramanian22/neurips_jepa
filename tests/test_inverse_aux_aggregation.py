"""The inverse sweep does not pick a weight from the test split."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.inverse_aux import NOT_MEASURED, aggregate_inverse_sweep


class InverseAggregationTests(unittest.TestCase):
    def test_missing_runs_stay_unmeasured(self) -> None:
        summary = aggregate_inverse_sweep({}, select_weight_on_test=False)
        self.assertIsNone(summary["selected_weight"])
        self.assertEqual(summary["selection_status"], "not_measured")
        self.assertEqual(len(summary["rows"]), 5)
        self.assertTrue(all(row["metrics"]["planning"] == NOT_MEASURED for row in summary["rows"]))
        self.assertFalse(summary["rows"][-1]["eligible_for_weight_selection"])

    def test_validation_rollout_selects_and_ignores_a_better_test_score(self) -> None:
        records = {}
        for weight, val, test in ((0.0, 0.4, 0.01), (0.01, 0.2, 0.9), (0.1, 0.3, 0.0), (1.0, 0.5, 0.0)):
            records[f"w{weight:g}"] = {
                "status": "COMPLETE",
                "val_rollout_mse": val,
                "test_rollout_mse": test,
                "metrics": {"rollout_error": val, "planning": None},
            }
        summary = aggregate_inverse_sweep(records, select_weight_on_test=False)
        self.assertEqual(summary["selected_weight"], 0.01)
        self.assertTrue(all(row.get("test_rollout_present_but_unused") for row in summary["rows"] if row["role"] == "joint"))

    def test_an_incomplete_sweep_does_not_select(self) -> None:
        records = {"w0": {"status": "COMPLETE", "val_rollout_mse": 0.1, "metrics": {}}}
        summary = aggregate_inverse_sweep(records, select_weight_on_test=False)
        self.assertIsNone(summary["selected_weight"])
        self.assertEqual(summary["selection_status"], "withheld_incomplete_sweep")

    def test_test_selection_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_inverse_sweep({}, select_weight_on_test=True)


if __name__ == "__main__":
    unittest.main()
