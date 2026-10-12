"""Selection rules stay the same if the table is reordered."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qualitative_select import FORK_SLICE, select_counterfactual_models, select_fork_examples


def _row(**overrides):
    base = {
        "model": "vicreg_action_s42",
        "environment": "mpm_legacy",
        "rollout_mode": "open_loop",
        "protocol": "one_step_replacement",
        "horizon": "1",
        "intervention_source": "force_reversal",
        "is_test_set": "False",
        "episode": "1",
        "t": "0",
        "seed": "42",
        "raster_mse": "1e-4",
        "fraction_affected_pixels": "0.2",
        "cosine": "0.1",
        "cosine_status": "ok",
        "rms_pos": "1e-4",
        "physically_distinct_observationally_identical": "False",
        "target_delta_l2": "0.2",
        "pred_delta_l2": "0.1",
    }
    base.update(overrides)
    return base


class SelectionTests(unittest.TestCase):
    def test_good_and_poor_are_different_rows_and_ignore_order(self) -> None:
        rows = [
            _row(episode="3", cosine="0.2"),
            _row(episode="2", cosine="-0.4"),
            _row(episode="9", cosine="0.9"),
            _row(episode="4", raster_mse="0.0", fraction_affected_pixels="0.0", cosine="", cosine_status="undefined_zero_vector", rms_pos="0.01", physically_distinct_observationally_identical="True"),
        ]
        forward = select_fork_examples(rows, 1e-6)
        backward = select_fork_examples(list(reversed(rows)), 1e-6)
        good = forward["selected"]["high_visibility_good_alignment"]
        poor = forward["selected"]["high_visibility_poor_alignment"]
        hidden = forward["selected"]["physical_but_invisible"]
        self.assertEqual(good["episode"], "9")
        self.assertEqual(poor["episode"], "2")
        self.assertEqual(hidden["episode"], "4")
        self.assertEqual(good["episode"], backward["selected"]["high_visibility_good_alignment"]["episode"])
        self.assertEqual(poor["episode"], backward["selected"]["high_visibility_poor_alignment"]["episode"])
        self.assertNotEqual((good["episode"], good["t"]), (poor["episode"], poor["t"]))

    def test_one_visible_row_does_not_become_both_roles(self) -> None:
        report = select_fork_examples([_row(episode="7", cosine="0.3")], 1e-6)
        self.assertEqual(report["selected"]["high_visibility_good_alignment"]["episode"], "7")
        self.assertIsNone(report["selected"]["high_visibility_poor_alignment"])
        self.assertIn("only one", report["reasons"]["high_visibility_poor_alignment"])

    def test_teacher_forced_and_below_threshold_are_excluded(self) -> None:
        rows = [
            _row(episode="1", rollout_mode="teacher_forced", cosine="1.0"),
            _row(episode="2", raster_mse="1e-12", cosine="1.0"),
        ]
        report = select_fork_examples(rows, 1e-6)
        self.assertIsNone(report["selected"]["high_visibility_good_alignment"])
        self.assertEqual(report["n_pool"], 1)

    def test_model_rank_uses_median_and_does_not_fill_missing_cosine_with_zero(self) -> None:
        rows = []
        for model, values in (("alpha", [0.2, 0.4]), ("beta", [-0.2, 0.0]), ("gamma", [])):
            for value in values:
                rows.append({**{key: value_ for key, value_ in {
                    "environment": "mpm_legacy",
                    "rollout_mode": "open_loop",
                    "protocol": "persistent_alternative",
                    "horizon": "10",
                    "is_test_set": "False",
                    "model": model,
                    "cosine": str(value),
                    "cosine_status": "ok",
                }.items()}})
        rows.append({
            "environment": "mpm_legacy",
            "rollout_mode": "open_loop",
            "protocol": "persistent_alternative",
            "horizon": "10",
            "is_test_set": "False",
            "model": "noaction",
            "cosine": "",
            "cosine_status": "undefined_zero_vector",
        })
        ranked = select_counterfactual_models(rows)
        self.assertEqual(ranked["strong"]["model"], "alpha")
        self.assertEqual(ranked["poor"]["model"], "beta")
        self.assertNotIn("noaction", (ranked["strong"]["model"], ranked["poor"]["model"]))

    def test_slice_is_not_the_test_set(self) -> None:
        self.assertEqual(FORK_SLICE["is_test_set"], "False")


if __name__ == "__main__":
    unittest.main()
