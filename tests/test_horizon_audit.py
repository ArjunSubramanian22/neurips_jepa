"""Toy dynamics for the multi-horizon counterfactual audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.horizon_audit import (
    counterfactual_actions,
    open_loop_rollout,
    score_horizon_pair,
    summarize_labeled,
    teacher_forced_rollout,
)


def _add(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float64) + np.asarray(action, dtype=np.float64)


def _biased(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return _add(state, action) + np.ones_like(np.asarray(state, dtype=np.float64))


def _raster(state: np.ndarray) -> np.ndarray:
    value = int(np.clip(round(float(state.reshape(-1)[0]) * 10.0 + 128.0), 0, 255))
    return np.full((1, 2, 2), value, dtype=np.uint8)


def _particle(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float64).reshape(1, -1)


class HorizonAuditTests(unittest.TestCase):
    def test_protocols_are_distinct_and_labeled(self) -> None:
        recorded = np.array(
            [[0.2, 0.1, 0.4, 0.0], [0.2, 0.1, 0.4, 0.0], [0.2, 0.1, 0.4, 0.0], [0.0, 0.2, -0.1, 0.3]],
            dtype=np.float64,
        )
        reversal = recorded[0].copy()
        reversal[2:4] *= -1.0
        shuffled = np.array([0.9, 0.1, 0.0, -0.5])
        one_step = counterfactual_actions(recorded, reversal, "one_step_replacement", 3)
        block = counterfactual_actions(recorded, reversal, "command_block_replacement", 3)
        persistent = counterfactual_actions(recorded, reversal, "persistent_alternative", 3)
        followed = counterfactual_actions(recorded, shuffled, "alternative_then_recorded", 3)
        self.assertTrue(np.allclose(one_step[0], reversal))
        self.assertTrue(np.allclose(one_step[1:], recorded[1:]))
        self.assertTrue(np.allclose(block[:3], reversal))
        self.assertTrue(np.allclose(block[3:], recorded[3:]))
        self.assertTrue(np.allclose(persistent, reversal))
        self.assertTrue(np.allclose(followed[0], shuffled))
        self.assertTrue(np.allclose(followed[1:], recorded[1:]))
        self.assertFalse(np.allclose(one_step, followed))
        with self.assertRaises(ValueError):
            counterfactual_actions(recorded, reversal, "command_block_replacement", 5)

    def test_open_loop_recurses_and_teacher_forcing_does_not(self) -> None:
        start = np.array([0.0])
        actions = np.zeros((3, 1))
        truth = np.zeros((4, 1))
        rolled = open_loop_rollout(_biased, start, actions)
        forced = teacher_forced_rollout(_biased, truth, actions)
        self.assertTrue(np.allclose(rolled.reshape(-1), [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(forced.reshape(-1), [1.0, 1.0, 1.0]))

    def test_exact_open_loop_matches_direction_and_retrieval(self) -> None:
        start = np.array([0.0])
        recorded = np.array([[1.0], [0.0], [0.0]])
        branch = counterfactual_actions(recorded, np.array([-1.0]), "one_step_replacement", 1)
        truth_a = np.concatenate([start.reshape(1, -1), open_loop_rollout(_add, start, recorded)], axis=0)
        truth_b = np.concatenate([start.reshape(1, -1), open_loop_rollout(_add, start, branch)], axis=0)
        pred_a = open_loop_rollout(_add, start, recorded)
        pred_b = open_loop_rollout(_add, start, branch)
        first = None
        for horizon in (1, 3):
            index = horizon - 1
            scored = score_horizon_pair(
                protocol="one_step_replacement",
                rollout_mode="open_loop",
                horizon=horizon,
                position_a=_particle(truth_a[horizon]),
                position_b=_particle(truth_b[horizon]),
                velocity_a=_particle(np.zeros(1)),
                velocity_b=_particle(np.zeros(1)),
                raster_a=_raster(truth_a[horizon]),
                raster_b=_raster(truth_b[horizon]),
                target_a=truth_a[horizon],
                target_b=truth_b[horizon],
                pred_a=pred_a[index],
                pred_b=pred_b[index],
                target_delta_h1=None if first is None else first,
            )
            self.assertEqual(scored["protocol"], "one_step_replacement")
            self.assertEqual(scored["rollout_mode"], "open_loop")
            self.assertAlmostEqual(scored["cosine"], 1.0, places=12)
            self.assertAlmostEqual(scored["magnitude_ratio"], 1.0, places=8)
            self.assertAlmostEqual(scored["normalized_vector_error"], 0.0, places=12)
            self.assertEqual(scored["counterfactual_ranking_top1"], 1.0)
            self.assertAlmostEqual(scored["rollout_mse_recorded"], 0.0, places=12)
            self.assertGreater(scored["rms_pos"], 0.0)
            self.assertFalse(scored["raster_identical"])
            if first is None:
                first = truth_b[horizon] - truth_a[horizon]
                self.assertEqual(scored["target_direction_status"], "not_applicable")
            else:
                self.assertAlmostEqual(scored["target_direction_cosine"], 1.0, places=12)

    def test_identical_futures_are_ambiguous_not_misses(self) -> None:
        state = np.array([0.5])
        scored = score_horizon_pair(
            protocol="persistent_alternative",
            rollout_mode="teacher_forced",
            horizon=1,
            position_a=_particle(state),
            position_b=_particle(state),
            velocity_a=_particle(np.zeros(1)),
            velocity_b=_particle(np.zeros(1)),
            raster_a=_raster(state),
            raster_b=_raster(state),
            target_a=state,
            target_b=state,
            pred_a=state,
            pred_b=state,
        )
        self.assertTrue(scored["observationally_ambiguous"])
        self.assertIsNone(scored["counterfactual_ranking_top1"])
        self.assertEqual(scored["cosine_status"], "undefined_zero_vector")
        self.assertIsNone(scored["cosine"])
        summary = summarize_labeled([scored])
        self.assertEqual(summary["n_total"], 1)
        self.assertEqual(summary["n_ranking_scored"], 0)
        self.assertEqual(summary["n_ranking_ambiguous"], 1)
        self.assertIsNone(summary["ranking_top1"])

    def test_action_blind_prediction_does_not_retrieve_the_counterfactual(self) -> None:
        recorded_future = np.array([1.0])
        counterfactual_future = np.array([3.0])
        blind = recorded_future
        scored = score_horizon_pair(
            protocol="alternative_then_recorded",
            rollout_mode="open_loop",
            horizon=1,
            position_a=_particle(recorded_future),
            position_b=_particle(counterfactual_future),
            velocity_a=_particle(np.array([1.0])),
            velocity_b=_particle(np.array([3.0])),
            raster_a=_raster(recorded_future),
            raster_b=_raster(counterfactual_future),
            target_a=recorded_future,
            target_b=counterfactual_future,
            pred_a=blind,
            pred_b=blind,
        )
        self.assertEqual(scored["counterfactual_ranking_top1"], 0.0)
        self.assertEqual(scored["cosine_status"], "undefined_zero_vector")
        self.assertAlmostEqual(scored["magnitude_ratio"], 0.0, places=12)
        self.assertGreater(scored["rms_vel"], 0.0)

    def test_summary_refuses_mixed_labels(self) -> None:
        base = dict(
            position_a=_particle(np.array([0.0])),
            position_b=_particle(np.array([1.0])),
            velocity_a=_particle(np.zeros(1)),
            velocity_b=_particle(np.zeros(1)),
            raster_a=_raster(np.array([0.0])),
            raster_b=_raster(np.array([1.0])),
            target_a=np.array([0.0]),
            target_b=np.array([1.0]),
            pred_a=np.array([0.0]),
            pred_b=np.array([1.0]),
        )
        open_row = score_horizon_pair(protocol="persistent_alternative", rollout_mode="open_loop", horizon=1, **base)
        forced_row = score_horizon_pair(protocol="persistent_alternative", rollout_mode="teacher_forced", horizon=1, **base)
        with self.assertRaises(ValueError):
            summarize_labeled([open_row, forced_row])


if __name__ == "__main__":
    unittest.main()
