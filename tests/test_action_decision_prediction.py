"""Frozen rules for the intervention-to-regret pilot."""

from __future__ import annotations

import unittest

import numpy as np

from src.metrics.action_decision_prediction import (
    HOLDOUT_EPISODES,
    TEST_EPISODE_MIN,
    assert_development_episodes,
    bootstrap_incremental_r2,
    decide,
    incremental_prediction,
    intervention_features,
)


def _row(episode: int, cosine: float, noise: float, seed: int = 0) -> dict:
    features = intervention_features(cosine, 2.0, 2.0)
    assert features is not None
    regret = (1.0 - cosine) + noise
    return {
        "episode": episode,
        "seed": seed,
        "model": "vicreg_action" if seed < 2 else "mse_action",
        "regret": regret,
        "validation_prediction_loss": 0.2 if seed < 2 else 0.02,
        **features,
    }


class InterventionFeatureTests(unittest.TestCase):
    def test_orthogonal_contrast(self) -> None:
        scored = intervention_features(0.0, 3.0, 4.0)
        self.assertIsNotNone(scored)
        assert scored is not None
        self.assertAlmostEqual(scored["response_ratio"], 0.75)
        self.assertAlmostEqual(scored["parallel_gain"], 0.0)
        self.assertAlmostEqual(scored["orthogonal_energy"], 0.5625)
        self.assertAlmostEqual(scored["effect_nve"], 1.5625)
        self.assertAlmostEqual(scored["effect_rmse"], 5.0)

    def test_scale_only_contrast_has_no_orthogonal_energy(self) -> None:
        scored = intervention_features(1.0, 1.0, 2.0)
        assert scored is not None
        self.assertAlmostEqual(scored["orthogonal_energy"], 0.0)
        self.assertAlmostEqual(scored["parallel_gain"], 0.5)
        self.assertAlmostEqual(scored["effect_nve"], 0.25)

    def test_undefined_when_the_target_is_null(self) -> None:
        self.assertIsNone(intervention_features(1.0, 1.0, 0.0))


class SplitAndDecisionTests(unittest.TestCase):
    def test_test_episodes_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            assert_development_episodes([TEST_EPISODE_MIN])
        assert_development_episodes(HOLDOUT_EPISODES)

    def test_duplicate_diagnostic_does_not_improve_holdout_fit(self) -> None:
        train = []
        test = []
        rng = np.random.default_rng(0)
        for episode in range(1440, 1452):
            for seed in range(4):
                cosine = float(rng.uniform(0.2, 0.9))
                train.append(_row(episode, cosine, float(rng.normal(0.0, 0.01)), seed))
        for episode in range(1452, 1464):
            for seed in range(4):
                cosine = float(rng.uniform(0.2, 0.9))
                test.append(_row(episode, cosine, float(rng.normal(0.0, 0.01)), seed))
        # effect_nve is determined by cosine when the ratio is fixed at 1.
        scored = incremental_prediction(
            train,
            test,
            ["cosine", "effect_nve"],
            ["cosine", "effect_nve", "parallel_gain"],
            "regret",
        )
        self.assertEqual(scored["status"], "ok")
        self.assertLess(abs(scored["incremental_r2"]), 1e-8)

    def test_a_new_signal_improves_holdout_r2(self) -> None:
        train = []
        test = []
        rng = np.random.default_rng(1)
        for split, bucket, start in (("train", train, 1440), ("test", test, 1452)):
            del split
            for episode in range(start, start + 12):
                for seed in range(4):
                    row = _row(episode, float(rng.uniform(0.2, 0.9)), 0.0, seed)
                    row["regret"] = row["orthogonal_energy"] + float(rng.normal(0.0, 0.001))
                    row["cosine"] = float(rng.normal(0.0, 1.0))
                    row["effect_nve"] = float(rng.normal(0.0, 1.0))
                    row["effect_rmse"] = float(rng.normal(0.0, 1.0))
                    row["validation_prediction_loss"] = float(rng.normal(0.0, 1.0))
                    bucket.append(row)
        scored = incremental_prediction(
            train,
            test,
            ["validation_prediction_loss", "cosine", "effect_nve", "effect_rmse"],
            ["validation_prediction_loss", "cosine", "effect_nve", "effect_rmse", "orthogonal_energy"],
            "regret",
        )
        self.assertGreater(scored["incremental_r2"], 0.5)
        interval = bootstrap_incremental_r2(
            train,
            test,
            ["validation_prediction_loss", "cosine", "effect_nve", "effect_rmse"],
            ["validation_prediction_loss", "cosine", "effect_nve", "effect_rmse", "orthogonal_energy"],
            "regret",
            draws=40,
            seed=2,
        )
        self.assertEqual(interval["status"], "ok")
        self.assertGreater(interval["low"], 0.0)

    def test_decision_is_no_go_without_a_stable_increment(self) -> None:
        block = {
            "status": "ok",
            "incremental_r2": 0.0,
            "full_r2": {"value": 0.2},
            "ci_low": -0.1,
        }
        decision = decide(
            {
                "n_rows": 240,
                "primary": block,
                "within_vicreg_action": block,
                "within_mse_action": block,
                "difficulty": block,
                "normalized": block,
                "reverse": block,
                "leave_one_seed_min_incremental_r2": 0.1,
            }
        )
        self.assertEqual(decision["decision"], "no-go")
        self.assertFalse(decision["primary_passed"])

    def test_decision_stays_no_go_when_a_control_removes_the_gain(self) -> None:
        passed = {
            "status": "ok",
            "incremental_r2": 0.2,
            "full_r2": {"value": 0.4},
            "ci_low": 0.05,
        }
        removed = {
            "status": "ok",
            "incremental_r2": 0.0,
            "full_r2": {"value": 0.4},
            "ci_low": 0.05,
        }
        decision = decide(
            {
                "n_rows": 240,
                "primary": passed,
                "within_vicreg_action": passed,
                "within_mse_action": removed,
                "difficulty": passed,
                "normalized": passed,
                "reverse": passed,
                "leave_one_seed_min_incremental_r2": 0.1,
            }
        )
        self.assertEqual(decision["decision"], "no-go")
        self.assertTrue(decision["primary_passed"])
        self.assertTrue(any("within_mse_action" in reason for reason in decision["reasons"]))


if __name__ == "__main__":
    unittest.main()
