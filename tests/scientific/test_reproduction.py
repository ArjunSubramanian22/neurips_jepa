"""The canonical rollout reduction stays inside the stored fixture tolerance."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.canonical_rollout import horizon_rollout_metrics

FIXTURE = ROOT / "tests" / "fixtures" / "canonical_rollout_v1.json"


class CanonicalEvaluatorFixtureTests(unittest.TestCase):
    def test_horizon_metrics_match_the_hand_derived_fixture(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        tolerance = float(payload["absolute_tolerance"])
        metrics = horizon_rollout_metrics(payload["predicted"], payload["target"])
        for key in ("rollout_mse", "target_variance", "vn_mse"):
            actual = metrics[key].detach().to(dtype=torch.float64)
            expected = torch.tensor(payload[key], dtype=torch.float64)
            self.assertTrue(torch.isfinite(actual).all(), key)
            self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=tolerance), key)


if __name__ == "__main__":
    unittest.main()
