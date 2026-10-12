"""The audit and plan commands write one schema and refuse a missing test split."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.audit import main as audit_main
from benchmark.environments import ForkableEnvironment, register_env, register_task_spec
from benchmark.plan import main as plan_main
from benchmark.schema import validate_result
from tests.test_benchmark_evaluator import _Toy


def _register() -> None:
    register_env("toy_cli", lambda task: ForkableEnvironment(_Toy(task)), ("slide",), {"val": ("validation", False)})

    def spec(_task: str) -> dict:
        return {
            "actions": {
                "along": np.array([0.4, 0.0]),
                "hold": np.zeros(2),
                "miss": np.array([0.0, 0.4]),
            },
            "positive": "along",
            "baseline": "hold",
            "alternate": "miss",
            "horizon": 2,
            "goal_cost": lambda _name, future: abs(float(future["positions"][0, 0]) - 1.0),
            "learned_reason": "toy has no checkpoint",
        }

    register_task_spec("toy_cli", spec)


class CommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _register()

    def test_audit_writes_a_schema_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "audit.json"
            audit_main(["--env", "toy_cli", "--split", "val", "--task", "slide", "--seed", "0", "--output", str(output)])
            payload = json.loads(output.read_text(encoding="utf-8"))
        validate_result(payload)
        self.assertEqual(payload["command"], "audit")
        self.assertEqual(payload["results"][0]["layers"]["planning"]["oracle"]["chosen"], "along")
        self.assertEqual(payload["results"][0]["layers"]["predictor_response"]["status"], "not_measured")
        self.assertEqual(payload["results"][0]["layers"]["planning"]["status"], "not_measured")
        self.assertIn("git_commit", payload)
        self.assertIn("config", payload)
        self.assertIsNone(payload["config"])

    def test_plan_uses_the_same_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            plan_main(["--env", "toy_cli", "--split", "val", "--output", str(output)])
            payload = json.loads(output.read_text(encoding="utf-8"))
        validate_result(payload)
        self.assertEqual(payload["command"], "plan")
        self.assertEqual(payload["results"][0]["task"], "slide")

    def test_split_test_on_the_arm_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "should_not_exist.json"
            with self.assertRaises(SystemExit) as raised:
                audit_main(
                    [
                        "--env",
                        "mujoco_arm",
                        "--split",
                        "test",
                        "--task",
                        "push_object",
                        "--output",
                        str(output),
                    ]
                )
        self.assertIn("no split", str(raised.exception))
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
