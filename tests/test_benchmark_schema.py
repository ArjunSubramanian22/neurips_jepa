"""The result schema rejects a partial record and a non-finite number."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.schema import LAYER_NAMES, validate_result, write_result


def _record() -> dict:
    layers = {name: {"status": "not_measured", "reason": "fixture"} for name in LAYER_NAMES}
    layers["planning"] = {
        "status": "not_measured",
        "oracle": {"status": "not_measured", "reason": "fixture"},
        "learned": {"status": "not_measured", "reason": "fixture"},
    }
    return {
        "schema_version": "counterfactual_benchmark_v1",
        "command": "audit",
        "env": "toy",
        "model": None,
        "checkpoint": None,
        "split": "val",
        "split_role": "validation",
        "is_test_set": False,
        "seed": 0,
        "results": [{"task": "push", "layers": layers}],
    }


class SchemaTests(unittest.TestCase):
    def test_a_complete_record_validates(self) -> None:
        validate_result(_record())

    def test_a_missing_layer_is_rejected(self) -> None:
        payload = _record()
        del payload["results"][0]["layers"]["direction_metrics"]
        with self.assertRaises(ValueError):
            validate_result(payload)

    def test_labeling_validation_as_test_is_rejected(self) -> None:
        payload = _record()
        payload["split"] = "val"
        payload["is_test_set"] = True
        with self.assertRaises(ValueError):
            validate_result(payload)

    def test_a_non_finite_value_is_not_written(self) -> None:
        payload = _record()
        payload["results"][0]["layers"]["physical_response"]["rms_pos"] = float("nan")
        with self.assertRaises(ValueError):
            write_result(Path("/tmp/should_not_write_benchmark.json"), payload)


if __name__ == "__main__":
    unittest.main()
