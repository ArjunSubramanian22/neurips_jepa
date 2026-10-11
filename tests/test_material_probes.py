"""Probe sanity checks and validation-only regime selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_protocol import ProtocolError
from src.material_probes import evaluate_probe, history_matrix, select_protocol, train_standardize


def _record(protocol_id: str, uint8: float, mech: float, *, selectable: bool = True, stable: bool = True) -> dict:
    return {
        "protocol_id": protocol_id,
        "selectable": selectable,
        "stable": stable,
        "val": {
            "privileged_stiffness_linear": 1.0,
            "geometry_uint8_linear": 1.0,
            "geometry_mechanical_linear": 1.0,
            "mechanical_history_stiffness_linear": mech,
            "uint8_history_stiffness_linear": uint8,
        },
    }


class ProbeTests(unittest.TestCase):
    def test_separable_labels_are_recovered(self) -> None:
        labels = np.array([0, 1, 2] * 12, dtype=np.int64)
        features = labels.astype(np.float32)[:, None]
        result = evaluate_probe("linear", features[:24], labels[:24], features[24:30], labels[24:30], features[30:], labels[30:], 3)
        self.assertEqual(result["val"]["accuracy"], 1.0)
        self.assertEqual(result["test"]["accuracy"], 1.0)

    def test_constant_features_stay_near_chance(self) -> None:
        labels = np.array([0, 1, 2] * 12, dtype=np.int64)
        features = np.zeros((labels.size, 4), dtype=np.float32)
        result = evaluate_probe("linear", features[:24], labels[:24], features[24:30], labels[24:30], features[30:], labels[30:], 3)
        self.assertLess(result["val"]["accuracy"], 0.5)
        self.assertEqual(result["val"]["n"], 6)

    def test_standardizer_uses_train_rows_only(self) -> None:
        train = np.zeros((4, 2), dtype=np.float64)
        val = np.ones((3, 2), dtype=np.float64) * 100.0
        test = np.ones((3, 2), dtype=np.float64) * -100.0
        scaled_train, scaled_val, _scaled_test = train_standardize(train, val, test)
        np.testing.assert_allclose(scaled_train, 0.0)
        self.assertTrue(np.all(scaled_val > 0.0))

    def test_history_concatenates_prefix(self) -> None:
        frames = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        packed = history_matrix(frames, 2)
        self.assertEqual(packed.shape, (2, 4))
        np.testing.assert_array_equal(packed[0], frames[0, :2].reshape(-1))


class SelectionTests(unittest.TestCase):
    def test_selection_uses_validation_and_requires_every_bar(self) -> None:
        records = [
            _record("weak", uint8=0.40, mech=0.80),
            _record("visible", uint8=0.70, mech=0.75),
            _record("higher_but_blocked", uint8=0.95, mech=0.95, selectable=False),
            _record("unstable", uint8=0.99, mech=0.99, stable=False),
        ]
        decision = select_protocol(records, chance=1.0 / 3.0, margin=0.20, privileged_min=0.99, geometry_min=0.90)
        self.assertEqual(decision["status"], "SELECTED")
        self.assertEqual(decision["protocol_id"], "visible")

    def test_selection_rejects_test_metrics(self) -> None:
        record = _record("leak", uint8=0.8, mech=0.8)
        record["test"] = {"uint8_history_stiffness_linear": 1.0}
        with self.assertRaises(ProtocolError):
            select_protocol([record], chance=0.333, margin=0.2, privileged_min=0.99, geometry_min=0.9)

    def test_none_when_raster_history_is_near_chance(self) -> None:
        decision = select_protocol(
            [_record("near_chance", uint8=0.40, mech=0.80)],
            chance=1.0 / 3.0,
            margin=0.20,
            privileged_min=0.99,
            geometry_min=0.90,
        )
        self.assertEqual(decision["status"], "NONE")
        self.assertIsNone(decision["protocol_id"])


if __name__ == "__main__":
    unittest.main()
