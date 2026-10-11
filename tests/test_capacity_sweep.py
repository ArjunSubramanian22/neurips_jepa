"""The width sweep changes one factor and does not invent a scale trend."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = _load("generate_capacity_configs", ROOT / "scripts" / "generate_capacity_configs.py")
capacity = _load("capacity_sweep", ROOT / "analysis" / "capacity_sweep.py")
FIXED = generate.FIXED
write_configs = generate.write_configs
scale_trend = capacity.scale_trend


class CapacitySweepTests(unittest.TestCase):
    def test_generated_configs_change_width_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_configs(Path(tmp))
            payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths if path.name.startswith("capacity_") and path.suffix == ".yaml" and "sweep" not in path.name and "seed" not in path.name]
        self.assertEqual(len(payloads), 3)
        trainings = [payload["training"] for payload in payloads]
        self.assertEqual(trainings[0], trainings[1])
        self.assertEqual(trainings[1], trainings[2])
        widths = [payload["model"]["embed_dim"] for payload in payloads]
        self.assertEqual(widths, [64, 128, 256])
        for payload in payloads:
            self.assertEqual(payload["model"]["predictor_embed_dim"], payload["model"]["embed_dim"])
            self.assertEqual(payload["model"]["encoder_depth"], FIXED["encoder_depth"])
            self.assertEqual(payload["model"]["encoder_num_heads"], FIXED["encoder_num_heads"])
            self.assertEqual(payload["model"]["predictor_depth"], FIXED["predictor_depth"])
            self.assertEqual(payload["data"]["split_version"], "legacy_v1")
            self.assertTrue(payload["data"]["normalize_actions"])
            self.assertFalse(payload["select_on_test"])

    def test_one_trained_width_is_not_a_scale_effect(self) -> None:
        self.assertEqual(
            scale_trend(
                [
                    {"scale": "small", "value": None},
                    {"scale": "base", "value": 0.07},
                    {"scale": "large", "value": None},
                ]
            ),
            "not_identified",
        )
        self.assertEqual(
            scale_trend(
                [
                    {"scale": "small", "value": 0.1},
                    {"scale": "base", "value": 0.2},
                ]
            ),
            "estimated",
        )


if __name__ == "__main__":
    unittest.main()
