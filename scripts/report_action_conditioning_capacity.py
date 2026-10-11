#!/usr/bin/env python3
"""Write parameter and FLOP counts for each action-conditioning mode.

    python3.13 scripts/report_action_conditioning_capacity.py

This does not train and does not score a test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.ac_jepa import ACJEPA
from src.models.capacity import capacity_report

CONFIG = ROOT / "configs" / "experiments" / "action_conditioning_v1.yaml"
OUT = ROOT / "experiments" / "action_conditioning" / "capacity.json"


def main() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if payload.get("select_architecture_on_test"):
        raise RuntimeError("architecture selection must not read the test set.")
    rows = []
    for mode in payload["fuse_modes"]:
        model_cfg = ModelConfig(**{**payload["model"], "fuse_mode": mode})
        report = capacity_report(ACJEPA(model_cfg))
        report["selection_role"] = payload["selection_role"]
        report["split_version"] = payload["split_version"]
        report["trained_metrics"] = "not_measured"
        rows.append(report)
    baseline = next(row for row in rows if row["fuse_mode"] == payload["baseline"])
    for row in rows:
        row["trainable_parameters_over_concat"] = row["trainable_parameters"] / baseline["trainable_parameters"]
        row["predictor_forward_flops_over_concat"] = row["predictor_forward_flops"] / baseline["predictor_forward_flops"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"modes": rows, "winner": None}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
