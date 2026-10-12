"""One result file for every audit and plan command.

A file is valid when :func:`validate_result` accepts it and
:func:`src.registry.to_jsonable` can encode it. Non-finite floats are refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark import SCHEMA_VERSION

LAYER_NAMES: tuple[str, ...] = (
    "physical_response",
    "observation_visibility",
    "target_response",
    "predictor_response",
    "direction_metrics",
    "ranking_metrics",
    "multi_step_metrics",
    "planning",
)

HEADER_KEYS: tuple[str, ...] = (
    "schema_version",
    "command",
    "env",
    "model",
    "checkpoint",
    "split",
    "split_role",
    "is_test_set",
    "seed",
    "results",
)


def validate_result(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a result file does not match the schema."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    missing = [key for key in HEADER_KEYS if key not in payload]
    if missing:
        raise ValueError(f"missing result fields: {missing}.")
    if payload["command"] not in ("audit", "plan"):
        raise ValueError("command must be audit or plan.")
    if not isinstance(payload["is_test_set"], bool):
        raise ValueError("is_test_set must be a boolean.")
    if payload["split"] == "test" and payload["is_test_set"] is not True:
        raise ValueError("split test requires is_test_set true.")
    if payload["split"] != "test" and payload["is_test_set"]:
        raise ValueError("is_test_set is true only for a declared test split.")
    results = payload["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a non-empty list.")
    for index, result in enumerate(results):
        if "task" not in result or "layers" not in result:
            raise ValueError(f"result {index} needs task and layers.")
        layers = result["layers"]
        for name in LAYER_NAMES:
            if name not in layers:
                raise ValueError(f"result {index} is missing layer {name}.")
            layer = layers[name]
            if not isinstance(layer, dict) or "status" not in layer:
                raise ValueError(f"layer {name} needs a status.")
        planning = layers["planning"]
        for side in ("oracle", "learned"):
            if side not in planning or "status" not in planning[side]:
                raise ValueError(f"planning.{side} needs a status.")


def write_result(path: str | Path, payload: dict[str, Any]) -> Path:
    """Validate, reject non-finite numbers, and write JSON."""
    from src.registry import to_jsonable

    validate_result(payload)
    encoded = to_jsonable(payload)
    validate_result(encoded)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(encoded, indent=2) + "\n", encoding="utf-8")
    return out
