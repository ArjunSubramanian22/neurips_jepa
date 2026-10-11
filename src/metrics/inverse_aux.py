"""Aggregate the predetermined inverse-auxiliary sweep.

Weight choice uses validation rollout only, and only after every joint
weight has a completed run. The frozen probe is a control.
"""

from __future__ import annotations

from typing import Any

WEIGHTS = (0.0, 0.01, 0.1, 1.0)
SELECTION_METRIC = "val_rollout_mse"
REPRESENTATION_METRICS = (
    "rollout_error",
    "latent_geometry",
    "action_magnitude_response",
    "directional_alignment",
    "retrieval",
    "planning",
)
NOT_MEASURED = "not_measured"


def _blank(weight: float | None, role: str) -> dict[str, Any]:
    return {
        "weight": weight,
        "role": role,
        "status": "NOT_RUN",
        "eligible_for_weight_selection": role == "joint",
        "metrics": {name: NOT_MEASURED for name in REPRESENTATION_METRICS},
        SELECTION_METRIC: None,
    }


def aggregate_inverse_sweep(
    records: dict[str, dict[str, Any] | None],
    *,
    select_weight_on_test: bool,
) -> dict[str, Any]:
    """Build the sweep table. ``records`` is keyed by ``w0.0`` ... and ``probe``."""
    if select_weight_on_test:
        raise ValueError("inverse weight selection must not read the test split.")
    rows: list[dict[str, Any]] = []
    for weight in WEIGHTS:
        key = f"w{weight:g}"
        rows.append(_row(key, weight, "joint", records.get(key)))
    rows.append(_row("probe", None, "probe", records.get("probe")))
    complete = [
        row
        for row in rows
        if row["role"] == "joint" and row["status"] == "COMPLETE" and row[SELECTION_METRIC] is not None
    ]
    selected = None
    selection_status = "not_measured"
    if len(complete) == len(WEIGHTS):
        selected = min(complete, key=lambda row: row[SELECTION_METRIC])["weight"]
        selection_status = "selected_on_validation_rollout"
    elif complete:
        selection_status = "withheld_incomplete_sweep"
    return {
        "weights": list(WEIGHTS),
        "selection_metric": SELECTION_METRIC,
        "selection_role": "val",
        "select_weight_on_test": False,
        "selected_weight": selected,
        "selection_status": selection_status,
        "probe_eligible_for_weight_selection": False,
        "rows": rows,
    }


def _row(key: str, weight: float | None, role: str, record: dict[str, Any] | None) -> dict[str, Any]:
    row = _blank(weight, role)
    row["key"] = key
    if record is None:
        return row
    status = str(record.get("status", "NOT_RUN"))
    row["status"] = status
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    for name in REPRESENTATION_METRICS:
        if name in metrics and metrics[name] is not None:
            row["metrics"][name] = metrics[name]
    value = record.get(SELECTION_METRIC)
    if value is None and isinstance(metrics, dict):
        value = metrics.get(SELECTION_METRIC)
    row[SELECTION_METRIC] = value
    if "test_rollout_mse" in record or (isinstance(metrics, dict) and "test_rollout_mse" in metrics):
        row["test_rollout_present_but_unused"] = True
    return row
