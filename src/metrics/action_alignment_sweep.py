"""Aggregate action-alignment ablations without treating sensitivity as success."""

from __future__ import annotations

from typing import Any

PRIMARY_VARIANTS = ("base", "direction", "magnitude", "direction_magnitude")
OPTIONAL_FRACTIONS = (0.5, 1.0)
METRICS = (
    "rollout_error",
    "effective_rank",
    "magnitude_response",
    "cosine_alignment",
    "retrieval",
    "multi_step_fidelity",
    "planning_success",
)
CORRECTNESS_METRICS = ("cosine_alignment", "retrieval")
NOT_MEASURED = "not_measured"


def _blank(name: str, eligible: bool) -> dict[str, Any]:
    return {
        "name": name,
        "status": "NOT_RUN",
        "eligible_for_selection": eligible,
        "metrics": {metric: NOT_MEASURED for metric in METRICS},
    }


def _read_metric(record: dict[str, Any], name: str) -> Any:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    if name in metrics and metrics[name] is not None:
        return metrics[name]
    return None


def aggregate_action_alignment(
    records: dict[str, dict[str, Any] | None],
    *,
    select_on_test: bool,
) -> dict[str, Any]:
    """Choose a variant from validation cosine and retrieval, or withhold the choice."""
    if select_on_test:
        raise ValueError("action-alignment selection must not read the test split.")
    rows = [_row(name, True, records.get(name)) for name in PRIMARY_VARIANTS]
    for fraction in OPTIONAL_FRACTIONS:
        key = f"fraction_{fraction:g}"
        rows.append(_row(key, False, records.get(key)))
    complete = [
        row
        for row in rows
        if row["eligible_for_selection"]
        and row["status"] == "COMPLETE"
        and all(isinstance(row["metrics"][name], (int, float)) for name in CORRECTNESS_METRICS)
    ]
    selected = None
    status = "not_measured"
    correctness = False
    if len(complete) == len(PRIMARY_VARIANTS):
        selected, status, correctness = _select(complete)
    elif complete:
        status = "withheld_incomplete_sweep"
    control = _control_status(rows, selected, correctness)
    return {
        "primary_family": "force_reversal",
        "primary_variants": list(PRIMARY_VARIANTS),
        "selection_role": "val",
        "selection_metrics": list(CORRECTNESS_METRICS),
        "magnitude_response_is_a_selector": False,
        "sensitivity_is_success": False,
        "select_variant_on_test": False,
        "selected_variant": selected,
        "selection_status": status,
        "counterfactual_correctness_improved": correctness,
        "control_improved": control,
        "rows": rows,
    }


def _row(name: str, eligible: bool, record: dict[str, Any] | None) -> dict[str, Any]:
    row = _blank(name, eligible)
    if record is None:
        return row
    row["status"] = str(record.get("status", "NOT_RUN"))
    for metric in METRICS:
        value = _read_metric(record, metric)
        if value is not None:
            row["metrics"][metric] = value
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    test_keys = [key for key in list(record) + list(metrics) if str(key).startswith("test_")]
    if test_keys:
        row["test_metrics_present_but_unused"] = True
    return row


def _select(rows: list[dict[str, Any]]) -> tuple[str | None, str, bool]:
    by_name = {row["name"]: row for row in rows}
    base = by_name["base"]["metrics"]
    improved = []
    for row in rows:
        if row["name"] == "base":
            continue
        metrics = row["metrics"]
        better = all(float(metrics[name]) > float(base[name]) for name in CORRECTNESS_METRICS)
        if better:
            improved.append(row)
    if not improved:
        return "base", "no_variant_improved_counterfactual_correctness", False
    ranked = sorted(
        improved,
        key=lambda row: (float(row["metrics"]["cosine_alignment"]), float(row["metrics"]["retrieval"])),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) > 1:
        other = ranked[1]
        tied = all(float(best["metrics"][name]) == float(other["metrics"][name]) for name in CORRECTNESS_METRICS)
        if tied:
            return None, "tied_on_validation_correctness", True
    return best["name"], "selected_on_validation_correctness", True


def _control_status(rows: list[dict[str, Any]], selected: str | None, correctness: bool) -> str | bool:
    if not correctness or selected in (None, "base"):
        return False
    by_name = {row["name"]: row for row in rows}
    chosen = by_name[selected]["metrics"]["planning_success"]
    base = by_name["base"]["metrics"]["planning_success"]
    if chosen == NOT_MEASURED or base == NOT_MEASURED:
        return NOT_MEASURED
    return float(chosen) > float(base)
