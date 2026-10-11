"""Action-dependent target masks. An ablation, not required infrastructure.

The canonical run is CLS uniform. Weighted pooling with uniform weights is
the control for the soft and soft-top-k masks. Selection, when runs exist,
reads validation only and does not treat action sensitivity as success.
"""

from __future__ import annotations

from typing import Any

CANONICAL = "cls_uniform"
CONTROL = "pool_uniform"
CANDIDATES = ("pool_soft", "pool_topk")
PRIMARY = (CANONICAL, CONTROL, *CANDIDATES)
OPTIONAL = ("patch_tokens", "local_head")
METRICS = (
    "rollout_error",
    "action_sensitivity",
    "directional_correctness",
    "retrieval",
    "planning_success",
)
NOT_MEASURED = "not_measured"


def _blank(name: str, eligible: bool) -> dict[str, Any]:
    return {
        "name": name,
        "status": "NOT_RUN",
        "eligible_for_selection": eligible,
        "metrics": {metric: NOT_MEASURED for metric in METRICS},
    }


def _metric(record: dict[str, Any], name: str) -> Any:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    value = metrics.get(name)
    return value if value is not None else None


def aggregate_target_masks(
    records: dict[str, dict[str, Any] | None],
    *,
    select_on_test: bool,
) -> dict[str, Any]:
    if select_on_test:
        raise ValueError("target-mask selection must not read the test split.")
    rows = [_row(name, name in CANDIDATES or name == CONTROL, records.get(name)) for name in PRIMARY]
    for name in OPTIONAL:
        rows.append(_row(name, False, records.get(name)))
    complete = [
        row
        for row in rows
        if row["name"] in PRIMARY
        and row["status"] == "COMPLETE"
        and all(isinstance(row["metrics"][name], (int, float)) for name in ("rollout_error", "directional_correctness", "retrieval"))
    ]
    selected = None
    status = "not_measured"
    correctness = False
    if len(complete) == len(PRIMARY):
        selected, status, correctness = _select(complete)
    elif complete:
        status = "withheld_incomplete_sweep"
    return {
        "canonical_baseline": CANONICAL,
        "readout_control": CONTROL,
        "replaces_canonical_baseline": False,
        "action_sensitivity_is_a_selector": False,
        "selection_role": "val",
        "select_variant_on_test": False,
        "selected_variant": selected,
        "selection_status": status,
        "counterfactual_correctness_improved": correctness,
        "control_improved": _control(rows, selected, correctness),
        "rows": rows,
    }


def _row(name: str, eligible: bool, record: dict[str, Any] | None) -> dict[str, Any]:
    row = _blank(name, eligible)
    if record is None:
        return row
    row["status"] = str(record.get("status", "NOT_RUN"))
    for metric in METRICS:
        value = _metric(record, metric)
        if value is not None:
            row["metrics"][metric] = value
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    if any(str(key).startswith("test_") for key in list(record) + list(metrics)):
        row["test_metrics_present_but_unused"] = True
    return row


def _beats_control(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    return (
        float(candidate["metrics"]["rollout_error"]) <= float(control["metrics"]["rollout_error"])
        and float(candidate["metrics"]["directional_correctness"]) > float(control["metrics"]["directional_correctness"])
        and float(candidate["metrics"]["retrieval"]) > float(control["metrics"]["retrieval"])
    )


def _select(rows: list[dict[str, Any]]) -> tuple[str | None, str, bool]:
    by_name = {row["name"]: row for row in rows}
    control = by_name[CONTROL]
    improved = [row for row in rows if row["name"] in CANDIDATES and _beats_control(row, control)]
    if not improved:
        return CONTROL, "no_weighting_improved_correctness_without_hurting_prediction", False
    ranked = sorted(
        improved,
        key=lambda row: (
            float(row["metrics"]["directional_correctness"]),
            float(row["metrics"]["retrieval"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) > 1 and all(
        float(best["metrics"][name]) == float(ranked[1]["metrics"][name])
        for name in ("directional_correctness", "retrieval")
    ):
        return None, "tied_on_validation_correctness", True
    return best["name"], "selected_on_validation_correctness", True


def _control(rows: list[dict[str, Any]], selected: str | None, correctness: bool) -> str | bool:
    if not correctness or selected in (None, CONTROL, CANONICAL):
        return False
    by_name = {row["name"]: row for row in rows}
    chosen = by_name[selected]["metrics"]["planning_success"]
    control = by_name[CONTROL]["metrics"]["planning_success"]
    if chosen == NOT_MEASURED or control == NOT_MEASURED:
        return NOT_MEASURED
    return float(chosen) > float(control)
