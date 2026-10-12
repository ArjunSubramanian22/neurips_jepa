"""Deterministic example selection for qualitative figures.

Rules are fixed here. They do not search for a photogenic frame, and they do
not use a test split. The visibility cutoff is the saved validation threshold.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Open-loop one-step force reversal on the historical simulator.
# Teacher-forced rows are excluded because they read the true next frame.
FORK_SLICE: dict[str, str] = {
    "model": "vicreg_action_s42",
    "environment": "mpm_legacy",
    "rollout_mode": "open_loop",
    "protocol": "one_step_replacement",
    "horizon": "1",
    "intervention_source": "force_reversal",
    "is_test_set": "False",
}

# Same counterfactual slice used for the metric-planning summary, widened to
# every checkpoint so strong and poor are chosen by median cosine.
RANK_SLICE: dict[str, str] = {
    "environment": "mpm_legacy",
    "rollout_mode": "open_loop",
    "protocol": "persistent_alternative",
    "horizon": "10",
    "is_test_set": "False",
}

ROLES: tuple[str, ...] = (
    "high_visibility_good_alignment",
    "high_visibility_poor_alignment",
    "physical_but_invisible",
)


def _matches(row: Mapping[str, Any], spec: Mapping[str, str]) -> bool:
    return all(str(row.get(key, "")) == value for key, value in spec.items())


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _order(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row["episode"]), int(row["t"]))


def select_fork_examples(
    rows: Sequence[Mapping[str, Any]],
    visibility_mse_threshold: float,
) -> dict[str, Any]:
    """Pick at most one row per role.

    High visibility means raster MSE above the saved threshold and a defined
    open-loop cosine. Good alignment is the largest cosine. Poor alignment is
    the smallest cosine among the other visible rows. The invisible role is
    the largest particle RMS among rows whose raster does not change.
    Ties break toward the smaller episode, then the smaller time.
    """
    if visibility_mse_threshold < 0.0:
        raise ValueError("visibility threshold must be >= 0.")
    pool = [row for row in rows if _matches(row, FORK_SLICE)]
    visible: list[Mapping[str, Any]] = []
    for row in pool:
        mse = _finite(row.get("raster_mse"))
        cosine = _finite(row.get("cosine"))
        if mse is None or mse <= visibility_mse_threshold:
            continue
        if str(row.get("cosine_status")) != "ok" or cosine is None:
            continue
        visible.append(row)
    selected: dict[str, Any] = {role: None for role in ROLES}
    reasons: dict[str, str] = {}
    if not visible:
        reasons["high_visibility_good_alignment"] = "no open-loop row above the visibility threshold with a defined cosine"
        reasons["high_visibility_poor_alignment"] = "no second visible row"
    else:
        good = min(visible, key=lambda row: (-float(row["cosine"]), *_order(row)))
        selected["high_visibility_good_alignment"] = good
        others = [row for row in visible if _order(row) != _order(good)]
        if not others:
            reasons["high_visibility_poor_alignment"] = "only one visible row, so poor alignment is not a second copy"
        else:
            selected["high_visibility_poor_alignment"] = min(
                others, key=lambda row: (float(row["cosine"]), *_order(row))
            )
    hidden = []
    for row in pool:
        rms = _finite(row.get("rms_pos"))
        affected = _finite(row.get("fraction_affected_pixels"))
        distinct = str(row.get("physically_distinct_observationally_identical")) == "True"
        if distinct and rms is not None and rms > 0.0 and affected == 0.0:
            hidden.append(row)
    if not hidden:
        reasons["physical_but_invisible"] = "no row with positive particle RMS and an unchanged raster"
    else:
        selected["physical_but_invisible"] = min(hidden, key=lambda row: (-float(row["rms_pos"]), *_order(row)))
    identities = [
        _order(row)
        for row in selected.values()
        if row is not None
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("selection assigned one episode time to two roles.")
    return {"selected": selected, "reasons": reasons, "n_pool": len(pool), "n_visible": len(visible)}


def select_counterfactual_models(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Highest and lowest median open-loop cosine.

    Models with no defined cosine are not given a median of zero. Ties break
    toward the lexicographically smaller checkpoint name.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if not _matches(row, RANK_SLICE):
            continue
        if str(row.get("cosine_status")) != "ok":
            continue
        cosine = _finite(row.get("cosine"))
        if cosine is None:
            continue
        grouped.setdefault(str(row["model"]), []).append(cosine)
    if not grouped:
        return {"strong": None, "poor": None, "reason": "no defined cosine in the ranking slice"}
    medians = []
    for model, values in grouped.items():
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            median = ordered[mid]
        else:
            median = 0.5 * (ordered[mid - 1] + ordered[mid])
        medians.append((median, model, len(ordered)))
    strong = min(medians, key=lambda item: (-item[0], item[1]))
    poor = min(medians, key=lambda item: (item[0], item[1]))
    result: dict[str, Any] = {
        "strong": {"model": strong[1], "median_cosine": strong[0], "n": strong[2]},
        "poor": {"model": poor[1], "median_cosine": poor[0], "n": poor[2]},
        "reason": None,
    }
    if strong[1] == poor[1]:
        result["poor"] = None
        result["reason"] = "only one model has a defined cosine"
    return result
