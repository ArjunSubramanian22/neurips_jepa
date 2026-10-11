"""Visibility cuts that do not get chosen by the outcome.

Percentile cuts describe the upper tail of a visibility sample. The primary
cut is an absolute threshold fit earlier, on validation visibility only.
This module does not compare outcome metrics across cuts, and it does not
drop rows that fall outside a cut: every summary carries the excluded count.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

TOP_FRACTIONS: tuple[float, ...] = (0.10, 0.20, 0.25, 0.33, 0.50, 0.75, 1.0)
PRIMARY_QUANTILE = 0.67
TARGET_TAU = 1e-8


def top_fraction_name(fraction: float) -> str:
    if float(fraction) >= 1.0:
        return "all_nonzero"
    percent = int(round(float(fraction) * 100.0))
    return f"top_{percent:02d}"


def nonzero_mask(visibility: Any) -> np.ndarray:
    values = np.asarray(visibility, dtype=np.float64).reshape(-1)
    return np.isfinite(values) & (values > 0.0)


def percentile_cutoff(visibility: Any, top_fraction: float) -> dict[str, Any]:
    """Cutoff for the upper tail of the nonzero sample.

    ``top_fraction=0.33`` uses quantile 0.67, the historical top-third cut.
    ``top_fraction=1`` is every strictly positive value and has no upper cutoff.
    Ties at the cutoff stay inside the selected set.
    """
    values = np.asarray(visibility, dtype=np.float64).reshape(-1)
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1].")
    base = values[nonzero_mask(values)]
    if float(top_fraction) >= 1.0:
        return {
            "name": "all_nonzero",
            "top_fraction": 1.0,
            "quantile": None,
            "threshold": 0.0,
            "comparison": ">",
            "n_nonzero": int(base.size),
            "population": "nonzero",
        }
    if base.size == 0:
        return {
            "name": top_fraction_name(top_fraction),
            "top_fraction": float(top_fraction),
            "quantile": 1.0 - float(top_fraction),
            "threshold": None,
            "comparison": ">=",
            "n_nonzero": 0,
            "population": "nonzero",
        }
    quantile = 1.0 - float(top_fraction)
    if abs(float(top_fraction) - 0.33) < 1e-9:
        quantile = PRIMARY_QUANTILE
    return {
        "name": top_fraction_name(top_fraction),
        "top_fraction": float(top_fraction),
        "quantile": float(quantile),
        "threshold": float(np.quantile(base, quantile)),
        "comparison": ">=",
        "n_nonzero": int(base.size),
        "population": "nonzero",
    }


def select_visibility(visibility: Any, *, threshold: float | None, comparison: str) -> np.ndarray:
    """Boolean mask. Non-finite visibility is never selected."""
    values = np.asarray(visibility, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if threshold is None:
        return np.zeros(values.shape[0], dtype=bool)
    cut = float(threshold)
    if comparison == ">":
        return finite & (values > cut)
    if comparison == ">=":
        return finite & (values >= cut)
    raise ValueError(f"unsupported comparison {comparison!r}.")


def _finite(values: Sequence[Any]) -> np.ndarray:
    kept = []
    for value in values:
        if value is None or value == "":
            continue
        number = float(value)
        if np.isfinite(number):
            kept.append(number)
    return np.asarray(kept, dtype=np.float64)


def _median(values: Sequence[Any]) -> tuple[float | None, int]:
    array = _finite(values)
    if array.size == 0:
        return None, 0
    return float(np.median(array)), int(array.size)


def summarize_selected(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[bool],
    *,
    label: str,
    visibility_metric: str,
    threshold: float | None,
    comparison: str,
    family: str,
) -> dict[str, Any]:
    """Metrics on one cut. Rows outside the cut remain in ``n_excluded``."""
    mask = np.asarray(list(selected), dtype=bool)
    if mask.shape[0] != len(rows):
        raise ValueError("selection mask length does not match the rows.")
    chosen = [row for row, keep in zip(rows, mask) if keep]
    ratio_values = []
    cosine_values = []
    ranking_values = []
    ranking_ambiguous = 0
    state_values = []
    state_status = "not_available"
    for row in chosen:
        target = row.get("target_delta_l2")
        pred = row.get("pred_delta_l2")
        if target is not None and pred is not None and float(target) > TARGET_TAU and np.isfinite(float(target)) and np.isfinite(float(pred)):
            ratio_values.append(float(pred) / float(target))
        if row.get("cosine_status") == "ok" and row.get("cosine") is not None:
            cosine_values.append(float(row["cosine"]))
        status = row.get("counterfactual_ranking_status")
        if status in {"scored", "distance_tie"} and row.get("counterfactual_ranking_top1") is not None:
            ranking_values.append(float(row["counterfactual_ranking_top1"]))
        elif status == "observationally_ambiguous" or row.get("observationally_ambiguous") is True:
            ranking_ambiguous += 1
        if "state_action_ratio" in row:
            state_status = "measured"
            if row.get("state_action_ratio") is not None and np.isfinite(float(row["state_action_ratio"])):
                state_values.append(float(row["state_action_ratio"]))
    ratio_median, n_ratio = _median(ratio_values)
    cosine_median, n_cosine = _median(cosine_values)
    ranking = _finite(ranking_values)
    state_median, n_state = _median(state_values)
    rms_median, n_rms = _median([row.get("rms_pos") for row in chosen])
    raster_median, n_raster = _median([row.get("raster_mse") for row in chosen])
    rgb_median, n_rgb = _median([row.get("rgb_mse") for row in chosen])
    affected_median, n_affected = _median([row.get("fraction_affected_pixels") for row in chosen])
    target_median, n_target = _median(
        [row.get("target_delta_l2") for row in chosen if row.get("target_delta_l2") is not None and float(row["target_delta_l2"]) > TARGET_TAU]
    )
    pred_median, n_pred = _median([row.get("pred_delta_l2") for row in chosen])
    return {
        "family": family,
        "label": label,
        "visibility_metric": visibility_metric,
        "threshold": threshold,
        "comparison": comparison,
        "selected_by_outcome": False,
        "n_total": int(len(rows)),
        "n_selected": int(mask.sum()),
        "n_excluded": int(len(rows) - mask.sum()),
        "rms_pos_median": rms_median,
        "n_rms": n_rms,
        "raster_mse_median": raster_median,
        "n_raster": n_raster,
        "rgb_mse_median": rgb_median,
        "n_rgb": n_rgb,
        "fraction_affected_median": affected_median,
        "n_affected": n_affected,
        "target_delta_l2_median": target_median,
        "n_target_defined": n_target,
        "pred_delta_l2_median": pred_median,
        "n_pred_defined": n_pred,
        "response_ratio_median": ratio_median,
        "n_ratio_defined": n_ratio,
        "cosine_median": cosine_median,
        "n_cosine_defined": n_cosine,
        "n_cosine_undefined": int(len(chosen) - n_cosine),
        "ranking_top1": None if ranking.size == 0 else float(ranking.mean()),
        "n_ranking_scored": int(ranking.size),
        "n_ranking_ambiguous": int(ranking_ambiguous),
        "state_action_ratio_median": state_median,
        "n_state_action_ratio": n_state,
        "state_action_ratio_status": state_status,
    }


def equal_frequency_bins(visibility: Any, n_bins: int = 10) -> tuple[np.ndarray, list[float]]:
    """Bin labels 1..n_bins on finite values. Ties stay in the lower bin."""
    values = np.asarray(visibility, dtype=np.float64).reshape(-1)
    if int(n_bins) < 2:
        raise ValueError("n_bins must be >= 2.")
    finite = np.isfinite(values)
    labels = np.zeros(values.shape[0], dtype=np.int64)
    usable = values[finite]
    if usable.size == 0:
        return labels, []
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    edges = [float(np.quantile(usable, q)) for q in quantiles]
    assigned = np.ones(usable.size, dtype=np.int64)
    for edge in edges:
        assigned += (usable > edge).astype(np.int64)
    labels[finite] = assigned
    return labels, edges


def _spearman(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.size < 2:
        return {"spearman": None, "n": int(x.size), "status": "too_few_pairs"}
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"spearman": None, "n": int(x.size), "status": "undefined_nonfinite"}

    def _rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        ordered = values[order]
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and ordered[stop] == ordered[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    left = _rank(x)
    right = _rank(y)
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denom = float(np.sqrt(np.sum(centered_left ** 2) * np.sum(centered_right ** 2)))
    if denom == 0.0:
        return {"spearman": None, "n": int(x.size), "status": "undefined_zero_variance"}
    return {"spearman": float(np.sum(centered_left * centered_right) / denom), "n": int(x.size), "status": "ok"}


def spearman_visibility(visibility: Any, outcome: Any, *, nonzero_only: bool) -> dict[str, Any]:
    """Spearman on paired finite rows. Undefined cosine stays out and is counted."""
    left = np.asarray(visibility, dtype=np.float64).reshape(-1)
    right = np.asarray([np.nan if value is None else float(value) for value in outcome], dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("visibility and outcome must have the same length.")
    paired = np.isfinite(left) & np.isfinite(right)
    if nonzero_only:
        paired &= left > 0.0
    result = _spearman(left[paired], right[paired])
    result["n_total"] = int(left.size)
    result["n_excluded"] = int(left.size - paired.sum())
    result["nonzero_only"] = bool(nonzero_only)
    return result


def integrate_top_fraction_curve(fractions: Sequence[float], metrics: Sequence[float | None]) -> dict[str, Any]:
    """Trapezoid integral of a metric across top-fraction cuts.

    Missing metric values are not interpolated. The integral is null unless
    every fraction has a finite metric. ``mean_over_fractions`` divides the
    integral by the fraction span, so a constant curve integrates to itself.
    """
    grid = np.asarray(list(fractions), dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("the fraction curve needs at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("fractions must be strictly increasing.")
    values = []
    missing = []
    for fraction, metric in zip(grid.tolist(), metrics):
        if metric is None or not np.isfinite(float(metric)):
            missing.append(float(fraction))
            values.append(np.nan)
        else:
            values.append(float(metric))
    array = np.asarray(values, dtype=np.float64)
    if missing:
        return {
            "integral": None,
            "mean_over_fractions": None,
            "status": "incomplete",
            "n_points": int(grid.size),
            "missing_fractions": missing,
        }
    integral = float(np.trapezoid(array, grid))
    span = float(grid[-1] - grid[0])
    return {
        "integral": integral,
        "mean_over_fractions": float(integral / span),
        "status": "ok",
        "n_points": int(grid.size),
        "missing_fractions": [],
    }
