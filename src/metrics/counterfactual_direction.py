"""Directional counterfactual metrics for a paired action intervention.

``delta_target`` is the target-encoder move between the two futures.
``delta_pred`` is the predictor move from the same latent under the two actions.
A cosine is stored only when both vectors clear ``norm_tau``. Near-zero cases
keep a null cosine and a status; they are not filled with 0.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.stats import episode_bootstrap_mean

NORM_TAU = 1e-8
EPSILON = 1e-12
HIGH_VISIBILITY_AFFECTED = 0.05

CONTRAST_NEAR_ZERO = "target_near_zero"
CONTRAST_INVISIBLE = "target_invisible"
CONTRAST_VALID = "valid_directional_contrast"
CONTRAST_NONFINITE = "nonfinite"


def _as_batch(delta: Any, name: str) -> np.ndarray:
    array = np.asarray(delta, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, D) or (D,).")
    return array


def directional_contrast(
    delta_pred: Any,
    delta_target: Any,
    *,
    fraction_affected: Any | None = None,
    norm_tau: float = NORM_TAU,
    epsilon: float = EPSILON,
    invisible_affected_max: float = 0.0,
) -> dict[str, Any]:
    """Score one or many paired interventions.

    Normalized vector error and the magnitude ratio stay defined when the
    target norm is small because their denominators add ``epsilon``. Cosine,
    angular error, and signed projection do not: a near-zero vector has no
    direction, so those fields are null.
    """
    pred = _as_batch(delta_pred, "delta_pred")
    target = _as_batch(delta_target, "delta_target")
    if pred.shape != target.shape:
        raise ValueError(f"delta shapes differ: {pred.shape} vs {target.shape}.")
    if float(norm_tau) < 0.0 or float(epsilon) <= 0.0:
        raise ValueError("norm_tau must be >= 0 and epsilon must be > 0.")
    n_rows = int(pred.shape[0])
    if fraction_affected is None:
        affected = np.ones(n_rows, dtype=np.float64)
    else:
        affected = np.asarray(fraction_affected, dtype=np.float64).reshape(-1)
        if affected.shape != (n_rows,):
            raise ValueError("fraction_affected must have one value per row.")
    finite = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(target), axis=1) & np.isfinite(affected)
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    dot = np.sum(pred * target, axis=1)
    near_target = finite & (target_norm <= float(norm_tau))
    near_pred = finite & (pred_norm <= float(norm_tau))
    invisible = finite & (affected <= float(invisible_affected_max))
    status = np.full(n_rows, CONTRAST_VALID, dtype=object)
    status[~finite] = CONTRAST_NONFINITE
    status[near_target] = CONTRAST_NEAR_ZERO
    status[finite & ~near_target & invisible] = CONTRAST_INVISIBLE

    cosine_ok = finite & ~near_target & ~near_pred
    cosine = np.full(n_rows, np.nan, dtype=np.float64)
    cosine[cosine_ok] = np.clip(dot[cosine_ok] / (pred_norm[cosine_ok] * target_norm[cosine_ok]), -1.0, 1.0)
    cosine_status = np.full(n_rows, "undefined_zero_vector", dtype=object)
    cosine_status[~finite] = "undefined_nonfinite"
    cosine_status[cosine_ok] = "ok"

    vector_error = np.full(n_rows, np.nan, dtype=np.float64)
    magnitude_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    vector_error[finite] = np.sum((pred[finite] - target[finite]) ** 2, axis=1) / (target_norm[finite] ** 2 + float(epsilon))
    magnitude_ratio[finite] = pred_norm[finite] / (target_norm[finite] + float(epsilon))

    angular = np.full(n_rows, np.nan, dtype=np.float64)
    angular[cosine_ok] = np.arccos(cosine[cosine_ok])
    projection = np.full(n_rows, np.nan, dtype=np.float64)
    projection_ok = finite & ~near_target
    projection[projection_ok] = dot[projection_ok] / target_norm[projection_ok]
    projection_status = np.full(n_rows, "undefined_zero_vector", dtype=object)
    projection_status[~finite] = "undefined_nonfinite"
    projection_status[projection_ok] = "ok"

    return {
        "n": n_rows,
        "target_norm": target_norm,
        "pred_norm": pred_norm,
        "target_near_zero": near_target,
        "target_invisible": invisible & finite,
        "contrast_status": status,
        "cosine": cosine,
        "cosine_status": cosine_status,
        "normalized_vector_error": vector_error,
        "magnitude_ratio": magnitude_ratio,
        "angular_error_rad": angular,
        "angular_status": cosine_status.copy(),
        "signed_projection": projection,
        "projection_status": projection_status,
        "norm_tau": float(norm_tau),
        "epsilon": float(epsilon),
    }


def _defined(values: np.ndarray, statuses: np.ndarray) -> np.ndarray:
    ok = np.asarray(statuses) == "ok"
    numeric = np.asarray(values, dtype=np.float64)
    return ok & np.isfinite(numeric)


def summarize_metric(
    values: Sequence[float | None],
    statuses: Sequence[str],
    contrast_status: Sequence[str],
) -> dict[str, Any]:
    """Aggregate without dropping rows. Undefined values stay out of the mean and are counted."""
    numeric = np.array([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    status_arr = np.asarray(list(statuses), dtype=object)
    contrast = np.asarray(list(contrast_status), dtype=object)
    if not (numeric.shape[0] == status_arr.shape[0] == contrast.shape[0]):
        raise ValueError("metric summaries require equal-length columns.")
    defined = _defined(numeric, status_arr)
    taken = numeric[defined]
    counts = {name: int(np.sum(contrast == name)) for name in (
        CONTRAST_VALID,
        CONTRAST_INVISIBLE,
        CONTRAST_NEAR_ZERO,
        CONTRAST_NONFINITE,
    )}
    summary: dict[str, Any] = {
        "n_total": int(numeric.shape[0]),
        "n_defined": int(defined.sum()),
        "n_undefined": int((~defined).sum()),
        "contrast_counts": counts,
        "median": None if taken.size == 0 else float(np.median(taken)),
        "mean_defined": None if taken.size == 0 else float(taken.mean()),
    }
    for percentile in (10, 25, 50, 75, 90):
        summary[f"p{percentile}"] = None if taken.size == 0 else float(np.percentile(taken, percentile))
    return summary


def assign_quintiles(values: Sequence[float]) -> tuple[np.ndarray, list[float]]:
    """Quintile labels 1..5. Edges are the empirical 20/40/60/80 percentiles of this sample."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("quintiles need a finite 1-d sample.")
    edges = [float(np.quantile(array, q)) for q in (0.2, 0.4, 0.6, 0.8)]
    labels = np.ones(array.shape[0], dtype=np.int64)
    for edge in edges:
        labels += (array > edge).astype(np.int64)
    return labels, edges


def bootstrap_valid_cosine(
    episode_ids: Sequence[int],
    cosine: Sequence[float | None],
    cosine_status: Sequence[str],
    contrast_status: Sequence[str],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Bootstrap the mean of per-episode mean valid cosines.

    Episodes with no valid directional cosine are counted and left out of the
    resample. They are not imputed.
    """
    episodes = np.asarray(episode_ids)
    values = np.array([np.nan if value is None else float(value) for value in cosine], dtype=np.float64)
    ok = (np.asarray(list(cosine_status)) == "ok") & (np.asarray(list(contrast_status)) == CONTRAST_VALID) & np.isfinite(values)
    unique = np.unique(episodes)
    means = []
    excluded = []
    for episode in unique.tolist():
        taken = values[(episodes == episode) & ok]
        if taken.size == 0:
            excluded.append(int(episode))
        else:
            means.append(float(taken.mean()))
    result: dict[str, Any] = {
        "n_episodes": int(unique.size),
        "n_episodes_with_valid": len(means),
        "n_episodes_without_valid": len(excluded),
        "excluded_episode_ids": excluded,
    }
    if len(means) < 1:
        result["bootstrap"] = None
        return result
    result["bootstrap"] = episode_bootstrap_mean(np.asarray(means), n_resamples, rng)
    return result
