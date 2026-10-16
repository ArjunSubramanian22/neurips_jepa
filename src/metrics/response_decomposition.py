"""Decompose an intervention response into direction, scale, and tracking.

``delta_pred`` and ``delta_target`` are the two-branch latent displacements
for one shared start state. Cosine is undefined when either displacement
has norm at or below ``norm_tau``. The magnitude ratio and the normalized
vector error stay numeric for a finite target above that threshold.

The median ratio can sit near one while the normalized error is large,
because an orthogonal residual does not change the ratio. Ranking two
conditions by closeness of the median ratio to one can therefore disagree
with ranking them by normalized error.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NORM_TAU = 1e-8
EPSILON = 1e-12


def _as_batch(delta: Any, name: str) -> np.ndarray:
    array = np.asarray(delta, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, D) or (D,).")
    return array


def decompose_response(
    delta_pred: Any,
    delta_target: Any,
    *,
    norm_tau: float = NORM_TAU,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Score paired intervention displacements.

    ``parallel_gain`` is the coefficient ``alpha`` in
    ``delta_pred = alpha * delta_target + residual``, with the residual
    orthogonal to ``delta_target``. It is undefined when the target norm
    is at or below ``norm_tau``.
    """
    pred = _as_batch(delta_pred, "delta_pred")
    target = _as_batch(delta_target, "delta_target")
    if pred.shape != target.shape:
        raise ValueError(f"delta shapes differ: {pred.shape} vs {target.shape}.")
    if float(norm_tau) < 0.0 or float(epsilon) <= 0.0:
        raise ValueError("norm_tau must be >= 0 and epsilon must be > 0.")
    finite = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(target), axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    dot = np.sum(pred * target, axis=1)
    target_ok = finite & (target_norm > float(norm_tau))
    cosine_ok = target_ok & (pred_norm > float(norm_tau))

    cosine = np.full(pred.shape[0], np.nan, dtype=np.float64)
    cosine[cosine_ok] = np.clip(dot[cosine_ok] / (pred_norm[cosine_ok] * target_norm[cosine_ok]), -1.0, 1.0)
    ratio = np.full(pred.shape[0], np.nan, dtype=np.float64)
    ratio[target_ok] = pred_norm[target_ok] / target_norm[target_ok]
    gain = np.full(pred.shape[0], np.nan, dtype=np.float64)
    gain[target_ok] = dot[target_ok] / np.square(target_norm[target_ok])
    nve = np.full(pred.shape[0], np.nan, dtype=np.float64)
    nve[target_ok] = np.sum(np.square(pred[target_ok] - target[target_ok]), axis=1) / np.square(target_norm[target_ok])
    orth = np.full(pred.shape[0], np.nan, dtype=np.float64)
    if np.any(target_ok):
        parallel = gain[target_ok, None] * target[target_ok]
        orth[target_ok] = np.linalg.norm(pred[target_ok] - parallel, axis=1)

    cosine_status = np.full(pred.shape[0], "undefined_zero_vector", dtype=object)
    cosine_status[~finite] = "undefined_nonfinite"
    cosine_status[cosine_ok] = "ok"
    return {
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "cosine": cosine,
        "cosine_status": cosine_status,
        "response_ratio": ratio,
        "parallel_gain": gain,
        "orthogonal_norm": orth,
        "normalized_vector_error": nve,
        "norm_tau": float(norm_tau),
    }


def _finite(values: Sequence[float | None]) -> np.ndarray:
    kept = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            kept.append(number)
    return np.asarray(kept, dtype=np.float64)


def median_or_none(values: Sequence[float | None]) -> float | None:
    taken = _finite(values)
    if taken.size == 0:
        return None
    return float(np.median(taken))


def log_ratio_distance(ratio: float) -> float:
    """How far a positive response ratio is from one, in log space."""
    if not np.isfinite(ratio) or float(ratio) <= 0.0:
        raise ValueError("response ratio must be finite and positive.")
    return float(abs(np.log(float(ratio))))


def ratio_rank_disagrees_with_error(
    ratio_a: float,
    error_a: float,
    ratio_b: float,
    error_b: float,
) -> dict[str, Any]:
    """Whether closeness of the median ratio to one reverses the error ranking.

    ``a`` is preferred by the ratio rule when ``|log ratio_a|`` is smaller.
    ``a`` is preferred by the error rule when ``error_a`` is smaller.
    A disagreement means those two preferences differ.
    """
    distance_a = log_ratio_distance(ratio_a)
    distance_b = log_ratio_distance(ratio_b)
    if not np.isfinite(error_a) or not np.isfinite(error_b):
        raise ValueError("errors must be finite.")
    ratio_tie = distance_a == distance_b
    error_tie = float(error_a) == float(error_b)
    ratio_prefers_a = distance_a < distance_b
    error_prefers_a = float(error_a) < float(error_b)
    return {
        "ratio_distance_a": distance_a,
        "ratio_distance_b": distance_b,
        "ratio_prefers": "tie" if ratio_tie else ("a" if ratio_prefers_a else "b"),
        "error_prefers": "tie" if error_tie else ("a" if error_prefers_a else "b"),
        "disagrees": (not ratio_tie) and (not error_tie) and (ratio_prefers_a != error_prefers_a),
    }
