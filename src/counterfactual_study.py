"""Counterfactual metrics for the high-excitation study.

Branch times are aggregated inside an episode before any bootstrap.
Undefined cosine stays null. Episode 1620 and above of the high-excitation
dump are rejected by :func:`assert_high_validation_episodes`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

NORM_TAU = 1e-8
PHYS_TAU = 1e-5
SUCCESS_COSINE = 0.5
SUCCESS_RATIO = 0.2
HIGH_TEST_START = 1620
HIGH_VAL_START = 1440


def assert_high_validation_episodes(episodes) -> list[int]:
    """Reject anything outside high-excitation validation 1440-1619."""
    ids = [int(episode) for episode in episodes]
    if not ids:
        raise ValueError("episode list is empty.")
    if any(episode < HIGH_VAL_START or episode >= HIGH_TEST_START for episode in ids):
        raise ValueError("high-excitation evaluation must stay inside episodes 1440-1619.")
    return ids


def pair_metrics(delta_pred: np.ndarray, delta_target: np.ndarray, tau: float = NORM_TAU) -> dict:
    """Cosine, response ratio, and directional projection of one contrast."""
    pred = np.asarray(delta_pred, dtype=np.float64).reshape(-1)
    target = np.asarray(delta_target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError("contrast vectors must share a shape.")
    pred_norm = float(np.linalg.norm(pred))
    target_norm = float(np.linalg.norm(target))
    cosine = None
    status = "undefined_zero_vector"
    ratio = None
    projection = None
    if np.isfinite(pred_norm) and np.isfinite(target_norm) and target_norm > tau:
        ratio = pred_norm / target_norm
        projection = float(np.dot(pred, target) / (target_norm ** 2))
    if (
        np.isfinite(pred_norm)
        and np.isfinite(target_norm)
        and pred_norm > tau
        and target_norm > tau
    ):
        cosine = float(np.clip(np.dot(pred, target) / (pred_norm * target_norm), -1.0, 1.0))
        status = "ok"
    return {
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "cosine": cosine,
        "cosine_status": status,
        "response_ratio": None if ratio is None else float(ratio),
        "projection": projection,
    }


def classify_bottleneck(row: dict) -> str:
    """Predeclared validation labels. Thresholds are not fit to a model score."""
    if float(row["rms_pos"]) <= PHYS_TAU:
        return "PHYSICAL_FAILURE"
    if float(row["identical_raster"]) == 1.0:
        return "OBSERVATION_BOTTLENECK"
    if float(row["target_norm"]) <= NORM_TAU:
        return "REPRESENTATION_BOTTLENECK"
    cosine = row.get("cosine")
    ratio = row.get("response_ratio")
    aligned = (
        cosine is not None
        and float(cosine) >= SUCCESS_COSINE
        and ratio is not None
        and float(ratio) >= SUCCESS_RATIO
    )
    if aligned:
        return "SUCCESSFUL_COUNTERFACTUAL_TRACKING"
    return "PREDICTOR_BOTTLENECK"


def _finite(values) -> np.ndarray:
    kept = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            kept.append(number)
    return np.asarray(kept, dtype=np.float64)


def episode_aggregate(rows: list[dict], keys: tuple[str, ...]) -> dict:
    """Median of branch-time values inside one episode. Cosine skips nulls."""
    out = {"n_branch": len(rows), "n_defined_cosine": 0}
    for key in keys:
        if key == "cosine":
            defined = _finite(row.get("cosine") for row in rows if row.get("cosine_status") == "ok")
            out["n_defined_cosine"] = int(defined.size)
            out["cosine"] = None if defined.size == 0 else float(np.median(defined))
        else:
            values = _finite(row.get(key) for row in rows)
            out[key] = None if values.size == 0 else float(np.median(values))
    return out


def bootstrap_median(episode_values: list, rng: np.random.Generator, n_resamples: int = 10000):
    """Percentile interval of the median, resampling episodes."""
    values = _finite(episode_values)
    if values.size == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_episodes": 0, "status": "undefined"}
    estimate = float(np.median(values))
    if values.size == 1:
        return {
            "estimate": estimate,
            "ci_low": estimate,
            "ci_high": estimate,
            "n_episodes": 1,
            "status": "single_episode",
        }
    draws = rng.integers(0, values.size, size=(n_resamples, values.size))
    medians = np.median(values[draws], axis=1)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(medians, 0.025)),
        "ci_high": float(np.quantile(medians, 0.975)),
        "n_episodes": int(values.size),
        "status": "episode_bootstrap_median",
    }


def group_episode_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """One row per episode after aggregating that episode's branch times."""
    buckets: dict = defaultdict(list)
    for row in rows:
        buckets[int(row["episode"])].append(row)
    aggregated = []
    for episode, group in sorted(buckets.items()):
        item = episode_aggregate(group, keys)
        item["episode"] = episode
        aggregated.append(item)
    return aggregated


def average_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, starting at 1. Ties share the mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def rank_correlation(left, right) -> dict:
    """Average-rank Pearson correlation. Null pairs are dropped."""
    pairs = []
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        if np.isfinite(float(a)) and np.isfinite(float(b)):
            pairs.append((float(a), float(b)))
    if len(pairs) < 3:
        return {"rho": None, "n": len(pairs), "status": "too_few_pairs"}
    xs = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    ys = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    rx = average_rank(xs)
    ry = average_rank(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if denom <= 0:
        return {"rho": None, "n": len(pairs), "status": "undefined_constant"}
    return {"rho": float(np.dot(rx, ry) / denom), "n": len(pairs), "status": "measured"}


def visibility_threshold(raster_mse: np.ndarray) -> dict:
    """Top third of nonzero raster-MSE forks. One threshold for every model."""
    values = np.asarray(raster_mse, dtype=np.float64)
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return {"top_third_mse": None, "n_nonzero": 0}
    return {
        "top_third_mse": float(np.quantile(nonzero, 2.0 / 3.0)),
        "n_nonzero": int(nonzero.size),
    }
