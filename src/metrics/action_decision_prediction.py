"""Predict finite-candidate physical regret from intervention probes.

The unit is one checkpoint on one validation episode. The outcome is the
physical endpoint regret of matched final-image candidate ranking. Probe
features come from the open-loop force-reversal contrast at the same horizon.
They do not include the regret, the chosen index, or the planner's candidate
costs.

Coefficients are fit on episodes 1440-1451 only. Episodes 1452-1463 are the
holdout. Episodes at or above 1620 are refused. A linear model on the
baselines is fit before any model that adds the intervention decomposition.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

TEST_EPISODE_MIN = 1620
FIT_EPISODES = tuple(range(1440, 1452))
HOLDOUT_EPISODES = tuple(range(1452, 1464))
PRIMARY_HORIZON = 10
PRIMARY_LIBRARY = "sequence"
PRIMARY_MODELS = ("vicreg_action", "mse_action")
BASELINE_FEATURES = (
    "validation_prediction_loss",
    "cosine",
    "effect_nve",
    "effect_rmse",
)
DIAGNOSTIC_FEATURES = (
    "log_ratio_distance",
    "orthogonal_energy",
    "parallel_gain",
)
MIN_INCREMENTAL_R2 = 0.05
MIN_ROWS = 200
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
NORM_TAU = 1e-8


def assert_development_episodes(episodes: Sequence[int]) -> None:
    """Refuse the opened high-excitation test and anything outside validation."""
    for episode in episodes:
        number = int(episode)
        if number >= TEST_EPISODE_MIN or number < 1440 or number > 1463:
            raise ValueError(f"episode {number} is outside the development planning block 1440-1463.")


def intervention_features(cosine: float, pred_norm: float, target_norm: float) -> dict[str, float] | None:
    """Derive the baseline effect error and the decomposition from one contrast.

    ``effect_nve`` and ``effect_rmse`` are the normalized and raw distances
    between the predicted and target displacements. The diagnostic triple is
    the log distance of the response ratio from one, the squared orthogonal
    residual in units of the target norm, and the parallel gain. All three are
    functions of the same contrast as cosine and the effect error.
    """
    cosine_value = float(cosine)
    pred = float(pred_norm)
    target = float(target_norm)
    if not np.isfinite(cosine_value) or not np.isfinite(pred) or not np.isfinite(target):
        return None
    if target <= NORM_TAU or pred <= NORM_TAU:
        return None
    cosine_value = float(np.clip(cosine_value, -1.0, 1.0))
    ratio = pred / target
    if ratio <= 0.0:
        return None
    effect_nve = float(ratio * ratio - 2.0 * cosine_value * ratio + 1.0)
    if effect_nve < 0.0 and effect_nve > -1e-10:
        effect_nve = 0.0
    if effect_nve < 0.0:
        return None
    orthogonal = float(ratio * ratio * max(0.0, 1.0 - cosine_value * cosine_value))
    return {
        "cosine": cosine_value,
        "response_ratio": float(ratio),
        "effect_nve": effect_nve,
        "effect_rmse": float(np.sqrt(effect_nve) * target),
        "log_ratio_distance": float(abs(np.log(ratio))),
        "orthogonal_energy": orthogonal,
        "parallel_gain": float(cosine_value * ratio),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("spearman inputs must be one-dimensional and the same length.")
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 3:
        return {"value": None, "n": int(left.size), "status": "too_few"}
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return {"value": None, "n": int(left.size), "status": "undefined_zero_variance"}
    ranked = np.corrcoef(average_ranks(left), average_ranks(right))[0, 1]
    return {"value": float(ranked), "n": int(left.size), "status": "ok"}


def r2_score(y: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    truth = np.asarray(list(y), dtype=np.float64)
    pred = np.asarray(list(predicted), dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("r2 inputs must be one-dimensional and the same length.")
    if truth.size < 2 or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        return {"value": None, "n": int(truth.size), "status": "invalid"}
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    if total == 0.0:
        return {"value": None, "n": int(truth.size), "status": "undefined_zero_variance"}
    residual = float(np.sum((truth - pred) ** 2))
    return {"value": float(1.0 - residual / total), "n": int(truth.size), "status": "ok"}


def _column(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _scale(train: Sequence[dict[str, Any]], columns: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Training-fold mean and scale. A constant column is marked unusable."""
    stats: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = _column(train, column)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} has a non-finite training value.")
        center = float(np.mean(values))
        spread = float(np.std(values))
        stats[column] = (center, spread)
    return stats


def _design(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    stats: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, list[str]]:
    used = [column for column in columns if stats[column][1] > 0.0]
    design = np.empty((len(rows), 1 + len(used)), dtype=np.float64)
    design[:, 0] = 1.0
    for index, column in enumerate(used):
        center, spread = stats[column]
        design[:, index + 1] = (_column(rows, column) - center) / spread
    return design, used


def fit_linear(
    train: Sequence[dict[str, Any]],
    columns: Sequence[str],
    target: str,
) -> dict[str, Any]:
    """Ordinary least squares with an intercept. Scales come from ``train`` only."""
    if len(train) < 4:
        raise ValueError("linear fit needs at least four rows.")
    outcome = _column(train, target)
    if not np.all(np.isfinite(outcome)):
        raise ValueError(f"{target} has a non-finite training value.")
    stats = _scale(train, columns)
    design, used = _design(train, columns, stats)
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return {
        "target": target,
        "stats": stats,
        "used": used,
        "coefficients": coefficients,
        "rank": int(rank),
        "n": len(train),
    }


def predict_linear(model: dict[str, Any], rows: Sequence[dict[str, Any]]) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(0, dtype=np.float64)
    design, used = _design(rows, model["used"], model["stats"])
    if used != model["used"]:
        raise RuntimeError("prediction columns do not match the fit.")
    return design @ model["coefficients"]


def incremental_prediction(
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    baseline: Sequence[str],
    full: Sequence[str],
    target: str,
) -> dict[str, Any]:
    """Fit both linear models on ``train`` and score both on ``test``."""
    if len(test) < 2:
        return {"status": "too_few_test_rows", "n_test": len(test)}
    baseline_model = fit_linear(train, baseline, target)
    full_model = fit_linear(train, full, target)
    baseline_pred = predict_linear(baseline_model, test)
    full_pred = predict_linear(full_model, test)
    truth = _column(test, target)
    baseline_fit = r2_score(truth, baseline_pred)
    full_fit = r2_score(truth, full_pred)
    incremental = None
    if baseline_fit["value"] is not None and full_fit["value"] is not None:
        incremental = float(full_fit["value"] - baseline_fit["value"])
    return {
        "status": "ok",
        "n_train": len(train),
        "n_test": len(test),
        "baseline_r2": baseline_fit,
        "full_r2": full_fit,
        "incremental_r2": incremental,
        "baseline_spearman": spearman(baseline_pred, truth),
        "full_spearman": spearman(full_pred, truth),
        "baseline_pred": baseline_pred,
        "full_pred": full_pred,
        "baseline_rank": baseline_model["rank"],
        "full_rank": full_model["rank"],
        "baseline_used": baseline_model["used"],
        "full_used": full_model["used"],
        "full_coefficients": {
            name: float(value)
            for name, value in zip(["intercept", *full_model["used"]], full_model["coefficients"])
        },
    }


def _episode_index(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    return grouped


def bootstrap_incremental_r2(
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    baseline: Sequence[str],
    full: Sequence[str],
    target: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Episode-cluster interval for holdout incremental R^2. The fit stays fixed.

    Each draw resamples holdout episodes with replacement. Rows that share an
    episode stay together, so seeds of one physical scene are not treated as
    independent.
    """
    point = incremental_prediction(train, test, baseline, full, target)
    if point["status"] != "ok" or point["incremental_r2"] is None:
        return {"low": None, "high": None, "status": point["status"], "n_draws": 0}
    grouped = _episode_index(test)
    episodes = np.asarray(sorted(grouped), dtype=np.int64)
    if episodes.size < 4:
        return {"low": None, "high": None, "status": "too_few_episodes", "n_draws": 0}
    base_model = fit_linear(train, baseline, target)
    full_model = fit_linear(train, full, target)
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(draws)):
        chosen = rng.choice(episodes, size=episodes.size, replace=True)
        sampled: list[dict[str, Any]] = []
        for episode in chosen.tolist():
            sampled.extend(grouped[int(episode)])
        truth = _column(sampled, target)
        base_r2 = r2_score(truth, predict_linear(base_model, sampled))
        full_r2 = r2_score(truth, predict_linear(full_model, sampled))
        if base_r2["value"] is None or full_r2["value"] is None:
            continue
        samples.append(float(full_r2["value"] - base_r2["value"]))
    if len(samples) < int(draws) * 0.8:
        return {"low": None, "high": None, "status": "unstable_bootstrap", "n_draws": len(samples)}
    low, high = np.percentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    return {
        "low": float(low),
        "high": float(high),
        "status": "ok",
        "n_draws": len(samples),
        "point": point["incremental_r2"],
    }


def linear_redundancy(
    train: Sequence[dict[str, Any]],
    diagnostic: str,
    baseline: Sequence[str],
) -> dict[str, Any]:
    """How much of a diagnostic the baselines already span, on the fit block."""
    model = fit_linear(train, baseline, diagnostic)
    predicted = predict_linear(model, train)
    score = r2_score(_column(train, diagnostic), predicted)
    return {"feature": diagnostic, "train_r2_from_baselines": score["value"], "status": score["status"]}


def _finite_check(name: str, block: dict[str, Any] | None) -> list[str]:
    if block is None or block.get("status") != "ok" or block.get("incremental_r2") is None:
        return [f"{name} did not return a finite incremental R^2"]
    reasons = []
    incremental = float(block["incremental_r2"])
    full = block.get("full_r2") or {}
    if incremental < MIN_INCREMENTAL_R2:
        reasons.append(f"{name} incremental R^2 is {incremental:.4f}, below {MIN_INCREMENTAL_R2:.2f}")
    if full.get("value") is None or float(full["value"]) < MIN_INCREMENTAL_R2:
        reasons.append(f"{name} full-model R^2 is below {MIN_INCREMENTAL_R2:.2f}")
    ci_low = block.get("ci_low")
    if ci_low is None or float(ci_low) <= 0.0:
        reasons.append(f"{name} bootstrap interval does not stay above zero")
    return reasons


def decide(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen rule. Controls veto a primary pass. They cannot create one."""
    reasons: list[str] = []
    if int(report.get("n_rows", 0)) < MIN_ROWS:
        reasons.append(f"fewer than {MIN_ROWS} primary rows")
    reasons.extend(_finite_check("primary", report.get("primary")))
    if reasons:
        return {"decision": "no-go", "reasons": reasons, "primary_passed": False}
    for name in ("within_vicreg_action", "within_mse_action", "difficulty", "normalized", "reverse"):
        reasons.extend(_finite_check(name, report.get(name)))
    seed_min = report.get("leave_one_seed_min_incremental_r2")
    if seed_min is None or float(seed_min) < 0.0:
        reasons.append("holding out one seed from the evaluation makes the incremental R^2 negative")
    return {
        "decision": "no-go" if reasons else "go",
        "reasons": reasons,
        "primary_passed": True,
    }
