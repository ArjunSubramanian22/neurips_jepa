"""Do simple audit metrics predict closed-loop planning better than ordinary ones?

The unit is a trained checkpoint. Audit metrics come from legacy validation.
Planning outcomes come from the report seeds of the predeclared CEM benchmark.
Neither source is the cvpr_v1 test split. Univariate fits only.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HORIZON_QUERIES = ROOT / "experiments" / "runs" / "horizon_audit_v1" / "queries.csv"
HELDOUT_DIR = ROOT / "experiments" / "runs" / "heldout_reproduction_v1" / "evaluation"
PLANNING_EPISODES = ROOT / "experiments" / "runs" / "planning_benchmark_v1" / "episodes.json"
PLANNING_PROTOCOL = ROOT / "configs" / "experiments" / "planning_benchmark_v1.yaml"
HEADLINE_CSV = ROOT / "analysis" / "metric_planning_headline.csv"
HEADLINE_JSON = ROOT / "analysis" / "metric_planning_headline.json"
MODEL_CSV = ROOT / "analysis" / "metric_planning_models.csv"
FIGURE = ROOT / "figures" / "metric_planning" / "predictors_vs_planning.png"

# Open-loop persistent reversal at horizon 10. Chosen because it is the
# existing multi-step audit slice, not because of a planning correlation.
CF_PROTOCOL = "persistent_alternative"
CF_HORIZON = "10"
CF_MODE = "open_loop"
CF_ENVIRONMENT = "mpm_legacy"
NORM_TAU = 1e-8
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 0

# Higher values are worse planning. A useful error metric should correlate
# positively with them. A useful alignment metric should correlate negatively.
WORSE_WHEN_LARGER = {
    "one_step_latent_mse": True,
    "multi_step_latent_mse": True,
    "variance_normalized_mse": True,
    "effective_rank": None,
    "magnitude_response_ratio": None,
    "directional_cosine": False,
    "normalized_vector_error": True,
    "retrieval_accuracy": False,
    "multi_step_counterfactual_fidelity": True,
}
CONVENTIONAL = (
    "one_step_latent_mse",
    "multi_step_latent_mse",
    "variance_normalized_mse",
    "effective_rank",
)
COUNTERFACTUAL = (
    "magnitude_response_ratio",
    "directional_cosine",
    "normalized_vector_error",
    "retrieval_accuracy",
    "multi_step_counterfactual_fidelity",
)
PLOT_METRICS = (
    "one_step_latent_mse",
    "effective_rank",
    "directional_cosine",
    "retrieval_accuracy",
    "normalized_vector_error",
)


def family_and_seed(name: str) -> tuple[str, int]:
    family, seed = name.rsplit("_s", 1)
    return family, int(seed)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks. Tied values share the average of their ranks."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop) + 1.0
        ranks[order[start : stop + 1]] = rank
        start = stop + 1
    return ranks


def _finite_pair(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray] | None:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("x and y must be one-dimensional and the same length.")
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 3:
        return None
    return left[keep], right[keep]


def pearson(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    pair = _finite_pair(x, y)
    if pair is None:
        return {"value": None, "n": 0, "status": "too_few"}
    left, right = pair
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return {"value": None, "n": int(left.size), "status": "undefined_zero_variance"}
    value = float(np.corrcoef(left, right)[0, 1])
    return {"value": value, "n": int(left.size), "status": "ok"}


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    pair = _finite_pair(x, y)
    if pair is None:
        return {"value": None, "n": 0, "status": "too_few"}
    left, right = pair
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return {"value": None, "n": int(left.size), "status": "undefined_zero_variance"}
    ranked = pearson(average_ranks(left), average_ranks(right))
    ranked["n"] = int(left.size)
    return ranked


def bootstrap_spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    point = spearman(x, y)
    if point["value"] is None:
        return {"low": None, "high": None, "status": point["status"], "n": point["n"]}
    pair = _finite_pair(x, y)
    assert pair is not None
    left, right = pair
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(draws)):
        index = rng.integers(0, left.size, size=left.size)
        draw = spearman(left[index], right[index])
        if draw["value"] is not None:
            samples.append(draw["value"])
    if len(samples) < int(draws) * 0.8:
        return {"low": None, "high": None, "status": "unstable_bootstrap", "n": point["n"]}
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "status": "ok", "n": point["n"]}


def loo_r2(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Leave-one-checkpoint-out R^2 of a univariate line. Negative means worse than the mean."""
    pair = _finite_pair(x, y)
    if pair is None:
        return {"value": None, "n": 0, "status": "too_few"}
    left, right = pair
    if left.size < 4 or float(np.std(right)) == 0.0:
        return {"value": None, "n": int(left.size), "status": "undefined_zero_variance" if float(np.std(right)) == 0.0 else "too_few"}
    predicted = np.empty(left.size, dtype=np.float64)
    for index in range(left.size):
        train = np.ones(left.size, dtype=bool)
        train[index] = False
        x_train, y_train = left[train], right[train]
        if float(np.std(x_train)) == 0.0:
            predicted[index] = float(np.mean(y_train))
            continue
        slope = float(np.cov(x_train, y_train, ddof=1)[0, 1] / np.var(x_train, ddof=1))
        intercept = float(np.mean(y_train) - slope * np.mean(x_train))
        predicted[index] = intercept + slope * left[index]
    residual = float(np.sum((right - predicted) ** 2))
    total = float(np.sum((right - np.mean(right)) ** 2))
    return {"value": 1.0 - residual / total, "n": int(left.size), "status": "ok"}


def _column(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) if row.get(key) is not None else np.nan for row in rows]


def leave_one_label_out(
    rows: Sequence[dict[str, Any]],
    metric: str,
    outcome: str,
    label: str,
) -> dict[str, Any]:
    groups = sorted({str(row[label]) for row in rows})
    values = []
    for group in groups:
        kept = [row for row in rows if str(row[label]) != group]
        result = spearman(_column(kept, metric), _column(kept, outcome))
        values.append({"left_out": group, "spearman": result["value"], "n": result["n"], "status": result["status"]})
    finite = [item["spearman"] for item in values if item["spearman"] is not None]
    return {
        "folds": values,
        "min": None if not finite else float(min(finite)),
        "max": None if not finite else float(max(finite)),
    }


def _median(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.median(array))


def _mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.mean(array))


def load_conventional(names: Sequence[str]) -> dict[str, dict[str, float | None]]:
    loaded: dict[str, dict[str, float | None]] = {}
    for name in names:
        path = HELDOUT_DIR / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluation_scope", "").find("1800") < 0:
            raise ValueError(f"{name} is not the legacy held-out evaluation.")
        mse = payload["rollout_mse"]
        vn = payload["vn_mse"]
        loaded[name] = {
            "one_step_latent_mse": float(mse[0]),
            "multi_step_latent_mse": float(mse[9]),
            "variance_normalized_mse": float(vn[9]),
            "effective_rank": float(payload["effective_rank"]["pred"]["participation_ratio"]),
        }
    return loaded


def load_counterfactual(names: Sequence[str]) -> dict[str, dict[str, float | None]]:
    buckets: dict[str, dict[str, list[float]]] = {
        name: {"ratio": [], "cosine": [], "vector": [], "retrieval": [], "fidelity": []} for name in names
    }
    with HORIZON_QUERIES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in buckets:
                continue
            if row["environment"] != CF_ENVIRONMENT or row["rollout_mode"] != CF_MODE:
                continue
            if row["protocol"] != CF_PROTOCOL or row["horizon"] != CF_HORIZON:
                continue
            if row["is_test_set"] != "False":
                raise ValueError("horizon audit row is marked as a test set.")
            slot = buckets[row["model"]]
            slot["fidelity"].append(float(row["rollout_mse_counterfactual"]))
            if row["contrast_status"] != "valid_directional_contrast":
                continue
            target = float(row["target_delta_l2"])
            if target <= NORM_TAU:
                continue
            slot["ratio"].append(float(row["pred_delta_l2"]) / target)
            slot["vector"].append(float(row["normalized_vector_error"]))
            if row["cosine_status"] == "ok" and row["cosine"] != "":
                slot["cosine"].append(float(row["cosine"]))
            if row["counterfactual_ranking_status"] == "scored" and row["counterfactual_ranking_top1"] != "":
                slot["retrieval"].append(float(row["counterfactual_ranking_top1"]))
    return {
        name: {
            "magnitude_response_ratio": _median(slot["ratio"]),
            "directional_cosine": _median(slot["cosine"]),
            "normalized_vector_error": _median(slot["vector"]),
            "retrieval_accuracy": _mean(slot["retrieval"]),
            "multi_step_counterfactual_fidelity": _median(slot["fidelity"]),
            "n_valid_direction": len(slot["ratio"]),
            "n_cosine": len(slot["cosine"]),
            "n_retrieval": len(slot["retrieval"]),
        }
        for name, slot in buckets.items()
    }


def load_planning(names: Sequence[str], report_seeds: Sequence[int]) -> dict[str, dict[str, float | None]]:
    episodes = json.loads(PLANNING_EPISODES.read_text(encoding="utf-8"))
    seeds = {int(seed) for seed in report_seeds}
    out: dict[str, dict[str, float | None]] = {}
    for name in names:
        rows = [row for row in episodes if row["controller"] == name and int(row["seed"]) in seeds and row["scored"]]
        if not rows:
            raise ValueError(f"no scored planning rows for {name}.")
        success = [1.0 if row["success"] else 0.0 for row in rows]
        out[name] = {
            "mpc_success_rate": float(np.mean(success)),
            "final_goal_distance": float(np.mean([row["final_distance"] for row in rows])),
            "cumulative_cost": float(np.mean([row["cumulative_cost"] for row in rows])),
            "n_scored_planning": len(rows),
        }
    return out


def assemble_rows(names: Sequence[str], report_seeds: Sequence[int]) -> list[dict[str, Any]]:
    conventional = load_conventional(names)
    counterfactual = load_counterfactual(names)
    planning = load_planning(names, report_seeds)
    rows = []
    for name in names:
        family, seed = family_and_seed(name)
        row: dict[str, Any] = {"model": name, "family": family, "training_seed": seed}
        row.update(conventional[name])
        row.update({key: counterfactual[name][key] for key in COUNTERFACTUAL})
        row.update(planning[name])
        rows.append(row)
    return rows


def _sign_status(metric: str, correlation: float | None) -> str:
    expected = WORSE_WHEN_LARGER[metric]
    if correlation is None or expected is None:
        return "no_predeclared_direction" if expected is None else "undefined"
    agrees = (correlation > 0.0) if expected else (correlation < 0.0)
    return "matches_predeclared_direction" if agrees else "opposite_predeclared_direction"


def evaluate_metric(rows: Sequence[dict[str, Any]], metric: str, outcome: str) -> dict[str, Any]:
    x = _column(rows, metric)
    y = _column(rows, outcome)
    rho = spearman(x, y)
    linear = pearson(x, y)
    interval = bootstrap_spearman_ci(x, y)
    held_seed = leave_one_label_out(rows, metric, outcome, "training_seed")
    held_family = leave_one_label_out(rows, metric, outcome, "family")
    fit = loo_r2(x, y)
    supported = bool(
        interval["low"] is not None
        and interval["high"] is not None
        and interval["low"] * interval["high"] > 0.0
        and _sign_status(metric, rho["value"]) == "matches_predeclared_direction"
    )
    return {
        "metric": metric,
        "group": "conventional" if metric in CONVENTIONAL else "counterfactual",
        "outcome": outcome,
        "n": rho["n"],
        "spearman": rho["value"],
        "spearman_status": rho["status"],
        "spearman_ci_low": interval["low"],
        "spearman_ci_high": interval["high"],
        "ci_status": interval["status"],
        "pearson": linear["value"],
        "pearson_status": linear["status"],
        "loo_r2": fit["value"],
        "loo_r2_status": fit["status"],
        "direction": _sign_status(metric, rho["value"]),
        "ci_excludes_zero_and_sign_matches": supported,
        "leave_one_training_seed_out_min": held_seed["min"],
        "leave_one_training_seed_out_max": held_seed["max"],
        "leave_one_training_seed_out": held_seed["folds"],
        "leave_one_family_out_min": held_family["min"],
        "leave_one_family_out_max": held_family["max"],
        "leave_one_family_out": held_family["folds"],
    }


def headline_sentence(distance_rows: Sequence[dict[str, Any]], success_status: str) -> str:
    if success_status == "undefined_zero_variance":
        text = "MPC success is the same for every checkpoint, so it has no correlation with any audit metric. "
    else:
        text = f"MPC success correlations are {success_status}. "
    excluding = [
        row
        for row in distance_rows
        if row.get("spearman_ci_low") is not None
        and row.get("spearman_ci_high") is not None
        and float(row["spearman_ci_low"]) * float(row["spearman_ci_high"]) > 0.0
    ]
    if not excluding:
        text += "No Spearman interval against final goal distance excludes zero."
    else:
        described = []
        for row in excluding:
            sign = "positive" if float(row["spearman"]) > 0.0 else "negative"
            described.append(f"{row['metric']} ({sign}, {float(row['spearman']):.2f})")
        text += "Spearman intervals against final goal distance that exclude zero: " + ", ".join(described) + ". "
    if not any(row.get("ci_excludes_zero_and_sign_matches") for row in distance_rows):
        text += "No interval supports a predeclared better-planning direction. "
    mse_rows = [row for row in distance_rows if row["metric"] in ("one_step_latent_mse", "multi_step_latent_mse")]
    if mse_rows and all(row.get("spearman") is not None and float(row["spearman"]) < 0.0 for row in mse_rows):
        text += "Latent MSE point estimates are negative, opposite to lower error meaning a closer goal, and their intervals include zero. "
    magnitude = next((row for row in distance_rows if row["metric"] == "magnitude_response_ratio"), None)
    if magnitude is not None:
        reduced = next((fold for fold in magnitude.get("leave_one_family_out", []) if fold["left_out"] == "vicreg_noaction"), None)
        if reduced is not None and reduced["spearman"] is not None and magnitude.get("spearman") is not None:
            text += (
                f"Leaving out the no-action family changes the magnitude Spearman from {float(magnitude['spearman']):.2f} "
                f"to {float(reduced['spearman']):.2f}."
            )
    return " ".join(text.split())


def write_headline(rows: Sequence[dict[str, Any]], evaluations: Sequence[dict[str, Any]], sentence: str) -> None:
    HEADLINE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "group",
        "n",
        "spearman",
        "spearman_ci_low",
        "spearman_ci_high",
        "pearson",
        "loo_r2",
        "direction",
        "ci_excludes_zero_and_sign_matches",
        "leave_one_training_seed_out_min",
        "leave_one_training_seed_out_max",
        "leave_one_family_out_min",
        "leave_one_family_out_max",
    ]
    with HEADLINE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluations:
            writer.writerow({key: row.get(key) for key in fieldnames})
    payload = {
        "unit": "checkpoint",
        "n_checkpoints": len(rows),
        "audit_split": "legacy validation episodes 1800-1999",
        "is_test_set": False,
        "planning_environment": CF_ENVIRONMENT,
        "counterfactual_slice": {
            "protocol": CF_PROTOCOL,
            "horizon": int(CF_HORIZON),
            "rollout_mode": CF_MODE,
        },
        "planning_outcome": "mean over report seeds, scored episodes only",
        "cumulative_column": "cumulative_cost",
        "reward_logged": False,
        "models_are_univariate": True,
        "headline": sentence,
        "evaluations": evaluations,
        "checkpoints": rows,
    }
    HEADLINE_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    model_fields = [
        "model",
        "family",
        "training_seed",
        *CONVENTIONAL,
        *COUNTERFACTUAL,
        "mpc_success_rate",
        "final_goal_distance",
        "cumulative_cost",
        "n_scored_planning",
    ]
    with MODEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=model_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_relationships(rows: Sequence[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"vicreg_action": "#4c78a8", "vicreg_noaction": "#f58518", "mse_action": "#54a24b"}
    figure, axes = plt.subplots(2, len(PLOT_METRICS), figsize=(14.5, 6.2), squeeze=False)
    outcomes = (("mpc_success_rate", "MPC success"), ("final_goal_distance", "final goal distance"))
    for column, metric in enumerate(PLOT_METRICS):
        for row_index, (outcome, ylabel) in enumerate(outcomes):
            axis = axes[row_index][column]
            for family, color in colors.items():
                subset = [item for item in rows if item["family"] == family and item.get(metric) is not None]
                axis.scatter(
                    [item[metric] for item in subset],
                    [item[outcome] for item in subset],
                    s=28,
                    color=color,
                    label=family if column == 0 and row_index == 0 else None,
                )
            axis.set_title(metric.replace("_", " "), fontsize=8)
            if outcome == "mpc_success_rate":
                axis.set_ylim(-0.05, 1.05)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    figure.suptitle("Top row: MPC success is 0 for every checkpoint. Bottom row: final goal distance.", fontsize=10, y=1.14)
    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import yaml

    protocol = yaml.safe_load(PLANNING_PROTOCOL.read_text(encoding="utf-8"))
    report_seeds = [int(seed) for seed in protocol["report_seeds"]]
    names = sorted(
        path.stem
        for path in HELDOUT_DIR.glob("*.json")
        if path.name != "completed_evaluation.json"
    )
    rows = assemble_rows(names, report_seeds)
    distance_evals = [evaluate_metric(rows, metric, "final_goal_distance") for metric in (*CONVENTIONAL, *COUNTERFACTUAL)]
    success = spearman(_column(rows, "one_step_latent_mse"), _column(rows, "mpc_success_rate"))
    cost_spearman = {
        metric: spearman(_column(rows, metric), _column(rows, "cumulative_cost"))
        for metric in (*CONVENTIONAL, *COUNTERFACTUAL)
    }
    sentence = headline_sentence(distance_evals, success["status"])
    write_headline(rows, distance_evals, sentence)
    # The compact table is final distance. Cost signs are stored beside it.
    payload = json.loads(HEADLINE_JSON.read_text(encoding="utf-8"))
    payload["logistic_mpc_success"] = "not_fit_single_class" if success["status"] == "undefined_zero_variance" else "not_in_headline"
    payload["cumulative_cost_spearman"] = cost_spearman
    payload["reward_logged"] = False
    HEADLINE_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_relationships(rows)
    print(sentence)
    print(HEADLINE_CSV)


if __name__ == "__main__":
    main()
