"""Planning-benchmark summaries. Sensitivity is not a substitute for success."""

from __future__ import annotations

from typing import Any

import numpy as np

PRINCIPAL_CHECKPOINTS = (
    {"name": "vicreg_action_s42", "path": "outputs/vicreg_nocrop/last.pt", "family": "vicreg_action"},
    {"name": "vicreg_action_s43", "path": "outputs/vicreg_nocrop_seed43/last.pt", "family": "vicreg_action"},
    {"name": "vicreg_action_s44", "path": "results/reviewer_strengthening/05_more_seeds/vicreg_action_s44/last.pt", "family": "vicreg_action"},
    {"name": "vicreg_action_s45", "path": "results/reviewer_strengthening/05_more_seeds/vicreg_action_s45/last.pt", "family": "vicreg_action"},
    {"name": "vicreg_noaction_s42", "path": "outputs/no_action/last.pt", "family": "vicreg_noaction"},
    {"name": "vicreg_noaction_s43", "path": "outputs/vicreg_nocrop_noaction_seed43/last.pt", "family": "vicreg_noaction"},
    {"name": "vicreg_noaction_s44", "path": "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s44/last.pt", "family": "vicreg_noaction"},
    {"name": "vicreg_noaction_s45", "path": "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s45/last.pt", "family": "vicreg_noaction"},
    {"name": "mse_action_s42", "path": "results/reviewer_strengthening/06_corrected_objective/mse_action_s42/last.pt", "family": "mse_action"},
    {"name": "mse_action_s43", "path": "results/reviewer_strengthening/06_corrected_objective/mse_action_s43/last.pt", "family": "mse_action"},
)
POOR_ACTION_FAMILY = "vicreg_noaction"


def record_episode(controller: str, task: str, seed: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "controller": controller,
        "task": task,
        "seed": int(seed),
        "status": result["status"],
        "scored": bool(result["scored"]),
        "success": bool(result["success"]) if result["scored"] else None,
        "final_distance": None if not result["scored"] else float(result["final_distance"]),
        "initial_distance": float(result["initial_distance"]),
        "cumulative_cost": None if not result["scored"] else float(result["cumulative_cost"]),
        "actions_to_success": result["actions_to_success"] if result["success"] else None,
        "planning_runtime_s": float(result["planning_runtime_s"]) if result["scored"] else None,
        "error": result.get("error"),
    }


def aggregate_controller(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one controller. Unscored goals stay listed and out of the rate."""
    scored = [row for row in records if row["scored"]]
    successes = [row for row in scored if row["success"]]
    failures = [row for row in records if row["status"] != "success"]
    distances = [float(row["final_distance"]) for row in scored if row["final_distance"] is not None]
    costs = [float(row["cumulative_cost"]) for row in scored if row["cumulative_cost"] is not None]
    runtimes = [float(row["planning_runtime_s"]) for row in scored if row["planning_runtime_s"] is not None]
    steps = [int(row["actions_to_success"]) for row in successes if row["actions_to_success"] is not None]
    return {
        "n_episodes": len(records),
        "n_scored": len(scored),
        "n_unscored": len(records) - len(scored),
        "n_success": len(successes),
        "success_rate": None if not scored else len(successes) / len(scored),
        "final_distance_mean": None if not distances else float(np.mean(distances)),
        "final_distance_median": None if not distances else float(np.median(distances)),
        "cumulative_cost_mean": None if not costs else float(np.mean(costs)),
        "actions_to_success_mean": None if not steps else float(np.mean(steps)),
        "planning_runtime_s_mean": None if not runtimes else float(np.mean(runtimes)),
        "failure_counts": _counts(row["status"] for row in failures),
        "unscored_statuses": _counts(row["status"] for row in records if not row["scored"]),
    }


def select_planning_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick one oracle config from validation rows. The same config is then used everywhere.

    Higher mean success wins. Ties break toward lower mean final distance.
    A test-split row is refused rather than ignored.
    """
    if any(str(row.get("split_role")) == "test" for row in rows):
        raise ValueError("planning configuration selection must not read the test split.")
    validation = [row for row in rows if str(row.get("split_role")) == "val"]
    if not validation:
        return {"selected_config": None, "selection_status": "not_run", "same_config_for_every_model": True}
    names = sorted({str(row["config"]) for row in validation})
    summary = []
    for name in names:
        group = [row for row in validation if row["config"] == name]
        summary.append(
            {
                "config": name,
                "oracle_success_mean": float(np.mean([float(row["oracle_success"]) for row in group])),
                "oracle_final_distance_mean": float(np.mean([float(row["oracle_final_distance"]) for row in group])),
                "n": len(group),
            }
        )
    summary.sort(key=lambda item: (-item["oracle_success_mean"], item["oracle_final_distance_mean"], item["config"]))
    winner = summary[0]
    tied = [
        item
        for item in summary
        if item["oracle_success_mean"] == winner["oracle_success_mean"]
        and item["oracle_final_distance_mean"] == winner["oracle_final_distance_mean"]
    ]
    if len(tied) > 1:
        return {
            "selected_config": None,
            "selection_status": "tied_on_validation_oracle",
            "same_config_for_every_model": True,
            "candidates": summary,
        }
    return {
        "selected_config": winner["config"],
        "selection_status": "selected_on_validation_oracle",
        "same_config_for_every_model": True,
        "candidates": summary,
    }


def choose_highlight(
    records: list[dict[str, Any]],
    validation_seeds: list[int] | tuple[int, ...],
    eligible: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Choose a learned model from validation episodes only."""
    seeds = {int(seed) for seed in validation_seeds}
    ranked = []
    for name in eligible:
        rows = [row for row in records if row["controller"] == name and int(row["seed"]) in seeds and row["scored"]]
        if not rows:
            continue
        ranked.append(
            {
                "name": name,
                "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
                "final_distance_mean": float(np.mean([float(row["final_distance"]) for row in rows])),
                "n_scored": len(rows),
            }
        )
    if not ranked:
        return {"highlight": None, "status": "not_measured", "candidates": []}
    ranked.sort(key=lambda item: (-item["success_rate"], item["final_distance_mean"], item["name"]))
    best = ranked[0]
    tied = [
        item
        for item in ranked
        if item["success_rate"] == best["success_rate"] and item["final_distance_mean"] == best["final_distance_mean"]
    ]
    if len(tied) > 1:
        return {"highlight": None, "status": "tied_on_validation", "candidates": ranked}
    return {"highlight": best["name"], "status": "selected_on_validation", "candidates": ranked}


def population_action_moments(actions: Any, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation on the training episodes only."""
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("actions must have shape (episodes, time, action_dim).")
    if not 0 < int(n_train) <= array.shape[0]:
        raise ValueError("n_train must select a non-empty proper or full prefix.")
    flat = array[: int(n_train)].reshape(-1, array.shape[-1])
    return flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-6)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
