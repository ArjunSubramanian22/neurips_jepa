"""Summaries across training seeds.

A training-seed confidence interval is not computed here. Two-seed
half-ranges are not intervals. The hierarchical bootstrap is a separate
statistic and is only available once every registry seed is complete.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

COMPLETE = "COMPLETE"
FAILED = "FAILED"
NOT_RUN = "NOT_RUN"
UNDEFINED = "UNDEFINED"
MIN_TRAINING_SEEDS = 5


def _required(seeds: Sequence[int]) -> list[int]:
    required = [int(seed) for seed in seeds]
    if len(required) < MIN_TRAINING_SEEDS:
        raise ValueError("a training-seed registry needs at least 5 seeds.")
    if len(set(required)) != len(required):
        raise ValueError("seed registry contains a duplicate.")
    return required


def _index(records: Sequence[Mapping[str, Any]], required: Sequence[int]) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        seed = int(record["seed"])
        if seed not in required:
            raise ValueError(f"seed {seed} is outside the registry and cannot be added.")
        if seed in indexed:
            raise ValueError(f"seed {seed} is duplicated.")
        status = str(record["status"])
        if status not in {COMPLETE, FAILED, NOT_RUN, UNDEFINED}:
            raise ValueError(f"unknown seed status {status!r}.")
        indexed[seed] = record
    for seed in required:
        indexed.setdefault(seed, {"seed": seed, "status": NOT_RUN, "value": None})
    return indexed


def _complete_values(indexed: Mapping[int, Mapping[str, Any]], required: Sequence[int]) -> list[float]:
    values = []
    for seed in required:
        record = indexed[seed]
        if record["status"] != COMPLETE:
            continue
        value = record.get("value")
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"complete seed {seed} has no finite value; refusing to drop it.")
        values.append(float(value))
    return values


def aggregate_seeds(records: Sequence[Mapping[str, Any]], required_seeds: Sequence[int]) -> dict[str, Any]:
    """Mean, median, and standard deviation of complete seeds.

    The training-seed interval is left empty unless every registry seed is
    complete. This function does not turn ``(max-min)/2`` into an interval.
    """
    required = _required(required_seeds)
    indexed = _index(records, required)
    values = _complete_values(indexed, required)
    array = np.asarray(values, dtype=np.float64)
    failed = [seed for seed in required if indexed[seed]["status"] == FAILED]
    not_run = [seed for seed in required if indexed[seed]["status"] == NOT_RUN]
    undefined = [seed for seed in required if indexed[seed]["status"] == UNDEFINED]
    complete = [seed for seed in required if indexed[seed]["status"] == COMPLETE]
    std = None
    if array.size >= 2:
        std = float(array.std(ddof=1))
    return {
        "seeds": [
            {
                "seed": seed,
                "status": indexed[seed]["status"],
                "value": None if indexed[seed].get("value") is None else float(indexed[seed]["value"]),
            }
            for seed in required
        ],
        "n_required": len(required),
        "n_complete": len(complete),
        "n_failed": len(failed),
        "n_not_run": len(not_run),
        "n_undefined": len(undefined),
        "failed_seeds": failed,
        "not_run_seeds": not_run,
        "undefined_seeds": undefined,
        "mean": None if array.size == 0 else float(array.mean()),
        "median": None if array.size == 0 else float(np.median(array)),
        "std": std,
        "std_ddof": 1,
        "training_seed_ci": None,
        "training_seed_ci_status": "not_computed_here",
        "half_range_is_confidence_interval": False,
    }


def paired_seed_comparison(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    required_seeds: Sequence[int],
) -> dict[str, Any]:
    """Paired seed differences on the registry intersection that both completed.

    The effect size is descriptive. It is not a confidence interval, and it
    is withheld from any seed that did not complete on both configurations.
    """
    required = _required(required_seeds)
    left_index = _index(left, required)
    right_index = _index(right, required)
    shared = [
        seed
        for seed in required
        if left_index[seed]["status"] == COMPLETE and right_index[seed]["status"] == COMPLETE
    ]
    differences = [float(left_index[seed]["value"]) - float(right_index[seed]["value"]) for seed in shared]
    array = np.asarray(differences, dtype=np.float64)
    effect: dict[str, Any]
    if array.size < 2:
        effect = {"cohens_d": None, "mean_difference": None if array.size == 0 else float(array.mean()), "status": "too_few_seeds"}
    else:
        deviation = float(array.std(ddof=1))
        effect = {
            "cohens_d": None if deviation == 0.0 else float(array.mean() / deviation),
            "mean_difference": float(array.mean()),
            "status": "undefined_zero_variance" if deviation == 0.0 else "descriptive",
        }
    incomplete = [seed for seed in required if seed not in shared]
    return {
        "required_seeds": required,
        "paired_seeds": shared,
        "unpaired_seeds": incomplete,
        "n_paired": len(shared),
        "differences": [{"seed": seed, "difference": difference} for seed, difference in zip(shared, differences)],
        "effect_size": effect,
        "training_seed_ci": None,
        "training_seed_ci_status": "insufficient_training_seeds" if incomplete else "not_computed_here",
        "half_range_is_confidence_interval": False,
    }
