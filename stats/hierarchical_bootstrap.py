"""Training-seed bootstrap, with episode resampling nested inside each seed.

The interval is the uncertainty of the mean of per-seed means. It is not the
episode bootstrap of one checkpoint. That conditional interval is returned
beside it and is labeled separately. A registry with any failed or unrun
seed does not receive a training-seed interval.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.stats import episode_bootstrap_mean
from stats.seed_aggregation import COMPLETE, FAILED, MIN_TRAINING_SEEDS, NOT_RUN


def hierarchical_bootstrap_mean(
    episodes_by_seed: Mapping[int, Sequence[float]],
    *,
    required_seeds: Sequence[int],
    seed_status: Mapping[int, str],
    n_resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    excluded_episode_ids: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Resample seeds, then resample episodes inside each drawn seed."""
    required = [int(seed) for seed in required_seeds]
    if len(required) < MIN_TRAINING_SEEDS or len(set(required)) != len(required):
        raise ValueError("training-seed intervals require at least 5 distinct registry seeds.")
    resamples = int(n_resamples)
    if resamples < 1:
        raise ValueError("n_resamples must be >= 1.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    status = {int(seed): str(seed_status.get(seed, NOT_RUN)) for seed in required}
    failed = [seed for seed in required if status[seed] == FAILED]
    not_run = [seed for seed in required if status[seed] == NOT_RUN]
    complete = [seed for seed in required if status[seed] == COMPLETE]
    unknown = [seed for seed in required if status[seed] not in {COMPLETE, FAILED, NOT_RUN}]
    if unknown:
        raise ValueError(f"unknown seed status for {unknown}.")
    usable: dict[int, np.ndarray] = {}
    conditional: dict[str, Any] = {}
    excluded = excluded_episode_ids or {}
    for seed in complete:
        values = np.asarray(episodes_by_seed.get(seed, []), dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"complete seed {seed} has no episodes; refusing to impute them.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"seed {seed} contains a non-finite episode value; refusing to drop it.")
        usable[seed] = values
        episode_interval = episode_bootstrap_mean(values, resamples, np.random.default_rng(seed), alpha=alpha)
        episode_interval["interval_kind"] = "conditional_on_checkpoint"
        episode_interval["n_episodes_excluded"] = len(list(excluded.get(seed, [])))
        episode_interval["excluded_episode_ids"] = [int(episode) for episode in excluded.get(seed, [])]
        conditional[str(seed)] = episode_interval
    descriptive = [float(usable[seed].mean()) for seed in complete]
    point = None if not descriptive else float(np.mean(descriptive))
    training: dict[str, Any] = {
        "status": "insufficient_training_seeds",
        "interval_kind": None,
        "ci_low": None,
        "ci_high": None,
        "mean": point,
        "alpha": float(alpha),
        "n_resamples": resamples,
        "weighting": "equal_seed_then_equal_episode",
        "failed_seeds": failed,
        "not_run_seeds": not_run,
        "n_complete": len(complete),
        "n_required": len(required),
    }
    if len(complete) == len(required):
        arrays = [usable[seed] for seed in required]
        replicates = np.empty(resamples, dtype=np.float64)
        n_seeds = len(arrays)
        for draw in range(resamples):
            seed_index = rng.integers(0, n_seeds, size=n_seeds)
            seed_means = np.empty(n_seeds, dtype=np.float64)
            for position, index in enumerate(seed_index.tolist()):
                values = arrays[index]
                episode_index = rng.integers(0, values.size, size=values.size)
                seed_means[position] = float(values[episode_index].mean())
            replicates[draw] = float(seed_means.mean())
        training.update(
            {
                "status": "ok",
                "interval_kind": "training_seed",
                "ci_low": float(np.quantile(replicates, float(alpha) / 2.0)),
                "ci_high": float(np.quantile(replicates, 1.0 - float(alpha) / 2.0)),
                "mean": point,
            }
        )
    return {
        "training_seed_interval": training,
        "conditional_on_checkpoint": conditional,
    }
