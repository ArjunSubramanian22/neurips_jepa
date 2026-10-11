"""Equal-episode bootstrap for clustered rollout statistics.

Each episode contributes its within-episode mean, and episodes are resampled
with replacement. Observations inside an episode are not treated as
independent. Non-finite values raise; they are not dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def episode_bootstrap_mean(
    episode_values: Any,
    n_resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile interval for the mean of per-episode means.

    ``episode_values`` is ``(n_episodes,)`` or ``(n_episodes, n_obs)``.
    Episodes have equal weight after the within-episode mean.
    """
    values = np.asarray(episode_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("episode_values must have shape (n_episodes, n_obs) with both positive.")
    if not np.all(np.isfinite(values)):
        raise ValueError("episode_values contain a non-finite value; refusing to drop it.")
    resamples = int(n_resamples)
    if resamples < 1:
        raise ValueError("n_resamples must be >= 1.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    episode_means = values.mean(axis=1)
    point = float(episode_means.mean())
    n_episodes = int(episode_means.shape[0])
    draws = rng.integers(0, n_episodes, size=(resamples, n_episodes))
    samples = episode_means[draws].mean(axis=1)
    low = float(np.quantile(samples, float(alpha) / 2.0))
    high = float(np.quantile(samples, 1.0 - float(alpha) / 2.0))
    return {
        "mean": point,
        "ci_low": low,
        "ci_high": high,
        "alpha": float(alpha),
        "n_episodes": n_episodes,
        "n_observations_per_episode": int(values.shape[1]),
        "n_resamples": resamples,
        "weighting": "equal_episode",
        "n_undefined_dropped": 0,
    }
