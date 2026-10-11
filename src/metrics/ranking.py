"""Decision-utility summaries for a discrete set of candidate actions.

These functions score whether a planner recovers a designated correct action.
They do not convert that hit into a latent distance.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def chance_top1(n_candidates: int) -> float:
    """Uniform chance of selecting one distinguished candidate."""
    if int(n_candidates) < 1:
        raise ValueError("n_candidates must be >= 1.")
    return 1.0 / float(n_candidates)


def decision_utility(
    chosen_index: int,
    correct_index: int,
    n_candidates: int,
) -> dict[str, Any]:
    """One-trial top-1 recovery against a uniform chance baseline."""
    n = int(n_candidates)
    chosen = int(chosen_index)
    correct = int(correct_index)
    if n < 1:
        raise ValueError("n_candidates must be >= 1.")
    if not 0 <= chosen < n or not 0 <= correct < n:
        raise ValueError("indexes must lie in the candidate range.")
    return {
        "top1": float(chosen == correct),
        "chance": chance_top1(n),
        "n_candidates": n,
        "chosen_index": chosen,
        "correct_index": correct,
        "status": "measured",
    }


def random_ranking_top1(
    n_trials: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Top-1 rate when scores are i.i.d. uniform and the correct index is 0.

    Ties in the scores are broken by the first maximum (``argmax``). With
    continuous scores, ties have probability zero.
    """
    trials = int(n_trials)
    n = int(n_candidates)
    if trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if n < 2:
        raise ValueError("n_candidates must be >= 2 for a ranking check.")
    scores = rng.random((trials, n))
    chosen = np.argmax(scores, axis=1)
    hits = int(np.sum(chosen == 0))
    return {
        "n_trials": trials,
        "n_candidates": n,
        "n_hits": hits,
        "top1": float(hits / trials),
        "chance": chance_top1(n),
        "correct_index": 0,
        "status": "measured",
    }
