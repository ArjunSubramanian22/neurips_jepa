"""Action planning on a caller-supplied dynamics function.

Random shooting scores a fixed candidate list. Cross-entropy minimization
searches a Gaussian over continuous actions. Neither routine trains a world
model. Oracle checks use linear residual dynamics whose correct action is
known before the planner runs.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .metrics.ranking import decision_utility

CostFn = Callable[[np.ndarray], np.ndarray]


def select_lowest_cost(costs: Any) -> int:
    """Index of the lowest finite cost. Ties break toward the smaller index."""
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("costs must be non-empty and finite.")
    return int(np.argmin(values))


def linear_residual_next(state: Any, actions: Any, weight: Any) -> np.ndarray:
    """One-step residual dynamics ``z + action @ weight``.

    ``state`` is ``(D,)``, ``actions`` is ``(K, A)``, and ``weight`` is ``(A, D)``.
    """
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    act = np.asarray(actions, dtype=np.float64)
    if act.ndim == 1:
        act = act.reshape(1, -1)
    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.shape != (act.shape[1], z.shape[0]):
        raise ValueError(
            f"weight shape {matrix.shape} does not match actions {act.shape} and state {z.shape}."
        )
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(act)) or not np.all(np.isfinite(matrix)):
        raise ValueError("dynamics inputs must be finite.")
    return z.reshape(1, -1) + act @ matrix


def squared_error_cost(predicted: Any, goal: Any) -> np.ndarray:
    """Mean squared error of each predicted row against one goal vector."""
    pred = np.asarray(predicted, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    target = np.asarray(goal, dtype=np.float64).reshape(1, -1)
    if pred.shape[1] != target.shape[1]:
        raise ValueError("predicted and goal dimensions differ.")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
        raise ValueError("predicted states and goal must be finite.")
    return np.mean((pred - target) ** 2, axis=-1)


def cem_minimize(
    cost_fn: CostFn,
    mean: Any,
    std: Any,
    *,
    population: int,
    n_iterations: int,
    elite_fraction: float,
    rng: np.random.Generator,
    lower: Any | None = None,
    upper: Any | None = None,
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """Minimize ``cost_fn`` by refitting a diagonal Gaussian to elite samples.

    ``cost_fn`` receives ``(population, action_dim)`` and returns ``(population,)``.
    The returned ``cost`` is the lowest cost among samples drawn during the
    search, not a cost recomputed on the elite mean.
    """
    center = np.asarray(mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(std, dtype=np.float64).reshape(-1)
    if center.shape != scale.shape:
        raise ValueError("mean and std must share a shape.")
    if np.any(scale <= 0.0):
        raise ValueError("std must be positive.")
    pop = int(population)
    steps = int(n_iterations)
    if pop < 2 or steps < 1:
        raise ValueError("population must be >= 2 and n_iterations >= 1.")
    if not 0.0 < float(elite_fraction) < 1.0:
        raise ValueError("elite_fraction must lie in (0, 1).")
    elite_count = max(1, int(round(pop * float(elite_fraction))))
    elite_count = min(elite_count, pop - 1)
    lo = None if lower is None else np.asarray(lower, dtype=np.float64)
    hi = None if upper is None else np.asarray(upper, dtype=np.float64)
    if (lo is None) ^ (hi is None):
        raise ValueError("lower and upper bounds must be passed together.")
    history: list[dict[str, float | int]] = []
    best_action = center.copy()
    best_cost = float("inf")
    for iteration in range(steps):
        samples = rng.normal(loc=center, scale=scale, size=(pop, center.size))
        if lo is not None:
            samples = np.clip(samples, lo, hi)
        costs = np.asarray(cost_fn(samples), dtype=np.float64).reshape(-1)
        if costs.shape != (pop,) or not np.all(np.isfinite(costs)):
            raise ValueError("cost_fn must return one finite cost per sample.")
        order = np.argsort(costs, kind="mergesort")
        elites = samples[order[:elite_count]]
        elite_costs = costs[order[:elite_count]]
        if float(elite_costs[0]) < best_cost:
            best_cost = float(elite_costs[0])
            best_action = samples[order[0]].copy()
        center = elites.mean(axis=0)
        scale = np.maximum(elites.std(axis=0), float(min_std))
        history.append(
            {
                "iteration": iteration,
                "best_cost": best_cost,
                "elite_mean_cost": float(elite_costs.mean()),
                "population_mean_cost": float(costs.mean()),
            }
        )
    return {
        "action": best_action,
        "cost": best_cost,
        "mean": center,
        "std": scale,
        "history": history,
        "elite_count": elite_count,
        "population": pop,
        "n_iterations": steps,
    }


def oracle_linear_action_recovery(
    *,
    seed: int = 0,
    n_candidates: int = 8,
    latent_dim: int = 4,
    correct_index: int = 3,
) -> dict[str, Any]:
    """Score random candidates under identity residual dynamics.

    The goal is the state reached by ``candidates[correct_index]``. The
    planner sees only costs, not the index. Identity dynamics make that
    candidate the unique minimizer of squared error.
    """
    if not 0 <= int(correct_index) < int(n_candidates):
        raise ValueError("correct_index is outside the candidate list.")
    rng = np.random.default_rng(int(seed))
    state = rng.normal(size=int(latent_dim))
    candidates = rng.normal(size=(int(n_candidates), int(latent_dim)))
    weight = np.eye(int(latent_dim))
    goal = state + candidates[int(correct_index)]
    predicted = linear_residual_next(state, candidates, weight)
    costs = squared_error_cost(predicted, goal)
    chosen = select_lowest_cost(costs)
    n_min = int(np.sum(np.isclose(costs, costs[chosen])))
    return {
        "seed": int(seed),
        "chosen_index": chosen,
        "correct_index": int(correct_index),
        "n_minima": n_min,
        "unique_minimum": n_min == 1,
        "costs": [float(v) for v in costs],
        "utility": decision_utility(chosen, int(correct_index), int(n_candidates)),
        "dynamics": "linear_residual_identity",
    }
