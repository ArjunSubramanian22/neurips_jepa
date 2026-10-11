"""Closed-loop model-predictive control.

Each environment step proposes an action, executes only the first one, and
may warm-start the next plan from the unused tail. The loop scores the true
task distance, not the planner's internal cost.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.planning import shift_plan

Propose = Callable[[np.ndarray | None], tuple[np.ndarray, np.ndarray | None, float]]
DISTANCE_FLOOR = 1e-8


def mean_force_action(action_dim: int = 4) -> np.ndarray:
    """Contact at the box center and zero force. No particle state."""
    action = np.zeros(int(action_dim), dtype=np.float64)
    if action.size >= 2:
        action[0] = 0.5
        action[1] = 0.5
    return action


def heuristic_push(
    positions: Any,
    goal_positions: Any,
    *,
    force_scale: float = 0.8,
    action_dim: int = 4,
) -> np.ndarray:
    """Push the current centroid toward the goal centroid.

    This baseline reads particle positions. It is not a learned planner.
    """
    current = np.asarray(positions, dtype=np.float64)
    goal = np.asarray(goal_positions, dtype=np.float64)
    if current.ndim != 2 or goal.shape != current.shape:
        raise ValueError("positions and goal_positions must share shape (N, D).")
    delta = goal.mean(axis=0) - current.mean(axis=0)
    norm = float(np.linalg.norm(delta))
    action = np.zeros(int(action_dim), dtype=np.float64)
    contact = np.clip(current.mean(axis=0), 0.08, 0.92)
    action[: contact.size] = contact
    if norm > DISTANCE_FLOOR and action.size >= contact.size + delta.size:
        action[contact.size : contact.size + delta.size] = force_scale * delta / norm
    return action


def random_action(rng: np.random.Generator, lower: Any, upper: Any) -> np.ndarray:
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError("bounds must share a shape.")
    return rng.uniform(lo, hi)


def run_closed_loop(
    propose: Propose,
    step: Callable[[np.ndarray], None],
    distance: Callable[[], float],
    *,
    initial_distance: float,
    max_steps: int,
    success_distance: float,
    ambiguous: bool = False,
) -> dict[str, Any]:
    """Execute at most ``max_steps`` closed-loop actions."""
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1.")
    initial = float(initial_distance)
    if ambiguous:
        return _finish("observationally_ambiguous", scored=False, initial=initial, success_distance=float(success_distance))
    if not np.isfinite(initial) or initial <= DISTANCE_FLOOR:
        return _finish("goal_already_satisfied", scored=False, initial=initial, success_distance=float(success_distance))
    threshold = float(success_distance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("success_distance must be finite and >= 0.")
    distances: list[float] = []
    actions: list[np.ndarray] = []
    runtimes: list[float] = []
    planner_costs: list[float | None] = []
    warm: np.ndarray | None = None
    status = "timeout"
    steps_to_success: int | None = None
    error: str | None = None
    for index in range(int(max_steps)):
        started = time.perf_counter()
        try:
            action, plan, planner_cost = propose(warm)
        except Exception as exc:
            status = "planner_exception"
            error = f"{type(exc).__name__}: {exc}"
            runtimes.append(time.perf_counter() - started)
            break
        runtimes.append(time.perf_counter() - started)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(action)):
            status = "nonfinite_action"
            break
        step(action)
        current = float(distance())
        if not np.isfinite(current):
            status = "nonfinite_distance"
            actions.append(action)
            break
        distances.append(current)
        actions.append(action)
        planner_costs.append(float(planner_cost) if np.isfinite(planner_cost) else None)
        warm = None if plan is None else shift_plan(plan)
        if current <= threshold:
            status = "success"
            steps_to_success = index + 1
            break
    final = distances[-1] if distances else initial
    return {
        "status": status,
        "scored": True,
        "success": status == "success",
        "initial_distance": initial,
        "final_distance": float(final),
        "success_distance": threshold,
        "cumulative_cost": float(np.sum(distances)) if distances else 0.0,
        "actions_to_success": steps_to_success,
        "planning_runtime_s": float(np.sum(runtimes)),
        "n_actions": len(actions),
        "distances": distances,
        "actions": actions,
        "planner_costs": planner_costs,
        "error": error,
    }


def _finish(status: str, *, scored: bool, initial: float, success_distance: float) -> dict[str, Any]:
    return {
        "status": status,
        "scored": scored,
        "success": False,
        "initial_distance": initial,
        "final_distance": initial,
        "success_distance": success_distance,
        "cumulative_cost": 0.0,
        "actions_to_success": None,
        "planning_runtime_s": 0.0,
        "n_actions": 0,
        "distances": [],
        "actions": [],
        "planner_costs": [],
        "error": None,
    }


def oracle_sequence_cost(world: Any, sequences: Any, score_fn: Callable[[Any], float]) -> np.ndarray:
    """Roll each sequence on a clone of ``world`` and restore the live state."""
    seq = np.asarray(sequences, dtype=np.float64)
    if seq.ndim != 3:
        raise ValueError(f"sequences must have shape (P, H, A), got {seq.shape}.")
    snapshot = world.snapshot()
    costs = np.empty(seq.shape[0], dtype=np.float64)
    try:
        for index in range(seq.shape[0]):
            world.restore(snapshot)
            for action in seq[index]:
                world.step_action(action)
            costs[index] = float(score_fn(world))
    finally:
        world.restore(snapshot)
    if not np.all(np.isfinite(costs)):
        raise ValueError("oracle costs must be finite.")
    return costs


def oracle_propose(
    world: Any,
    score_fn: Callable[[Any], float],
    cem: dict[str, Any],
    rng: np.random.Generator,
    warm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CEM using true dynamics. ``score_fn`` reads the rolled world."""
    from src.planning import cem_plan_sequence

    mean = cem["mean"] if warm is None else warm

    def cost(samples: np.ndarray) -> np.ndarray:
        return oracle_sequence_cost(world, samples, score_fn)

    result = cem_plan_sequence(
        cost,
        mean,
        cem["std"],
        horizon=int(cem["horizon"]),
        population=int(cem["population"]),
        n_iterations=int(cem["n_iterations"]),
        elite_fraction=float(cem["elite_fraction"]),
        action_dim=int(cem["action_dim"]),
        lower=cem["lower"],
        upper=cem["upper"],
        temporal_smooth=float(cem["temporal_smooth"]),
        rng=rng,
    )
    return result["action"], result["sequence"], float(result["cost"])
