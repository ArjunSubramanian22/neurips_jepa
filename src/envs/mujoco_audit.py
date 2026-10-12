"""Box-push fork audit.

The comparisons live in :mod:`src.envs.fork_audit`. This module only supplies
the box-push actions and the goal, which is a coordinate of the free box.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.fork_audit import audit_forks
from src.envs.mujoco_box_push import ACTION_LIMIT, TASKS, MujocoBoxPush

HORIZON = 12
GOAL_COORDINATE = 1.0
_UNTRAINED = "no checkpoint was trained on mujoco_box_push observations"


def task_actions(task: str) -> dict[str, np.ndarray]:
    """Three commands. ``along`` contacts the box. ``miss`` does not."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}.")
    axis = int(TASKS[task]["axis"])
    along = np.zeros(2, dtype=np.float64)
    miss = np.zeros(2, dtype=np.float64)
    along[axis] = 0.008
    miss[1 - axis] = ACTION_LIMIT
    return {"along": along, "miss": miss, "hold": np.zeros(2, dtype=np.float64)}


def _goal(axis: int):
    def cost(name: str, future: dict[str, Any]) -> float:
        del name
        return abs(float(future["positions"][0, axis]) - GOAL_COORDINATE)

    return cost


def audit_task(task: str, seed: int = 0, represent=None) -> dict[str, Any]:
    """Fork three actions from one reset and keep each diagnostic layer separate."""
    env = MujocoBoxPush(task)
    try:
        actions = task_actions(task)
        report = audit_forks(
            env,
            actions,
            positive="along",
            baseline="hold",
            alternate="miss",
            horizon=HORIZON,
            goal_cost=_goal(int(TASKS[task]["axis"])),
            learned_reason=_UNTRAINED,
            seed=seed,
            represent=represent,
        )
        oracle = report["planning"]["oracle"]
        return {
            "environment_id": report["environment_id"],
            "task": task,
            "seed": int(seed),
            "horizon": HORIZON,
            "restoration_exact": True,
            "box_xy": {name: value[:2] for name, value in report["positions"].items()},
            "along_vs_hold": report["positive_vs_baseline"],
            "along_vs_miss": report["positive_vs_alternate"],
            "retrieval": report["retrieval"],
            "planning": {
                "learned": report["planning"]["learned"],
                "oracle_goal_coordinate": GOAL_COORDINATE,
                "oracle_costs": oracle["costs"],
                "oracle_chosen": oracle["chosen"],
                "status": oracle["status"],
            },
        }
    finally:
        env.close()
