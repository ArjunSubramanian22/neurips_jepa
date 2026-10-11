"""Arm fork audit. Comparisons come from the shared fork evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.fork_audit import audit_forks
from src.envs.mujoco_arm import FOLD, NAMED_ACTIONS, TASK_ACTIONS, MujocoArm, post_visible_fraction

HORIZON = 5
OBJECT_GOAL_X = 0.55
_UNTRAINED = "no checkpoint was trained on mujoco_arm observations"


def _goal(task: str):
    def cost(name: str, future: dict[str, Any]) -> float:
        del name
        if task == "push_object":
            return abs(float(future["positions"][0, 0]) - OBJECT_GOAL_X)
        if task == "fold_clear":
            return float(np.linalg.norm(np.asarray(future["control"]) - FOLD))
        if task == "cover_post":
            return post_visible_fraction(future["image"])
        raise ValueError(f"unknown task {task!r}.")

    return cost


def audit_task(task: str, seed: int = 0, **callbacks: Any) -> dict[str, Any]:
    """Fork push, fold, and hold. The task only changes which outcome is the goal."""
    spec = TASK_ACTIONS[task]
    env = MujocoArm(task)
    try:
        report = audit_forks(
            env,
            NAMED_ACTIONS,
            positive=spec["positive"],
            baseline="hold",
            alternate=spec["alternate"],
            horizon=HORIZON,
            goal_cost=_goal(task),
            learned_reason=_UNTRAINED,
            seed=seed,
            predict=callbacks.get("predict"),
            encode=callbacks.get("encode"),
        )
        report["task"] = task
        return report
    finally:
        env.close()
