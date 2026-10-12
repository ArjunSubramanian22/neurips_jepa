"""Wrap an existing simulator in the fork interface.

``clone_state`` and ``restore_state`` call ``get_state`` and ``set_state``.
The simulators themselves are unchanged, so the older audit scripts keep working.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Declared dataset roles. A name that is absent has no such split.
SPLITS: dict[str, dict[str, tuple[str, bool]]] = {
    "mujoco_arm": {
        "train": ("train", False),
        "val": ("validation", False),
    },
    "mujoco_box_push": {
        "live": ("not_applicable", False),
    },
}

TASKS: dict[str, tuple[str, ...]] = {
    "mujoco_arm": ("push_object", "fold_clear", "cover_post"),
    "mujoco_box_push": ("slide_x", "slide_y", "glance_x"),
}


def resolve_split(env: str, split: str) -> dict[str, Any]:
    """Return the split record. An undeclared split, including a missing test set, is an error."""
    if env not in SPLITS:
        raise KeyError(f"unknown environment {env!r}.")
    table = SPLITS[env]
    if split not in table:
        declared = ", ".join(sorted(table))
        raise ValueError(f"{env} has no split {split!r}. Declared splits: {declared}.")
    role, is_test = table[split]
    return {"split": split, "split_role": role, "is_test_set": bool(is_test)}


class ForkableEnvironment:
    """Delegate to a simulator that already implements ``get_state`` and ``set_state``."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self, seed: int) -> None:
        self.env.reset(int(seed))

    def clone_state(self) -> dict[str, Any]:
        return self.env.get_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.env.set_state(state)

    def get_state(self) -> dict[str, Any]:
        return self.clone_state()

    def set_state(self, state: dict[str, Any]) -> None:
        self.restore_state(state)

    def step(self, action: np.ndarray) -> None:
        self.env.step(action)

    def render(self) -> np.ndarray:
        return self.env.render()

    def physical_metrics(self, other: dict[str, Any]) -> dict[str, float]:
        return self.env.physical_state_distance(other)

    def sample_counterfactual_action(self) -> np.ndarray:
        sampler = getattr(self.env, "sample_counterfactual_action", None)
        if sampler is not None:
            return sampler()
        return self.env.sample_action()

    def close(self) -> None:
        closer = getattr(self.env, "close", None)
        if closer is not None:
            closer()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_env(name: str, task: str) -> ForkableEnvironment:
    """Construct one registered environment. MuJoCo is imported only here."""
    if name not in TASKS or task not in TASKS[name]:
        raise KeyError(f"unknown environment task {name}:{task}.")
    if name == "mujoco_arm":
        from src.envs.mujoco_arm import MujocoArm

        return ForkableEnvironment(MujocoArm(task))
    if name == "mujoco_box_push":
        from src.envs.mujoco_box_push import MujocoBoxPush

        return ForkableEnvironment(MujocoBoxPush(task))
    raise KeyError(name)


_SPECS: dict[str, Callable[[str], dict[str, Any]]] = {}


def register_task_spec(name: str, spec_for_task: Callable[[str], dict[str, Any]]) -> None:
    """Supply actions and a goal for an environment that is not built in."""
    _SPECS[name] = spec_for_task


def task_spec(name: str, task: str) -> dict[str, Any]:
    """Actions and goal used by the existing audits. Not a second definition."""
    if name in _SPECS:
        return _SPECS[name](task)
    if name == "mujoco_arm":
        from src.envs.arm_audit import HORIZON, _goal
        from src.envs.mujoco_arm import NAMED_ACTIONS, TASK_ACTIONS

        spec = TASK_ACTIONS[task]
        return {
            "actions": NAMED_ACTIONS,
            "positive": spec["positive"],
            "baseline": "hold",
            "alternate": spec["alternate"],
            "horizon": HORIZON,
            "goal_cost": _goal(task),
            "learned_reason": "no checkpoint was trained on mujoco_arm observations",
        }
    if name == "mujoco_box_push":
        from src.envs.mujoco_audit import HORIZON, _goal, task_actions
        from src.envs.mujoco_box_push import TASKS as BOX_TASKS

        return {
            "actions": task_actions(task),
            "positive": "along",
            "baseline": "hold",
            "alternate": "miss",
            "horizon": HORIZON,
            "goal_cost": _goal(int(BOX_TASKS[task]["axis"])),
            "learned_reason": "no checkpoint was trained on mujoco_box_push observations",
        }
    raise KeyError(name)


Factory = Callable[[str], ForkableEnvironment]
_EXTRA: dict[str, tuple[Factory, tuple[str, ...], dict[str, tuple[str, bool]]]] = {}


def register_env(
    name: str,
    factory: Factory,
    tasks: tuple[str, ...],
    splits: dict[str, tuple[str, bool]],
) -> None:
    """Register an environment that is not one of the built-in simulators."""
    _EXTRA[name] = (factory, tasks, splits)
    TASKS[name] = tasks
    SPLITS[name] = splits


def make_registered(name: str, task: str) -> ForkableEnvironment:
    if name in _EXTRA:
        factory, tasks, _splits = _EXTRA[name]
        if task not in tasks:
            raise KeyError(f"unknown environment task {name}:{task}.")
        return factory(task)
    return make_env(name, task)
