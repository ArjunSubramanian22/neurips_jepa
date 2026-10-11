"""Small action-fork audit for the MuJoCo box push.

Physical and image layers use the shared cascade metrics. The latent layers
use the box coordinates as a privileged target. No checkpoint trained on
these images is scored. A copy of the true next coordinate is the positive
predictor control. A predictor that ignores the action is the negative control.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.mujoco_box_push import ACTION_LIMIT, TASKS, MujocoBoxPush
from src.metrics.cascade import (
    cosine_alignment,
    not_measured,
    observation_visibility,
    physical_response,
)

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
    return {
        "along": along,
        "miss": miss,
        "hold": np.zeros(2, dtype=np.float64),
    }


def _roll(env: MujocoBoxPush, state: dict[str, Any], action: np.ndarray) -> dict[str, Any]:
    env.set_state(state)
    for _ in range(HORIZON):
        env.step(action)
    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "box_xy": env.box_xy(),
        "image": env.render(),
    }


def _pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    physical = physical_response(
        left["qpos"][:3].reshape(1, 3),
        right["qpos"][:3].reshape(1, 3),
        left["qvel"][:3].reshape(1, 3),
        right["qvel"][:3].reshape(1, 3),
    )
    image = observation_visibility(left["image"], right["image"])
    target = cosine_alignment(right["box_xy"] - left["box_xy"], right["box_xy"] - left["box_xy"])
    ignored = np.zeros(2, dtype=np.float64)
    constant = cosine_alignment(ignored, right["box_xy"] - left["box_xy"])
    return {
        "physical": physical,
        "observation": image,
        "target_latent": {
            "status": "measured",
            "source": "privileged_box_xy",
            "delta_l2": float(np.linalg.norm(right["box_xy"] - left["box_xy"])),
        },
        "predictor": {
            "learned": not_measured(_UNTRAINED),
            "oracle_copies_target": target,
            "constant_ignores_action": constant,
        },
        "direction": {
            "learned": not_measured(_UNTRAINED),
            "oracle_cosine": target["cosine"],
            "oracle_status": target["status"],
            "constant_cosine": constant["cosine"],
            "constant_status": constant["status"],
        },
    }


def _retrieval(futures: dict[str, np.ndarray], predict) -> dict[str, Any]:
    """Two-way nearest neighbor. A tie is not a hit."""
    names = list(futures)
    hits = 0
    ties = 0
    for name in names:
        predicted = np.asarray(predict(name), dtype=np.float64)
        distances = [
            float(np.mean((predicted - np.asarray(futures[other], dtype=np.float64)) ** 2))
            for other in names
        ]
        best = min(distances)
        winners = [names[i] for i, dist in enumerate(distances) if np.isclose(dist, best)]
        if len(winners) != 1:
            ties += 1
        elif winners[0] == name:
            hits += 1
    if ties == len(names):
        return {
            "top1": None,
            "n": len(names),
            "n_ties": ties,
            "status": "ambiguous_identical_target",
        }
    return {
        "top1": float(hits / len(names)),
        "n": len(names),
        "n_ties": ties,
        "status": "measured",
    }


def _planning(futures: dict[str, np.ndarray], axis: int) -> dict[str, Any]:
    costs = {
        name: float(abs(float(position[axis]) - GOAL_COORDINATE))
        for name, position in futures.items()
    }
    chosen = min(costs, key=costs.get)
    return {
        "learned": not_measured(_UNTRAINED),
        "oracle_goal_coordinate": GOAL_COORDINATE,
        "oracle_costs": costs,
        "oracle_chosen": chosen,
        "status": "measured",
    }


def audit_task(task: str, seed: int = 0) -> dict[str, Any]:
    """Fork three actions from one reset and keep each diagnostic layer separate."""
    env = MujocoBoxPush(task)
    try:
        env.reset(seed)
        start = env.get_state()
        actions = task_actions(task)
        futures = {name: _roll(env, start, action) for name, action in actions.items()}
        env.set_state(start)
        restored = env.get_state()
        if not np.array_equal(restored["vector"], start["vector"]):
            raise RuntimeError("set_state did not restore the fork origin.")
        box = {name: value["box_xy"] for name, value in futures.items()}
        images = {
            name: value["image"].astype(np.float64) / 255.0 for name, value in futures.items()
        }
        return {
            "environment_id": env.environment_id,
            "task": task,
            "seed": int(seed),
            "horizon": HORIZON,
            "restoration_exact": True,
            "box_xy": {name: value.tolist() for name, value in box.items()},
            "along_vs_hold": _pair(futures["hold"], futures["along"]),
            "along_vs_miss": _pair(futures["miss"], futures["along"]),
            "retrieval": {
                "learned": not_measured(_UNTRAINED),
                "space": "rendered_rgb",
                "oracle_copies_target": _retrieval(images, lambda name: images[name]),
                "constant_ignores_action": _retrieval(
                    images, lambda _name: images["hold"]
                ),
                "privileged_box_xy": {
                    "oracle_copies_target": _retrieval(box, lambda name: box[name]),
                    "note": (
                        "A miss that does not move the box can share coordinates "
                        "with hold, so this space can be ambiguous."
                    ),
                },
            },
            "planning": _planning(box, int(TASKS[task]["axis"])),
        }
    finally:
        env.close()
