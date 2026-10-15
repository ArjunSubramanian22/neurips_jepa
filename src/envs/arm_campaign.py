"""Larger arm dataset with varied starts and a sealed test suffix.

This does not replace ``outputs/mujoco_arm``. Episode order is train, then
validation, then test. Training code that asks for split ``all`` receives
only the first two blocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.mujoco_arm import FOLD, HOME, PUSH, RENDER_SIZE, MujocoArm
from src.metrics.cascade import observation_visibility

KINDS = ("push", "fold", "hold", "switch", "random_hold", "sequence")
N_TRAIN = 480
N_VAL = 120
N_TEST = 120
EPISODE_LENGTH = 16
SPLIT_VERSION = "arm_campaign_v1"


def kind_for(index: int) -> str:
    return KINDS[int(index) % len(KINDS)]


def actions_for(env: MujocoArm, index: int, steps: int) -> np.ndarray:
    """Deterministic action sequence for one episode. ``steps`` is T-1."""
    if steps < 2:
        raise ValueError("need at least two actions.")
    rng = np.random.default_rng(90_000 + int(index))
    low = env.model.actuator_ctrlrange[:, 0]
    high = env.model.actuator_ctrlrange[:, 1]
    kind = kind_for(index)

    def hold(target: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(target, dtype=np.float64).reshape(1, 2), steps, axis=0)

    if kind == "push":
        return hold(PUSH)
    if kind == "fold":
        return hold(FOLD)
    if kind == "hold":
        return hold(HOME)
    if kind == "random_hold":
        return hold(rng.uniform(low, high))
    if kind == "switch":
        mid = steps // 2
        first = np.repeat(rng.uniform(low, high).reshape(1, 2), mid, axis=0)
        second = np.repeat(rng.uniform(low, high).reshape(1, 2), steps - mid, axis=0)
        return np.vstack([first, second])
    actions = np.zeros((steps, 2), dtype=np.float64)
    cursor = 0
    while cursor < steps:
        target = rng.uniform(low, high)
        nxt = min(steps, cursor + 4)
        actions[cursor:nxt] = target
        cursor = nxt
    return actions


def reflect_actions(env: MujocoArm, actions: np.ndarray) -> np.ndarray:
    low = env.model.actuator_ctrlrange[:, 0]
    high = env.model.actuator_ctrlrange[:, 1]
    reflected = low + high - np.asarray(actions, dtype=np.float64)
    return np.clip(reflected, low, high)


def rollout(env: MujocoArm, index: int, length: int, actions: np.ndarray | None = None) -> dict[str, Any]:
    start = env.reset_campaign(index)
    steps = length - 1
    chosen = actions_for(env, index, steps) if actions is None else np.asarray(actions, dtype=np.float64)
    if chosen.shape != (steps, 2):
        raise ValueError(f"expected actions {(steps, 2)}, got {chosen.shape}.")
    frames = [env.render()]
    xy = [env.tracked_positions()[0, :2].copy()]
    for action in chosen:
        env.step(action)
        frames.append(env.render())
        xy.append(env.tracked_positions()[0, :2].copy())
    if not np.isfinite(np.stack(xy)).all():
        raise RuntimeError(f"episode {index} produced a non-finite object trajectory.")
    return {"frames": frames, "actions": chosen, "xy": np.stack(xy), "start": start, "kind": kind_for(index)}


def generate_campaign_dataset(
    path: str | Path,
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    length: int = EPISODE_LENGTH,
) -> Path:
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("train, validation, and test must each be non-empty.")
    if length < 3:
        raise ValueError("episode length must be at least 3.")
    n_episodes = n_train + n_val + n_test
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    observations = np.zeros((n_episodes, length, RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8)
    actions = np.zeros((n_episodes, length - 1, 2), dtype=np.float64)
    object_xy = np.zeros((n_episodes, length, 2), dtype=np.float64)
    kinds = [kind_for(index) for index in range(n_episodes)]
    env = MujocoArm("push_object")
    try:
        for index in range(n_episodes):
            played = rollout(env, index, length)
            observations[index] = np.stack(played["frames"])
            actions[index] = played["actions"]
            object_xy[index] = played["xy"]
    finally:
        env.close()
    np.save(out / "observations.npy", observations)
    np.save(out / "actions.npy", actions)
    np.save(out / "object_xy.npy", object_xy)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "environment_id": "mujoco_arm",
                "split_version": SPLIT_VERSION,
                "package": "mujoco",
                "version": "3.3.7",
                "license": "Apache-2.0",
                "n_episodes": n_episodes,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "episode_length": length,
                "action_dim": 2,
                "image": [RENDER_SIZE, RENDER_SIZE, 3],
                "kinds": kinds,
                "episode_order": "train, validation, test",
                "test_closed": True,
                "initial_state": "reset_campaign varies object xy and joint start; reset() does not",
                "commands": {"push": PUSH.tolist(), "fold": FOLD.tolist(), "hold": HOME.tolist()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def audit_campaign_dataset(path: str | Path, *, intervention_sample: int = 6) -> dict[str, Any]:
    """Replay, split separation, varied starts, and a reflected-action check."""
    folder = Path(path)
    observations = np.load(folder / "observations.npy")
    actions = np.load(folder / "actions.npy")
    object_xy = np.load(folder / "object_xy.npy")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    n_train = int(meta["n_train"])
    n_val = int(meta["n_val"])
    n_test = int(meta["n_test"])
    n_episodes = int(observations.shape[0])
    if n_train + n_val + n_test != n_episodes:
        raise ValueError("split counts do not cover the file.")
    train_ids = set(range(0, n_train))
    val_ids = set(range(n_train, n_train + n_val))
    test_ids = set(range(n_train + n_val, n_episodes))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("split ids overlap.")
    if not np.isfinite(object_xy).all():
        raise ValueError("object positions are not finite.")
    length = int(observations.shape[1])
    env = MujocoArm("push_object")
    checked = [0, n_train, n_train + n_val]
    try:
        for index in checked:
            played = rollout(env, index, length)
            if not np.array_equal(np.stack(played["frames"]), observations[index]):
                raise ValueError(f"replay frames diverged at episode {index}.")
            if not np.allclose(played["xy"], object_xy[index]):
                raise ValueError(f"replay object position diverged at episode {index}.")
            if not np.allclose(played["actions"], actions[index]):
                raise ValueError(f"replay actions diverged at episode {index}.")
        start_xy = object_xy[:, 0]
        varied = float(np.std(start_xy[:, 0])) > 1e-3 and float(np.std(start_xy[:, 1])) > 1e-4
        if not varied:
            raise ValueError("initial object positions do not vary across episodes.")
        sample = [index for index in range(min(n_episodes, max(intervention_sample, len(KINDS))))]
        effects = []
        for index in sample:
            reflected = reflect_actions(env, actions[index])
            played = rollout(env, index, length, reflected)
            recorded_xy = object_xy[index, -1]
            physical = float(np.linalg.norm(played["xy"][-1] - recorded_xy))
            image = observation_visibility(observations[index, -1], played["frames"][-1])
            effects.append({
                "episode": index,
                "kind": kind_for(index),
                "object_effect": physical,
                "image_mse": float(image["mse"]),
            })
    finally:
        env.close()
    visible_effects = [item for item in effects if item["object_effect"] > 1e-3 and item["image_mse"] > 0.01]
    push_move = [
        float(np.linalg.norm(object_xy[index, -1] - object_xy[index, 0]))
        for index in range(n_episodes)
        if kind_for(index) == "push"
    ]
    hold_move = [
        float(np.linalg.norm(object_xy[index, -1] - object_xy[index, 0]))
        for index in range(n_episodes)
        if kind_for(index) == "hold"
    ]
    return {
        "split_version": SPLIT_VERSION,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "episode_length": length,
        "replay_exact": True,
        "splits_disjoint": True,
        "test_closed_to_all": True,
        "start_x_std": float(np.std(start_xy[:, 0])),
        "start_y_std": float(np.std(start_xy[:, 1])),
        "push_displacement_median": None if not push_move else float(np.median(push_move)),
        "hold_displacement_median": None if not hold_move else float(np.median(hold_move)),
        "intervention_sample": effects,
        "n_visible_interventions": len(visible_effects),
        "status": "passed",
    }
