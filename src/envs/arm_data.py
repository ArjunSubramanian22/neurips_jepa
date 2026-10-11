"""Modest arm dataset. The tail split is validation, not a test set.

Episodes cycle through a contact command, a non-contact fold, and a seeded
random joint target. Thirty-two episodes is enough to check that those
commands change the object, the image, and the oracle's choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.arm_audit import audit_task
from src.envs.mujoco_arm import FOLD, PUSH, RENDER_SIZE, MujocoArm
from src.metrics.cascade import observation_visibility

N_EPISODES = 32
EPISODE_LENGTH = 8
VAL_FRACTION = 0.25


def episode_kind(index: int) -> str:
    return ("push", "fold", "random")[int(index) % 3]


def _action_for(env: MujocoArm, index: int) -> np.ndarray:
    kind = episode_kind(index)
    if kind == "push":
        return PUSH.copy()
    if kind == "fold":
        return FOLD.copy()
    return env.sample_action()


def generate_arm_dataset(path: str | Path, n_episodes: int = N_EPISODES, length: int = EPISODE_LENGTH) -> Path:
    """Write observations, actions, and object positions. Does not train."""
    if n_episodes < 3 or length < 2:
        raise ValueError("need at least 3 episodes and 2 frames.")
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    observations = np.zeros((n_episodes, length, RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8)
    actions = np.zeros((n_episodes, length - 1, 2), dtype=np.float64)
    object_xy = np.zeros((n_episodes, length, 2), dtype=np.float64)
    kinds = [episode_kind(i) for i in range(n_episodes)]
    env = MujocoArm("push_object")
    try:
        for index in range(n_episodes):
            env.reset(index)
            observations[index, 0] = env.render()
            object_xy[index, 0] = env.tracked_positions()[0, :2]
            for time in range(length - 1):
                action = _action_for(env, index)
                actions[index, time] = action
                env.step(action)
                observations[index, time + 1] = env.render()
                object_xy[index, time + 1] = env.tracked_positions()[0, :2]
    finally:
        env.close()
    np.save(out / "observations.npy", observations)
    np.save(out / "actions.npy", actions)
    np.save(out / "object_xy.npy", object_xy)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "environment_id": "mujoco_arm",
                "package": "mujoco",
                "version": "3.3.7",
                "license": "Apache-2.0",
                "n_episodes": n_episodes,
                "episode_length": length,
                "action_dim": 2,
                "image": [RENDER_SIZE, RENDER_SIZE, 3],
                "kinds": kinds,
                "val_fraction": VAL_FRACTION,
                "split": "contiguous tail is validation, not a test set",
                "commands": {"push": PUSH.tolist(), "fold": FOLD.tolist()},
            },
            indent=2,
        )
        + "\n"
    )
    return out


def validate_arm_dataset(path: str | Path) -> dict[str, Any]:
    """Physical effect, image change, exact replay, and oracle action choice."""
    folder = Path(path)
    observations = np.load(folder / "observations.npy")
    actions = np.load(folder / "actions.npy")
    object_xy = np.load(folder / "object_xy.npy")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    kinds = list(meta["kinds"])
    if observations.shape[0] != len(kinds):
        raise ValueError("metadata kinds do not match the observations.")
    push_move = []
    fold_move = []
    push_mse = []
    fold_mse = []
    for index, kind in enumerate(kinds):
        delta = float(np.linalg.norm(object_xy[index, -1] - object_xy[index, 0]))
        image = observation_visibility(observations[index, 0], observations[index, -1])
        if kind == "push":
            push_move.append(delta)
            push_mse.append(image["mse"])
        elif kind == "fold":
            fold_move.append(delta)
            fold_mse.append(image["mse"])
    if not push_move or not fold_move:
        raise ValueError("the dataset needs both push and fold episodes.")
    if min(push_move) <= 0.05:
        raise ValueError(f"a push episode did not move the object: {min(push_move)}.")
    if max(fold_move) >= 1e-6:
        raise ValueError(f"a fold episode moved the object: {max(fold_move)}.")
    if min(push_mse) <= 0.01 or min(fold_mse) <= 0.01:
        raise ValueError("a scripted episode did not change the image.")
    env = MujocoArm("push_object")
    try:
        env.reset(0)
        if not np.array_equal(env.render(), observations[0, 0]):
            raise ValueError("replaying episode 0 did not match the saved first frame.")
        for time, action in enumerate(actions[0]):
            env.step(action)
            if not np.array_equal(env.render(), observations[0, time + 1]):
                raise ValueError(f"replay diverged at frame {time + 1}.")
            if not np.allclose(env.tracked_positions()[0, :2], object_xy[0, time + 1]):
                raise ValueError(f"replay object position diverged at frame {time + 1}.")
    finally:
        env.close()
    audits = {task: audit_task(task, seed=0) for task in ("push_object", "fold_clear", "cover_post")}
    if audits["push_object"]["planning"]["oracle"]["chosen"] != "push":
        raise ValueError("oracle did not select the push.")
    if audits["fold_clear"]["planning"]["oracle"]["chosen"] != "fold":
        raise ValueError("oracle did not select the fold.")
    if audits["cover_post"]["planning"]["oracle"]["chosen"] != "fold":
        raise ValueError("oracle did not cover the post.")
    n_episodes = int(observations.shape[0])
    n_val = max(1, int(round(n_episodes * VAL_FRACTION))) if n_episodes > 1 else 0
    n_val = min(n_val, n_episodes - 1) if n_episodes > 1 else 0
    return {
        "environment_id": "mujoco_arm",
        "n_episodes": n_episodes,
        "episode_length": int(observations.shape[1]),
        "n_train": n_episodes - n_val,
        "n_val": n_val,
        "is_test_set": False,
        "split_note": "the contiguous tail is validation, not a test set",
        "push_object_displacement_min": float(min(push_move)),
        "fold_object_displacement_max": float(max(fold_move)),
        "push_image_mse_min": float(min(push_mse)),
        "fold_image_mse_min": float(min(fold_mse)),
        "replay_exact": True,
        "oracle_push": audits["push_object"]["planning"]["oracle"]["chosen"],
        "oracle_fold": audits["fold_clear"]["planning"]["oracle"]["chosen"],
        "oracle_cover": audits["cover_post"]["planning"]["oracle"]["chosen"],
        "learned_model": "not_measured",
        "status": "passed",
    }
