"""Goal-reaching tasks whose targets are produced by an action, not the initial shape."""

from __future__ import annotations

from typing import Any

import numpy as np

TASKS = ("center_target", "region_target", "configuration", "bar_orientation")
SUCCESS_FRACTION = 0.25
SCRIPTED_FORCE = 0.8


def principal_angle(points: Any) -> float:
    """Undirected angle of the leading principal axis."""
    cloud = np.asarray(points, dtype=np.float64)
    centered = cloud - cloud.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    return float(np.arctan2(axis[1], axis[0]))


def angle_distance(left: float, right: float) -> float:
    """Smallest undirected difference, in ``[0, pi/2]``."""
    delta = (float(left) - float(right) + np.pi / 2.0) % np.pi - np.pi / 2.0
    return abs(float(delta))


def task_distance(name: str, positions: Any, goal_positions: Any, region: Any) -> float:
    """Physical distance for one named task. Lower is closer to the goal."""
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}.")
    current = np.asarray(positions, dtype=np.float64)
    goal = np.asarray(goal_positions, dtype=np.float64)
    if current.shape != goal.shape or current.ndim != 2:
        raise ValueError("positions and goal_positions must share shape (N, D).")
    if name == "center_target":
        return float(np.linalg.norm(current.mean(axis=0) - goal.mean(axis=0)))
    if name == "region_target":
        index = np.asarray(region, dtype=np.int64).reshape(-1)
        if index.size < 1:
            raise ValueError("region_target requires at least one particle.")
        return float(np.linalg.norm(current[index].mean(axis=0) - goal[index].mean(axis=0)))
    if name == "configuration":
        return float(np.sqrt(np.mean((current - goal) ** 2)))
    return angle_distance(principal_angle(current), principal_angle(goal))


def designated_region(start_positions: Any) -> np.ndarray:
    """Particles on the high-x side of the initial object. Fixed for the episode."""
    start = np.asarray(start_positions, dtype=np.float64)
    median = float(np.median(start[:, 0]))
    index = np.flatnonzero(start[:, 0] >= median)
    if index.size < 1:
        index = np.arange(start.shape[0])
    return index.astype(np.int64)


def make_reachable_goal(
    name: str,
    *,
    seed: int,
    scripted_steps: int,
    mpm: Any = None,
) -> dict[str, Any]:
    """Build a start state and a goal reached by a scripted contact force.

    The scripted action is stored so a test can replay it. Planners receive
    the goal state and goal image, not the action that produced them.
    """
    from scripts.generate_trajectories import (
        MLSMPM2D,
        MPMConfig,
        apply_action_frames,
        render_rgb,
        sample_geometry,
    )

    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}.")
    if int(scripted_steps) < 1:
        raise ValueError("scripted_steps must be >= 1.")
    cfg = mpm if mpm is not None else MPMConfig()
    geometry = "bar" if name == "bar_orientation" else "disk"
    rng = np.random.default_rng(int(seed))
    sim = MLSMPM2D(cfg)
    sim.reset(sample_geometry(geometry, cfg.particle_spacing, rng), 70.0)
    centroid = sim.x.mean(axis=0)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    force = np.float32(SCRIPTED_FORCE) * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    contact = np.clip(centroid, 0.08, 0.92).astype(np.float32)
    action = np.array([contact[0], contact[1], force[0], force[1]], dtype=np.float32)
    start = sim.x.copy()
    region = designated_region(start)
    start_frame = render_rgb(sim.x, sim.v, cfg.img_size, cfg.render_sigma)
    branch = sim.clone()
    for _ in range(int(scripted_steps)):
        apply_action_frames(branch, action, 1)
    goal = branch.x.copy()
    goal_frame = render_rgb(branch.x, branch.v, cfg.img_size, cfg.render_sigma)
    initial = task_distance(name, start, goal, region)
    return {
        "task": name,
        "geometry": geometry,
        "sim": sim,
        "goal_positions": goal,
        "region": region,
        "start_frame": start_frame,
        "goal_frame": goal_frame,
        "scripted_action": action,
        "scripted_steps": int(scripted_steps),
        "initial_distance": float(initial),
        "observationally_ambiguous": bool(np.array_equal(start_frame, goal_frame)),
        "seed": int(seed),
    }
