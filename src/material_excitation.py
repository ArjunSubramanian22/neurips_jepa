"""Open-loop force schedules for material identifiability.

The waveform depends on time and the rest centroid. It does not read Young's
modulus, mu, or lambda. ``legacy_random`` is the historical state-dependent
sampler and is not an open-loop protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.generate_trajectories import (
    GEOMETRIES,
    MLSMPM2D,
    STIFFNESS_E,
    apply_action_frames,
    episode_rng_seed,
    render_rgb,
    sample_action,
    sample_geometry,
)
from src.environments import EnvironmentSpec
from src.excitation import stability_report

OPEN_LOOP_SCHEDULES = ("orbit", "compression", "push_release", "bend")


def rest_centroid(positions: np.ndarray) -> np.ndarray:
    center = np.asarray(positions, dtype=np.float64).mean(axis=0)
    return np.clip(center, 0.08, 0.92).astype(np.float32)


def open_loop_action(
    schedule: str,
    step: int,
    centroid: np.ndarray,
    *,
    amplitude: float,
    period: int,
    bend_offset: float,
) -> np.ndarray:
    """Return one action. ``step`` is the observation index being executed."""
    if schedule not in OPEN_LOOP_SCHEDULES:
        raise ValueError(f"Unknown open-loop schedule {schedule!r}.")
    if period < 1:
        raise ValueError("period must be >= 1.")
    center = np.asarray(centroid, dtype=np.float32)
    amp = np.float32(amplitude)
    phase = 2.0 * np.pi * (float(step) / float(period))
    action = np.zeros(4, dtype=np.float32)
    action[0] = center[0]
    action[1] = center[1]
    if schedule == "orbit":
        action[2] = amp * np.float32(np.sin(phase))
        action[3] = amp * np.float32(np.cos(phase))
    elif schedule == "compression":
        action[3] = -amp
    elif schedule == "push_release":
        on = (int(step) // int(period)) % 2 == 0
        action[3] = -amp if on else np.float32(0.0)
    elif schedule == "bend":
        action[0] = np.float32(np.clip(float(center[0]) + float(bend_offset), 0.08, 0.92))
        action[2] = amp * np.float32(np.sin(phase))
    return action


def class_geometry(episode_index: int) -> tuple[int, int, float, str]:
    """Same class/geometry cycle as the historical dump generator."""
    class_id = int(episode_index) % len(STIFFNESS_E)
    geom_id = (int(episode_index) // len(STIFFNESS_E)) % len(GEOMETRIES)
    return class_id, geom_id, float(STIFFNESS_E[class_id]), GEOMETRIES[geom_id]


def simulate_episode(
    spec: EnvironmentSpec,
    schedule: str,
    episode_index: int,
    *,
    episode_length: int,
    save_times: tuple[int, ...],
    dump_seed: int,
    amplitude: float,
    period: int,
    bend_offset: float,
    max_particles: int,
    youngs_override: float | None = None,
) -> dict[str, Any]:
    """Roll one episode and keep raster plus mechanical features at ``save_times``."""
    if episode_length < 2:
        raise ValueError("episode_length must be >= 2.")
    class_id, geom_id, youngs, geometry = class_geometry(episode_index)
    if youngs_override is not None:
        youngs = float(youngs_override)
    rng = np.random.default_rng(episode_rng_seed(dump_seed, episode_index))
    sim = MLSMPM2D(spec.mpm)
    points = sample_geometry(geometry, spec.mpm.particle_spacing, rng)
    sim.reset(points, youngs)
    centroid = rest_centroid(sim.x)
    order = np.lexsort((sim.x[:, 1], sim.x[:, 0]))
    if int(sim.x.shape[0]) > int(max_particles):
        order = order[np.linspace(0, order.size - 1, int(max_particles)).astype(np.int64)]
    actions = np.zeros((episode_length - 1, 4), dtype=np.float32)
    saved_mech: list[np.ndarray] = []
    saved_uint8: list[np.ndarray] = []
    finite = True
    for step in range(episode_length):
        if step in save_times:
            saved_mech.append(_pack_mech(sim, order, max_particles))
            saved_uint8.append(render_rgb(sim.x, sim.v, sim.cfg.img_size, sim.cfg.render_sigma))
            if not np.all(np.isfinite(sim.x)):
                finite = False
        if step == episode_length - 1:
            break
        if schedule == "legacy_random":
            if step % int(spec.command_hold) == 0:
                current = sample_action(rng, sim.x, spec.mpm.action_dim)
            action = current
        else:
            action = open_loop_action(
                schedule,
                step,
                centroid,
                amplitude=amplitude,
                period=period,
                bend_offset=bend_offset,
            )
        actions[step] = action
        apply_action_frames(sim, action, 1)
    stable = bool(finite and stability_report(sim)["stable"])
    return {
        "episode_index": int(episode_index),
        "class_id": class_id,
        "geom_id": geom_id,
        "youngs": youngs,
        "mu": float(sim.mu),
        "lam": float(sim.lam),
        "actions": actions,
        "mech": np.stack(saved_mech, axis=0),
        "uint8": np.stack(saved_uint8, axis=0),
        "stable": stable,
        "n_particles": int(points.shape[0]),
    }


def _pack_mech(sim: MLSMPM2D, order: np.ndarray, max_particles: int) -> np.ndarray:
    """Pack x, v, C, F only. Material parameters are not included."""
    take = order[: int(max_particles)]
    feat = np.zeros((int(max_particles), 12), dtype=np.float32)
    n = int(take.size)
    feat[:n, 0:2] = sim.x[take]
    feat[:n, 2:4] = sim.v[take]
    feat[:n, 4:8] = sim.C[take].reshape(n, 4)
    feat[:n, 8:12] = sim.F[take].reshape(n, 4)
    return feat.reshape(-1)
