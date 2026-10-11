"""Generate AC-JEPA physics dumps with a custom 2D MLS-MPM simulator.

Simulator
---------
**Custom Taichi-style MLS-MPM** (Hu et al., 2018; the 88-line Taichi demo).
This is a continuum MPM dough/plasticine analogue used by PhysWorld / PhysTwin
style deformable-object work. A vectorized NumPy backend ships in-repo so
dumps can be produced without installing Taichi, PlasticineLab, or MuJoCo.
When ``taichi`` is importable the same constitutive model can be swapped in
later; the on-disk schema does not change.

Each episode is a contact-conditioned rollout:

* 2 geometries: ``disk`` (blob) and ``bar`` (slab)
* 3 Young's-modulus classes (soft / medium / stiff) → ``stiffness_labels``
* Continuous Young's modulus ``stiffness_values`` (the E used in simulation)
* Actions: ``[contact_x, contact_y, force_x, force_y]`` in unit box
* Observations: rasterized ``(C, H, W)`` RGB-style frames, default 3×64×64

Output is the NPY-directory layout already parsed by
:func:`src.dataset.load_physics_trajectories` / :class:`DeformablePhysicsDataset`::

    <out>/observations.npy      (N, T, C, H, W) uint8
    <out>/actions.npy           (N, T-1, 4) float32
    <out>/stiffness_labels.npy  (N,) int64
    <out>/stiffness_values.npy  (N,) float32  # Young's modulus E
    <out>/geometry_labels.npy   (N,) int64   # extra; ignored by the loader
    <out>/metadata.json

Run::

    python scripts/generate_trajectories.py
    python scripts/generate_trajectories.py --n-episodes 2000 --output outputs/physics_mpm
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FloatArr = npt.NDArray[np.float32]

GEOMETRIES: tuple[str, ...] = ("disk", "bar")
# Three Young's moduli (Pa-like, simulation units). Maps onto n_stiffness_classes=3.
STIFFNESS_E: tuple[float, ...] = (18.0, 70.0, 220.0)
POISSON = 0.2


@dataclass(frozen=True)
class MPMConfig:
    n_grid: int = 32
    dt: float = 1.0e-3
    gravity: float = 0.0
    particle_spacing: float = 0.026
    p_rho: float = 1.0
    action_gain: float = 55.0
    contact_radius: float = 0.10
    damping: float = 0.08
    img_size: int = 64
    in_channels: int = 3
    action_dim: int = 4
    render_sigma: float = 1.35
    substeps: int = 8
    # Defaults reproduce the historical actuator. Other values take a separate path.
    contact_weight_mode: Literal["normalized", "local_peak"] = "normalized"
    force_influence: float = 1.0
    center_influence: float = 1.0

    def __post_init__(self) -> None:
        if self.contact_weight_mode not in ("normalized", "local_peak"):
            raise ValueError("contact_weight_mode must be 'normalized' or 'local_peak'.")
        if not 0.0 <= float(self.center_influence) <= 1.0:
            raise ValueError("center_influence must lie in [0, 1].")
        if float(self.force_influence) <= 0.0:
            raise ValueError("force_influence must be positive.")
        if not 0.0 <= float(self.damping) < 1.0:
            raise ValueError("damping must lie in [0, 1).")
        if float(self.action_gain) <= 0.0 or float(self.contact_radius) <= 0.0:
            raise ValueError("action_gain and contact_radius must be positive.")


def lame(youngs: float, nu: float = POISSON) -> tuple[float, float]:
    mu = youngs / (2.0 * (1.0 + nu))
    lam = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(mu), float(lam)


def sample_geometry(
    name: str,
    spacing: float,
    rng: np.random.Generator,
) -> FloatArr:
    """Return particle rest positions in the unit box for ``disk`` or ``bar``."""
    jitter = 0.25 * spacing
    if name == "disk":
        radius = 0.16
        cx, cy = 0.50, 0.46
        xs = np.arange(cx - radius, cx + radius + spacing * 0.5, spacing)
        ys = np.arange(cy - radius, cy + radius + spacing * 0.5, spacing)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        pts = np.stack([xx[mask], yy[mask]], axis=-1).astype(np.float32)
    elif name == "bar":
        width, height = 0.46, 0.11
        cx, cy = 0.50, 0.40
        xs = np.arange(cx - width / 2, cx + width / 2 + spacing * 0.5, spacing)
        ys = np.arange(cy - height / 2, cy + height / 2 + spacing * 0.5, spacing)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown geometry '{name}'. Expected one of {GEOMETRIES}.")
    pts = pts + rng.normal(0.0, jitter, size=pts.shape).astype(np.float32)
    return np.clip(pts, 0.08, 0.92).astype(np.float32)


class MLSMPM2D:
    """Vectorized 2D MLS-MPM (Taichi 88-line constitutive model)."""

    def __init__(self, cfg: MPMConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else MPMConfig()
        self.n_grid = int(self.cfg.n_grid)
        self.dx = 1.0 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * float(self.cfg.p_rho)
        self.x = np.zeros((0, 2), dtype=np.float32)
        self.v = np.zeros((0, 2), dtype=np.float32)
        self.C = np.zeros((0, 2, 2), dtype=np.float32)
        self.F = np.zeros((0, 2, 2), dtype=np.float32)
        self.mu = 0.0
        self.lam = 0.0

    def reset(self, positions: FloatArr, youngs: float) -> None:
        n = int(positions.shape[0])
        self.x = np.asarray(positions, dtype=np.float32).copy()
        self.v = np.zeros((n, 2), dtype=np.float32)
        self.C = np.zeros((n, 2, 2), dtype=np.float32)
        self.F = np.repeat(np.eye(2, dtype=np.float32)[None, ...], n, axis=0)
        self.mu, self.lam = lame(youngs)

    def snapshot(self) -> dict[str, Any]:
        """Copy the complete particle state. Grid mass/velocity is rebuilt each substep."""
        return {
            "x": self.x.copy(),
            "v": self.v.copy(),
            "C": self.C.copy(),
            "F": self.F.copy(),
            "mu": float(self.mu),
            "lam": float(self.lam),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore particle state from :meth:`snapshot`. Does not copy the config."""
        self.x = np.asarray(snap["x"], dtype=np.float32).copy()
        self.v = np.asarray(snap["v"], dtype=np.float32).copy()
        self.C = np.asarray(snap["C"], dtype=np.float32).copy()
        self.F = np.asarray(snap["F"], dtype=np.float32).copy()
        self.mu = float(snap["mu"])
        self.lam = float(snap["lam"])

    def clone(self) -> "MLSMPM2D":
        """Independent copy that shares only the frozen ``MPMConfig``."""
        other = MLSMPM2D(self.cfg)
        other.restore(self.snapshot())
        return other

    def contact_diagnostics(self, action: FloatArr) -> dict[str, float]:
        """Mechanics-derived contact statistics (predeclared, not result-tuned).

        The actuator always applies the full scaled force after *normalizing*
        Gaussian weights, so a far contact still injects the same total
        impulse, only more uniformly. ``effective_contact`` is therefore
        ``min_particle_distance <= 2 * contact_radius`` (Gaussian support).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        contact = action[:2]
        force = action[2:4] * np.float32(self.cfg.action_gain)
        offset = self.x - contact[None, :]
        dist2 = np.sum(offset * offset, axis=-1)
        radius = float(self.cfg.contact_radius)
        radius2 = np.float32(radius ** 2)
        w_raw = np.exp(-dist2 / (np.float32(2.0) * radius2 + 1e-8))
        z = float(np.sum(w_raw) + 1e-8)
        w = w_raw / z
        min_dist = float(np.sqrt(np.min(dist2) + 1e-12))
        pr = float(1.0 / max(float(np.sum(w * w)), 1e-12))
        return {
            "min_particle_distance": min_dist,
            "raw_weight_sum": z,
            "max_normalized_weight": float(np.max(w)),
            "weight_participation": pr,
            "force_norm": float(np.linalg.norm(force)),
            "contact_radius": radius,
            "effective_contact": float(min_dist <= 2.0 * radius),
        }

    def _uses_legacy_actuator(self) -> bool:
        """True when the historical normalized actuator arithmetic must be used."""
        cfg = self.cfg
        return (
            cfg.contact_weight_mode == "normalized"
            and float(cfg.force_influence) == 1.0
            and float(cfg.center_influence) == 1.0
        )

    def _legacy_action_force(self, action: FloatArr) -> FloatArr:
        """Historical actuator. Total impulse is independent of contact location."""
        contact = action[:2]
        force = action[2:4] * np.float32(self.cfg.action_gain)
        offset = self.x - contact[None, :]
        dist2 = np.sum(offset * offset, axis=-1)
        radius2 = np.float32(self.cfg.contact_radius**2)
        weights = np.exp(-dist2 / (np.float32(2.0) * radius2 + 1e-8))
        weights = weights / (np.sum(weights) + 1e-8)
        return (weights[:, None] * force[None, :]).astype(np.float32)

    def _configured_action_force(self, action: FloatArr) -> FloatArr:
        """Actuator with explicit center influence, force influence, and weight mode.

        ``center_influence=1`` uses the commanded contact. Smaller values pull
        the contact toward the particle centroid, which reduces how much
        ``(cx, cy)`` move the force distribution. ``force_influence`` multiplies
        the force channels on top of ``action_gain``. ``local_peak`` divides
        Gaussian weights by their maximum instead of their sum, so a tight
        contact injects more total impulse than a distant one.
        """
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float32)
        center = action[:2].astype(np.float32, copy=True)
        if float(cfg.center_influence) < 1.0:
            centroid = self.x.mean(axis=0).astype(np.float32)
            mix = np.float32(cfg.center_influence)
            center = (np.float32(1.0) - mix) * centroid + mix * center
        force = action[2:4] * np.float32(cfg.action_gain) * np.float32(cfg.force_influence)
        offset = self.x - center[None, :]
        dist2 = np.sum(offset * offset, axis=-1)
        radius2 = np.float32(cfg.contact_radius**2)
        weights = np.exp(-dist2 / (np.float32(2.0) * radius2 + 1e-8))
        if cfg.contact_weight_mode == "normalized":
            weights = weights / (np.sum(weights) + 1e-8)
        elif cfg.contact_weight_mode == "local_peak":
            weights = weights / (np.max(weights) + 1e-8)
        else:
            raise ValueError(f"Unknown contact_weight_mode {cfg.contact_weight_mode!r}.")
        return (weights[:, None] * force[None, :]).astype(np.float32)

    def _action_force(self, action: FloatArr) -> FloatArr:
        if self._uses_legacy_actuator():
            return self._legacy_action_force(action)
        return self._configured_action_force(action)

    def substep(self, action: FloatArr) -> None:
        cfg = self.cfg
        n_grid = self.n_grid
        dx = self.dx
        inv_dx = self.inv_dx
        dt = np.float32(cfg.dt)
        grid_v = np.zeros((n_grid, n_grid, 2), dtype=np.float32)
        grid_m = np.zeros((n_grid, n_grid), dtype=np.float32)

        base = (self.x * inv_dx - 0.5).astype(np.int32)
        fx = self.x * inv_dx - base.astype(np.float32)
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        w = np.stack([w0, w1, w2], axis=1)  # (N, 3, 2)

        u, s, vh = np.linalg.svd(self.F, full_matrices=False)
        r = u @ vh
        j = np.linalg.det(self.F).astype(np.float32)
        cauchy = (2.0 * self.mu * (self.F - r)) @ np.transpose(self.F, (0, 2, 1))
        cauchy = cauchy + (self.lam * (j - 1.0) * j)[:, None, None] * np.eye(2, dtype=np.float32)
        stress = (-dt * 4.0 * inv_dx * inv_dx * self.p_vol) * cauchy
        affine = stress + self.p_mass * self.C
        momentum = self.p_mass * self.v
        mass = np.float32(self.p_mass)

        for i in range(3):
            for j in range(3):
                dpos = (np.array([i, j], dtype=np.float32) - fx) * dx
                weight = w[:, i, 0] * w[:, j, 1]
                gi = np.clip(base[:, 0] + i, 0, n_grid - 1)
                gj = np.clip(base[:, 1] + j, 0, n_grid - 1)
                mv = weight[:, None] * (momentum + (affine @ dpos[..., None])[..., 0])
                np.add.at(grid_v, (gi, gj), mv)
                np.add.at(grid_m, (gi, gj), weight * mass)

        nonzero = grid_m > 1e-8
        grid_v[nonzero] /= grid_m[nonzero, None]
        grid_v[..., 1] += np.float32(cfg.gravity) * dt
        grid_v[:3] = 0
        grid_v[-3:] = 0
        grid_v[:, :3] = 0
        grid_v[:, -3:] = 0

        new_v = np.zeros_like(self.v)
        new_C = np.zeros_like(self.C)
        for i in range(3):
            for j in range(3):
                dpos = np.array([i, j], dtype=np.float32) - fx
                weight = w[:, i, 0] * w[:, j, 1]
                gi = np.clip(base[:, 0] + i, 0, n_grid - 1)
                gj = np.clip(base[:, 1] + j, 0, n_grid - 1)
                g_v = grid_v[gi, gj]
                new_v += weight[:, None] * g_v
                new_C += 4.0 * inv_dx * weight[:, None, None] * (g_v[:, :, None] * dpos[:, None, :])

        new_v = new_v + dt * self._action_force(action)
        new_v *= np.float32(1.0 - cfg.damping)
        new_F = (np.eye(2, dtype=np.float32)[None, ...] + dt * new_C) @ self.F
        u, sig, vh = np.linalg.svd(new_F, full_matrices=False)
        sig = np.clip(sig, 0.2, 4.0)
        new_F = (u * sig[:, None, :]) @ vh
        self.v = new_v.astype(np.float32)
        self.C = new_C.astype(np.float32)
        self.F = new_F.astype(np.float32)
        self.x = np.clip(self.x + dt * self.v, 0.02, 0.98).astype(np.float32)


def render_rgb(
    positions: FloatArr,
    velocities: FloatArr,
    img_size: int,
    sigma: float,
) -> npt.NDArray[np.uint8]:
    """Local Gaussian splat → uint8 ``(3, H, W)`` (density, v_x, v_y)."""
    h = int(img_size)
    density = np.zeros((h, h), dtype=np.float32)
    mom_x = np.zeros((h, h), dtype=np.float32)
    mom_y = np.zeros((h, h), dtype=np.float32)
    px = positions[:, 0] * np.float32(h - 1)
    py = positions[:, 1] * np.float32(h - 1)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    inv_two_s2 = 1.0 / (2.0 * sigma * sigma + 1e-8)
    ix = np.rint(px).astype(np.int32)
    iy = np.rint(py).astype(np.int32)
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    for dy in range(-radius, radius + 1):
        yy = np.clip(iy + dy, 0, h - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(ix + dx, 0, h - 1)
            w = np.exp(-(np.float32(dx * dx + dy * dy)) * inv_two_s2)
            np.add.at(density, (yy, xx), w)
            np.add.at(mom_x, (yy, xx), w * vx)
            np.add.at(mom_y, (yy, xx), w * vy)
    mass = density + 1e-8
    ch0 = density / (density.max() + 1e-8)
    ch1 = 0.5 + 0.5 * np.tanh(mom_x / mass)
    ch2 = 0.5 + 0.5 * np.tanh(mom_y / mass)
    rgb = np.stack([ch0, ch1, ch2], axis=0)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def sample_action(
    rng: np.random.Generator,
    positions: FloatArr,
    action_dim: int = 4,
) -> FloatArr:
    """Draw ``[contact_x, contact_y, force_x, force_y]`` near the object."""
    action = np.zeros((action_dim,), dtype=np.float32)
    centroid = positions.mean(axis=0)
    span = positions.max(axis=0) - positions.min(axis=0) + 1e-3
    contact = centroid + rng.uniform(-0.35, 0.35, size=2).astype(np.float32) * span
    action[:2] = np.clip(contact, 0.08, 0.92)
    action[2:4] = rng.normal(0.0, 0.40, size=2).astype(np.float32)
    return action


def generate_episode(
    sim: MLSMPM2D,
    *,
    geometry: str,
    youngs: float,
    episode_length: int,
    rng: np.random.Generator,
    hold: int = 3,
) -> tuple[npt.NDArray[np.uint8], FloatArr]:
    cfg = sim.cfg
    pts = sample_geometry(geometry, cfg.particle_spacing, rng)
    sim.reset(pts, youngs)
    frames = np.empty((episode_length, cfg.in_channels, cfg.img_size, cfg.img_size), dtype=np.uint8)
    actions = np.empty((episode_length - 1, cfg.action_dim), dtype=np.float32)
    frames[0] = render_rgb(sim.x, sim.v, cfg.img_size, cfg.render_sigma)
    current = sample_action(rng, sim.x, cfg.action_dim)
    for t in range(episode_length - 1):
        if t % hold == 0:
            current = sample_action(rng, sim.x, cfg.action_dim)
        actions[t] = current
        for _ in range(cfg.substeps):
            sim.substep(current)
        frames[t + 1] = render_rgb(sim.x, sim.v, cfg.img_size, cfg.render_sigma)
    return frames, actions


def episode_rng_seed(dump_seed: int, episode_index: int) -> int:
    """Per-episode RNG seed used by :func:`generate_dataset`."""
    return int(dump_seed) + 10007 * int(episode_index)


def episode_class_geom(episode_index: int) -> tuple[int, int]:
    """Stiffness class id and geometry id for ``episode_index`` (dump layout)."""
    n_class = len(STIFFNESS_E)
    n_geom = len(GEOMETRIES)
    class_id = int(episode_index % n_class)
    geom_id = int((episode_index // n_class) % n_geom)
    return class_id, geom_id


def apply_action_frames(sim: MLSMPM2D, action: FloatArr, n_frames: int) -> None:
    """Advance ``n_frames`` observation steps under a constant command."""
    if n_frames < 1:
        return
    action = np.asarray(action, dtype=np.float32)
    for _ in range(int(n_frames)):
        for _sub in range(sim.cfg.substeps):
            sim.substep(action)


def replay_episode(
    episode_index: int,
    *,
    dump_seed: int = 0,
    episode_length: int = 16,
    mpm: MPMConfig | None = None,
    hold: int = 3,
) -> dict[str, Any]:
    """Re-simulate one dump episode, capturing a snapshot before every action.

    Returns frames, actions, geometry/stiffness labels, and ``snapshots[t]``
    for ``t = 0 .. T-2`` (state immediately before applying ``actions[t]``),
    plus ``snapshots[T-1]`` after the last command. Dynamics in ``substep``
    are deterministic given ``(x, v, C, F, μ, λ, a)``; the RNG is used only
    for rest-shape jitter and action sampling.
    """
    class_id, geom_id = episode_class_geom(episode_index)
    rng = np.random.default_rng(episode_rng_seed(dump_seed, episode_index))
    sim = MLSMPM2D(mpm if mpm is not None else MPMConfig())
    geometry = GEOMETRIES[geom_id]
    youngs = STIFFNESS_E[class_id]
    pts = sample_geometry(geometry, sim.cfg.particle_spacing, rng)
    sim.reset(pts, youngs)
    frames = np.empty(
        (episode_length, sim.cfg.in_channels, sim.cfg.img_size, sim.cfg.img_size),
        dtype=np.uint8,
    )
    actions = np.empty((episode_length - 1, sim.cfg.action_dim), dtype=np.float32)
    snapshots: list[dict[str, Any]] = []
    frames[0] = render_rgb(sim.x, sim.v, sim.cfg.img_size, sim.cfg.render_sigma)
    snapshots.append(sim.snapshot())
    current = sample_action(rng, sim.x, sim.cfg.action_dim)
    for t in range(episode_length - 1):
        if t % hold == 0:
            current = sample_action(rng, sim.x, sim.cfg.action_dim)
        actions[t] = current
        apply_action_frames(sim, current, 1)
        frames[t + 1] = render_rgb(sim.x, sim.v, sim.cfg.img_size, sim.cfg.render_sigma)
        snapshots.append(sim.snapshot())
    return {
        "episode_index": int(episode_index),
        "frames": frames,
        "actions": actions,
        "snapshots": snapshots,
        "class_id": class_id,
        "geom_id": geom_id,
        "geometry": geometry,
        "youngs": float(youngs),
        "hold": int(hold),
        "dump_seed": int(dump_seed),
    }


def _simulate_index(
    item: tuple[int, int, int, int, int, dict[str, Any]],
) -> tuple[int, npt.NDArray[np.uint8], FloatArr, int, int]:
    """Worker entry: one episode (picklable for ProcessPoolExecutor)."""
    ep, seed, geom_id, class_id, episode_length, cfg_kwargs = item
    rng = np.random.default_rng(seed)
    sim = MLSMPM2D(MPMConfig(**cfg_kwargs))
    frames, acts = generate_episode(
        sim,
        geometry=GEOMETRIES[geom_id],
        youngs=STIFFNESS_E[class_id],
        episode_length=episode_length,
        rng=rng,
    )
    return ep, frames, acts, class_id, geom_id


def generate_dataset(
    output_dir: str | Path,
    *,
    n_episodes: int = 2000,
    episode_length: int = 16,
    seed: int = 0,
    img_size: int = 64,
    mpm: MPMConfig | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Simulate ``n_episodes`` and write an NPY dump :class:`DeformablePhysicsDataset` can load."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1.")
    if episode_length < 2:
        raise ValueError("episode_length must be >= 2.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = mpm if mpm is not None else MPMConfig(img_size=img_size)
    if img_size != cfg.img_size:
        cfg = MPMConfig(**{**asdict(cfg), "img_size": img_size})
    n_class = len(STIFFNESS_E)
    n_geom = len(GEOMETRIES)
    cfg_kwargs = asdict(cfg)
    jobs = []
    for ep in range(n_episodes):
        class_id = int(ep % n_class)
        geom_id = int((ep // n_class) % n_geom)
        jobs.append((ep, int(seed) + 10007 * ep, geom_id, class_id, episode_length, cfg_kwargs))

    observations = np.empty(
        (n_episodes, episode_length, cfg.in_channels, cfg.img_size, cfg.img_size),
        dtype=np.uint8,
    )
    actions = np.empty((n_episodes, episode_length - 1, cfg.action_dim), dtype=np.float32)
    stiffness = np.empty((n_episodes,), dtype=np.int64)
    geometry = np.empty((n_episodes,), dtype=np.int64)

    if workers is None:
        workers = 1 if n_episodes <= 12 else min(8, max(1, (os.cpu_count() or 2) - 1))
    workers = max(1, int(workers))
    if workers > 1 and os.name != "nt":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    started = time.perf_counter()
    if workers == 1:
        iterator = (_simulate_index(job) for job in jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_simulate_index, jobs, chunksize=max(1, n_episodes // (workers * 4)))
    n_done = 0
    try:
        for ep, frames, acts, class_id, geom_id in iterator:
            observations[ep] = frames
            actions[ep] = acts
            stiffness[ep] = class_id
            geometry[ep] = geom_id
            n_done += 1
            if n_done % max(1, n_episodes // 10) == 0 or n_done == 1:
                elapsed = time.perf_counter() - started
                print(f"  episode {n_done}/{n_episodes}  ({elapsed:.1f}s)", flush=True)
    finally:
        if workers > 1:
            pool.shutdown(wait=True)
    stiffness_values = np.asarray(STIFFNESS_E, dtype=np.float32)[stiffness]
    np.save(output_dir / "observations.npy", observations)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "stiffness_labels.npy", stiffness)
    np.save(output_dir / "stiffness_values.npy", stiffness_values)
    np.save(output_dir / "geometry_labels.npy", geometry)
    meta = {
        "simulator": "taichi_mls_mpm_2d",
        "backend": "numpy",
        "constitutive": "MLS-MPM neo-Hookean (Hu et al. 2018 / Taichi 88-line)",
        "n_episodes": int(n_episodes),
        "episode_length": int(episode_length),
        "img_size": int(cfg.img_size),
        "in_channels": int(cfg.in_channels),
        "action_dim": int(cfg.action_dim),
        "action_layout": ["contact_x", "contact_y", "force_x", "force_y"],
        "geometries": list(GEOMETRIES),
        "stiffness_E": list(STIFFNESS_E),
        "n_stiffness_classes": n_class,
        "seed": int(seed),
        "workers": int(workers),
        "seconds": float(time.perf_counter() - started),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir}  N={n_episodes} T={episode_length}  in {meta['seconds']:.1f}s")
    return meta


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MLS-MPM trajectory dumps for AC-JEPA.")
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--output", default="outputs/physics_mpm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="Process workers; 0 = auto.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run scripts/validate_dataset.py after writing the dump.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    meta = generate_dataset(
        args.output,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
        img_size=args.img_size,
        workers=None if args.workers == 0 else args.workers,
    )
    if args.validate:
        from scripts.validate_dataset import validate_physics_dataset

        report = validate_physics_dataset(
            args.output,
            report_path=_ROOT / "outputs" / "dataset_report.json",
        )
        meta["validation"] = report
    return meta


if __name__ == "__main__":
    main(sys.argv[1:])
