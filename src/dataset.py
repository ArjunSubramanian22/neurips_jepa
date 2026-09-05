"""Synthetic and physical-simulation datasets for AC-JEPA.

Each training sample is a transition ``(x_t, a_t, x_{t+1})``:

* ``x_t``     — observation frame, unbatched shape ``(C, H, W)``
* ``a_t``     — action vector, unbatched shape ``(A_dim,)``
* ``x_{t+1}`` — future observation, unbatched shape ``(C, H, W)``

When ``DataConfig.dataset_path`` is ``None``, :class:`DeformableObjectDataset`
synthesizes mass-spring rollouts. Otherwise :class:`DeformablePhysicsDataset`
ingests HDF5 / NPY / NPZ dumps from Taichi, PlasticineLab, or MuJoCo
soft-body environments, chunks them into windows of length ``T``, applies
time-consistent vision augmentations, and z-score-normalizes actions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .config import DataConfig, TrainingConfig
else:
    from config import DataConfig, TrainingConfig

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
Split = Literal["train", "val", "all"]
Transition = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _grid_positions(n_grid: int, spatial_dims: int) -> FloatArray:
    """Build a regular lattice occupying the central 50% of the unit box."""
    axis = np.linspace(0.25, 0.75, n_grid, dtype=np.float32)
    meshes = np.meshgrid(*([axis] * spatial_dims), indexing="ij")
    stacked = np.stack(meshes, axis=-1).reshape(-1, spatial_dims)
    return stacked.astype(np.float32, copy=False)


def _edge_list(n_grid: int, spatial_dims: int) -> IntArray:
    """Structural (axis) and shear (planar diagonal) spring connectivity."""
    edges: list[tuple[int, int]] = []

    def index(coord: tuple[int, ...]) -> int:
        idx = 0
        stride = 1
        for c in reversed(coord):
            idx += c * stride
            stride *= n_grid
        return idx

    def in_bounds(coord: tuple[int, ...]) -> bool:
        return all(0 <= c < n_grid for c in coord)

    axis_offsets: list[tuple[int, ...]]
    shear_offsets: list[tuple[int, ...]]
    if spatial_dims == 2:
        axis_offsets = [(1, 0), (0, 1)]
        shear_offsets = [(1, 1), (1, -1)]
    else:
        axis_offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        shear_offsets = [
            (1, 1, 0),
            (1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
        ]

    ranges = [range(n_grid)] * spatial_dims
    # Cartesian product without itertools to keep the module self-contained.
    coords: list[tuple[int, ...]] = [()]
    for r in ranges:
        coords = [prefix + (v,) for prefix in coords for v in r]

    seen: set[tuple[int, int]] = set()
    for coord in coords:
        i = index(coord)
        for offset in axis_offsets + shear_offsets:
            nbr = tuple(c + o for c, o in zip(coord, offset))
            if not in_bounds(nbr):
                continue
            j = index(nbr)
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return np.asarray(edges, dtype=np.int64)


class MassSpringSimulator:
    """Explicit semi-implicit Euler mass-spring system on a regular lattice."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.rest_pos = _grid_positions(cfg.n_grid, cfg.spatial_dims)
        self.edges = _edge_list(cfg.n_grid, cfg.spatial_dims)
        i, j = self.edges[:, 0], self.edges[:, 1]
        rest = np.linalg.norm(self.rest_pos[j] - self.rest_pos[i], axis=-1)
        self.rest_length = rest.astype(np.float32)
        self._i = i
        self._j = j

    def reset(self, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        jitter = rng.standard_normal(self.rest_pos.shape).astype(np.float32)
        positions = np.clip(self.rest_pos + 0.01 * jitter, 0.05, 0.95)
        velocities = np.zeros_like(positions)
        return positions, velocities

    def _spring_forces(self, positions: FloatArray, stiffness: float | None = None) -> FloatArray:
        spring_k = np.float32(self.cfg.stiffness if stiffness is None else stiffness)
        delta = positions[self._j] - positions[self._i]
        dist = np.linalg.norm(delta, axis=-1, keepdims=True).astype(np.float32)
        direction = delta / (dist + 1e-8)
        stretch = (dist[..., 0] - self.rest_length).astype(np.float32)
        hookean = (spring_k * stretch)[:, None] * direction
        forces = np.zeros_like(positions)
        np.add.at(forces, self._i, hookean)
        np.add.at(forces, self._j, -hookean)
        return forces

    def _wall_forces(self, positions: FloatArray) -> FloatArray:
        margin = np.float32(0.06)
        wall_k = np.float32(120.0)
        lower = np.clip(margin - positions, 0.0, None)
        upper = np.clip(positions - (np.float32(1.0) - margin), 0.0, None)
        return wall_k * lower - wall_k * upper

    def _action_forces(self, positions: FloatArray, action: FloatArray) -> FloatArray:
        cfg = self.cfg
        dims = cfg.spatial_dims
        contact = action[:dims]
        force = action[dims : 2 * dims] * np.float32(cfg.action_gain)
        offset = positions - contact[None, :]
        dist2 = np.sum(offset * offset, axis=-1)
        radius2 = np.float32(cfg.contact_radius**2)
        weights = np.exp(-dist2 / (np.float32(2.0) * radius2 + 1e-8))
        weights = weights / (np.sum(weights) + 1e-8)
        return (weights[:, None] * force[None, :]).astype(np.float32)

    def substep(
        self,
        positions: FloatArray,
        velocities: FloatArray,
        action: FloatArray,
        stiffness: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        cfg = self.cfg
        dt = np.float32(cfg.internal_dt)
        forces = self._spring_forces(positions, stiffness=stiffness)
        forces += self._wall_forces(positions)
        forces += self._action_forces(positions, action)
        gravity = np.zeros(cfg.spatial_dims, dtype=np.float32)
        gravity[-1] = np.float32(-cfg.gravity)
        accel = forces / np.float32(cfg.mass) + gravity[None, :]
        velocities = (velocities + dt * accel) * np.float32(1.0 - cfg.damping)
        positions = np.clip(positions + dt * velocities, 0.0, 1.0)
        return positions, velocities

    def sample_action(self, rng: np.random.Generator, positions: FloatArray) -> FloatArray:
        cfg = self.cfg
        dims = cfg.spatial_dims
        action = np.zeros(cfg.action_dim, dtype=np.float32)
        centroid = positions.mean(axis=0)
        span = positions.max(axis=0) - positions.min(axis=0) + 1e-3
        contact = centroid + rng.uniform(-0.35, 0.35, size=dims).astype(np.float32) * span
        action[:dims] = np.clip(contact, 0.1, 0.9)
        action[dims : 2 * dims] = rng.normal(0.0, 0.35, size=dims).astype(np.float32)
        if cfg.action_dim > 2 * dims:
            action[2 * dims :] = rng.uniform(-1.0, 1.0, size=cfg.action_dim - 2 * dims).astype(
                np.float32
            )
        return action


def render_frame(
    positions: FloatArray,
    velocities: FloatArray,
    img_size: int,
    in_channels: int,
    sigma: float,
) -> FloatArray:
    """Splat particles into a ``(C, H, W)`` observation.

    Channel layout (prefix of ``in_channels``):

    0. mass density
    1. image-plane velocity ``u`` (tanh-compressed)
    2. image-plane velocity ``v`` (tanh-compressed)
    3. mean depth (3D) or speed (2D)
    """
    spatial_dims = positions.shape[-1]
    uv = positions[:, :2]
    pixels = uv * np.float32(img_size - 1)
    u = pixels[:, 0]
    v = pixels[:, 1]

    ys = np.arange(img_size, dtype=np.float32)
    xs = np.arange(img_size, dtype=np.float32)
    dx = xs[None, None, :] - u[:, None, None]
    dy = ys[None, :, None] - v[:, None, None]
    kernel = np.exp(-(dx * dx + dy * dy) / (np.float32(2.0) * np.float32(sigma) ** 2 + 1e-8))

    if spatial_dims == 3:
        # Closer particles (smaller z) contribute more, a cheap occlusion prior.
        depth_w = (np.float32(1.25) - positions[:, 2]).clip(0.25, 1.25)[:, None, None]
        kernel = kernel * depth_w

    density = kernel.sum(axis=0)
    density = density / (density.max() + 1e-8)

    vx = (kernel * velocities[:, 0][:, None, None]).sum(axis=0)
    vy = (kernel * velocities[:, 1][:, None, None]).sum(axis=0)
    mass = kernel.sum(axis=0) + 1e-8
    vx = np.tanh(vx / mass)
    vy = np.tanh(vy / mass)

    channels: list[FloatArray] = [density.astype(np.float32)]
    if in_channels >= 2:
        channels.append(vx.astype(np.float32))
    if in_channels >= 3:
        channels.append(vy.astype(np.float32))
    if in_channels >= 4:
        if spatial_dims == 3:
            depth = (kernel * positions[:, 2][:, None, None]).sum(axis=0) / mass
            channels.append(depth.astype(np.float32))
        else:
            speed = np.sqrt(vx * vx + vy * vy).astype(np.float32)
            channels.append(speed)
    while len(channels) < in_channels:
        channels.append(np.zeros((img_size, img_size), dtype=np.float32))
    frame = np.stack(channels[:in_channels], axis=0)
    return np.clip(frame, -1.0, 1.0).astype(np.float32)


def _stiffness_levels(cfg: DataConfig) -> list[float]:
    n_classes = cfg.n_stiffness_classes
    if n_classes == 1:
        return [float(cfg.stiffness)]
    multipliers = np.linspace(0.45, 1.8, n_classes, dtype=np.float32)
    return [float(cfg.stiffness * m) for m in multipliers]


def resolve_stiffness_values(
    labels: torch.Tensor,
    *,
    source: str | Path | None = None,
    codebook: Sequence[float] | None = None,
) -> torch.Tensor:
    """Continuous stiffness (Young's modulus / spring k) aligned with class labels.

    Preference order: on-disk ``stiffness_values`` array, ``metadata.json``
    ``stiffness_E`` lookup, then ``codebook[label]``. Always returns ``(N,)``.
    """
    n_ep = int(labels.reshape(-1).shape[0])
    values: torch.Tensor | None = None
    path = Path(source) if source is not None else None
    if path is not None and path.is_dir():
        npy = path / "stiffness_values.npy"
        if npy.is_file():
            values = torch.as_tensor(np.load(npy, allow_pickle=False)).float().reshape(-1)
        meta_path = path / "metadata.json"
        if codebook is None and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            raw = meta.get("stiffness_E") or meta.get("stiffness_levels")
            if isinstance(raw, (list, tuple)) and raw:
                codebook = [float(v) for v in raw]
    elif path is not None and path.is_file() and path.suffix.lower() == ".npz":
        with np.load(path) as handle:
            for key in _VALUE_KEYS:
                if key in handle:
                    values = torch.as_tensor(handle[key]).float().reshape(-1)
                    break
    if values is not None:
        if int(values.shape[0]) != n_ep:
            raise ValueError(
                f"stiffness_values length {int(values.shape[0])} != episode count {n_ep}."
            )
        return values.contiguous()
    if codebook is None or len(codebook) < 1:
        n_classes = int(max(int(labels.max().item()) + 1, 1)) if labels.numel() else 1
        codebook = [float(i) for i in range(n_classes)]
    table = torch.tensor([float(v) for v in codebook], dtype=torch.float32)
    idx = labels.long().reshape(-1).clamp(0, table.numel() - 1)
    return table[idx].contiguous()


def _load_trajectory_archive(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Load ``observations`` ``(N, T, C, H, W)`` and ``actions`` ``(N, T-1, A)``."""
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory archive not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as handle:
            if "observations" not in handle or "actions" not in handle:
                raise KeyError(
                    f"{path} must contain arrays 'observations' and 'actions'."
                )
            observations = torch.from_numpy(np.asarray(handle["observations"]))
            actions = torch.from_numpy(np.asarray(handle["actions"]))
            raw_labels = (
                np.array(handle["stiffness_labels"]) if "stiffness_labels" in handle else None
            )
    elif suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "observations" not in payload or "actions" not in payload:
            raise KeyError(f"{path} must be a dict with keys 'observations' and 'actions'.")
        observations = payload["observations"]
        actions = payload["actions"]
        raw_labels = payload.get("stiffness_labels")
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
    else:
        raise ValueError(f"Unsupported archive type '{path.suffix}'; use .npz, .pt, or .pth.")

    observations = observations.float().contiguous()
    actions = actions.float().contiguous()
    if observations.ndim != 5:
        raise ValueError(
            "observations must have shape (N, T, C, H, W); "
            f"got {tuple(observations.shape)}."
        )
    if actions.ndim != 3:
        raise ValueError(
            "actions must have shape (N, T-1, A_dim); "
            f"got {tuple(actions.shape)}."
        )
    n_obs, t_obs = observations.shape[0], observations.shape[1]
    n_act, t_act = actions.shape[0], actions.shape[1]
    if n_obs != n_act:
        raise ValueError(f"Episode count mismatch: observations N={n_obs}, actions N={n_act}.")
    if t_obs != t_act + 1:
        raise ValueError(
            f"Time mismatch: observations T={t_obs} requires actions T-1={t_obs - 1}, "
            f"got {t_act}."
        )
    labels: torch.Tensor | None
    if raw_labels is None:
        labels = None
    else:
        labels = torch.as_tensor(raw_labels).long().reshape(-1)
        if labels.shape[0] != n_obs:
            raise ValueError(
                f"stiffness_labels length {labels.shape[0]} != episode count {n_obs}."
            )
    return observations, actions, labels


class DeformableObjectDataset(Dataset[Transition]):
    """PyTorch dataset of action-conditioned deformable-object transitions.

    Args:
        cfg: Data generation / loading configuration.
        split: Episode subset. ``train`` / ``val`` slice along the episode axis;
            ``all`` keeps every generated (or loaded) episode.
    """

    def __init__(self, cfg: DataConfig | None = None, split: Split = "all") -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else DataConfig()
        self.split = split
        if self.cfg.dataset_path is None:
            observations, actions, stiffness_labels, positions, velocities = self._simulate_all()
        else:
            observations, actions, loaded_labels = _load_trajectory_archive(Path(self.cfg.dataset_path))
            self._validate_loaded_shapes(observations, actions)
            if loaded_labels is None:
                stiffness_labels = torch.zeros(observations.shape[0], dtype=torch.long)
            else:
                stiffness_labels = loaded_labels
            positions, velocities = None, None
        stiffness_values = resolve_stiffness_values(
            stiffness_labels,
            source=self.cfg.dataset_path,
            codebook=_stiffness_levels(self.cfg),
        )
        n_full = int(observations.shape[0])
        n_train_ep, n_val_ep = _episode_split_counts(n_full, self.cfg.val_fraction)
        self.observations, self.actions, self.stiffness_labels, self.stiffness_values = self._apply_split(
            observations, actions, stiffness_labels, stiffness_values
        )
        if positions is not None and velocities is not None:
            if self.split == "train":
                positions, velocities = positions[:n_train_ep], velocities[:n_train_ep]
            elif self.split == "val":
                positions, velocities = positions[n_train_ep:], velocities[n_train_ep:]
            self.positions = positions.contiguous()
            self.velocities = velocities.contiguous()
        else:
            self.positions = None
            self.velocities = None
        self.pairs_per_episode = int(self.observations.shape[1] - 1)
        self.n_episodes = int(self.observations.shape[0])

    def _validate_loaded_shapes(self, observations: torch.Tensor, actions: torch.Tensor) -> None:
        _, _, channels, height, width = observations.shape
        action_dim = actions.shape[-1]
        cfg = self.cfg
        if (height, width) != (cfg.img_size, cfg.img_size):
            raise ValueError(
                f"Loaded frames are {height}x{width} but DataConfig.img_size={cfg.img_size}."
            )
        if channels != cfg.in_channels:
            raise ValueError(
                f"Loaded C={channels} but DataConfig.in_channels={cfg.in_channels}."
            )
        if action_dim != cfg.action_dim:
            raise ValueError(
                f"Loaded A_dim={action_dim} but DataConfig.action_dim={cfg.action_dim}."
            )

    def _apply_split(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.split == "all":
            return observations, actions, labels, values
        n_episodes = observations.shape[0]
        n_val = max(1, int(round(n_episodes * self.cfg.val_fraction))) if n_episodes > 1 else 0
        n_val = min(n_val, n_episodes - 1) if n_episodes > 1 else 0
        n_train = n_episodes - n_val
        if self.split == "train":
            return observations[:n_train], actions[:n_train], labels[:n_train], values[:n_train]
        if self.split == "val":
            if n_val == 0:
                raise ValueError("val split is empty; increase n_episodes or val_fraction.")
            return observations[n_train:], actions[n_train:], labels[n_train:], values[n_train:]
        raise ValueError(f"Unknown split '{self.split}'.")

    def _simulate_all(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        sim = MassSpringSimulator(cfg)
        levels = _stiffness_levels(cfg)
        n_classes = len(levels)
        n, t, c, h = cfg.n_episodes, cfg.episode_length, cfg.in_channels, cfg.img_size
        n_particles = cfg.n_particles
        dims = cfg.spatial_dims
        obs = np.empty((n, t, c, h, h), dtype=np.float32)
        act = np.empty((n, t - 1, cfg.action_dim), dtype=np.float32)
        labels = np.empty((n,), dtype=np.int64)
        positions_out = np.empty((n, t, n_particles, dims), dtype=np.float32)
        velocities_out = np.empty((n, t, n_particles, dims), dtype=np.float32)

        for ep in range(n):
            class_id = int(rng.integers(0, n_classes))
            labels[ep] = class_id
            stiffness = levels[class_id]
            positions, velocities = sim.reset(rng)
            positions_out[ep, 0] = positions
            velocities_out[ep, 0] = velocities
            obs[ep, 0] = render_frame(positions, velocities, h, c, cfg.gaussian_sigma)
            for step in range(t - 1):
                action = sim.sample_action(rng, positions)
                act[ep, step] = action
                for _ in range(cfg.substeps):
                    positions, velocities = sim.substep(
                        positions, velocities, action, stiffness=stiffness
                    )
                positions_out[ep, step + 1] = positions
                velocities_out[ep, step + 1] = velocities
                obs[ep, step + 1] = render_frame(
                    positions, velocities, h, c, cfg.gaussian_sigma
                )
        return (
            torch.from_numpy(obs),
            torch.from_numpy(act),
            torch.from_numpy(labels),
            torch.from_numpy(positions_out),
            torch.from_numpy(velocities_out),
        )

    def split_sizes(self) -> tuple[int, int]:
        """Return ``(n_train_episodes, n_val_episodes)`` for the configured fraction."""
        n_episodes = int(self.observations.shape[0]) if self.split == "all" else self.n_episodes
        if self.split != "all":
            # Already sliced; report this split vs the complement is not available.
            return (self.n_episodes, 0) if self.split == "train" else (0, self.n_episodes)
        n_val = max(1, int(round(n_episodes * self.cfg.val_fraction))) if n_episodes > 1 else 0
        n_val = min(n_val, n_episodes - 1) if n_episodes > 1 else 0
        return n_episodes - n_val, n_val

    def indices_for_episodes(self, start: int, stop: int) -> list[int]:
        """Map an episode half-open interval to flat transition indices."""
        pairs = self.pairs_per_episode
        start = max(0, start)
        stop = min(self.n_episodes, stop)
        return list(range(start * pairs, stop * pairs))

    def __len__(self) -> int:
        return self.n_episodes * self.pairs_per_episode

    def __getitem__(self, index: int) -> Transition:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = index // self.pairs_per_episode
        t = index % self.pairs_per_episode
        x_t = self.observations[episode, t]
        a_t = self.actions[episode, t]
        x_tp1 = self.observations[episode, t + 1]
        return x_t, a_t, x_tp1

    def sample_batch(self, batch_size: int, rng: torch.Generator | None = None) -> Transition:
        """Draw a stacked mini-batch with shapes ``(B, C, H, W)``, ``(B, A)``, ``(B, C, H, W)``."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        indices = torch.randint(len(self), (batch_size,), generator=rng)
        frames, actions, futures = zip(*(self[int(i)] for i in indices))
        return torch.stack(frames, 0), torch.stack(actions, 0), torch.stack(futures, 0)


class ParticleTransitionDataset(Dataset[tuple[torch.Tensor, ...]]):
    """``(pos_t, vel_t, a_t, pos_{t+1}, vel_{t+1})`` pairs for the GNN baseline."""

    def __init__(self, base: DeformableObjectDataset, indices: Sequence[int]) -> None:
        if base.positions is None or base.velocities is None:
            raise ValueError("ParticleTransitionDataset requires stored lattice trajectories.")
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.indices[index]
        episode = flat // self.base.pairs_per_episode
        t = flat % self.base.pairs_per_episode
        assert self.base.positions is not None and self.base.velocities is not None
        return (
            self.base.positions[episode, t],
            self.base.velocities[episode, t],
            self.base.actions[episode, t],
            self.base.positions[episode, t + 1],
            self.base.velocities[episode, t + 1],
        )


# ---------------------------------------------------------------------------
# Physical-simulation ingestion (HDF5 / NPY / NPZ)
# ---------------------------------------------------------------------------

_OBS_KEYS = (
    "observations",
    "observation",
    "images",
    "image",
    "rgb",
    "frames",
    "obs",
    "obses",
    "video",
    "states",
)
_ACT_KEYS = ("actions", "action", "acts", "ctrl", "control", "u", "action_seq")
_LABEL_KEYS = ("stiffness_labels", "labels", "stiffness", "material", "class_id")
_VALUE_KEYS = ("stiffness_values", "youngs_modulus", "youngs", "stiffness_e")


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Loading HDF5 physics dumps requires h5py. Install with `pip install h5py`."
        ) from exc
    return h5py


def _as_float_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.float()
    return torch.from_numpy(np.asarray(value)).float()


def _is_channel_last(shape: Sequence[int]) -> bool:
    if len(shape) < 3:
        return False
    return shape[-1] in (1, 2, 3, 4) and shape[-2] > 4 and shape[-3] > 4


def canonicalize_observations(obs: Any, img_size: int) -> torch.Tensor:
    """Map heterogeneous image dumps to ``(N, T, C, H, W)`` float tensors.

    Accepts channel-last RGB, uint8 [0, 255], missing batch axes, and
    resizes spatially to ``img_size``.
    """
    tensor = _as_float_tensor(obs)
    if tensor.ndim == 4:
        if tensor.shape[1] in (1, 2, 3, 4) and tensor.shape[-1] > 4 and tensor.shape[-2] > 4:
            tensor = tensor.unsqueeze(0)
        elif _is_channel_last(tuple(tensor.shape)):
            tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(2)
    if tensor.ndim != 5:
        raise ValueError(
            "Observations must broadcast to (N, T, C, H, W) or (N, T, H, W, C); "
            f"got {tuple(tensor.shape)}."
        )
    if _is_channel_last(tuple(tensor.shape)):
        tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
    if tensor.shape[-2] != img_size or tensor.shape[-1] != img_size:
        n_ep, n_t, channels = tensor.shape[:3]
        flat = tensor.reshape(n_ep * n_t, channels, tensor.shape[-2], tensor.shape[-1])
        flat = F.interpolate(flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
        tensor = flat.reshape(n_ep, n_t, channels, img_size, img_size)
    max_abs = float(tensor.abs().max().clamp_min(1e-8))
    if max_abs > 1.5:
        tensor = tensor / 255.0
    return tensor.contiguous()


def canonicalize_actions(act: Any, n_frames: int) -> torch.Tensor:
    """Map action dumps to ``(N, T-1, A)``, dropping a trailing aligned action if needed."""
    tensor = _as_float_tensor(act)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Actions must have shape (N, T, A) or (N, T-1, A); got {tuple(tensor.shape)}.")
    n_steps = tensor.shape[1]
    if n_steps == n_frames:
        tensor = tensor[:, :-1]
    elif n_steps != n_frames - 1:
        raise ValueError(
            f"Action time {n_steps} is incompatible with {n_frames} frames "
            "(expected T or T-1)."
        )
    return tensor.contiguous()


def _lookup_mapping(store: Any, names: tuple[str, ...]) -> Any | None:
    keys = list(store.keys()) if hasattr(store, "keys") else []
    lowered = {str(k).lower(): k for k in keys}
    for name in names:
        if name in store:
            return store[name]
        if name.lower() in lowered:
            return store[lowered[name.lower()]]
    return None


def _load_h5_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        obs = _lookup_mapping(handle, _OBS_KEYS)
        act = _lookup_mapping(handle, _ACT_KEYS)
        labels = _lookup_mapping(handle, _LABEL_KEYS)
        if obs is not None and act is not None:
            label_arr = None if labels is None else np.asarray(labels[()])
            return np.asarray(obs[()]), np.asarray(act[()]), label_arr

        groups = [handle[k] for k in handle.keys() if isinstance(handle[k], h5py.Group)]
        if not groups:
            raise KeyError(
                f"{path} has no recognizable observation/action datasets. "
                f"Looked for {_OBS_KEYS[:6]} and {_ACT_KEYS[:4]}."
            )
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        label_list: list[int] = []
        for group in groups:
            nested_obs = _lookup_mapping(group, _OBS_KEYS)
            nested_act = _lookup_mapping(group, _ACT_KEYS)
            if nested_obs is None or nested_act is None:
                continue
            o = np.asarray(nested_obs[()])
            a = np.asarray(nested_act[()])
            if o.ndim == 3:
                o = o[None, ...]
            if a.ndim == 1:
                a = a[None, ...]
            obs_list.append(o)
            act_list.append(a)
            nested_lab = _lookup_mapping(group, _LABEL_KEYS)
            label_list.append(int(np.asarray(nested_lab[()]).reshape(-1)[0]) if nested_lab is not None else 0)
        if not obs_list:
            raise KeyError(f"{path} episode groups did not contain observation/action arrays.")
        max_t = max(o.shape[0] for o in obs_list)
        c_shape = obs_list[0].shape[1:]
        a_dim = act_list[0].shape[-1]
        obs_pad = np.zeros((len(obs_list), max_t, *c_shape), dtype=np.float32)
        act_len = max(a.shape[0] for a in act_list)
        act_pad = np.zeros((len(act_list), act_len, a_dim), dtype=np.float32)
        for i, (o, a) in enumerate(zip(obs_list, act_list, strict=True)):
            obs_pad[i, : o.shape[0]] = o.astype(np.float32, copy=False)
            act_pad[i, : a.shape[0]] = a.astype(np.float32, copy=False)
        labels_arr = np.asarray(label_list, dtype=np.int64) if any(label_list) else None
        return obs_pad, act_pad, labels_arr


def _load_npy_directory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    files = {p.name.lower(): p for p in path.iterdir() if p.is_file()}

    def _first(names: tuple[str, ...]) -> Path | None:
        for name in names:
            for suffix in (".npy", ".npz"):
                candidate = name + suffix
                if candidate in files:
                    return files[candidate]
            if name in files:
                return files[name]
        return None

    obs_path = _first(_OBS_KEYS)
    act_path = _first(_ACT_KEYS)
    if obs_path is not None and act_path is not None:
        obs = np.load(obs_path, allow_pickle=False)
        act = np.load(act_path, allow_pickle=False)
        lab_path = _first(_LABEL_KEYS)
        labels = None if lab_path is None else np.load(lab_path, allow_pickle=False)
        return np.asarray(obs), np.asarray(act), None if labels is None else np.asarray(labels)

    h5_files = sorted(path.glob("*.h5")) + sorted(path.glob("*.hdf5"))
    if h5_files:
        obs_parts, act_parts, lab_parts = [], [], []
        for h5_path in h5_files:
            o, a, lab = _load_h5_arrays(h5_path)
            obs_parts.append(o)
            act_parts.append(a)
            if lab is not None:
                lab_parts.append(lab)
        obs = np.concatenate(obs_parts, axis=0)
        act = np.concatenate(act_parts, axis=0)
        labels = np.concatenate(lab_parts, axis=0) if lab_parts else None
        return obs, act, labels

    raise FileNotFoundError(
        f"{path} is not a physics dump directory. Expected observations.npy + actions.npy "
        "or one or more .h5 files."
    )


def load_physics_trajectories(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a Taichi / PlasticineLab / MuJoCo-style trajectory archive.

    Supported sources:

    * ``.h5`` / ``.hdf5`` — flat datasets or one group per episode
    * ``.npz`` / ``.pt`` — keys ``observations`` / ``actions``
    * ``.npy`` directory — ``observations.npy`` + ``actions.npy``
    * directory of ``.h5`` shards
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Physics dataset path not found: {path}")
    suffix = path.suffix.lower()
    labels_np: np.ndarray | None
    if path.is_dir():
        obs_np, act_np, labels_np = _load_npy_directory(path)
        observations = torch.from_numpy(np.asarray(obs_np))
        actions = torch.from_numpy(np.asarray(act_np))
        raw_labels = None if labels_np is None else torch.as_tensor(labels_np)
    elif suffix in {".h5", ".hdf5"}:
        obs_np, act_np, labels_np = _load_h5_arrays(path)
        observations = torch.from_numpy(np.asarray(obs_np))
        actions = torch.from_numpy(np.asarray(act_np))
        raw_labels = None if labels_np is None else torch.as_tensor(labels_np)
    elif suffix in {".npz", ".pt", ".pth"}:
        observations, actions, raw_labels = _load_trajectory_archive(path)
    elif suffix == ".npy":
        raise ValueError(
            "A lone .npy file is ambiguous; use a directory with observations.npy and actions.npy."
        )
    else:
        raise ValueError(f"Unsupported physics dataset type '{path}'. Use HDF5, NPZ, or an NPY directory.")

    if raw_labels is None:
        stiffness = torch.zeros(observations.shape[0], dtype=torch.long)
    else:
        stiffness = torch.as_tensor(raw_labels).long().reshape(-1)
    return observations, actions, stiffness


def inspect_physics_source(path: str | Path) -> dict[str, int]:
    """Read array shapes without running the simulator (may load arrays)."""
    observations, actions, _ = load_physics_trajectories(path)
    frames = canonicalize_observations(observations, img_size=int(observations.shape[-2]))
    n_frames = int(frames.shape[1])
    acts = canonicalize_actions(actions, n_frames=n_frames)
    return {
        "n_episodes": int(frames.shape[0]),
        "episode_length": n_frames,
        "in_channels": int(frames.shape[2]),
        "img_size": int(frames.shape[-1]),
        "action_dim": int(acts.shape[-1]),
    }


class ConsistentFrameAugment:
    """Vision augmentations with parameters tied across time.

    Random resized crop, color jitter, and Gaussian blur use one draw per
    window so ``x_t`` and ``x_{t+k}`` stay photometrically / geometrically
    consistent — the standard SSL recipe for video / world-model encoders.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.img_size = cfg.img_size
        self.crop_scale_min = cfg.crop_scale_min
        self.jitter = cfg.color_jitter
        self.blur_prob = cfg.blur_prob
        self.blur_sigma_max = cfg.blur_sigma_max

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected frames (T, C, H, W); got {tuple(frames.shape)}.")
        frames = self._random_resized_crop(frames)
        frames = self._color_jitter(frames)
        if torch.rand(()) < self.blur_prob:
            frames = self._gaussian_blur(frames)
        return frames

    def _random_resized_crop(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        scale = float(torch.empty(()).uniform_(self.crop_scale_min, 1.0))
        crop_h = max(1, int(round(height * scale)))
        crop_w = max(1, int(round(width * scale)))
        top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
        left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
        cropped = frames[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(
            cropped, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

    def _color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        strength = self.jitter
        if strength <= 0.0:
            return frames
        signed = bool(frames.min() < -1e-3)
        work = frames * 0.5 + 0.5 if signed else frames
        brightness = 1.0 + float(torch.empty(()).uniform_(-strength, strength))
        contrast = 1.0 + float(torch.empty(()).uniform_(-strength, strength))
        saturation = 1.0 + float(torch.empty(()).uniform_(-strength, strength))
        work = work * brightness
        mean = work.mean(dim=(-2, -1), keepdim=True)
        work = (work - mean) * contrast + mean
        if work.shape[1] >= 3:
            gray = (
                0.299 * work[:, 0:1] + 0.587 * work[:, 1:2] + 0.114 * work[:, 2:3]
            )
            rgb = work[:, :3] * saturation + gray * (1.0 - saturation)
            work = torch.cat([rgb, work[:, 3:]], dim=1) if work.shape[1] > 3 else rgb
        work = work.clamp(0.0, 1.0)
        return work * 2.0 - 1.0 if signed else work

    def _gaussian_blur(self, frames: torch.Tensor) -> torch.Tensor:
        sigma = float(torch.empty(()).uniform_(0.1, self.blur_sigma_max))
        radius = max(1, int(round(3.0 * sigma)))
        k = 2 * radius + 1
        coords = torch.arange(k, dtype=frames.dtype, device=frames.device) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, k, k)
        channels = frames.shape[1]
        weight = kernel_2d.expand(channels, 1, k, k)
        padded = F.pad(frames, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(padded, weight, groups=channels)


def _episode_split_counts(n_episodes: int, val_fraction: float) -> tuple[int, int]:
    n_val = max(1, int(round(n_episodes * val_fraction))) if n_episodes > 1 else 0
    n_val = min(n_val, n_episodes - 1) if n_episodes > 1 else 0
    return n_episodes - n_val, n_val


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


class DeformablePhysicsDataset(Dataset[Transition]):
    """HDF5/NPY loader with chunking, time-consistent augs, and action z-scoring.

    Index space is every adjacent pair inside each length-``T`` window, so a
    :class:`~torch.utils.data.DataLoader` still yields
    ``(x_t, a_t, x_{t+1})``. Full windows are available via :meth:`get_chunk`.
    """

    def __init__(self, cfg: DataConfig | None = None, split: Split = "all") -> None:
        super().__init__()
        if cfg is None or cfg.dataset_path is None:
            raise ValueError("DeformablePhysicsDataset requires DataConfig.dataset_path.")
        self.cfg = cfg
        self.split = split
        raw_obs, raw_act, labels = load_physics_trajectories(cfg.dataset_path)
        observations = canonicalize_observations(raw_obs, img_size=cfg.img_size)
        actions = canonicalize_actions(raw_act, n_frames=int(observations.shape[1]))
        if labels.shape[0] != observations.shape[0]:
            labels = torch.zeros(observations.shape[0], dtype=torch.long)
        values = resolve_stiffness_values(
            labels, source=cfg.dataset_path, codebook=_stiffness_levels(cfg)
        )
        n_train, n_val = _episode_split_counts(int(observations.shape[0]), cfg.val_fraction)
        scaler_src = actions[: max(1, n_train)]
        action_mean = scaler_src.reshape(-1, actions.shape[-1]).mean(dim=0)
        action_std = scaler_src.reshape(-1, actions.shape[-1]).std(dim=0, unbiased=False).clamp_min(1e-6)
        if split == "train":
            observations, actions, labels, values = (
                observations[:n_train],
                actions[:n_train],
                labels[:n_train],
                values[:n_train],
            )
            self._global_episode_offset = 0
        elif split == "val":
            if n_val == 0:
                raise ValueError("val split is empty; increase n_episodes or val_fraction.")
            observations, actions, labels, values = (
                observations[n_train:],
                actions[n_train:],
                labels[n_train:],
                values[n_train:],
            )
            self._global_episode_offset = n_train
        else:
            self._global_episode_offset = 0
        self.observations = observations.contiguous()
        self.actions = actions.contiguous()
        self.stiffness_labels = labels.long().contiguous()
        self.stiffness_values = values.float().contiguous()
        self.n_episodes = int(self.observations.shape[0])
        self.pairs_per_episode = int(self.observations.shape[1] - 1)
        self.action_mean = action_mean
        self.action_std = action_std
        self.augmentor = ConsistentFrameAugment(cfg)
        self._pair_index = self._build_pair_index()
        self._n_train_episodes = n_train if split == "all" else (self.n_episodes if split == "train" else 0)

    @property
    def model_actions(self) -> torch.Tensor:
        """Actions in the coordinates the predictor was trained on (optionally z-scored)."""
        return self._normalize_action(self.actions)

    def _build_pair_index(self) -> list[tuple[int, int]]:
        horizon = int(self.observations.shape[1])
        chunk = min(self.cfg.chunk_length, horizon)
        if chunk < 2:
            raise ValueError("Native episode length must be >= 2 to form transitions.")
        stride = self.cfg.chunk_stride
        index: list[tuple[int, int]] = []
        for ep in range(self.n_episodes):
            last_start = horizon - chunk
            for start in range(0, last_start + 1, stride):
                for offset in range(chunk - 1):
                    index.append((ep, start + offset))
        if not index:
            raise ValueError("No trajectory chunks produced; check chunk_length vs episode length.")
        return index

    def split_sizes(self) -> tuple[int, int]:
        if self.split == "train":
            return self.n_episodes, 0
        if self.split == "val":
            return 0, self.n_episodes
        return _episode_split_counts(self.n_episodes, self.cfg.val_fraction)

    def indices_for_episodes(self, start: int, stop: int) -> list[int]:
        start = max(0, start)
        stop = min(self.n_episodes, stop)
        return [i for i, (ep, _) in enumerate(self._pair_index) if start <= ep < stop]

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_actions:
            return action
        return (action - self.action_mean) / self.action_std

    def _should_augment(self, episode: int) -> bool:
        if not self.cfg.augment:
            return False
        if self.split == "train":
            return True
        if self.split == "val":
            return False
        return episode < self._n_train_episodes

    def get_chunk(self, episode: int, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_{t:t+T}, a_{t:t+T-1})`` with optional train-time augmentation."""
        chunk = min(self.cfg.chunk_length, int(self.observations.shape[1]))
        frames = self.observations[episode, start : start + chunk].clone()
        actions = self.actions[episode, start : start + chunk - 1].clone()
        if self._should_augment(episode):
            frames = self.augmentor(frames)
        actions = self._normalize_action(actions)
        return frames, actions

    def __len__(self) -> int:
        return len(self._pair_index)

    def __getitem__(self, index: int) -> Transition:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode, t = self._pair_index[index]
        frames = self.observations[episode, t : t + 2]
        if self._should_augment(episode):
            frames = self.augmentor(frames)
        x_t, x_tp1 = frames[0], frames[1]
        a_t = self._normalize_action(self.actions[episode, t])
        return x_t, a_t, x_tp1


def build_dataset(cfg: DataConfig | None = None, split: Split = "all") -> Dataset[Transition]:
    """Synthetic mass-spring data, or a physics dump when ``dataset_path`` is set."""
    cfg = cfg if cfg is not None else DataConfig()
    if cfg.dataset_path is None:
        return DeformableObjectDataset(cfg, split=split)
    return DeformablePhysicsDataset(cfg, split=split)


def dataloader_worker_kwargs(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    *,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Multi-worker prefetch settings (``prefetch_factor`` requires workers > 0)."""
    workers = train_cfg.num_workers if num_workers is None else num_workers
    pin_memory = bool(data_cfg.pin_memory and torch.cuda.is_available())
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(data_cfg.prefetch_factor)
        kwargs["worker_init_fn"] = _worker_init_fn
    return kwargs


def build_dataloader(
    data_cfg: DataConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    *,
    split: Split = "train",
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
) -> DataLoader[Transition]:
    """Construct a ``DataLoader`` that yields batched AC-JEPA transitions."""
    data_cfg = data_cfg if data_cfg is not None else DataConfig()
    train_cfg = train_cfg if train_cfg is not None else TrainingConfig()
    dataset = build_dataset(data_cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else train_cfg.batch_size,
        shuffle=True if shuffle is None else shuffle,
        **dataloader_worker_kwargs(data_cfg, train_cfg, num_workers=num_workers),
    )


def generate_dummy_batch(
    batch_size: int = 4,
    cfg: DataConfig | None = None,
) -> Transition:
    """Synthesize a batched transition without an external dataset path.

    This is the smoke-test entry used by ``python src/dataset.py``.
    """
    cfg = cfg if cfg is not None else DataConfig(dataset_path=None)
    if cfg.dataset_path is not None:
        raise ValueError("generate_dummy_batch expects dataset_path=None.")
    smoke = replace(
        cfg,
        dataset_path=None,
        n_episodes=max(batch_size, 4),
        episode_length=max(cfg.episode_length, 2),
        val_fraction=0.0,
        pin_memory=False,
        augment=False,
        normalize_actions=False,
    )
    dataset = DeformableObjectDataset(smoke, split="all")
    return dataset.sample_batch(batch_size)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dummy AC-JEPA tensors or load a physics trajectory dump."
    )
    parser.add_argument(
        "--dataset_path",
        "--dataset-path",
        dest="dataset_path",
        default=None,
        help="HDF5 / NPY directory / NPZ archive. Omit to synthesize mass-spring data.",
    )
    parser.add_argument("--spatial-dims", type=int, default=2, choices=(2, 3))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=4)
    parser.add_argument("--n-grid", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate or load a batched transition and print tensor statistics."""
    args = _parse_args(argv)
    action_dim = 4 if args.spatial_dims == 2 else 6
    cfg = DataConfig(
        dataset_path=args.dataset_path,
        spatial_dims=args.spatial_dims,
        img_size=args.img_size,
        in_channels=3,
        action_dim=action_dim,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        n_grid=args.n_grid,
        seed=args.seed,
        val_fraction=0.0,
        pin_memory=False,
        chunk_length=args.chunk_length,
        augment=bool(args.dataset_path),
    )
    if args.dataset_path is not None:
        meta = inspect_physics_source(args.dataset_path)
        cfg = replace(
            cfg,
            in_channels=meta["in_channels"],
            action_dim=meta["action_dim"],
            n_episodes=meta["n_episodes"],
            episode_length=meta["episode_length"],
        )
    dataset = build_dataset(cfg, split="all")
    train_cfg = TrainingConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_worker_kwargs(cfg, train_cfg, num_workers=args.num_workers),
    )
    x_t, a_t, x_tp1 = next(iter(loader))
    source = "synthetic mass-spring" if cfg.dataset_path is None else str(cfg.dataset_path)
    delta = (x_tp1 - x_t).abs().mean().item()
    print("AC-JEPA deformable-object batch")
    print(f"  source        : {source}")
    print(f"  spatial_dims  : {cfg.spatial_dims}")
    print(f"  dataset len   : {len(dataset)} transitions  ({dataset.n_episodes} episodes)")
    print(f"  x_t           : {tuple(x_t.shape)}  dtype={x_t.dtype}")
    print(f"  a_t           : {tuple(a_t.shape)}  dtype={a_t.dtype}")
    print(f"  x_{{t+1}}       : {tuple(x_tp1.shape)}  dtype={x_tp1.dtype}")
    print(f"  mean |x_{{t+1}} - x_t| : {delta:.6f}")
    print(f"  x_t range     : [{x_t.min().item():.4f}, {x_t.max().item():.4f}]")
    print(f"  a_t range     : [{a_t.min().item():.4f}, {a_t.max().item():.4f}]")
    if delta < 1e-6:
        print("  warning       : consecutive frames are nearly identical (check action_gain / stiffness).")


if __name__ == "__main__":
    main(sys.argv[1:])
