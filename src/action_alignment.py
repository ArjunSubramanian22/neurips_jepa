"""Counterfactual action alignment on simulator forks.

``delta_pred`` is the predictor's change under an intervened action.
``delta_target`` is the stop-gradient target encoder's change between the
recorded future and the future rolled under that same intervention.
Families are not mixed in one loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

FAMILIES = (
    "force_reversal",
    "force_rotation",
    "force_scaling",
    "center_shift",
    "shuffled_same_time",
)
PRIMARY_FAMILY = "force_reversal"
FORCE_ROTATION_RADIANS = float(np.pi / 2.0)
FORCE_SCALE = 0.5
CENTER_SHIFT = (0.10, 0.0)
CENTER_CLIP = (0.08, 0.92)
NORM_TAU = 1e-8
MAGNITUDE_EPS = 1e-8


def assert_one_family(families: Sequence[str]) -> str:
    """Refuse a batch that mixes intervention families."""
    unique = {str(name) for name in families}
    if len(unique) != 1:
        raise ValueError(f"refusing to pool intervention families: {sorted(unique)}")
    name = next(iter(unique))
    if name not in FAMILIES:
        raise ValueError(f"unknown intervention family {name!r}")
    return name


def same_time_donor(episode_ids: Sequence[int], episode: int) -> int:
    """Next episode in sorted order at the same timestep. A fixed derangement."""
    ordered = sorted({int(item) for item in episode_ids})
    if len(ordered) < 2:
        raise ValueError("shuffled_same_time needs at least two episodes.")
    if int(episode) not in ordered:
        raise ValueError(f"episode {episode} is not in the donor pool.")
    position = ordered.index(int(episode))
    return ordered[(position + 1) % len(ordered)]


def apply_intervention(
    action: np.ndarray,
    family: str,
    *,
    donor: np.ndarray | None = None,
    spatial_dims: int = 2,
) -> np.ndarray:
    """Intervene in raw simulator coordinates. Center and force are not mixed."""
    if family not in FAMILIES:
        raise ValueError(f"unknown intervention family {family!r}")
    raw = np.asarray(action, dtype=np.float32).copy()
    if raw.shape[-1] < 2 * int(spatial_dims):
        raise ValueError("action is shorter than center plus force.")
    if family == "shuffled_same_time":
        if donor is None:
            raise ValueError("shuffled_same_time requires the donor action at the same timestep.")
        return np.asarray(donor, dtype=np.float32).copy()
    center = raw[..., :spatial_dims]
    force = raw[..., spatial_dims : 2 * spatial_dims]
    if family == "force_reversal":
        force = -force
    elif family == "force_rotation":
        if spatial_dims != 2:
            raise ValueError("force_rotation is defined for 2D force vectors.")
        cosine = np.float32(np.cos(FORCE_ROTATION_RADIANS))
        sine = np.float32(np.sin(FORCE_ROTATION_RADIANS))
        rotated = np.empty_like(force)
        rotated[..., 0] = cosine * force[..., 0] - sine * force[..., 1]
        rotated[..., 1] = sine * force[..., 0] + cosine * force[..., 1]
        force = rotated
    elif family == "force_scaling":
        force = force * np.float32(FORCE_SCALE)
    elif family == "center_shift":
        shifted = center.copy()
        shifted[..., 0] = shifted[..., 0] + np.float32(CENTER_SHIFT[0])
        shifted[..., 1] = shifted[..., 1] + np.float32(CENTER_SHIFT[1])
        center = np.clip(shifted, np.float32(CENTER_CLIP[0]), np.float32(CENTER_CLIP[1]))
    else:
        raise ValueError(f"unknown intervention family {family!r}")
    out = raw.copy()
    out[..., :spatial_dims] = center
    out[..., spatial_dims : 2 * spatial_dims] = force
    return out


def intervene_normalized(
    action: Tensor,
    mean: Tensor,
    std: Tensor,
    family: str,
    *,
    donor: Tensor | None = None,
    spatial_dims: int = 2,
) -> Tensor:
    """Apply a raw-space intervention, then return to the training coordinate system."""
    raw = action * std + mean
    donor_raw = None if donor is None else (donor * std + mean).detach().cpu().numpy()
    prime = apply_intervention(
        raw.detach().cpu().numpy(),
        family,
        donor=donor_raw,
        spatial_dims=spatial_dims,
    )
    prime_t = torch.as_tensor(prime, dtype=action.dtype, device=action.device)
    return (prime_t - mean) / std.clamp_min(1e-6)


def action_alignment_losses(
    delta_pred: Tensor,
    delta_target: Tensor,
    *,
    norm_tau: float = NORM_TAU,
    eps: float = MAGNITUDE_EPS,
) -> dict[str, Tensor]:
    """Direction, magnitude, and unit-vector losses on valid target moves.

    The target delta is detached here. Rows with ``||delta_target|| <= norm_tau``
    are excluded from every component: a near-zero target has no direction, and
    a log ratio against it would collapse toward zero when both norms sit on
    ``eps``.
    """
    if delta_pred.shape != delta_target.shape or delta_pred.ndim != 2:
        raise ValueError("delta_pred and delta_target must share shape (B, D).")
    if float(norm_tau) < 0.0 or float(eps) <= 0.0:
        raise ValueError("norm_tau must be >= 0 and eps must be > 0.")
    target = delta_target.detach()
    pred_norm = delta_pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    valid = torch.isfinite(pred_norm) & torch.isfinite(target_norm) & (target_norm > float(norm_tau))
    cosine = F.cosine_similarity(delta_pred, target, dim=-1, eps=float(eps))
    direction = 1.0 - cosine
    magnitude = torch.abs(torch.log(pred_norm + float(eps)) - torch.log(target_norm + float(eps)))
    pred_unit = delta_pred / (pred_norm.unsqueeze(-1) + float(eps))
    target_unit = target / (target_norm.unsqueeze(-1) + float(eps))
    vector = (pred_unit - target_unit).norm(dim=-1)

    def _mean(component: Tensor) -> Tensor:
        if not bool(valid.any().item()):
            return delta_pred.new_zeros(())
        return component[valid].mean()

    return {
        "dir": _mean(direction),
        "mag": _mean(magnitude),
        "vec": _mean(vector),
        "n_valid": valid.sum().to(dtype=delta_pred.dtype),
        "n_excluded": (~valid).sum().to(dtype=delta_pred.dtype),
    }


def combine_base_and_alignment(
    base: Tensor,
    parts: dict[str, Tensor],
    *,
    lambda_dir: float,
    lambda_mag: float,
    lambda_vec: float,
) -> Tensor:
    """Add whichever alignment terms have a positive weight."""
    if lambda_dir < 0.0 or lambda_mag < 0.0 or lambda_vec < 0.0:
        raise ValueError("alignment weights must be >= 0.")
    loss = base
    if lambda_dir > 0.0:
        loss = loss + base.new_tensor(float(lambda_dir)) * parts["dir"]
    if lambda_mag > 0.0:
        loss = loss + base.new_tensor(float(lambda_mag)) * parts["mag"]
    if lambda_vec > 0.0:
        loss = loss + base.new_tensor(float(lambda_vec)) * parts["vec"]
    return loss


def apply_action_alignment(
    base: Tensor,
    predictor: Callable[[Tensor, Tensor], Tensor] | nn.Module,
    z_t: Tensor,
    z_hat: Tensor,
    z_target: Tensor,
    action_prime: Tensor,
    future_prime: Tensor,
    target_encoder: Callable[[Tensor], Tensor] | nn.Module,
    valid: Tensor,
    *,
    lambda_dir: float,
    lambda_mag: float,
    lambda_vec: float,
    norm_tau: float = NORM_TAU,
    eps: float = MAGNITUDE_EPS,
) -> tuple[Tensor, dict[str, float]]:
    """One extra predictor call and one extra target encode, on valid rows only."""
    mask = valid.reshape(-1) > 0.5
    logs: dict[str, float] = {
        "cf_applied": float(mask.float().mean().item()) if mask.numel() else 0.0,
        "cf_n_valid": float(mask.sum().item()),
        "cf_n_excluded": float((~mask).sum().item()),
    }
    if not bool(mask.any().item()):
        return base, logs
    index = mask.nonzero(as_tuple=False).flatten()
    delta_pred = predictor(z_t.index_select(0, index), action_prime.index_select(0, index)) - z_hat.index_select(0, index)
    with torch.no_grad():
        future_latent = target_encoder(future_prime.index_select(0, index))
    delta_target = future_latent - z_target.index_select(0, index)
    parts = action_alignment_losses(delta_pred, delta_target, norm_tau=norm_tau, eps=eps)
    loss = combine_base_and_alignment(
        base,
        parts,
        lambda_dir=lambda_dir,
        lambda_mag=lambda_mag,
        lambda_vec=lambda_vec,
    )
    logs["cf_dir"] = float(parts["dir"].detach().item())
    logs["cf_mag"] = float(parts["mag"].detach().item())
    logs["cf_vec"] = float(parts["vec"].detach().item())
    logs["cf_n_valid"] = float(parts["n_valid"].detach().item())
    logs["cf_n_excluded"] = float(parts["n_excluded"].detach().item()) + float((~mask).sum().item())
    return loss, logs


def included_in_fraction(episode: int, time: int, fraction: float, seed: int) -> bool:
    """Deterministic subset of transitions. The draw does not depend on process hash salt."""
    scale = float(fraction)
    if scale >= 1.0:
        return True
    if scale <= 0.0:
        return False
    key = (int(episode) * 1_000_003 + int(time) * 9176 + int(seed) * 131) % 1_000_000
    return key < int(round(scale * 1_000_000))


class CounterfactualCache:
    """Paired futures for one intervention family, keyed by global episode and time."""

    def __init__(
        self,
        family: str,
        episode: np.ndarray,
        time: np.ndarray,
        action_prime: np.ndarray,
        frame_prime: np.ndarray,
    ) -> None:
        self.family = assert_one_family([family])
        episode = np.asarray(episode, dtype=np.int64).reshape(-1)
        time = np.asarray(time, dtype=np.int64).reshape(-1)
        action_prime = np.asarray(action_prime, dtype=np.float32)
        frame_prime = np.asarray(frame_prime)
        if episode.shape != time.shape or action_prime.shape[0] != episode.shape[0]:
            raise ValueError("cache columns must share a length.")
        if frame_prime.shape[0] != episode.shape[0]:
            raise ValueError("frame_prime must have one frame per pair.")
        self.episode = episode
        self.time = time
        self.action_prime = action_prime
        self.frame_prime = frame_prime
        self._index: dict[tuple[int, int], int] = {}
        for row, (ep, step) in enumerate(zip(episode.tolist(), time.tolist())):
            key = (int(ep), int(step))
            if key in self._index:
                raise ValueError(f"duplicate counterfactual pair at episode {ep} t {step}.")
            self._index[key] = row

    def __len__(self) -> int:
        return int(self.episode.shape[0])

    def get(self, episode: int, time: int) -> tuple[np.ndarray, np.ndarray]:
        key = (int(episode), int(time))
        if key not in self._index:
            raise KeyError(f"no {self.family} pair for episode {episode} t {time}.")
        row = self._index[key]
        return self.action_prime[row], self.frame_prime[row]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            family=np.asarray(self.family),
            episode=self.episode,
            time=self.time,
            action_prime=self.action_prime,
            frame_prime=self.frame_prime,
        )


def load_counterfactual_cache(path: Path) -> CounterfactualCache:
    archive = np.load(Path(path), allow_pickle=False)
    family = str(np.asarray(archive["family"]).reshape(-1)[0])
    return CounterfactualCache(
        family,
        archive["episode"],
        archive["time"],
        archive["action_prime"],
        archive["frame_prime"],
    )


def simulate_fork(
    snapshot: dict[str, Any],
    action: np.ndarray,
    action_prime: np.ndarray,
    mpm: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the recorded action and the intervention from one particle snapshot."""
    from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, render_rgb

    sim = MLSMPM2D(mpm)
    sim.restore(snapshot)
    other = sim.clone()
    apply_action_frames(sim, action, 1)
    apply_action_frames(other, action_prime, 1)
    factual = render_rgb(sim.x, sim.v, mpm.img_size, mpm.render_sigma)
    counter = render_rgb(other.x, other.v, mpm.img_size, mpm.render_sigma)
    return factual, counter


def fork_episode(
    episode_index: int,
    times: Sequence[int],
    family: str,
    *,
    dump_seed: int = 0,
    episode_length: int = 16,
    mpm: Any = None,
    donor_actions: dict[int, np.ndarray] | None = None,
) -> CounterfactualCache:
    """Replay one dump episode and fork the requested times under one family."""
    from scripts.generate_trajectories import MPMConfig, replay_episode

    family = assert_one_family([family])
    cfg = mpm if mpm is not None else MPMConfig()
    replay = replay_episode(
        int(episode_index),
        dump_seed=int(dump_seed),
        episode_length=int(episode_length),
        mpm=cfg,
    )
    episodes: list[int] = []
    steps: list[int] = []
    primes: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for time in times:
        step = int(time)
        action = np.asarray(replay["actions"][step], dtype=np.float32)
        donor = None if donor_actions is None else donor_actions[step]
        prime = apply_intervention(action, family, donor=donor)
        factual, counter = simulate_fork(replay["snapshots"][step], action, prime, cfg)
        recorded = np.asarray(replay["frames"][step + 1])
        if not np.array_equal(factual, recorded):
            raise RuntimeError(f"fork factual frame disagrees with the replay at t={step}.")
        episodes.append(int(episode_index))
        steps.append(step)
        primes.append(prime)
        frames.append(counter)
    return CounterfactualCache(
        family,
        np.asarray(episodes, dtype=np.int64),
        np.asarray(steps, dtype=np.int64),
        np.stack(primes, axis=0),
        np.stack(frames, axis=0),
    )


def _as_frame(frame: np.ndarray) -> Tensor:
    tensor = torch.as_tensor(np.asarray(frame))
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    tensor = tensor.float()
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor


class PairedCounterfactualDataset(torch.utils.data.Dataset):
    """Yield factual transitions plus a cached fork on a deterministic subset.

    Excluded rows still return a tensor of the right shape so the loader can
    collate, with a zero validity flag. The alignment loss skips those rows
    and does not encode the placeholder frame.
    """

    def __init__(
        self,
        base: Any,
        cache: CounterfactualCache,
        indices: Sequence[int],
        *,
        fraction: float,
        seed: int,
    ) -> None:
        self.base = base
        self.cache = cache
        self.indices = [int(index) for index in indices]
        self.fraction = float(fraction)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _locate(self, index: int) -> tuple[int, int, int]:
        if hasattr(self.base, "_pair_index"):
            local_episode, time = self.base._pair_index[index]
        else:
            local_episode = index // int(self.base.pairs_per_episode)
            time = index % int(self.base.pairs_per_episode)
        ids = getattr(self.base, "global_episode_ids", None)
        episode = int(ids[local_episode]) if ids is not None else int(local_episode)
        return int(local_episode), int(time), episode

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        flat = self.indices[index]
        local_episode, time, episode = self._locate(flat)
        use_pair = included_in_fraction(episode, time, self.fraction, self.seed)
        if not use_pair:
            x_t, a_t, x_tp1 = self.base[flat]
            return x_t, a_t, x_tp1, torch.zeros_like(a_t), torch.zeros_like(x_t), torch.zeros(())
        raw_prime, frame_prime = self.cache.get(episode, time)
        frames = self.base.observations[local_episode, time : time + 2].clone()
        factual = self.base.actions[local_episode, time].clone()
        prime = torch.as_tensor(np.asarray(raw_prime), dtype=factual.dtype)
        actions = torch.stack([factual, prime], dim=0)
        future = _as_frame(frame_prime).to(dtype=frames.dtype)
        window = torch.cat([frames, future.unsqueeze(0)], dim=0)
        if self.base._should_augment(local_episode):
            window, actions = self.base.augmentor(window, actions)
        x_t, x_tp1, x_prime = window[0], window[1], window[2]
        a_t = self.base._normalize_action(actions[0])
        a_prime = self.base._normalize_action(actions[1])
        return x_t, a_t, x_tp1, a_prime, x_prime, torch.ones(())
