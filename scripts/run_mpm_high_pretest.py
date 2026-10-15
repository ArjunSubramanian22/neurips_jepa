#!/usr/bin/env python3
"""Validation multi-horizon counterfactuals for legacy and high-excitation MPM.

Does not read high-excitation episodes 1620-1799. RSSM cosine uses the
stochastic mean. The deterministic state is stored separately because the
horizon-1 audit showed it is shared with the target.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, render_rgb, replay_episode
from src.config import DataConfig
from src.counterfactual_study import assert_high_validation_episodes, pair_metrics
from src.dataset import DeformablePhysicsDataset
from src.environments import environment_from_dict
from src.registry import utc_now
from src.train import load_model_from_checkpoint

BRANCH_TIMES = (0, 1, 2, 3, 4, 5)
HORIZONS = (1, 5, 10)
SIM_HORIZON = 10
METRIC_VERSION = "mpm_multih_v2_physindex"


def physical_step(horizon: int) -> int:
    """Cache frame for this horizon. Horizon 1 is index 0."""
    if int(horizon) < 1:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    return int(horizon) - 1


def latent_step(horizon: int, *, teacher_forced_jepa: bool, horizons: tuple[int, ...] = HORIZONS) -> int:
    """Index into the latent tensor.

    Teacher-forced JEPA keeps one vector per entry of ``horizons``. That list
    index is not a frame number: horizons (1, 5, 10) occupy slots 0, 1, and 2.
    Physical and raster rows have to use :func:`physical_step` instead.
    """
    if teacher_forced_jepa:
        return horizons.index(int(horizon))
    return physical_step(horizon)
CACHE_ROOT = ROOT / "outputs" / "counterfactual_cache"
ROW_PATH = ROOT / "experiments" / "runs" / "mpm_high_multih" / "branch_rows.csv"
FIELDNAMES = [
    "fork_id", "regime", "model", "seed", "protocol", "horizon", "episode", "branch_time",
    "rms_pos", "rms_vel", "com_displacement", "raster_mse", "raster_l1", "identical_raster",
    "changed_pixel_fraction", "pred_norm", "target_norm", "cosine", "cosine_status",
    "response_ratio", "projection", "cosine_h", "cosine_z", "cosine_concat",
    "same_state_action_norm", "git_commit", "checkpoint_sha256", "metric_version",
]


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raster_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    a = np.asarray(left)
    b = np.asarray(right)
    diff = a.astype(np.float64) - b.astype(np.float64)
    changed = np.any(a != b, axis=0)
    identical = float(np.mean(a == b))
    return {
        "raster_mse": float(np.mean((diff / 255.0) ** 2)),
        "raster_l1": float(np.mean(np.abs(diff) / 255.0)),
        "identical_raster": 1.0 if np.array_equal(a, b) else 0.0,
        "changed_pixel_fraction": float(np.mean(changed)),
        "fraction_identical": identical,
    }


def _state_metrics(pos_a, vel_a, pos_b, vel_b) -> dict:
    pos_a = np.asarray(pos_a, dtype=np.float64)
    pos_b = np.asarray(pos_b, dtype=np.float64)
    vel_a = np.asarray(vel_a, dtype=np.float64)
    vel_b = np.asarray(vel_b, dtype=np.float64)
    return {
        "rms_pos": float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=-1)))),
        "rms_vel": float(np.sqrt(np.mean(np.sum((vel_a - vel_b) ** 2, axis=-1)))),
        "com_displacement": float(np.linalg.norm(pos_a.mean(axis=0) - pos_b.mean(axis=0))),
    }


def _reverse_force(actions: np.ndarray) -> np.ndarray:
    out = np.array(actions, dtype=np.float32, copy=True)
    out[..., 2:] *= np.float32(-1.0)
    return out


def _simulate(snapshot, actions, mpm) -> tuple[list, list, list]:
    sim = MLSMPM2D(mpm)
    sim.restore(snapshot)
    frames, positions, velocities = [], [], []
    for action in actions:
        apply_action_frames(sim, action, 1)
        frames.append(render_rgb(sim.x, sim.v, mpm.img_size, mpm.render_sigma))
        positions.append(np.array(sim.x, copy=True))
        velocities.append(np.array(sim.v, copy=True))
    return frames, positions, velocities


def _episode_path(regime: str, episode: int) -> Path:
    return CACHE_ROOT / regime / f"ep{episode}.npz"


def build_cache(regime: str, episodes: list[int], env_yaml: str, dump: str, hold: int, *, allow_test: bool = False) -> Path:
    """Write one file per episode. Particle counts differ across episodes."""
    if allow_test:
        if list(episodes) != list(range(1620, 1800)):
            raise RuntimeError("the test cache must be exactly episodes 1620-1799.")
        regime = "mpm_high_test"
    elif regime == "mpm_high":
        episodes = assert_high_validation_episodes(episodes)
    spec = environment_from_dict(yaml.safe_load((ROOT / env_yaml).read_text(encoding="utf-8")))
    if int(spec.command_hold) != int(hold):
        raise RuntimeError(f"{regime} hold {spec.command_hold} does not match {hold}.")
    out_dir = CACHE_ROOT / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    observations = np.load(ROOT / dump / "observations.npy", mmap_mode="r")
    for index, episode in enumerate(episodes):
        path = _episode_path(regime, episode)
        if path.is_file():
            continue
        if regime == "mpm_high" and episode >= 1620:
            raise RuntimeError("refusing a high-excitation test episode.")
        if regime == "mpm_high_test" and (episode < 1620 or episode > 1799):
            raise RuntimeError("the test cache received a non-test episode.")
        replay = replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=hold)
        if not np.array_equal(replay["frames"], observations[episode]):
            raise RuntimeError(f"replay does not match the dump at episode {episode}.")
        channels, height, width = replay["frames"].shape[1:]
        action_dim = replay["actions"].shape[-1]
        n_branch = len(BRANCH_TIMES)
        frames_rec = np.zeros((n_branch, SIM_HORIZON, channels, height, width), dtype=np.uint8)
        frames_rev = np.zeros_like(frames_rec)
        frame_now = np.zeros((n_branch, channels, height, width), dtype=np.uint8)
        actions_rec = np.zeros((n_branch, SIM_HORIZON, action_dim), dtype=np.float32)
        actions_rev = np.zeros_like(actions_rec)
        pos_rec, pos_rev, vel_rec, vel_rev = [], [], [], []
        for branch_index, branch in enumerate(BRANCH_TIMES):
            recorded = replay["actions"][branch : branch + SIM_HORIZON].astype(np.float32)
            reversed_actions = _reverse_force(recorded)
            actions_rec[branch_index] = recorded
            actions_rev[branch_index] = reversed_actions
            frame_now[branch_index] = replay["frames"][branch]
            branch_pos_rec, branch_vel_rec = [], []
            for step in range(SIM_HORIZON):
                frames_rec[branch_index, step] = replay["frames"][branch + 1 + step]
                branch_pos_rec.append(np.asarray(replay["snapshots"][branch + 1 + step]["x"], dtype=np.float32))
                branch_vel_rec.append(np.asarray(replay["snapshots"][branch + 1 + step]["v"], dtype=np.float32))
            rev_frames, rev_pos, rev_vel = _simulate(replay["snapshots"][branch], reversed_actions, spec.mpm)
            for step in range(SIM_HORIZON):
                frames_rev[branch_index, step] = rev_frames[step]
            pos_rec.append(np.stack(branch_pos_rec))
            vel_rec.append(np.stack(branch_vel_rec))
            pos_rev.append(np.stack(rev_pos))
            vel_rev.append(np.stack(rev_vel))
        np.savez_compressed(
            path,
            frames_rec=frames_rec,
            frames_rev=frames_rev,
            frame_now=frame_now,
            actions_rec=actions_rec,
            actions_rev=actions_rev,
            full_frames=replay["frames"],
            full_actions=np.asarray(replay["actions"], dtype=np.float32),
            pos_rec=np.stack(pos_rec),
            pos_rev=np.stack(pos_rev),
            vel_rec=np.stack(vel_rec),
            vel_rev=np.stack(vel_rev),
        )
        if index % 10 == 0:
            print(f"cache {regime} {index + 1}/{len(episodes)}", flush=True)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "regime": regime,
        "env_yaml": env_yaml,
        "dump": dump,
        "hold": hold,
        "episodes": episodes,
        "branch_times": list(BRANCH_TIMES),
        "sim_horizon": SIM_HORIZON,
        "git_commit": _git(),
        "timestamp": utc_now(),
        "test_episodes_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    return out_dir


def load_cache(regime: str, episodes: list[int]) -> dict:
    parts = [np.load(_episode_path(regime, episode)) for episode in episodes]
    def stack(key):
        return np.stack([part[key] for part in parts])
    return {
        "episodes": episodes,
        "frames_rec": stack("frames_rec"),
        "frames_rev": stack("frames_rev"),
        "frame_now": stack("frame_now"),
        "actions_rec": stack("actions_rec"),
        "actions_rev": stack("actions_rev"),
        "full_frames": stack("full_frames"),
        "full_actions": stack("full_actions"),
        "pos_rec": [part["pos_rec"] for part in parts],
        "pos_rev": [part["pos_rev"] for part in parts],
        "vel_rec": [part["vel_rec"] for part in parts],
        "vel_rev": [part["vel_rev"] for part in parts],
    }


def _norm_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray, device) -> torch.Tensor:
    raw = torch.as_tensor(actions, dtype=torch.float32, device=device)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (raw - center) / scale


def _as_video(frames: np.ndarray, device) -> torch.Tensor:
    tensor = torch.as_tensor(frames, dtype=torch.float32, device=device)
    return tensor / 255.0


def _jepa_open(model, frame_now, future, actions):
    z0 = model.encode(frame_now, target=False)
    pred = model.rollout(z0, actions)
    flat = future.reshape(-1, *future.shape[2:])
    target = model.encode(flat, target=True).reshape(future.shape[0], future.shape[1], -1)
    return pred, target, z0


def _jepa_teacher(model, frame_now, future, actions, other_actions):
    """One-step predictions from the real fork frame at t+H-1."""
    preds, targets, gates = [], [], []
    for horizon in HORIZONS:
        if horizon == 1:
            state_frame = frame_now
        else:
            state_frame = future[:, horizon - 2]
        state = model.encode(state_frame, target=False)
        action = actions[:, horizon - 1]
        other = other_actions[:, horizon - 1]
        preds.append(model.predictor(state, action))
        targets.append(model.encode(future[:, horizon - 1], target=True))
        gates.append(torch.linalg.norm(model.predictor(state, action) - model.predictor(state, other), dim=-1))
    return preds, targets, gates


def _rssm_filter(model, frames, actions, start: int):
    deter, stochastic = model.initial_state(frames.shape[0], frames.device)
    for time in range(start + 1):
        action = None if time == 0 else actions[:, time - 1]
        deter, stochastic, _feature = model.observe(deter, stochastic, action, frames[:, time])
    return deter, stochastic


def _rssm_sequences(model, frames, actions, start: int, protocol: str):
    deter, stochastic = _rssm_filter(model, frames[:, : start + 1], actions, start)
    pred_h, pred_z, tgt_h, tgt_z = [], [], [], []
    if protocol == "open_loop":
        td, ts = deter.detach().clone(), stochastic.detach().clone()
        for step in range(SIM_HORIZON):
            action = actions[:, start + step]
            deter, stochastic, pred = model.imagine(deter, stochastic, action)
            td, ts, tgt = model.observe(td, ts, action, frames[:, start + 1 + step])
            ph, pz = model.split_feature(pred)
            th, tz = model.split_feature(tgt)
            pred_h.append(ph)
            pred_z.append(pz)
            tgt_h.append(th)
            tgt_z.append(tz)
    elif protocol == "teacher_forced":
        for step in range(SIM_HORIZON):
            action = actions[:, start + step]
            _d, _s, pred = model.imagine(deter.detach().clone(), stochastic.detach().clone(), action)
            deter, stochastic, tgt = model.observe(deter, stochastic, action, frames[:, start + 1 + step])
            ph, pz = model.split_feature(pred)
            th, tz = model.split_feature(tgt)
            pred_h.append(ph)
            pred_z.append(pz)
            tgt_h.append(th)
            tgt_z.append(tz)
    else:
        raise ValueError(protocol)
    return (
        torch.stack(pred_z, dim=1),
        torch.stack(tgt_z, dim=1),
        torch.stack(pred_h, dim=1),
        torch.stack(tgt_h, dim=1),
    )


def _append_rows(handle, writer, rows: list[dict]) -> None:
    for row in rows:
        stored = dict(row)
        for key in ("cosine", "response_ratio", "projection", "cosine_h", "cosine_z", "cosine_concat"):
            if stored.get(key) is None:
                stored[key] = ""
        writer.writerow(stored)
    handle.flush()


def _done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((row["regime"], row["model"], row["seed"], row["protocol"]))
    return done


def score_checkpoint(regime: str, episodes: list[int], checkpoint: Path, model_name: str, seed: int, mean, std, device, git: str, row_path: Path | None = None) -> None:
    if regime == "mpm_high":
        episodes = assert_high_validation_episodes(episodes)
    cache = load_cache(regime, episodes)
    model, _cfg = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    digest = _sha256(checkpoint)
    is_rssm = type(model).__name__ == "CompactRSSM"
    protocols = ("open_loop", "teacher_forced")
    n = len(episodes)
    n_branch = len(BRANCH_TIMES)
    rows = []
    with torch.no_grad():
        for protocol in protocols:
            for branch_index, branch in enumerate(BRANCH_TIMES):
                future_rec = _as_video(cache["frames_rec"][:, branch_index], device)
                future_rev = _as_video(cache["frames_rev"][:, branch_index], device)
                now = _as_video(cache["frame_now"][:, branch_index], device)
                # Full videos are required for the RSSM filter. JEPA uses now + future.
                if is_rssm:
                    recorded_video = _as_video(cache["full_frames"], device)
                    recorded_actions = _norm_actions(cache["full_actions"], mean, std, device)
                    reversed_video = recorded_video.clone()
                    reversed_actions = recorded_actions.clone()
                    reversed_video[:, branch + 1 : branch + 1 + SIM_HORIZON] = future_rev
                    reversed_actions[:, branch : branch + SIM_HORIZON] = _norm_actions(
                        cache["actions_rev"][:, branch_index], mean, std, device
                    )
                    pred_z_r, tgt_z_r, pred_h_r, tgt_h_r = _rssm_sequences(
                        model, recorded_video, recorded_actions, branch, protocol
                    )
                    pred_z_v, tgt_z_v, pred_h_v, tgt_h_v = _rssm_sequences(
                        model, reversed_video, reversed_actions, branch, protocol
                    )
                    bundles = {
                        "z": (pred_z_v - pred_z_r, tgt_z_v - tgt_z_r),
                        "h": (pred_h_v - pred_h_r, tgt_h_v - tgt_h_r),
                        "concat": (
                            torch.cat([pred_h_v, pred_z_v], dim=-1) - torch.cat([pred_h_r, pred_z_r], dim=-1),
                            torch.cat([tgt_h_v, tgt_z_v], dim=-1) - torch.cat([tgt_h_r, tgt_z_r], dim=-1),
                        ),
                    }
                    gate = torch.linalg.norm(pred_z_v[:, 0] - pred_z_r[:, 0], dim=-1)
                else:
                    acts_rec = _norm_actions(cache["actions_rec"][:, branch_index], mean, std, device)
                    acts_rev = _norm_actions(cache["actions_rev"][:, branch_index], mean, std, device)
                    if protocol == "open_loop":
                        pred_r, tgt_r, z0 = _jepa_open(model, now, future_rec, acts_rec)
                        pred_v, tgt_v, _z0 = _jepa_open(model, now, future_rev, acts_rev)
                        bundles = {"z": (pred_v - pred_r, tgt_v - tgt_r)}
                        gate = torch.linalg.norm(
                            model.predictor(z0, acts_rec[:, 0]) - model.predictor(z0, acts_rev[:, 0]),
                            dim=-1,
                        )
                    else:
                        pred_r, tgt_r, gate_r = _jepa_teacher(model, now, future_rec, acts_rec, acts_rev)
                        pred_v, tgt_v, _gate_v = _jepa_teacher(model, now, future_rev, acts_rev, acts_rec)
                        bundles = {"z": (
                            torch.stack(pred_v, dim=1) - torch.stack(pred_r, dim=1),
                            torch.stack(tgt_v, dim=1) - torch.stack(tgt_r, dim=1),
                        )}
                        gate = torch.stack(gate_r, dim=1)
                for episode_index, episode in enumerate(episodes):
                    for horizon in HORIZONS:
                        teacher_forced_jepa = (not is_rssm) and protocol == "teacher_forced"
                        step = latent_step(horizon, teacher_forced_jepa=teacher_forced_jepa)
                        frame = physical_step(horizon)
                        if frame >= SIM_HORIZON:
                            raise RuntimeError(f"horizon {horizon} does not fit a cache of length {SIM_HORIZON}.")
                        physical = _state_metrics(
                            cache["pos_rec"][episode_index][branch_index, frame],
                            cache["vel_rec"][episode_index][branch_index, frame],
                            cache["pos_rev"][episode_index][branch_index, frame],
                            cache["vel_rev"][episode_index][branch_index, frame],
                        )
                        raster = _raster_metrics(
                            cache["frames_rec"][episode_index, branch_index, frame],
                            cache["frames_rev"][episode_index, branch_index, frame],
                        )
                        primary = pair_metrics(
                            bundles["z"][0][episode_index, step].detach().cpu().numpy(),
                            bundles["z"][1][episode_index, step].detach().cpu().numpy(),
                        )
                        extra = {"cosine_h": "", "cosine_z": primary["cosine"], "cosine_concat": ""}
                        if is_rssm:
                            extra["cosine_h"] = pair_metrics(
                                bundles["h"][0][episode_index, step].detach().cpu().numpy(),
                                bundles["h"][1][episode_index, step].detach().cpu().numpy(),
                            )["cosine"]
                            extra["cosine_concat"] = pair_metrics(
                                bundles["concat"][0][episode_index, step].detach().cpu().numpy(),
                                bundles["concat"][1][episode_index, step].detach().cpu().numpy(),
                            )["cosine"]
                        gate_value = float(gate[episode_index].detach().cpu()) if gate.ndim == 1 else float(gate[episode_index, HORIZONS.index(horizon)].detach().cpu())
                        rows.append({
                            "fork_id": f"{regime}:{episode}:{branch}:force_reversal",
                            "regime": regime,
                            "model": model_name,
                            "seed": seed,
                            "protocol": protocol,
                            "horizon": horizon,
                            "episode": episode,
                            "branch_time": branch,
                            **physical,
                            "raster_mse": raster["raster_mse"],
                            "raster_l1": raster["raster_l1"],
                            "identical_raster": raster["identical_raster"],
                            "changed_pixel_fraction": raster["changed_pixel_fraction"],
                            "pred_norm": primary["pred_norm"],
                            "target_norm": primary["target_norm"],
                            "cosine": primary["cosine"],
                            "cosine_status": primary["cosine_status"],
                            "response_ratio": primary["response_ratio"],
                            "projection": primary["projection"],
                            "cosine_h": extra["cosine_h"],
                            "cosine_z": extra["cosine_z"],
                            "cosine_concat": extra["cosine_concat"],
                            "same_state_action_norm": gate_value,
                            "git_commit": git,
                            "checkpoint_sha256": digest,
                            "metric_version": METRIC_VERSION,
                        })
                print(f"scored {regime} {model_name} s{seed} {protocol} t={branch}", flush=True)
    row_path = Path(os.environ.get("PRETEST_ROWS", str(row_path or ROW_PATH)))
    new_file = not row_path.is_file()
    row_path.parent.mkdir(parents=True, exist_ok=True)
    with row_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        _append_rows(handle, writer, rows)
    del model


def _moments(dump: str, split_version: str):
    cfg = DataConfig(
        dataset_path=str(ROOT / dump),
        img_size=64,
        augment=False,
        split_version=split_version,
        normalize_actions=True,
    )
    train = DeformablePhysicsDataset(cfg, split="train")
    return train.action_mean.numpy().astype(np.float32), train.action_std.numpy().astype(np.float32)


def main() -> None:
    git = _git()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device}", flush=True)
    high_episodes = list(range(1440, 1620))
    legacy_episodes = list(range(1800, 1980))
    limit = int(os.environ.get("PRETEST_LIMIT", "0"))
    if limit:
        high_episodes = high_episodes[:limit]
        legacy_episodes = legacy_episodes[:limit]
    build_cache("mpm_high", high_episodes, "configs/environments/mpm_high_excitation.yaml", "outputs/physics_mpm_high", 8)
    build_cache("mpm_legacy", legacy_episodes, "configs/environments/mpm_legacy.yaml", "outputs/physics_mpm", 3)
    high_mean, high_std = _moments("outputs/physics_mpm_high", "fresh_pool_v2")
    legacy_mean, legacy_std = _moments("outputs/physics_mpm", "legacy_v1")
    jobs = []
    for seed in (42, 43, 44, 45, 46):
        jobs.append(("mpm_high", f"outputs/mpm_high_runs/jepa_s{seed}/last.pt", "vicreg_action", seed, high_mean, high_std))
        jobs.append(("mpm_high", f"outputs/mpm_high_runs/noaction_s{seed}/last.pt", "vicreg_noaction", seed, high_mean, high_std))
        jobs.append(("mpm_high", f"outputs/mpm_high_runs/mse_s{seed}/last.pt", "mse_action", seed, high_mean, high_std))
        jobs.append(("mpm_high", f"outputs/mpm_high_runs/rssm_s{seed}/last.pt", "rssm", seed, high_mean, high_std))
    legacy_paths = {
        ("vicreg_action", 42): "outputs/vicreg_nocrop/last.pt",
        ("vicreg_action", 43): "outputs/vicreg_nocrop_seed43/last.pt",
        ("vicreg_action", 44): "results/reviewer_strengthening/05_more_seeds/vicreg_action_s44/last.pt",
        ("vicreg_action", 45): "results/reviewer_strengthening/05_more_seeds/vicreg_action_s45/last.pt",
        ("vicreg_noaction", 42): "outputs/no_action/last.pt",
        ("vicreg_noaction", 43): "outputs/vicreg_nocrop_noaction_seed43/last.pt",
        ("vicreg_noaction", 44): "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s44/last.pt",
        ("vicreg_noaction", 45): "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s45/last.pt",
        ("mse_action", 42): "results/reviewer_strengthening/06_corrected_objective/mse_action_s42/last.pt",
        ("mse_action", 43): "results/reviewer_strengthening/06_corrected_objective/mse_action_s43/last.pt",
    }
    for (name, seed), path in legacy_paths.items():
        jobs.append(("mpm_legacy", path, name, seed, legacy_mean, legacy_std))
    done = _done_keys(Path(os.environ.get("PRETEST_ROWS", str(ROW_PATH))))
    if limit:
        jobs = [job for job in jobs if job[0] == "mpm_high" and job[2] in {"vicreg_action", "rssm"} and job[3] == 42]
    for regime, path, name, seed, mean, std in jobs:
        # A completed model has both protocols.
        if (regime, name, str(seed), "open_loop") in done and (regime, name, str(seed), "teacher_forced") in done:
            print(f"skip {regime} {name} s{seed}", flush=True)
            continue
        score_checkpoint(regime, high_episodes if regime == "mpm_high" else legacy_episodes, ROOT / path, name, seed, mean, std, device, git)
    print("PRETEST_EVAL_DONE", flush=True)


def run_frozen_test() -> None:
    """Score episodes 1620-1799. The test script must create the lock first."""
    episodes = list(range(1620, 1800))
    build_cache(
        "mpm_high",
        episodes,
        "configs/environments/mpm_high_excitation.yaml",
        "outputs/physics_mpm_high",
        8,
        allow_test=True,
    )
    from src.config import DataConfig
    from src.dataset import DeformablePhysicsDataset
    cfg = DataConfig(
        dataset_path=str(ROOT / "outputs/physics_mpm_high"),
        img_size=64,
        augment=False,
        split_version="fresh_pool_v2",
        normalize_actions=True,
    )
    train = DeformablePhysicsDataset(cfg, split="train")
    mean = train.action_mean.numpy().astype(np.float32)
    std = train.action_std.numpy().astype(np.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    git = _git()
    out = ROOT / "experiments" / "runs" / "mpm_high_test_final" / "branch_rows.csv"
    jobs = []
    for seed in (42, 43, 44, 45, 46):
        jobs.append((f"outputs/mpm_high_runs/jepa_s{seed}/last.pt", "vicreg_action", seed))
        jobs.append((f"outputs/mpm_high_runs/noaction_s{seed}/last.pt", "vicreg_noaction", seed))
        jobs.append((f"outputs/mpm_high_runs/mse_s{seed}/last.pt", "mse_action", seed))
        jobs.append((f"outputs/mpm_high_runs/rssm_s{seed}/last.pt", "rssm", seed))
    for path, name, seed in jobs:
        score_checkpoint("mpm_high_test", episodes, ROOT / path, name, seed, mean, std, device, git, row_path=out)


if __name__ == "__main__":
    main()
