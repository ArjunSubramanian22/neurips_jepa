#!/usr/bin/env python3
"""Counterfactual check on the existing MuJoCo arm validation tail.

Actions are joint-position targets, so the intervention reflects each target
through the center of its control range. It does not reuse the MPM force flip.
Episodes 24-31 are validation. There is no arm test split.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.counterfactual_study import pair_metrics
from src.dataset import DeformablePhysicsDataset
from src.envs.mujoco_arm import MujocoArm
from src.registry import utc_now
from src.train import load_model_from_checkpoint

EPISODES = list(range(24, 32))
HORIZONS = (1, 7)
LOW = np.array([-1.6, -2.4], dtype=np.float64)
HIGH = np.array([1.6, 2.2], dtype=np.float64)
OUT = ROOT / "experiments" / "runs" / "arm_counterfactual"


def _reflect(action: np.ndarray) -> np.ndarray:
    reflected = LOW + HIGH - np.asarray(action, dtype=np.float64)
    return np.clip(reflected, LOW, HIGH)


def _rollout(env: MujocoArm, episode: int, actions: list[np.ndarray]):
    env.reset(episode)
    frames = [env.render()]
    positions = [env.tracked_positions().copy()]
    velocities = [env.tracked_velocities().copy()]
    for action in actions:
        env.step(action)
        frames.append(env.render())
        positions.append(env.tracked_positions().copy())
        velocities.append(env.tracked_velocities().copy())
    return frames, positions, velocities


def _chw(frame: np.ndarray, device) -> torch.Tensor:
    array = np.transpose(frame, (2, 0, 1))
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def main() -> None:
    cfg = DataConfig(
        dataset_path=str(ROOT / "outputs/mujoco_arm"),
        img_size=64,
        augment=False,
        split_version="arm_tail_v1",
        normalize_actions=True,
    )
    train = DeformablePhysicsDataset(cfg, split="train")
    mean = train.action_mean.numpy().astype(np.float32)
    std = train.action_std.numpy().astype(np.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env = MujocoArm("push_object")
    forks = []
    try:
        for episode in EPISODES:
            recorded_action = env.sample_action() if False else None
            env.reset(episode)
            # Replay the dataset command. Push and fold are constant; random
            # uses the environment RNG advanced by reset(episode).
            from src.envs.arm_data import _action_for
            actions = [_action_for(env, episode) for _ in range(7)]
            reflected = [_reflect(action) for action in actions]
            rec_frames, rec_pos, rec_vel = _rollout(env, episode, actions)
            rev_frames, rev_pos, rev_vel = _rollout(env, episode, reflected)
            forks.append({
                "episode": episode,
                "actions": actions,
                "reflected": reflected,
                "rec_frames": rec_frames,
                "rev_frames": rev_frames,
                "rec_pos": rec_pos,
                "rev_pos": rev_pos,
                "rec_vel": rec_vel,
                "rev_vel": rev_vel,
            })
    finally:
        env.close()
    jobs = []
    for seed in (42, 43, 44, 45, 46):
        jobs.append((f"outputs/mujoco_arm_runs/jepa_s{seed}/last.pt", "jepa", seed))
        jobs.append((f"outputs/mujoco_arm_runs/rssm_s{seed}/last.pt", "rssm", seed))
        jobs.append((f"outputs/mujoco_arm_runs/latent_tf_s{seed}/last.pt", "latent_tf", seed))
    rows = []
    for path, name, seed in jobs:
        model, _cfg = load_model_from_checkpoint(ROOT / path, device)
        model.eval()
        is_rssm = type(model).__name__ == "CompactRSSM"
        with torch.no_grad():
            for fork in forks:
                for horizon in HORIZONS:
                    def norm(actions):
                        raw = np.stack(actions[:horizon], axis=0).astype(np.float32)
                        tensor = torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
                        center = torch.as_tensor(mean, dtype=torch.float32, device=device)
                        scale = torch.as_tensor(std, dtype=torch.float32, device=device)
                        return (tensor - center) / scale
                    now = _chw(fork["rec_frames"][0], device)
                    if is_rssm:
                        def rssm_z(frames, actions):
                            video = torch.cat([_chw(frame, device) for frame in frames[: horizon + 1]], dim=0).unsqueeze(0)
                            deter, stochastic = model.initial_state(1, device)
                            deter, stochastic, _ = model.observe(deter, stochastic, None, video[:, 0])
                            pred = None
                            for step in range(horizon):
                                deter, stochastic, pred = model.imagine(deter, stochastic, actions[:, step])
                            _h, pred_z = model.split_feature(pred)
                            td, ts = model.initial_state(1, device)
                            td, ts, _ = model.observe(td, ts, None, video[:, 0])
                            tgt = None
                            for step in range(horizon):
                                td, ts, tgt = model.observe(td, ts, actions[:, step], video[:, step + 1])
                            _h, tgt_z = model.split_feature(tgt)
                            return pred_z[0].detach().cpu().numpy(), tgt_z[0].detach().cpu().numpy()
                        pred_r, tgt_r = rssm_z(fork["rec_frames"], norm(fork["actions"]))
                        pred_v, tgt_v = rssm_z(fork["rev_frames"], norm(fork["reflected"]))
                    else:
                        z0 = model.encode(now, target=False)
                        pred_r = model.rollout(z0, norm(fork["actions"]))[0, horizon - 1]
                        pred_v = model.rollout(z0, norm(fork["reflected"]))[0, horizon - 1]
                        tgt_r = model.encode(_chw(fork["rec_frames"][horizon], device), target=True)[0]
                        tgt_v = model.encode(_chw(fork["rev_frames"][horizon], device), target=True)[0]
                        pred_r = pred_r.detach().cpu().numpy()
                        pred_v = pred_v.detach().cpu().numpy()
                        tgt_r = tgt_r.detach().cpu().numpy()
                        tgt_v = tgt_v.detach().cpu().numpy()
                    metrics = pair_metrics(pred_v - pred_r, tgt_v - tgt_r)
                    pos_a = fork["rec_pos"][horizon]
                    pos_b = fork["rev_pos"][horizon]
                    vel_a = fork["rec_vel"][horizon]
                    vel_b = fork["rev_vel"][horizon]
                    raster_a = fork["rec_frames"][horizon].astype(np.float64)
                    raster_b = fork["rev_frames"][horizon].astype(np.float64)
                    rows.append({
                        "episode": fork["episode"],
                        "model": name,
                        "seed": seed,
                        "horizon": horizon,
                        "protocol": "open_loop",
                        "rms_pos": float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=-1)))),
                        "rms_vel": float(np.sqrt(np.mean(np.sum((vel_a - vel_b) ** 2, axis=-1)))),
                        "raster_mse": float(np.mean(((raster_a - raster_b) / 255.0) ** 2)),
                        "pred_norm": metrics["pred_norm"],
                        "target_norm": metrics["target_norm"],
                        "cosine": metrics["cosine"],
                        "response_ratio": metrics["response_ratio"],
                        "projection": metrics["projection"],
                    })
        del model
        print(f"arm {name} s{seed}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            stored = dict(row)
            for key in ("cosine", "response_ratio", "projection"):
                if stored[key] is None:
                    stored[key] = ""
            writer.writerow(stored)
    lines = [
        "# Arm validation counterfactual",
        "",
        "Episodes 24-31. Intervention reflects shoulder and elbow targets through the control-range center.",
        "This is not the MPM force reversal. Open loop only. Horizons 1 and 7, the longest horizon at T=8.",
        "",
    ]
    for name in ("jepa", "rssm", "latent_tf"):
        for horizon in HORIZONS:
            cosines = []
            for seed in (42, 43, 44, 45, 46):
                group = [row for row in rows if row["model"] == name and row["seed"] == seed and row["horizon"] == horizon]
                defined = [row["cosine"] for row in group if row["cosine"] is not None]
                value = None if not defined else float(np.median(defined))
                cosines.append(value)
                lines.append(f"- {name} seed {seed} H{horizon}: cosine {value} pred {np.median([row['pred_norm'] for row in group]):.4g}")
            lines.append(f"- {name} H{horizon} seed cosines {cosines}")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "protocol.json").write_text(json.dumps({
        "episodes": EPISODES,
        "horizons": list(HORIZONS),
        "intervention": "reflect joint targets through control-range center",
        "control_low": LOW.tolist(),
        "control_high": HIGH.tolist(),
        "timestamp": utc_now(),
        "sealed_dump": False,
        "reason": "A new sealed arm dump would be extra infrastructure. Existing validation tail is reported instead.",
    }, indent=2) + "\n", encoding="utf-8")
    print("ARM_DONE")


if __name__ == "__main__":
    main()
