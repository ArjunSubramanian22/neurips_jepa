#!/usr/bin/env python3
"""Finite-candidate planning on high-excitation validation episodes 1440-1463.

The candidate library is drawn from training episodes only. The goal action
is held constant for the horizon. Test episodes are not read.
"""

from __future__ import annotations

import csv
import json
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
from src.counterfactual_study import assert_high_validation_episodes, rank_correlation
from src.dataset import DeformablePhysicsDataset
from src.environments import environment_from_dict
from src.registry import utc_now
from src.train import load_model_from_checkpoint

EPISODES = list(range(1440, 1464))
HORIZONS = (5, 10)
LIBRARY_EPISODES = (0, 80, 160, 240, 320, 400, 480, 560)
OUT = ROOT / "experiments" / "runs" / "mpm_high_planning"


def _simulate(snapshot, action, horizon, mpm):
    sim = MLSMPM2D(mpm)
    sim.restore(snapshot)
    frames = []
    for _ in range(horizon):
        apply_action_frames(sim, action, 1)
        frames.append(render_rgb(sim.x, sim.v, mpm.img_size, mpm.render_sigma))
    return frames, np.array(sim.x, copy=True), np.array(sim.v, copy=True)


def _video(frame0, frames, device):
    stack = np.stack([frame0, *frames], axis=0)
    return torch.as_tensor(stack, dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def _norm(action, mean, std, device, horizon):
    raw = np.repeat(action.reshape(1, 1, -1), horizon, axis=1)
    tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (tensor - center) / scale


def _score_jepa(model, frame0, goal_frames, action, mean, std, device, horizon):
    video_now = torch.as_tensor(frame0, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    goal = torch.as_tensor(goal_frames[horizon - 1], dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    z0 = model.encode(video_now, target=False)
    pred = model.rollout(z0, _norm(action, mean, std, device, horizon))
    target = model.encode(goal, target=True)
    return float(torch.mean((pred[:, horizon - 1] - target) ** 2).item())


def _score_rssm(model, frame0, goal_frames, candidate_action, goal_action, mean, std, device, horizon):
    now = torch.as_tensor(frame0, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    goal = torch.as_tensor(np.stack(goal_frames[:horizon], axis=0), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    candidate = _norm(candidate_action, mean, std, device, horizon)
    goal_actions = _norm(goal_action, mean, std, device, horizon)
    deter, stochastic = model.initial_state(1, device)
    deter, stochastic, _ = model.observe(deter, stochastic, None, now)
    pred = None
    for step in range(horizon):
        deter, stochastic, pred = model.imagine(deter, stochastic, candidate[:, step])
    _ph, pred_z = model.split_feature(pred)
    td, ts = model.initial_state(1, device)
    td, ts, _ = model.observe(td, ts, None, now)
    tgt = None
    for step in range(horizon):
        td, ts, tgt = model.observe(td, ts, goal_actions[:, step], goal[:, step])
    _th, tgt_z = model.split_feature(tgt)
    return float(torch.mean((pred_z - tgt_z) ** 2).item())


def main() -> None:
    assert_high_validation_episodes(EPISODES)
    spec = environment_from_dict(yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text(encoding="utf-8")))
    cfg = DataConfig(dataset_path=str(ROOT / "outputs/physics_mpm_high"), img_size=64, augment=False, split_version="fresh_pool_v2", normalize_actions=True)
    train = DeformablePhysicsDataset(cfg, split="train")
    mean = train.action_mean.numpy().astype(np.float32)
    std = train.action_std.numpy().astype(np.float32)
    library = []
    for episode in LIBRARY_EPISODES:
        if episode >= 1440:
            raise RuntimeError("planning library must come from training episodes.")
        replay = replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        library.append(replay["actions"][0].astype(np.float32))
    mean_action = np.mean(np.stack(library), axis=0).astype(np.float32)
    futures = {}
    for episode in EPISODES:
        replay = replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        goal_index = EPISODES.index(episode) % len(library)
        for horizon in HORIZONS:
            for index, action in enumerate(library):
                frames, pos, vel = _simulate(replay["snapshots"][0], action, horizon, spec.mpm)
                futures[(episode, horizon, index)] = {
                    "frame0": replay["frames"][0],
                    "frames": frames,
                    "pos": pos,
                    "vel": vel,
                    "goal_index": goal_index,
                }
            frames, pos, vel = _simulate(replay["snapshots"][0], mean_action, horizon, spec.mpm)
            futures[(episode, horizon, "mean")] = {"pos": pos}
        print(f"planned physics {episode}", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    jobs = []
    for seed in (42, 43, 44, 45, 46):
        jobs.append((f"outputs/mpm_high_runs/jepa_s{seed}/last.pt", "vicreg_action", seed))
        jobs.append((f"outputs/mpm_high_runs/noaction_s{seed}/last.pt", "vicreg_noaction", seed))
        jobs.append((f"outputs/mpm_high_runs/mse_s{seed}/last.pt", "mse_action", seed))
        jobs.append((f"outputs/mpm_high_runs/rssm_s{seed}/last.pt", "rssm", seed))
    raw = []
    rng = np.random.default_rng(0)
    for path, name, seed in jobs:
        model, _cfg = load_model_from_checkpoint(ROOT / path, device)
        model.eval()
        is_rssm = type(model).__name__ == "CompactRSSM"
        with torch.no_grad():
            for episode in EPISODES:
                for horizon in HORIZONS:
                    goal = futures[(episode, horizon, 0)]["goal_index"]
                    goal_item = futures[(episode, horizon, goal)]
                    scores = []
                    for index in range(len(library)):
                        if is_rssm:
                            distance = _score_rssm(
                                model, goal_item["frame0"], goal_item["frames"], library[index], library[goal], mean, std, device, horizon
                            )
                        else:
                            distance = _score_jepa(
                                model, goal_item["frame0"], goal_item["frames"], library[index], mean, std, device, horizon
                            )
                        scores.append(distance)
                    order = np.argsort(scores)
                    chosen = int(order[0])
                    random_choice = int(rng.integers(0, len(library)))
                    goal_pos = futures[(episode, horizon, goal)]["pos"]
                    def rms(index):
                        return float(np.sqrt(np.mean(np.sum((futures[(episode, horizon, index)]["pos"] - goal_pos) ** 2, axis=-1))))
                    raw.append({
                        "episode": episode,
                        "model": name,
                        "seed": seed,
                        "horizon": horizon,
                        "goal_index": goal,
                        "chosen_index": chosen,
                        "top1": int(chosen == goal),
                        "top3": int(goal in set(int(i) for i in order[:3])),
                        "random_top1": int(random_choice == goal),
                        "mean_action_rms": rms("mean"),
                        "chosen_rms": rms(chosen),
                        "oracle_rms": 0.0,
                        "regret": rms(chosen),
                        "rank_of_goal": int(np.where(order == goal)[0][0]) + 1,
                    })
        del model
        print(f"planned {name} s{seed}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0].keys()))
        writer.writeheader()
        writer.writerows(raw)
    summary = []
    for name in ("vicreg_action", "vicreg_noaction", "mse_action", "rssm"):
        for seed in (42, 43, 44, 45, 46):
            for horizon in HORIZONS:
                group = [row for row in raw if row["model"] == name and row["seed"] == seed and row["horizon"] == horizon]
                summary.append({
                    "model": name,
                    "seed": seed,
                    "horizon": horizon,
                    "top1": float(np.mean([row["top1"] for row in group])),
                    "top3": float(np.mean([row["top3"] for row in group])),
                    "mean_regret": float(np.mean([row["regret"] for row in group])),
                    "mean_rank": float(np.mean([row["rank_of_goal"] for row in group])),
                    "random_top1": float(np.mean([row["random_top1"] for row in group])),
                })
    with (OUT / "seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / "library.json").write_text(json.dumps({
        "training_episodes": list(LIBRARY_EPISODES),
        "validation_episodes": EPISODES,
        "horizons": list(HORIZONS),
        "timestamp": utc_now(),
        "test_episodes_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    lines = ["# Planning", "", "Validation episodes 1440-1463. Eight training-set actions, held constant.", ""]
    for row in summary:
        lines.append(
            f"- {row['model']} seed {row['seed']} H{row['horizon']}: top1 {row['top1']:.3f} "
            f"top3 {row['top3']:.3f} regret {row['mean_regret']:.6g} rank {row['mean_rank']:.2f}"
        )
    lines.append("")
    lines.append("Random top-1 is about 0.125 for eight candidates. Oracle regret is zero by construction.")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
