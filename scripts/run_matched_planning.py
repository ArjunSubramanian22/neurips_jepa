#!/usr/bin/env python3
"""Finite-candidate planning with the same goal image for every model.

JEPA and RSSM both see the current frame, the candidate action sequence, and
the final goal image. RSSM does not receive the goal trajectory or the actions
that generated it. Physical endpoint error is recorded separately from latent
top-1. Validation episodes only.
"""

from __future__ import annotations

import argparse
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
from src.counterfactual_study import assert_high_validation_episodes
from src.dataset import DeformablePhysicsDataset
from src.environments import environment_from_dict
from src.registry import utc_now
from src.train import load_model_from_checkpoint

LIBRARY_EPISODES = (0, 80, 160, 240, 320, 400, 480, 560)
HORIZONS = (5, 10)
CHECKPOINTS = {
    "vicreg_action": "outputs/mpm_high_runs/jepa_s{seed}/last.pt",
    "vicreg_noaction": "outputs/mpm_high_runs/noaction_s{seed}/last.pt",
    "mse_action": "outputs/mpm_high_runs/mse_s{seed}/last.pt",
    "rssm": "outputs/mpm_high_runs/rssm_s{seed}/last.pt",
}


def _simulate_sequence(snapshot, actions, mpm):
    sim = MLSMPM2D(mpm)
    sim.restore(snapshot)
    frames = []
    for action in actions:
        apply_action_frames(sim, np.asarray(action, dtype=np.float32), 1)
        frames.append(render_rgb(sim.x, sim.v, mpm.img_size, mpm.render_sigma))
    return frames, np.array(sim.x, copy=True), np.array(sim.v, copy=True)


def _norm_sequence(actions, mean, std, device) -> torch.Tensor:
    raw = np.asarray(actions, dtype=np.float32).reshape(1, -1, actions.shape[-1])
    tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (tensor - center) / scale


def _score_jepa_sequence(model, frame0, goal_frame, actions, mean, std, device) -> float:
    video_now = torch.as_tensor(frame0, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    goal = torch.as_tensor(goal_frame, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    z0 = model.encode(video_now, target=False)
    pred = model.rollout(z0, _norm_sequence(actions, mean, std, device))
    target = model.encode(goal, target=True)
    return float(torch.mean((pred[:, -1] - target) ** 2).item())


def _score_rssm_goal_image(model, frame0, goal_frame, actions, mean, std, device) -> float:
    """Imagine the candidate. The target is the final image from a fresh state.

    No goal-action tensor is built. ``observe`` is only used with ``action is
    None``, once for the current frame and once inside ``encode``.
    """
    now = torch.as_tensor(frame0, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    goal = torch.as_tensor(goal_frame, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    candidate = _norm_sequence(actions, mean, std, device)
    deter, stochastic = model.initial_state(1, device)
    deter, stochastic, _feature = model.observe(deter, stochastic, None, now)
    pred = None
    for step in range(candidate.shape[1]):
        deter, stochastic, pred = model.imagine(deter, stochastic, candidate[:, step])
    _pred_h, pred_z = model.split_feature(pred)
    target = model.encode(goal, target=True)
    _tgt_h, tgt_z = model.split_feature(target)
    return float(torch.mean((pred_z - tgt_z) ** 2).item())


def _endpoint_rms(pos, goal_pos) -> float:
    return float(np.sqrt(np.mean(np.sum((pos - goal_pos) ** 2, axis=-1))))


def _libraries(replays: list[dict], horizon: int) -> dict[str, list[np.ndarray]]:
    constant = []
    sequence = []
    for replay in replays:
        actions = np.asarray(replay["actions"], dtype=np.float32)
        if actions.shape[0] < horizon:
            raise RuntimeError(f"training episode has {actions.shape[0]} actions, need {horizon}.")
        constant.append(np.repeat(actions[0].reshape(1, -1), horizon, axis=0))
        sequence.append(actions[:horizon])
        sequence.append(actions[5 : 5 + horizon])
    return {"constant": constant, "sequence": sequence}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-information planning on high-excitation validation.")
    parser.add_argument("--episodes", type=int, nargs="+", default=[1440, 1441])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--models", nargs="+", default=["vicreg_action", "rssm"])
    parser.add_argument("--libraries", nargs="+", default=["constant", "sequence"])
    parser.add_argument("--out", default="experiments/runs/mpm_high_planning_matched_pilot")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    episodes = assert_high_validation_episodes(args.episodes)
    unknown = [name for name in args.models if name not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}.")
    for episode in LIBRARY_EPISODES:
        if episode >= 1440:
            raise RuntimeError("planning library must come from training episodes.")
    spec = environment_from_dict(
        yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text(encoding="utf-8"))
    )
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
    del train
    replays = [
        replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        for episode in LIBRARY_EPISODES
    ]
    libraries = {horizon: _libraries(replays, horizon) for horizon in HORIZONS}
    futures = {}
    for episode in episodes:
        replay = replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        for horizon in HORIZONS:
            for kind in args.libraries:
                library = libraries[horizon][kind]
                goal_index = episodes.index(episode) % len(library)
                for index, actions in enumerate(library):
                    frames, pos, vel = _simulate_sequence(replay["snapshots"][0], actions, spec.mpm)
                    futures[(episode, horizon, kind, index)] = {
                        "frame0": replay["frames"][0],
                        "frames": frames,
                        "pos": pos,
                        "vel": vel,
                        "goal_index": goal_index,
                    }
        print(f"matched physics {episode}", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    raw = []
    for name in args.models:
        for seed in args.seeds:
            path = ROOT / CHECKPOINTS[name].format(seed=seed)
            model, _cfg = load_model_from_checkpoint(path, device)
            model.eval()
            is_rssm = type(model).__name__ == "CompactRSSM"
            if name == "rssm" and not is_rssm:
                raise RuntimeError(f"{path} is not a CompactRSSM.")
            if name != "rssm" and is_rssm:
                raise RuntimeError(f"{path} is an RSSM but was requested as {name}.")
            with torch.no_grad():
                for episode in episodes:
                    for horizon in HORIZONS:
                        for kind in args.libraries:
                            library = libraries[horizon][kind]
                            goal = futures[(episode, horizon, kind, 0)]["goal_index"]
                            goal_item = futures[(episode, horizon, kind, goal)]
                            scores = []
                            for index in range(len(library)):
                                item = futures[(episode, horizon, kind, index)]
                                if is_rssm:
                                    distance = _score_rssm_goal_image(
                                        model, goal_item["frame0"], goal_item["frames"][horizon - 1], library[index], mean, std, device
                                    )
                                else:
                                    distance = _score_jepa_sequence(
                                        model, goal_item["frame0"], goal_item["frames"][horizon - 1], library[index], mean, std, device
                                    )
                                scores.append(distance)
                            order = np.argsort(np.asarray(scores))
                            chosen = int(order[0])
                            goal_pos = goal_item["pos"]
                            physical = [
                                _endpoint_rms(futures[(episode, horizon, kind, index)]["pos"], goal_pos)
                                for index in range(len(library))
                            ]
                            physical_order = np.argsort(np.asarray(physical))
                            raw.append({
                                "episode": episode,
                                "model": name,
                                "seed": seed,
                                "horizon": horizon,
                                "library": kind,
                                "n_candidates": len(library),
                                "goal_information": "final_image",
                                "goal_index": goal,
                                "chosen_index": chosen,
                                "latent_top1": int(chosen == goal),
                                "physical_top1": int(int(physical_order[0]) == goal),
                                "chosen_rms": physical[chosen],
                                "oracle_rms": physical[int(physical_order[0])],
                                "regret": physical[chosen] - physical[int(physical_order[0])],
                                "rank_of_goal": int(np.where(order == goal)[0][0]) + 1,
                            })
            del model
            print(f"matched {name} s{seed}", flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0].keys()))
        writer.writeheader()
        writer.writerows(raw)
    summary = []
    for name in args.models:
        for seed in args.seeds:
            for horizon in HORIZONS:
                for kind in args.libraries:
                    group = [
                        row for row in raw
                        if row["model"] == name and row["seed"] == seed and row["horizon"] == horizon and row["library"] == kind
                    ]
                    summary.append({
                        "model": name,
                        "seed": seed,
                        "horizon": horizon,
                        "library": kind,
                        "n_episodes": len(group),
                        "n_candidates": group[0]["n_candidates"],
                        "goal_information": "final_image",
                        "latent_top1": float(np.mean([row["latent_top1"] for row in group])),
                        "mean_chosen_rms": float(np.mean([row["chosen_rms"] for row in group])),
                        "mean_regret": float(np.mean([row["regret"] for row in group])),
                        "mean_rank": float(np.mean([row["rank_of_goal"] for row in group])),
                        "physical_oracle_top1": float(np.mean([row["physical_top1"] for row in group])),
                    })
    with (out / "seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (out / "protocol.json").write_text(json.dumps({
        "goal_information": "final_image",
        "rssm_receives_goal_actions": False,
        "rssm_receives_goal_trajectory": False,
        "training_episodes": list(LIBRARY_EPISODES),
        "validation_episodes": episodes,
        "seeds": list(args.seeds),
        "models": list(args.models),
        "libraries": list(args.libraries),
        "horizons": list(HORIZONS),
        "test_episodes_read": False,
        "timestamp": utc_now(),
        "note": "latent_top1 is candidate identity. mean_chosen_rms is the physical endpoint error of that choice. They are not interchangeable.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
