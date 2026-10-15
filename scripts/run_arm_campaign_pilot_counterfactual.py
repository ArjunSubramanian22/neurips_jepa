#!/usr/bin/env python3
"""Open-loop action contrast on the one-epoch arm pilots.

Validation episodes 480-485 only. Test episodes 600-719 are not read.
"""

from __future__ import annotations

import argparse
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
from src.envs.arm_campaign import kind_for, reflect_actions, rollout
from src.envs.mujoco_arm import MujocoArm
from src.train import load_model_from_checkpoint

EPISODES = tuple(range(480, 486))
HORIZONS = (1, 10)
JOBS = (
    ("outputs/mujoco_arm_campaign_runs/mse_epoch1/last.pt", "mse_action"),
    ("outputs/mujoco_arm_campaign_runs/mse_noaction_epoch1/last.pt", "mse_noaction"),
)


def _chw(frame: np.ndarray, device) -> torch.Tensor:
    array = np.transpose(frame, (2, 0, 1))
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot arm counterfactual. Validation only.")
    parser.add_argument("--checkpoint", nargs=2, action="append", metavar=("PATH", "NAME"))
    parser.add_argument(
        "--out",
        default="experiments/runs/arm_campaign_v2/epoch1_counterfactual.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    jobs = tuple((path, name) for path, name in args.checkpoint) if args.checkpoint else JOBS
    if any(episode >= 600 for episode in EPISODES):
        raise SystemExit("refusing: this pilot must stay out of the arm test suffix.")
    cfg = DataConfig(
        dataset_path=str(ROOT / "outputs/mujoco_arm_campaign"),
        img_size=64,
        augment=False,
        split_version="arm_campaign_v1",
        normalize_actions=True,
        action_dim=2,
        episode_length=16,
    )
    train = DeformablePhysicsDataset(cfg, split="train")
    mean = train.action_mean.numpy().astype(np.float32)
    std = train.action_std.numpy().astype(np.float32)
    if train.normalization_provenance["uses_test"]:
        raise SystemExit("refusing: action moments used the test split.")
    del train
    saved_actions = np.load(ROOT / "outputs/mujoco_arm_campaign/actions.npy")
    env = MujocoArm("push_object")
    forks = []
    try:
        for episode in EPISODES:
            actions = saved_actions[episode]
            recorded = rollout(env, episode, 16, actions)
            reflected = rollout(env, episode, 16, reflect_actions(env, actions))
            forks.append({"episode": episode, "kind": kind_for(episode), "recorded": recorded, "reflected": reflected})
    finally:
        env.close()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for path, name in jobs:
        model, _loaded = load_model_from_checkpoint(ROOT / path, device)
        model.eval()
        with torch.no_grad():
            for fork in forks:
                for horizon in HORIZONS:
                    def norm(actions: np.ndarray) -> torch.Tensor:
                        raw = np.asarray(actions[:horizon], dtype=np.float32)
                        tensor = torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
                        center = torch.as_tensor(mean, dtype=torch.float32, device=device)
                        scale = torch.as_tensor(std, dtype=torch.float32, device=device)
                        return (tensor - center) / scale

                    now = _chw(fork["recorded"]["frames"][0], device)
                    z0 = model.encode(now, target=False)
                    pred_r = model.rollout(z0, norm(fork["recorded"]["actions"]))[0, horizon - 1]
                    pred_v = model.rollout(z0, norm(fork["reflected"]["actions"]))[0, horizon - 1]
                    tgt_r = model.encode(_chw(fork["recorded"]["frames"][horizon], device), target=True)[0]
                    tgt_v = model.encode(_chw(fork["reflected"]["frames"][horizon], device), target=True)[0]
                    metrics = pair_metrics(
                        (pred_v - pred_r).detach().cpu().numpy(),
                        (tgt_v - tgt_r).detach().cpu().numpy(),
                    )
                    rows.append({
                        "episode": fork["episode"],
                        "kind": fork["kind"],
                        "model": name,
                        "horizon": horizon,
                        "cosine": metrics["cosine"],
                        "response_ratio": metrics["response_ratio"],
                        "pred_norm": metrics["pred_norm"],
                        "target_norm": metrics["target_norm"],
                    })
        del model
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for _path, name in jobs:
        for horizon in HORIZONS:
            group = [row for row in rows if row["model"] == name and row["horizon"] == horizon]
            defined = [row["cosine"] for row in group if row["cosine"] is not None]
            summary.append({
                "model": name,
                "horizon": horizon,
                "n": len(group),
                "n_defined_cosine": len(defined),
                "median_cosine": None if not defined else float(np.median(defined)),
                "median_pred_norm": float(np.median([row["pred_norm"] for row in group])),
                "median_target_norm": float(np.median([row["target_norm"] for row in group])),
                "median_response_ratio": None
                if any(row["response_ratio"] is None for row in group)
                else float(np.median([row["response_ratio"] for row in group])),
            })
    out.write_text(json.dumps({
        "episodes": list(EPISODES),
        "test_episodes_read": False,
        "checkpoints": [{"path": path, "name": name} for path, name in jobs],
        "rows": rows,
        "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
