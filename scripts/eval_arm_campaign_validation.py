#!/usr/bin/env python3
"""Open-loop arm diagnostic on validation episodes 480-599 only.

Scores best.pt and last.pt for the preregistered grid. Episodes 600-719
are refused.
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
from src.envs.arm_campaign import kind_for, reflect_actions, rollout
from src.envs.mujoco_arm import MujocoArm
from src.train import load_model_from_checkpoint

EPISODES = tuple(range(480, 600))
HORIZONS = (1, 5, 10)
RUNS = ROOT / "outputs" / "mujoco_arm_campaign_runs"
OUT = ROOT / "experiments" / "runs" / "arm_campaign_v2" / "validation_grid"
MODELS = ("vicreg", "mse", "noaction", "rssm")
SEEDS = (42, 43, 44, 45, 46)


def _chw(frame: np.ndarray, device) -> torch.Tensor:
    array = np.transpose(frame, (2, 0, 1))
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def _norm(actions: np.ndarray, mean: np.ndarray, std: np.ndarray, device, horizon: int) -> torch.Tensor:
    raw = np.asarray(actions[:horizon], dtype=np.float32)
    tensor = torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (tensor - center) / scale


def _rssm_z(model, frames, actions, device, horizon: int):
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


def main(episodes: tuple[int, ...] = EPISODES, out: Path = OUT) -> None:
    if episodes == EPISODES:
        if any(episode >= 600 or episode < 480 for episode in episodes):
            raise SystemExit("refusing: validation scoring must stay inside episodes 480-599.")
    elif list(episodes) != list(range(600, 720)):
        raise SystemExit("refusing: the only other allowed slice is the sealed test 600-719.")
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
    if train.normalization_provenance["uses_test"] or max(train.global_episode_ids) >= 480:
        raise SystemExit("refusing: action moments are not train-only.")
    mean = train.action_mean.numpy().astype(np.float32)
    std = train.action_std.numpy().astype(np.float32)
    del train
    saved = np.load(ROOT / "outputs/mujoco_arm_campaign/actions.npy")
    env = MujocoArm("push_object")
    forks = []
    try:
        for episode in episodes:
            actions = saved[episode]
            recorded = rollout(env, episode, 16, actions)
            reflected = rollout(env, episode, 16, reflect_actions(env, actions))
            forks.append({
                "episode": episode,
                "kind": kind_for(episode),
                "recorded": recorded,
                "reflected": reflected,
            })
            if episode % 20 == 0:
                print(f"fork {episode}", flush=True)
    finally:
        env.close()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for model_name in MODELS:
        for seed in SEEDS:
            run = RUNS / f"{model_name}_s{seed}"
            for checkpoint in ("best", "last"):
                path = run / f"{checkpoint}.pt"
                if not path.is_file():
                    raise SystemExit(f"missing {path}")
                model, _loaded = load_model_from_checkpoint(path, device)
                model.eval()
                is_rssm = type(model).__name__ == "CompactRSSM"
                with torch.no_grad():
                    for fork in forks:
                        for horizon in HORIZONS:
                            recorded_actions = _norm(fork["recorded"]["actions"], mean, std, device, horizon)
                            reflected_actions = _norm(fork["reflected"]["actions"], mean, std, device, horizon)
                            if is_rssm:
                                pred_r, tgt_r = _rssm_z(model, fork["recorded"]["frames"], recorded_actions, device, horizon)
                                pred_v, tgt_v = _rssm_z(model, fork["reflected"]["frames"], reflected_actions, device, horizon)
                            else:
                                now = _chw(fork["recorded"]["frames"][0], device)
                                z0 = model.encode(now, target=False)
                                pred_r = model.rollout(z0, recorded_actions)[0, horizon - 1].detach().cpu().numpy()
                                pred_v = model.rollout(z0, reflected_actions)[0, horizon - 1].detach().cpu().numpy()
                                tgt_r = model.encode(_chw(fork["recorded"]["frames"][horizon], device), target=True)[0].detach().cpu().numpy()
                                tgt_v = model.encode(_chw(fork["reflected"]["frames"][horizon], device), target=True)[0].detach().cpu().numpy()
                            metrics = pair_metrics(pred_v - pred_r, tgt_v - tgt_r)
                            pos_a = fork["recorded"]["xy"][horizon]
                            pos_b = fork["reflected"]["xy"][horizon]
                            image_a = fork["recorded"]["frames"][horizon].astype(np.float64)
                            image_b = fork["reflected"]["frames"][horizon].astype(np.float64)
                            rows.append({
                                "model": model_name,
                                "seed": seed,
                                "checkpoint": checkpoint,
                                "episode": fork["episode"],
                                "kind": fork["kind"],
                                "horizon": horizon,
                                "cosine": metrics["cosine"],
                                "cosine_status": metrics["cosine_status"],
                                "response_ratio": metrics["response_ratio"],
                                "pred_norm": metrics["pred_norm"],
                                "target_norm": metrics["target_norm"],
                                "object_l2": float(np.linalg.norm(pos_a - pos_b)),
                                "raster_mse": float(np.mean(((image_a - image_b) / 255.0) ** 2)),
                            })
                del model
                print(f"scored {model_name} s{seed} {checkpoint}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            stored = dict(row)
            for key in ("cosine", "response_ratio"):
                if stored[key] is None:
                    stored[key] = ""
            writer.writerow(stored)
    summary = []
    for model_name in MODELS:
        for checkpoint in ("best", "last"):
            for horizon in HORIZONS:
                seed_cosine = []
                seed_ratio = []
                seed_pred = []
                for seed in SEEDS:
                    group = [
                        row for row in rows
                        if row["model"] == model_name and row["seed"] == seed and row["checkpoint"] == checkpoint and row["horizon"] == horizon
                    ]
                    defined = [row["cosine"] for row in group if row["cosine"] is not None]
                    ratios = [row["response_ratio"] for row in group if row["response_ratio"] is not None]
                    seed_cosine.append(None if not defined else float(np.median(defined)))
                    seed_ratio.append(None if not ratios else float(np.median(ratios)))
                    seed_pred.append(float(np.median([row["pred_norm"] for row in group])))
                finite = [value for value in seed_cosine if value is not None]
                summary.append({
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "horizon": horizon,
                    "n_episodes": len(episodes),
                    "cosine_median_of_seeds": None if not finite else float(np.median(finite)),
                    "seed_cosine": seed_cosine,
                    "seed_response_ratio": seed_ratio,
                    "seed_pred_norm": seed_pred,
                    "object_l2_median": float(np.median([
                        row["object_l2"] for row in rows
                        if row["model"] == model_name and row["checkpoint"] == checkpoint and row["horizon"] == horizon
                    ])),
                    "raster_mse_median": float(np.median([
                        row["raster_mse"] for row in rows
                        if row["model"] == model_name and row["checkpoint"] == checkpoint and row["horizon"] == horizon
                    ])),
                })
    (out / "summary.json").write_text(json.dumps({
        "episodes": [episodes[0], episodes[-1]],
        "test_episodes_read": episodes[0] >= 600,
        "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def open_test_once() -> None:
    import os

    freeze = ROOT / "experiments" / "arm_protocol_freeze.json"
    out = ROOT / "experiments" / "runs" / "arm_campaign_v2" / "test_once"
    lock = out / "TEST_EVALUATED.lock"
    if not freeze.is_file():
        raise SystemExit("refusing: experiments/arm_protocol_freeze.json is missing.")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    if payload.get("episodes") != [600, 719] or payload.get("model_selection_on_test"):
        raise SystemExit("refusing: the freeze does not match the sealed 120-episode test.")
    if lock.exists():
        raise SystemExit(f"refusing: {lock} already exists.")
    out.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("ARM TEST 600-719 HAS BEEN OPENED\n")
    main(tuple(range(600, 720)), out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--open-test-once":
        open_test_once()
    else:
        main()
