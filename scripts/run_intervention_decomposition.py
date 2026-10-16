#!/usr/bin/env python3
"""Development sweep: does the median-ratio trap survive other interventions?

Validation episodes only. High-excitation test episodes 1620-1799 are refused.
The recorded branch is the dump rollout. The other branch changes the force
and is resimulated from the saved particle state. This does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, render_rgb, replay_episode
from src.config import DataConfig
from src.dataset import DeformablePhysicsDataset
from src.environments import environment_from_dict
from src.metrics.response_decomposition import (
    decompose_response,
    median_or_none,
    ratio_rank_disagrees_with_error,
)
from src.train import load_model_from_checkpoint

EPISODES = tuple(range(1440, 1456))
BRANCH_TIMES = (0, 4)
HORIZONS = (1, 10)
SIM_HORIZON = 10
FAMILIES = ("identity", "force_scaling", "force_zero", "force_reversal", "force_rotation")
MODEL_PATHS = {
    "vicreg_action": "outputs/mpm_high_runs/jepa_s{seed}/last.pt",
    "mse_action": "outputs/mpm_high_runs/mse_s{seed}/last.pt",
}


def _intervene(action: np.ndarray, family: str) -> np.ndarray:
    raw = np.asarray(action, dtype=np.float32).copy()
    force = raw[..., 2:4]
    if family == "identity":
        return raw
    if family == "force_scaling":
        force = force * np.float32(0.5)
    elif family == "force_zero":
        force = np.zeros_like(force)
    elif family == "force_reversal":
        force = -force
    elif family == "force_rotation":
        rotated = np.empty_like(force)
        rotated[..., 0] = -force[..., 1]
        rotated[..., 1] = force[..., 0]
        force = rotated
    else:
        raise ValueError(f"unknown family {family}")
    raw[..., 2:4] = force
    return raw


def _simulate(snapshot: dict, actions: np.ndarray, mpm) -> tuple[np.ndarray, np.ndarray]:
    sim = MLSMPM2D(mpm)
    sim.restore(snapshot)
    frames = []
    positions = []
    for action in actions:
        apply_action_frames(sim, action, 1)
        frames.append(render_rgb(sim.x, sim.v, mpm.img_size, mpm.render_sigma))
        positions.append(np.array(sim.x, copy=True))
    return np.stack(frames), np.stack(positions)


def _rms(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=-1))))


def _video(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.stack(frames), dtype=torch.float32, device=device)
    return tensor / 255.0


def _norm_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    raw = torch.as_tensor(actions, dtype=torch.float32, device=device)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (raw - center) / scale


def _score_model(model, frame_now, future_a, future_b, actions_a, actions_b) -> list[dict]:
    """Open-loop and teacher-forced contrasts at the requested horizons."""
    now = frame_now.unsqueeze(0)
    z0 = model.encode(now, target=False)
    open_a = model.rollout(z0, actions_a.unsqueeze(0))[0]
    open_b = model.rollout(z0, actions_b.unsqueeze(0))[0]
    target_a = model.encode(future_a, target=True)
    target_b = model.encode(future_b, target=True)
    rows = []
    for mode in ("open_loop", "teacher_forced"):
        for horizon in HORIZONS:
            index = int(horizon) - 1
            if mode == "open_loop":
                pred_a = open_a[index]
                pred_b = open_b[index]
            else:
                state_frame = now if horizon == 1 else future_a[index - 1].unsqueeze(0)
                state_b = now if horizon == 1 else future_b[index - 1].unsqueeze(0)
                state_a = model.encode(state_frame, target=False)
                state_b = model.encode(state_b, target=False)
                pred_a = model.predictor(state_a, actions_a[index].unsqueeze(0))[0]
                pred_b = model.predictor(state_b, actions_b[index].unsqueeze(0))[0]
            scored = decompose_response(
                (pred_b - pred_a).detach().cpu().numpy(),
                (target_b[index] - target_a[index]).detach().cpu().numpy(),
            )
            rows.append(
                {
                    "rollout_mode": mode,
                    "horizon": horizon,
                    "pred_norm": float(scored["pred_norm"][0]),
                    "target_norm": float(scored["target_norm"][0]),
                    "cosine": None if scored["cosine_status"][0] != "ok" else float(scored["cosine"][0]),
                    "cosine_status": str(scored["cosine_status"][0]),
                    "response_ratio": None if not np.isfinite(scored["response_ratio"][0]) else float(scored["response_ratio"][0]),
                    "parallel_gain": None if not np.isfinite(scored["parallel_gain"][0]) else float(scored["parallel_gain"][0]),
                    "orthogonal_norm": None if not np.isfinite(scored["orthogonal_norm"][0]) else float(scored["orthogonal_norm"][0]),
                    "normalized_vector_error": None
                    if not np.isfinite(scored["normalized_vector_error"][0])
                    else float(scored["normalized_vector_error"][0]),
                }
            )
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["seed"], row["family"], row["rollout_mode"], row["horizon"]), []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        defined = [row["cosine"] for row in group if row["cosine_status"] == "ok"]
        summary.append(
            {
                "model": key[0],
                "seed": key[1],
                "family": key[2],
                "rollout_mode": key[3],
                "horizon": key[4],
                "n_rows": len(group),
                "n_cosine": len(defined),
                "cosine_median": median_or_none(defined),
                "response_ratio_median": median_or_none([row["response_ratio"] for row in group]),
                "parallel_gain_median": median_or_none(
                    [row["parallel_gain"] for row in group if row["cosine_status"] == "ok"]
                ),
                "normalized_vector_error_median": median_or_none(
                    [row["normalized_vector_error"] for row in group if row["cosine_status"] == "ok"]
                ),
                "pred_norm_median": median_or_none([row["pred_norm"] for row in group]),
                "target_norm_median": median_or_none([row["target_norm"] for row in group]),
                "rms_pos_median": median_or_none([row["rms_pos"] for row in group]),
            }
        )
    return summary


def _comparisons(summary: list[dict], models: tuple[str, ...]) -> list[dict]:
    index = {
        (row["model"], row["seed"], row["family"], row["rollout_mode"], row["horizon"]): row for row in summary
    }
    out = []
    seeds = sorted({row["seed"] for row in summary})
    for model in models:
        for seed in seeds:
            for family in FAMILIES:
                open_row = index.get((model, seed, family, "open_loop", 10))
                teacher_row = index.get((model, seed, family, "teacher_forced", 10))
                if open_row is None or teacher_row is None:
                    continue
                if open_row["n_cosine"] < 8 or teacher_row["n_cosine"] < 8:
                    out.append(
                        {
                            "model": model,
                            "seed": seed,
                            "family": family,
                            "status": "too_few_defined_cosines",
                            "n_open": open_row["n_cosine"],
                            "n_teacher": teacher_row["n_cosine"],
                        }
                    )
                    continue
                decision = ratio_rank_disagrees_with_error(
                    open_row["response_ratio_median"],
                    open_row["normalized_vector_error_median"],
                    teacher_row["response_ratio_median"],
                    teacher_row["normalized_vector_error_median"],
                )
                gap = float(decision["ratio_distance_a"] - decision["ratio_distance_b"])
                out.append(
                    {
                        "model": model,
                        "seed": seed,
                        "family": family,
                        "status": "compared",
                        "ratio_prefers": decision["ratio_prefers"],
                        "error_prefers": decision["error_prefers"],
                        "disagrees": decision["disagrees"],
                        "log_ratio_gap_open_minus_teacher": gap,
                        "material_trap": bool(
                            decision["ratio_prefers"] == "a"
                            and decision["error_prefers"] == "b"
                            and gap <= -0.05
                        ),
                        "open_cosine": open_row["cosine_median"],
                        "teacher_cosine": teacher_row["cosine_median"],
                        "open_ratio": open_row["response_ratio_median"],
                        "teacher_ratio": teacher_row["response_ratio_median"],
                        "open_nve": open_row["normalized_vector_error_median"],
                        "teacher_nve": teacher_row["normalized_vector_error_median"],
                    }
                )
    return out


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, default="1440:1456")
    parser.add_argument("--dump-seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--out", type=str, default="experiments/runs/intervention_decomposition_dev")
    parser.add_argument("--role", type=str, default="development")
    parser.add_argument("--check-dump", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    start_ep, stop_ep = (int(part) for part in args.episodes.split(":"))
    episodes = tuple(range(start_ep, stop_ep))
    seeds = tuple(int(part) for part in args.seeds.split(","))
    out = ROOT / args.out
    if args.dump_seed == 0 and any(episode >= 1620 or episode < 1440 for episode in episodes):
        raise SystemExit("dump seed 0 scoring must stay inside validation episodes 1440-1619.")
    if args.dump_seed != 0 and any(episode < 0 or episode >= 1440 for episode in episodes):
        raise SystemExit("a new dump seed must not reuse training or validation indices above 1439.")
    out.mkdir(parents=True, exist_ok=True)
    spec = environment_from_dict(yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text()))
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
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"device {device}", flush=True)
    if args.check_dump:
        observations = np.load(ROOT / "outputs/physics_mpm_high/observations.npy", mmap_mode="r")
        probe = replay_episode(episodes[0], dump_seed=args.dump_seed, episode_length=16, mpm=spec.mpm, hold=8)
        if not np.array_equal(probe["frames"], observations[episodes[0]]):
            raise SystemExit(f"replay does not match the dump at episode {episodes[0]}.")
    rows: list[dict] = []
    started = time.perf_counter()
    for seed in seeds:
        for model_name, pattern in MODEL_PATHS.items():
            checkpoint = pattern.format(seed=seed)
            model, _loaded = load_model_from_checkpoint(ROOT / checkpoint, device)
            model.eval()
            for episode in episodes:
                replay = replay_episode(
                    episode, dump_seed=args.dump_seed, episode_length=16, mpm=spec.mpm, hold=8
                )
                for branch in BRANCH_TIMES:
                    recorded = replay["actions"][branch : branch + SIM_HORIZON].astype(np.float32)
                    snapshot = replay["snapshots"][branch]
                    recorded_frames = replay["frames"][branch + 1 : branch + 1 + SIM_HORIZON]
                    recorded_pos = np.stack(
                        [np.asarray(replay["snapshots"][branch + 1 + step]["x"]) for step in range(SIM_HORIZON)]
                    )
                    frame_now = _video([replay["frames"][branch]], device)[0]
                    future_rec = _video(recorded_frames, device)
                    actions_rec = _norm_actions(recorded, mean, std, device)
                    for family in FAMILIES:
                        other = np.stack([_intervene(step, family) for step in recorded])
                        if family == "identity":
                            future_np, pos_b = recorded_frames, recorded_pos
                        else:
                            future_np, pos_b = _simulate(snapshot, other, spec.mpm)
                        future_other = _video(future_np, device)
                        actions_other = _norm_actions(other, mean, std, device)
                        with torch.no_grad():
                            scored = _score_model(
                                model, frame_now, future_rec, future_other, actions_rec, actions_other
                            )
                        rms = [_rms(recorded_pos[step], pos_b[step]) for step in range(SIM_HORIZON)]
                        for item in scored:
                            item.update(
                                {
                                    "model": model_name,
                                    "seed": seed,
                                    "episode": episode,
                                    "branch_time": branch,
                                    "family": family,
                                    "rms_pos": rms[int(item["horizon"]) - 1],
                                }
                            )
                            rows.append(item)
                print(
                    f"scored {model_name} s{seed} episode {episode} in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            del model
    summary = _summarize(rows)
    comparisons = _comparisons(summary, tuple(MODEL_PATHS))
    fieldnames = list(rows[0].keys())
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "role": args.role,
        "episodes": [episodes[0], episodes[-1]],
        "n_episodes": len(episodes),
        "branch_times": list(BRANCH_TIMES),
        "seeds": list(seeds),
        "dump_seed": args.dump_seed,
        "material_trap_rule": "ratio prefers open loop, error prefers teacher forcing, and the open-loop log-ratio distance is at least 0.05 smaller",
        "test_episodes_read": False,
        "hypothesis": (
            "At horizon 10, closeness of the median response ratio to one prefers "
            "open-loop VICReg over teacher forcing, while normalized vector error "
            "and cosine prefer teacher forcing. The question is whether that "
            "disagreement holds for scaling, zeroing, and rotating the force."
        ),
        "summary": summary,
        "comparisons": comparisons,
        "seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
