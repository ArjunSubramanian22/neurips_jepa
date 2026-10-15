#!/usr/bin/env python3
"""Observation and action ablations for RSSM posterior targets.

Validation cache only. The imagined action sequence stays intact. Each mode
changes what the target filter is allowed to see.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mpm_high_pretest import BRANCH_TIMES, _as_video, _moments, _norm_actions, load_cache
from src.counterfactual_study import assert_high_validation_episodes, pair_metrics
from src.train import load_model_from_checkpoint

MODES = ("intact", "blank_image", "cross_fork_image", "zero_action", "swapped_action")
EPISODES = (1440, 1441, 1442, 1443)
HORIZONS = (1, 10)
BRANCHES = (0, 3)
OUT = ROOT / "experiments" / "runs" / "rssm_target_controls_pilot"


def ablate(
    frame: torch.Tensor,
    action: torch.Tensor,
    other_frame: torch.Tensor,
    other_action: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "intact":
        return frame, action
    if mode == "blank_image":
        return torch.zeros_like(frame), action
    if mode == "cross_fork_image":
        return other_frame, action
    if mode == "zero_action":
        return frame, torch.zeros_like(action)
    if mode == "swapped_action":
        return frame, other_action
    raise ValueError(f"unknown target ablation {mode!r}.")


def _filter(model, frames: torch.Tensor, actions: torch.Tensor, start: int):
    deter, stochastic = model.initial_state(frames.shape[0], frames.device)
    for time in range(start + 1):
        action = None if time == 0 else actions[:, time - 1]
        deter, stochastic, _feature = model.observe(deter, stochastic, action, frames[:, time])
    return deter.detach(), stochastic.detach()


def contrast_at_horizon(
    model,
    frames_a: torch.Tensor,
    actions_a: torch.Tensor,
    frames_b: torch.Tensor,
    actions_b: torch.Tensor,
    start: int,
    horizon: int,
    mode: str,
) -> dict:
    """Open-loop imagine versus an ablated posterior target at one horizon."""
    pred_a = _filter(model, frames_a, actions_a, start)
    pred_b = _filter(model, frames_b, actions_b, start)
    tgt_a = (pred_a[0].clone(), pred_a[1].clone())
    tgt_b = (pred_b[0].clone(), pred_b[1].clone())
    pred_feature_a = pred_feature_b = tgt_feature_a = tgt_feature_b = None
    for step in range(horizon):
        action_a = actions_a[:, start + step]
        action_b = actions_b[:, start + step]
        frame_a = frames_a[:, start + 1 + step]
        frame_b = frames_b[:, start + 1 + step]
        deter_a, stoch_a, pred_feature_a = model.imagine(pred_a[0], pred_a[1], action_a)
        pred_a = (deter_a, stoch_a)
        deter_b, stoch_b, pred_feature_b = model.imagine(pred_b[0], pred_b[1], action_b)
        pred_b = (deter_b, stoch_b)
        tgt_frame_a, tgt_action_a = ablate(frame_a, action_a, frame_b, action_b, mode)
        tgt_frame_b, tgt_action_b = ablate(frame_b, action_b, frame_a, action_a, mode)
        deter_ta, stoch_ta, tgt_feature_a = model.observe(tgt_a[0], tgt_a[1], tgt_action_a, tgt_frame_a)
        tgt_a = (deter_ta, stoch_ta)
        deter_tb, stoch_tb, tgt_feature_b = model.observe(tgt_b[0], tgt_b[1], tgt_action_b, tgt_frame_b)
        tgt_b = (deter_tb, stoch_tb)
    _pred_h, pred_z_a = model.split_feature(pred_feature_a)
    _pred_h, pred_z_b = model.split_feature(pred_feature_b)
    _tgt_h, tgt_z_a = model.split_feature(tgt_feature_a)
    _tgt_h, tgt_z_b = model.split_feature(tgt_feature_b)
    metrics = pair_metrics(
        (pred_z_b - pred_z_a).detach().cpu().numpy().reshape(-1),
        (tgt_z_b - tgt_z_a).detach().cpu().numpy().reshape(-1),
    )
    metrics["target_delta_norm"] = float(torch.linalg.norm(tgt_z_b - tgt_z_a).detach().cpu())
    return metrics


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSSM target ablations on high-excitation validation.")
    parser.add_argument("--episodes", type=int, nargs="+", default=list(EPISODES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    episodes = assert_high_validation_episodes(args.episodes)
    cache = load_cache("mpm_high", list(episodes))
    mean, std = _moments("outputs/physics_mpm_high", "fresh_pool_v2")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / args.out
    rows = []
    for seed in args.seeds:
        model, _cfg = load_model_from_checkpoint(ROOT / f"outputs/mpm_high_runs/rssm_s{seed}/last.pt", device)
        model.eval()
        with torch.no_grad():
            for episode_index, episode in enumerate(episodes):
                recorded = _as_video(cache["full_frames"][episode_index], device).unsqueeze(0)
                recorded_actions = _norm_actions(cache["full_actions"][episode_index], mean, std, device).unsqueeze(0)
                for branch in BRANCHES:
                    branch_index = BRANCH_TIMES.index(branch)
                    reversed_video = recorded.clone()
                    reversed_actions = recorded_actions.clone()
                    future_rev = _as_video(cache["frames_rev"][episode_index, branch_index], device).unsqueeze(0)
                    reversed_video[:, branch + 1 : branch + 1 + future_rev.shape[1]] = future_rev
                    reversed_actions[:, branch : branch + future_rev.shape[1]] = _norm_actions(
                        cache["actions_rev"][episode_index, branch_index], mean, std, device
                    ).unsqueeze(0)
                    for horizon in HORIZONS:
                        for mode in MODES:
                            metrics = contrast_at_horizon(
                                model,
                                recorded,
                                recorded_actions,
                                reversed_video,
                                reversed_actions,
                                branch,
                                horizon,
                                mode,
                            )
                            rows.append({
                                "episode": episode,
                                "seed": seed,
                                "branch_time": branch,
                                "horizon": horizon,
                                "mode": mode,
                                "cosine": metrics["cosine"],
                                "cosine_status": metrics["cosine_status"],
                                "response_ratio": metrics["response_ratio"],
                                "target_delta_norm": metrics["target_delta_norm"],
                                "pred_norm": metrics["pred_norm"],
                                "target_norm": metrics["target_norm"],
                            })
                print(f"rssm controls s{seed} episode {episode}", flush=True)
        del model
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
    for mode in MODES:
        for horizon in HORIZONS:
            group = [row for row in rows if row["mode"] == mode and row["horizon"] == horizon]
            cosines = [float(row["cosine"]) for row in group if row["cosine"] is not None]
            by_seed = []
            for seed in args.seeds:
                seed_rows = [row for row in group if row["seed"] == seed and row["cosine"] is not None]
                by_seed.append(None if not seed_rows else float(np.median([float(row["cosine"]) for row in seed_rows])))
            summary.append({
                "mode": mode,
                "horizon": horizon,
                "n": len(group),
                "median_cosine": None if not cosines else float(np.median(cosines)),
                "seed_median_cosine": by_seed,
                "median_target_delta_norm": float(np.median([row["target_delta_norm"] for row in group])),
            })
    (out / "summary.json").write_text(json.dumps({
        "seeds": list(args.seeds),
        "episodes": list(episodes),
        "test_episodes_read": False,
        "dependence": "swapped_action negates the target delta of cross_fork_image, so its cosine is the negation. It is not a second piece of evidence.",
        "modes": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
