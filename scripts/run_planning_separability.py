#!/usr/bin/env python3
"""Physical separability of the published matched-planning candidate sets.

Does not load MPM checkpoints. The goal index matches the 24-episode list
used by experiments/runs/mpm_high_planning_matched.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import replay_episode
from scripts.run_matched_planning import (
    HORIZONS,
    LIBRARY_EPISODES,
    _endpoint_rms,
    _libraries,
    _simulate_sequence,
)
from src.environments import environment_from_dict

EPISODES = list(range(1440, 1464))
PLAN_ROWS = ROOT / "experiments" / "runs" / "mpm_high_planning_matched" / "raw_results.csv"
OUT = ROOT / "experiments" / "runs" / "mpm_high_planning_separability"
# Declared before the margins are computed. A runner-up inside a quarter of
# the episode's typical candidate separation counts as genuinely close.
CLOSE_FRACTION = 0.25


def _tertile_labels(values: np.ndarray) -> list[str]:
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = []
    for value in values:
        if value <= low:
            labels.append("hard")
        elif value <= high:
            labels.append("medium")
        else:
            labels.append("easy")
    return labels


def main() -> None:
    if not PLAN_ROWS.is_file():
        raise SystemExit(f"missing published rankings: {PLAN_ROWS}")
    spec = environment_from_dict(
        yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text(encoding="utf-8"))
    )
    replays = [
        replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        for episode in LIBRARY_EPISODES
    ]
    libraries = {horizon: _libraries(replays, horizon) for horizon in HORIZONS}
    rows = []
    pair_rows = []
    for episode_index, episode in enumerate(EPISODES):
        replay = replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        for horizon in HORIZONS:
            for kind in ("constant", "sequence"):
                library = libraries[horizon][kind]
                goal = episode_index % len(library)
                positions = []
                for actions in library:
                    _frames, pos, _vel = _simulate_sequence(replay["snapshots"][0], actions, spec.mpm)
                    positions.append(pos)
                to_goal = np.asarray([_endpoint_rms(pos, positions[goal]) for pos in positions], dtype=np.float64)
                order = np.argsort(to_goal, kind="mergesort")
                best = float(to_goal[order[0]])
                second = float(to_goal[order[1]])
                pairs = []
                for left in range(len(positions)):
                    for right in range(left + 1, len(positions)):
                        distance = _endpoint_rms(positions[left], positions[right])
                        pairs.append(distance)
                        pair_rows.append({
                            "episode": episode,
                            "horizon": horizon,
                            "library": kind,
                            "distance": distance,
                        })
                pair_median = float(np.median(pairs))
                margin = second - best
                rows.append({
                    "episode": episode,
                    "horizon": horizon,
                    "library": kind,
                    "n_candidates": len(library),
                    "goal_index": goal,
                    "oracle_is_goal": int(int(order[0]) == goal),
                    "best_rms": best,
                    "second_rms": second,
                    "margin": margin,
                    "pairwise_median": pair_median,
                    "pairwise_p10": float(np.quantile(pairs, 0.10)),
                    "pairwise_p90": float(np.quantile(pairs, 0.90)),
                    "close": int(margin < CLOSE_FRACTION * pair_median),
                })
        print(f"separability {episode}", flush=True)
    for kind in ("constant", "sequence"):
        for horizon in HORIZONS:
            group = [row for row in rows if row["library"] == kind and row["horizon"] == horizon]
            labels = _tertile_labels(np.asarray([row["margin"] for row in group], dtype=np.float64))
            for row, label in zip(group, labels):
                row["difficulty"] = label
    rankings = list(csv.DictReader(PLAN_ROWS.open(encoding="utf-8")))
    by_task = {(int(row["episode"]), int(row["horizon"]), row["library"]): row for row in rows}
    stratified = []
    for rank in rankings:
        task = by_task[(int(rank["episode"]), int(rank["horizon"]), rank["library"])]
        stratified.append({
            "model": rank["model"],
            "seed": int(rank["seed"]),
            "episode": int(rank["episode"]),
            "horizon": int(rank["horizon"]),
            "library": rank["library"],
            "difficulty": task["difficulty"],
            "latent_top1": int(rank["latent_top1"]),
            "margin": task["margin"],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "episode_margins.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    accuracy = []
    for kind in ("constant", "sequence"):
        for horizon in HORIZONS:
            group = [row for row in rows if row["library"] == kind and row["horizon"] == horizon]
            margins = np.asarray([row["margin"] for row in group], dtype=np.float64)
            pairs = np.asarray(
                [row["distance"] for row in pair_rows if row["library"] == kind and row["horizon"] == horizon],
                dtype=np.float64,
            )
            summary.append({
                "library": kind,
                "horizon": horizon,
                "n_episodes": len(group),
                "n_candidates": group[0]["n_candidates"],
                "oracle_is_goal": float(np.mean([row["oracle_is_goal"] for row in group])),
                "median_best_rms": float(np.median([row["best_rms"] for row in group])),
                "median_margin": float(np.median(margins)),
                "margin_p10": float(np.quantile(margins, 0.10)),
                "margin_p90": float(np.quantile(margins, 0.90)),
                "min_margin": float(np.min(margins)),
                "pairwise_median": float(np.median(pairs)),
                "pairwise_p10": float(np.quantile(pairs, 0.10)),
                "pairwise_p90": float(np.quantile(pairs, 0.90)),
                "fraction_close": float(np.mean([row["close"] for row in group])),
                "close_rule": f"margin < {CLOSE_FRACTION} * episode pairwise median",
            })
            for model in ("vicreg_action", "vicreg_noaction", "mse_action", "rssm"):
                for difficulty in ("easy", "medium", "hard"):
                    chosen = [
                        row for row in stratified
                        if row["model"] == model and row["library"] == kind and row["horizon"] == horizon and row["difficulty"] == difficulty
                    ]
                    accuracy.append({
                        "model": model,
                        "library": kind,
                        "horizon": horizon,
                        "difficulty": difficulty,
                        "n_rows": len(chosen),
                        "n_episodes": len({row["episode"] for row in chosen}),
                        "top1": float(np.mean([row["latent_top1"] for row in chosen])),
                        "median_margin": float(np.median([row["margin"] for row in chosen])),
                    })
    (OUT / "summary.json").write_text(json.dumps({
        "episodes": EPISODES,
        "close_fraction": CLOSE_FRACTION,
        "difficulty": "tertiles of the best-vs-second margin within library and horizon; high margin is easy",
        "models_rescored": False,
        "summary": summary,
        "accuracy": accuracy,
    }, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Planning-task separability",
        "",
        "Same 24 validation episodes, libraries, and goal indices as the matched-planning table. Models were not reloaded. A candidate pair is close when the oracle margin is under "
        f"{CLOSE_FRACTION:.0%} of that episode's median pairwise endpoint distance. Easy, medium, and hard are tertiles of the margin.",
        "",
    ]
    for row in summary:
        lines.append(
            f"- {row['library']} H{row['horizon']}: median margin {row['median_margin']:.4g} "
            f"(p10 {row['margin_p10']:.4g}, p90 {row['margin_p90']:.4g}), "
            f"pairwise median {row['pairwise_median']:.4g}, "
            f"close {100 * row['fraction_close']:.1f}%, "
            f"oracle is goal {100 * row['oracle_is_goal']:.1f}%"
        )
    lines.append("")
    lines.append("## Top-1 by margin tertile")
    lines.append("")
    for row in accuracy:
        lines.append(
            f"- {row['model']} {row['library']} H{row['horizon']} {row['difficulty']}: "
            f"top1 {100 * row['top1']:.1f}% over {row['n_episodes']} episodes "
            f"(margin median {row['median_margin']:.4g})"
        )
    lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
