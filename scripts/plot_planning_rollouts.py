"""Draw the planning comparison from the saved qualitative episode."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

from src.planning_figures import plot_distance_curves, plot_task_distances, write_rollout_gif

PANELS = (
    ("heuristic", "heuristic"),
    ("oracle", "oracle"),
    ("vicreg_action_s44", "action s44"),
    ("vicreg_noaction_s42", "no-action s42"),
    ("random", "random"),
)
RATIO_ORDER = (
    "heuristic",
    "oracle",
    "mean_action",
    "vicreg_noaction_s42",
    "vicreg_action_s44",
    "random",
)


def main() -> None:
    run = ROOT / "experiments" / "runs" / "planning_benchmark_v1"
    protocol = yaml.safe_load((ROOT / "configs/experiments/planning_benchmark_v1.yaml").read_text(encoding="utf-8"))
    archive = np.load(run / "qualitative.npz")
    episodes = json.loads((run / "episodes.json").read_text(encoding="utf-8"))
    report_seeds = {int(seed) for seed in protocol["report_seeds"]}
    frames = {label: archive[f"{key}_frames"] for key, label in PANELS}
    distances = {label: archive[f"{key}_distances"] for key, label in PANELS}
    figure_dir = ROOT / "figures" / "planning"
    write_rollout_gif(frames, figure_dir / "qualitative_rollouts.gif")
    plot_distance_curves(
        distances,
        figure_dir / "qualitative_distances.png",
        title="Center target, report seed 2. s44 is the validation highlight; success rate 0.",
    )
    table: dict[str, dict[str, float]] = {}
    for task in protocol["tasks"]:
        table[task] = {}
        for name in RATIO_ORDER:
            rows = [
                row
                for row in episodes
                if row["controller"] == name and int(row["seed"]) in report_seeds and row["task"] == task and row["scored"]
            ]
            label = name.replace("vicreg_", "").replace("_", " ")
            table[task][label] = float(np.mean([row["final_distance"] for row in rows])) if rows else float("nan")
    plot_task_distances(
        table,
        figure_dir / "report_final_distance.png",
        title="Report seeds, mean final distance. Unscored identical-image goals are excluded.",
    )
    print(figure_dir)


if __name__ == "__main__":
    main()
