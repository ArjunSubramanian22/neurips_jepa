#!/usr/bin/env python3
"""Train the preregistered arm grid. Does not read episodes 600-719.

Four models, seeds 42-46, 20 epochs. An existing 20-epoch history is left
in place. Both best.pt and last.pt are produced by the trainer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mujoco_arm_campaign_runs"
PYTHON = sys.executable
SEEDS = (42, 43, 44, 45, 46)
CELLS = (
    {"run": "vicreg_s{seed}", "config": "configs/experiments/arm_campaign_jepa_v1.yaml", "loss": "vicreg", "no_action": False},
    {"run": "mse_s{seed}", "config": "configs/experiments/arm_campaign_jepa_v1.yaml", "loss": "mse", "no_action": False},
    {"run": "noaction_s{seed}", "config": "configs/experiments/arm_campaign_jepa_v1.yaml", "loss": "vicreg", "no_action": True},
    {"run": "rssm_s{seed}", "config": "configs/experiments/arm_campaign_rssm_v1.yaml", "loss": None, "no_action": False},
)


def _complete(run: str) -> bool:
    history = OUT / run / "history.json"
    if not history.is_file() or not (OUT / run / "best.pt").is_file() or not (OUT / run / "last.pt").is_file():
        return False
    payload = json.loads(history.read_text(encoding="utf-8"))
    return int(payload.get("epochs") or 0) >= 20 and len(payload.get("history") or []) >= 20


def main() -> None:
    for seed in SEEDS:
        for cell in CELLS:
            run = cell["run"].format(seed=seed)
            if _complete(run):
                print(f"skip {run}", flush=True)
                continue
            command = [
                PYTHON,
                "-m",
                "src.train",
                "--mode",
                "train",
                "--config",
                cell["config"],
                "--epochs",
                "20",
                "--seed",
                str(seed),
                "--run-name",
                run,
                "--output-dir",
                "outputs/mujoco_arm_campaign_runs",
                "--device",
                "mps",
                "--num-workers",
                "0",
            ]
            if cell["loss"] is not None:
                command.extend(["--loss-type", cell["loss"]])
            if cell["no_action"]:
                command.append("--no-action")
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    print("ARM_GRID_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
