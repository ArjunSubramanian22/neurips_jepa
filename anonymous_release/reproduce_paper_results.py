#!/usr/bin/env python3
"""Regenerate paper-facing analysis from cached artifacts when present.

Does not retrain. Does not overwrite original outputs/ checkpoints or JSON.
Dataset regeneration and training are explicit subcommands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.call(args, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("audit", "analysis", "counterfactual", "train-seed", "all-cached"),
        default="all-cached",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss", default="vicreg", choices=("vicreg", "mse"))
    parser.add_argument("--action", default="action", choices=("action", "noaction"))
    parser.add_argument("--max-val-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.stage in {"audit", "all-cached"}:
        rc = _run([PYTHON, "scripts/run_baseline_audit.py"])
        if rc:
            return rc
        rc = _run([PYTHON, "scripts/run_snapshot_validation.py"])
        if rc:
            return rc
        rc = _run([PYTHON, "scripts/run_action_path_audit.py"])
        if rc:
            return rc
    if args.stage == "counterfactual":
        rc = _run(
            [
                PYTHON,
                "scripts/run_simulator_counterfactual.py",
                "--device",
                "cpu",
                "--max-val-episodes",
                str(args.max_val_episodes),
            ]
        )
        if rc:
            return rc
        return _run([PYTHON, "scripts/analyze_simulator_counterfactual.py"])
    if args.stage in {"analysis", "all-cached"}:
        cf = ROOT / "results/reviewer_strengthening/01_simulator_counterfactual/physical_divergence_per_example.csv"
        if cf.is_file():
            rc = _run([PYTHON, "scripts/analyze_simulator_counterfactual.py"])
            if rc:
                return rc
        else:
            print("analysis: Phase 1 CSVs not present; skip", flush=True)
    if args.stage == "train-seed":
        if args.seed is None:
            raise SystemExit("--seed required")
        run = f"{args.loss}_{args.action}_s{args.seed}"
        root = (
            "results/reviewer_strengthening/05_more_seeds"
            if args.loss == "vicreg"
            else "results/reviewer_strengthening/06_corrected_objective"
        )
        return _run(
            [
                "bash",
                "scripts/train_reviewer_seed.sh",
                str(args.seed),
                args.loss,
                args.action,
                root,
                run,
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
