#!/usr/bin/env python3
"""Record the linear-dynamics planner sanity check.

    python3.13 scripts/run_planning_oracle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.ranking import random_ranking_top1
from src.planning import cem_minimize, oracle_linear_action_recovery
from src.registry import RunRecorder, to_jsonable


def main(argv: list[str] | None = None) -> None:
    config_path = ROOT / "configs" / "experiments" / "planning_oracle_sanity.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    command = "python3.13 scripts/run_planning_oracle.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed=int(config["seed"]),
        model={"name": "linear_residual_identity", "learned": None},
        environment={"name": "synthetic_latent", "simulator": None},
        dataset_split=config["dataset_split"],
        checkpoint={"path": None, "note": "no learned checkpoint"},
    ) as run:
        recovery = oracle_linear_action_recovery(
            seed=int(config["seed"]),
            n_candidates=int(config["n_candidates"]),
            latent_dim=int(config["latent_dim"]),
            correct_index=int(config["correct_index"]),
        )
        target = np.array([0.4, -0.2, 0.7, 0.1])
        initial_cost = float(np.sum((np.zeros(4) - target) ** 2))

        def quadratic(samples: np.ndarray) -> np.ndarray:
            return np.sum((samples - target) ** 2, axis=1)

        cem = cem_minimize(
            quadratic,
            mean=np.zeros(4),
            std=np.ones(4),
            population=int(config["cem"]["population"]),
            n_iterations=int(config["cem"]["n_iterations"]),
            elite_fraction=float(config["cem"]["elite_fraction"]),
            rng=np.random.default_rng(int(config["seed"])),
        )
        ranking_cfg = config["random_ranking"]
        ranking = random_ranking_top1(
            int(ranking_cfg["n_trials"]),
            int(ranking_cfg["n_candidates"]),
            np.random.default_rng(int(ranking_cfg["seed"])),
        )
        criterion = config["success_criterion"]
        ranking_error = abs(ranking["top1"] - ranking["chance"])
        checks = {
            "oracle_top1_is_one": recovery["utility"]["top1"] == criterion["oracle_top1"],
            "oracle_minimum_unique": recovery["unique_minimum"],
            "cem_cost_below_initial": cem["cost"] < initial_cost,
            "random_top1_near_chance": ranking_error <= float(criterion["random_top1_abs_error_max"]),
        }
        metrics = to_jsonable(
            {
                "oracle": recovery,
                "cem": {
                    "initial_cost": initial_cost,
                    "final_cost": cem["cost"],
                    "history": cem["history"],
                    "population": cem["population"],
                    "n_iterations": cem["n_iterations"],
                    "elite_count": cem["elite_count"],
                },
                "random_ranking": ranking,
                "success_checks": checks,
                "success_criterion_met": all(checks.values()),
                "learned_model_evaluated": False,
            }
        )
        (run.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        run.complete(metrics)
    print(run.run_dir)


if __name__ == "__main__":
    main()
