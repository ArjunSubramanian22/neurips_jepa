#!/usr/bin/env python3
"""Aggregate headline checkpoints by the fixed training-seed registry.

    python3.13 scripts/aggregate_headline_seeds.py

Episode intervals stay labeled as conditional on the checkpoint. A
training-seed interval is written only when every registry seed is complete.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import RunRecorder, to_jsonable
from src.seed_uncertainty_plots import plot_seed_values
from stats.hierarchical_bootstrap import hierarchical_bootstrap_mean
from stats.seed_aggregation import UNDEFINED, aggregate_seeds, paired_seed_comparison

QUERIES = ROOT / "experiments" / "runs" / "horizon_audit_v1" / "queries.parquet"
STATUS = ROOT / "experiments" / "seed_sweeps" / "headline_v1" / "status.json"
PROTOCOL = "persistent_alternative"
HORIZON = 10
MODES = ("open_loop", "teacher_forced")
METRICS = ("ranking_top1", "cosine")


def _configuration(model: str) -> str:
    return model.rsplit("_s", 1)[0]


def _episode_values(rows: list[dict[str, Any]], metric: str) -> tuple[dict[int, list[float]], list[int]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    seen = set()
    for row in rows:
        episode = int(row["episode"])
        seen.add(episode)
        if metric == "ranking_top1":
            if row.get("counterfactual_ranking_status") in {"scored", "distance_tie"} and row.get("counterfactual_ranking_top1") is not None:
                grouped[episode].append(float(row["counterfactual_ranking_top1"]))
        elif row.get("cosine_status") == "ok" and row.get("cosine") is not None:
            grouped[episode].append(float(row["cosine"]))
    means = {episode: float(np.mean(values)) for episode, values in grouped.items() if values}
    excluded = sorted(seen - set(means))
    return means, excluded


def _records_for(status_rows: list[dict[str, Any]], configuration: str, means_by_seed: dict[int, dict[int, float]]) -> list[dict[str, Any]]:
    records = []
    for row in status_rows:
        if row["configuration"] != configuration:
            continue
        seed = int(row["seed"])
        if row["status"] != "COMPLETE":
            records.append({"seed": seed, "status": row["status"], "value": None})
            continue
        episodes = means_by_seed.get(seed, {})
        if not episodes:
            records.append({"seed": seed, "status": UNDEFINED, "value": None})
            continue
        records.append({"seed": seed, "status": "COMPLETE", "value": float(np.mean(list(episodes.values())))})
    return records


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in status["seeds"]]
    rows = pq.read_table(QUERIES).to_pylist()
    selected = [
        row
        for row in rows
        if row["protocol"] == PROTOCOL and int(row["horizon"]) == HORIZON and row["rollout_mode"] in MODES
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(_configuration(row["model"]), int(row["seed"]), row["rollout_mode"])].append(row)
    blocks = []
    flat_rows = []
    paired_payload: dict[str, Any] = {}
    for mode in MODES:
        for metric in METRICS:
            configurations = []
            by_configuration: dict[str, list[dict[str, Any]]] = {}
            for configuration in dict.fromkeys(row["configuration"] for row in status["slots"]):
                status_rows = [row for row in status["slots"] if row["configuration"] == configuration]
                means_by_seed = {}
                excluded_by_seed = {}
                for row in status_rows:
                    seed = int(row["seed"])
                    if row["status"] != "COMPLETE":
                        continue
                    episodes, excluded = _episode_values(grouped.get((configuration, seed, mode), []), metric)
                    means_by_seed[seed] = episodes
                    excluded_by_seed[seed] = excluded
                records = _records_for(status_rows, configuration, means_by_seed)
                summary = aggregate_seeds(records, seeds)
                seed_status = {int(record["seed"]): record["status"] for record in records}
                episodes_by_seed = {seed: list(values.values()) for seed, values in means_by_seed.items() if values}
                hierarchical = hierarchical_bootstrap_mean(
                    episodes_by_seed,
                    required_seeds=seeds,
                    seed_status=seed_status,
                    n_resamples=2000,
                    rng=np.random.default_rng(0),
                    excluded_episode_ids=excluded_by_seed,
                )
                configurations.append({"id": configuration, "aggregate": summary, "hierarchical": hierarchical})
                by_configuration[configuration] = records
                for seed_row in summary["seeds"]:
                    flat_rows.append(
                        {
                            "protocol": PROTOCOL,
                            "horizon": HORIZON,
                            "rollout_mode": mode,
                            "metric": metric,
                            "configuration": configuration,
                            "seed": seed_row["seed"],
                            "status": seed_row["status"],
                            "value": seed_row["value"],
                            "training_seed_ci_status": hierarchical["training_seed_interval"]["status"],
                            "is_test_set": False,
                        }
                    )
            paired = None
            if metric == "ranking_top1" and mode == "open_loop":
                paired = {
                    "action_minus_noaction": paired_seed_comparison(
                        by_configuration["vicreg_action"],
                        by_configuration["vicreg_noaction"],
                        seeds,
                    ),
                    "vicreg_minus_mse": paired_seed_comparison(
                        by_configuration["vicreg_action"],
                        by_configuration["mse_action"],
                        seeds,
                    ),
                }
                paired_payload = paired
            blocks.append(
                {
                    "metric": metric,
                    "rollout_mode": mode,
                    "protocol": PROTOCOL,
                    "horizon": HORIZON,
                    "seeds": seeds,
                    "configurations": configurations,
                    "paired": paired,
                }
            )
    with RunRecorder(
        "headline_seed_uncertainty_v1",
        repo_root=ROOT,
        config={"seed_registry_id": "headline_v1", "seeds": seeds, "protocol": PROTOCOL, "horizon": HORIZON, "split_version": "legacy_v1", "is_test_set": False},
        command="python3.13 scripts/aggregate_headline_seeds.py",
        seed="headline_v1",
        model={"configurations": ["vicreg_action", "vicreg_noaction", "mse_action"]},
        environment={"id": "mpm_legacy"},
        dataset_split={"split_version": "legacy_v1", "headline_role": "legacy_validation", "is_test_set": False},
        checkpoint={"path": str(QUERIES.relative_to(ROOT)), "note": "existing checkpoints only"},
    ) as run:
        payload = {"blocks": blocks, "paired": paired_payload, "is_test_set": False}
        (run.run_dir / "seed_summaries.json").write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
        with (run.run_dir / "seed_values.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
        plot_seed_values(blocks, ROOT / "figures" / "seed_uncertainty")
        run.complete(
            to_jsonable(
                {
                    "is_test_set": False,
                    "training_seed_intervals_reported": False,
                    "n_not_run": status["n_not_run"],
                    "not_run": [
                        {"configuration": row["configuration"], "seed": row["seed"]}
                        for row in status["slots"]
                        if row["status"] == "NOT_RUN"
                    ],
                }
            )
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
