#!/usr/bin/env python3
"""Plan or launch the headline training-seed sweep.

    python3.13 scripts/launch_seed_sweep.py
    python3.13 scripts/launch_seed_sweep.py --launch

The seed list is fixed. A missing checkpoint stays ``NOT_RUN``. A failed
launch stays ``FAILED``. Neither is removed from the registry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import utc_now

REGISTRY = ROOT / "configs" / "seeds" / "headline_v1.yaml"
STATUS_DIR = ROOT / "experiments" / "seed_sweeps" / "headline_v1"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
NOT_RUN = "NOT_RUN"


def load_seed_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in payload["seeds"]]
    if len(seeds) < int(payload["minimum_complete_seeds"]):
        raise ValueError("the seed registry is shorter than the minimum complete count.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed registry contains a duplicate seed.")
    payload["seeds"] = seeds
    return payload


def slot_status(checkpoint: Path | None, failure_record: Path | None) -> str:
    """Classify one registry slot. Absence is ``NOT_RUN``, not removal."""
    if checkpoint is not None and checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return COMPLETE
    if failure_record is not None and failure_record.is_file():
        return FAILED
    return NOT_RUN


def plan_sweep(registry: dict[str, Any], repo_root: Path, failure_records: dict[tuple[str, int], Path] | None = None) -> list[dict[str, Any]]:
    """One row per configuration and registry seed, including seeds with no checkpoint."""
    failures = failure_records or {}
    rows = []
    for configuration in registry["configurations"]:
        checkpoints = {int(seed): path for seed, path in configuration["checkpoints"].items()}
        missing = [seed for seed in registry["seeds"] if seed not in checkpoints]
        if missing:
            raise ValueError(f"{configuration['id']} is missing registry seeds {missing}.")
        for seed in registry["seeds"]:
            raw = checkpoints[seed]
            checkpoint = None if raw in {None, ""} else repo_root / str(raw)
            failure = failures.get((configuration["id"], seed))
            status = slot_status(checkpoint, failure)
            rows.append(
                {
                    "seed_registry_id": registry["seed_registry_id"],
                    "configuration": configuration["id"],
                    "objective": configuration["objective"],
                    "condition_on_action": bool(configuration["condition_on_action"]),
                    "seed": int(seed),
                    "status": status,
                    "checkpoint": None if checkpoint is None else str(checkpoint.relative_to(repo_root)),
                    "failure_record": None if failure is None else str(failure),
                }
            )
    return rows


def launch_not_run(
    rows: list[dict[str, Any]],
    runner: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Start only ``NOT_RUN`` slots. A raised runner marks that slot ``FAILED`` and keeps it."""
    updated = []
    for row in rows:
        current = dict(row)
        if current["status"] != NOT_RUN:
            updated.append(current)
            continue
        try:
            runner(current)
        except Exception as exc:
            current["status"] = FAILED
            current["failure_reason"] = f"{type(exc).__name__}: {exc}"
        updated.append(current)
    return updated


def write_sweep_status(directory: Path, rows: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    """Write the current snapshot and append it. Earlier lines stay in the log."""
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "recorded_at": utc_now(),
        "seed_registry_id": registry["seed_registry_id"],
        "seeds": registry["seeds"],
        "minimum_complete_seeds": int(registry["minimum_complete_seeds"]),
        "split_version": registry.get("split_version"),
        "is_test_set": registry.get("is_test_set"),
        "n_slots": len(rows),
        "n_complete": sum(row["status"] == COMPLETE for row in rows),
        "n_failed": sum(row["status"] == FAILED for row in rows),
        "n_not_run": sum(row["status"] == NOT_RUN for row in rows),
        "slots": rows,
    }
    (directory / "status.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    with (directory / "registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot) + "\n")


def _train_runner(row: dict[str, Any]) -> None:
    action = "action" if row["condition_on_action"] else "noaction"
    out_root = ROOT / "results" / "seed_sweeps" / "headline_v1"
    run_name = f"{row['configuration']}_s{row['seed']}"
    command = [
        "bash",
        str(ROOT / "scripts" / "train_reviewer_seed.sh"),
        str(row["seed"]),
        row["objective"],
        action,
        str(out_root),
        run_name,
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    registry = load_seed_registry(REGISTRY)
    rows = plan_sweep(registry, ROOT)
    if "--launch" in sys.argv:
        rows = launch_not_run(rows, _train_runner)
    write_sweep_status(STATUS_DIR, rows, registry)
    print(STATUS_DIR / "status.json")


if __name__ == "__main__":
    main()
