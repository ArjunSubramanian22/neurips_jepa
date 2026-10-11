#!/usr/bin/env python3
"""Recompute the archived val-only table into a new run directory.

    python3.13 scripts/run_heldout_reproduction.py

Does not overwrite ``results/final_checkpoint_results.csv``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import RunRecorder, to_jsonable


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def main() -> None:
    config_path = ROOT / "configs" / "experiments" / "heldout_reproduction_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    command = "python3.13 scripts/run_heldout_reproduction.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed="scheduled:" + ",".join(config["scheduled_checkpoints"]),
        model={"checkpoints": config["scheduled_checkpoints"], "selection": "last.pt"},
        environment={"observations": "outputs/physics_mpm/observations.npy"},
        dataset_split={
            "protocol_id": config["protocol_id"],
            "eval_episodes": config["eval_episodes"],
            "test_defined": False,
        },
        checkpoint={"path": "see scheduled_checkpoints", "note": "ten no-crop last.pt files"},
    ) as run:
        out_dir = run.run_dir / "evaluation"
        out_csv = run.run_dir / "recomputed.csv"
        log_path = run.run_dir / "evaluate_final.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_final.py"),
                    "--out",
                    str(out_dir),
                    "--csv",
                    str(out_csv),
                    "--only",
                    *config["scheduled_checkpoints"],
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8")[-2000:]
            raise RuntimeError(f"evaluate_final.py exited {proc.returncode}: {tail}")
        fresh = _read_csv(out_csv)
        archive = _read_csv(ROOT / config["archive_csv"])
        tolerance = float(config["absolute_tolerance"])
        columns = list(config["compare_columns"])
        rows = []
        missing = []
        for name in config["scheduled_checkpoints"]:
            if name not in fresh:
                missing.append(name)
                rows.append({"model": name, "status": "MISSING"})
                continue
            if name not in archive:
                rows.append({"model": name, "status": "ABSENT_FROM_ARCHIVE"})
                continue
            diffs = {}
            within = True
            for column in columns:
                new_value = float(fresh[name][column])
                old_value = float(archive[name][column])
                delta = new_value - old_value
                diffs[column] = {
                    "recomputed": new_value,
                    "archived": old_value,
                    "delta": delta,
                    "within_tolerance": abs(delta) <= tolerance,
                }
                within = within and abs(delta) <= tolerance
            rows.append({"model": name, "status": "COMPARED", "within_tolerance": within, "columns": diffs})
        metrics = to_jsonable(
            {
                "n_scheduled": len(config["scheduled_checkpoints"]),
                "n_missing": len(missing),
                "missing": missing,
                "absolute_tolerance": tolerance,
                "rows": rows,
                "success_criterion_met": len(missing) == 0 and all(row.get("within_tolerance") for row in rows),
                "models_evaluated": True,
            }
        )
        (run.run_dir / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        run.complete(metrics)
    print(run.run_dir)


if __name__ == "__main__":
    main()
