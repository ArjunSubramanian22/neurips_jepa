#!/usr/bin/env python3
"""Hash a regenerated dump against DATA_MANIFEST.csv.

    python3.13 scripts/check_regenerated_dump.py

Observations are copied into ``outputs/physics_mpm/observations.npy`` only
when the regenerated observations and actions both match the manifest.
Existing action and label files are not overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import RunRecorder, to_jsonable

REQUIRED = (
    "outputs/physics_mpm/observations.npy",
    "outputs/physics_mpm/actions.npy",
    "outputs/physics_mpm/stiffness_labels.npy",
    "outputs/physics_mpm/stiffness_values.npy",
    "outputs/physics_mpm/geometry_labels.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    regen = ROOT / "outputs" / "_regen_seed0"
    manifest_rows = {
        row["path"]: row
        for row in csv.DictReader((ROOT / "DATA_MANIFEST.csv").open(encoding="utf-8"))
    }
    config = {
        "regen_dir": "outputs/_regen_seed0",
        "manifest": "DATA_MANIFEST.csv",
        "copy_observations_if_actions_match": True,
    }
    with RunRecorder(
        "dump_regeneration_hashcheck_v1",
        repo_root=ROOT,
        config=config,
        command="python3.13 scripts/check_regenerated_dump.py",
        seed=0,
        model={"name": None, "note": "simulator regeneration hash check"},
        environment={"simulator": "MLSMPM2D", "dump_seed": 0, "n_episodes": 2000},
        dataset_split={"protocol_id": "hist_v1_last_fraction_val", "evaluated": False},
        checkpoint={"path": None, "note": "no checkpoint"},
    ) as run:
        comparisons = []
        for rel in REQUIRED:
            name = Path(rel).name
            path = regen / name
            recorded = (manifest_rows[rel].get("sha256") or "").strip()
            if not path.is_file():
                comparisons.append(
                    {
                        "path": rel,
                        "regen_exists": False,
                        "measured_sha256": None,
                        "recorded_sha256": recorded or None,
                        "matches_manifest": False,
                    }
                )
                continue
            measured = _sha256(path)
            comparisons.append(
                {
                    "path": rel,
                    "regen_exists": True,
                    "measured_sha256": measured,
                    "recorded_sha256": recorded or None,
                    "matches_manifest": measured == recorded,
                }
            )
        by_name = {Path(row["path"]).name: row for row in comparisons}
        observations_match = bool(by_name["observations.npy"]["matches_manifest"])
        actions_match = bool(by_name["actions.npy"]["matches_manifest"])
        copied = False
        destination = ROOT / "outputs" / "physics_mpm" / "observations.npy"
        if observations_match and actions_match:
            if destination.is_file() and _sha256(destination) == by_name["observations.npy"]["measured_sha256"]:
                copied = False
                copy_note = "destination already has the matching observations tensor"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(regen / "observations.npy", destination)
                copied = _sha256(destination) == by_name["observations.npy"]["recorded_sha256"]
                copy_note = "copied regenerated observations.npy into outputs/physics_mpm/"
        else:
            copy_note = "refused to copy; observations and actions must both match the manifest"
        metrics = to_jsonable(
            {
                "comparisons": comparisons,
                "observations_match": observations_match,
                "actions_match": actions_match,
                "observations_copied": copied,
                "copy_note": copy_note,
                "models_evaluated": False,
            }
        )
        (run.run_dir / "hashcheck.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        run.complete(metrics)
    print(json.dumps({"observations_match": observations_match, "actions_match": actions_match, "copied": copied, "note": copy_note}))


if __name__ == "__main__":
    main()
