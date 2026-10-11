#!/usr/bin/env python3
"""Record which declared artifacts are present and whether their hashes match.

    python3.13 scripts/audit_reproducibility_baseline.py

The script does not train, does not evaluate a model, and does not fill in a
hash for a file that is absent.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _check_row(row: dict[str, str], path_key: str = "path") -> dict[str, object]:
    rel = row.get(path_key, "")
    recorded = (row.get("sha256") or "").strip()
    path = ROOT / rel if rel else None
    exists = bool(path is not None and path.is_file())
    measured = None
    match = None
    if exists and recorded:
        measured = _sha256(path)
        match = measured == recorded
    return {
        "path": rel,
        "exists": exists,
        "recorded_sha256": recorded or None,
        "measured_sha256": measured,
        "sha256_match": match,
        "bytes": path.stat().st_size if exists else None,
        "manifest_notes": row.get("notes") or row.get("status") or "",
    }


def main() -> None:
    checkpoint_rows = list(csv.DictReader((ROOT / "CHECKPOINT_MANIFEST.csv").open(encoding="utf-8")))
    data_rows = list(csv.DictReader((ROOT / "DATA_MANIFEST.csv").open(encoding="utf-8")))
    experiment_rows = list(csv.DictReader((ROOT / "EXPERIMENT_MANIFEST.csv").open(encoding="utf-8")))
    usage = shutil.disk_usage(ROOT)
    config = {
        "checkpoint_manifest": "CHECKPOINT_MANIFEST.csv",
        "data_manifest": "DATA_MANIFEST.csv",
        "experiment_manifest": "EXPERIMENT_MANIFEST.csv",
        "hashes_recomputed": True,
        "models_evaluated": False,
    }
    with RunRecorder(
        "reproducibility_baseline_v1",
        repo_root=ROOT,
        config=config,
        command="python3.13 scripts/audit_reproducibility_baseline.py",
        seed="not_applicable",
        model={"name": None, "note": "inventory only"},
        environment={"name": "worktree_inventory"},
        dataset_split={"protocol_id": "hist_v1_last_fraction_val", "evaluated": False},
        checkpoint={"path": None, "note": "hashes checked, weights not loaded"},
    ) as run:
        checkpoints = [_check_row(row) for row in checkpoint_rows]
        data = [_check_row(row) for row in data_rows]
        experiments = []
        for row in experiment_rows:
            artifact = row.get("path_or_artifact") or ""
            first = artifact.split()[0] if artifact else ""
            path = ROOT / first if first and not first.startswith("Downloads") else None
            experiments.append(
                {
                    "experiment_id": row.get("experiment_id"),
                    "manifest_status": row.get("status"),
                    "artifact": artifact,
                    "artifact_exists": bool(path is not None and path.exists()),
                }
            )
        present = [row for row in checkpoints if row["exists"]]
        matched = [row for row in present if row["sha256_match"] is True]
        mismatched = [row for row in present if row["sha256_match"] is False]
        metrics = to_jsonable(
            {
                "n_checkpoint_rows": len(checkpoints),
                "n_checkpoints_present": len(present),
                "n_checkpoint_hashes_matched": len(matched),
                "n_checkpoint_hashes_mismatched": len(mismatched),
                "checkpoints": checkpoints,
                "data_files": data,
                "experiment_manifest": experiments,
                "disk_free_bytes": usage.free,
                "models_evaluated": False,
                "observations_present": (ROOT / "outputs" / "physics_mpm" / "observations.npy").is_file(),
            }
        )
        (run.run_dir / "baseline.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        run.complete(metrics)
    print(run.run_dir)


if __name__ == "__main__":
    main()
