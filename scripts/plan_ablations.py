"""Dry-run the declared ablation catalog.

    python3.13 scripts/plan_ablations.py
    python3.13 scripts/plan_ablations.py --tier 2 --required-only
    python3.13 scripts/plan_ablations.py --output experiments/ablations/v1/dry_run.json

The default prints commands and does not train. ``--launch`` runs only
``NOT_RUN`` jobs, refuses a directory that already has files, and will not
replace an existing config. A new ``--version`` writes under a new root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation_plan import (
    NOT_RUN,
    annotate_status,
    assert_output_available,
    dry_run_summary,
    expand_jobs,
    format_dry_run,
    load_catalog,
    load_seed_registry,
    materialize_config,
    measured_reference_seconds,
)
from src.registry import utc_now

CATALOG = ROOT / "configs" / "ablations" / "catalog_v1.yaml"


def build_jobs(version: str | None = None) -> tuple[dict, list[dict]]:
    catalog = load_catalog(CATALOG)
    if version:
        catalog["version"] = version
    registry = load_seed_registry(ROOT / catalog["seed_registry"])
    reference = measured_reference_seconds(catalog, ROOT)
    jobs = annotate_status(expand_jobs(catalog, registry, reference_seconds=reference), ROOT)
    return catalog, jobs


def select_jobs(jobs: list[dict], *, tier: int | None, required_only: bool) -> list[dict]:
    chosen = jobs
    if tier is not None:
        chosen = [job for job in chosen if job["tier"] == tier]
    if required_only:
        chosen = [job for job in chosen if job["required"]]
    return chosen


def append_ledger(path: Path, summary: dict, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at": utc_now(),
        "summary": summary,
        "jobs": [
            {
                "job_id": job["job_id"],
                "template": job["template"],
                "seed": job["seed"],
                "tier": job["tier"],
                "status": job["status"],
                "output_dir": job["output_dir"],
            }
            for job in jobs
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def launch_jobs(jobs: list[dict]) -> None:
    for job in jobs:
        if job["status"] != NOT_RUN or not job["command"]:
            continue
        if job.get("overrides"):
            materialize_config(job, ROOT)
        assert_output_available(ROOT / job["output_dir"])
        subprocess.run(job["command"], check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="Dry-run the ablation catalog.")
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--required-only", action="store_true")
    parser.add_argument("--version", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    catalog, jobs = build_jobs(args.version)
    chosen = select_jobs(jobs, tier=args.tier, required_only=args.required_only)
    summary = dry_run_summary(chosen)
    text = format_dry_run(summary)
    version = str(catalog["version"])
    ledger = Path(args.ledger) if args.ledger else ROOT / "experiments" / "ablations" / version / "ledger.jsonl"
    append_ledger(ledger, summary, chosen)
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise SystemExit(f"refusing to overwrite {destination}.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"summary": summary, "jobs": chosen}, indent=2) + "\n", encoding="utf-8")
    if args.launch:
        launch_jobs(chosen)
    print(text, end="")
    return text


if __name__ == "__main__":
    main(sys.argv[1:])
