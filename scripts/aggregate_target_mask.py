"""Write the target-mask ablation table. Missing runs stay unmeasured."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.target_mask_sweep import OPTIONAL, PRIMARY, aggregate_target_masks


def load_records(run_dir: Path) -> dict[str, dict | None]:
    records: dict[str, dict | None] = {}
    for name in (*PRIMARY, *OPTIONAL):
        path = run_dir / name / "metrics.json"
        records[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/target_mask_v1.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "experiments/runs/target_mask_v1")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if protocol.get("select_variant_on_test"):
        raise SystemExit("refusing to aggregate a protocol that selects on test.")
    summary = aggregate_target_masks(load_records(args.run_dir), select_on_test=False)
    summary["experiment_id"] = protocol.get("experiment_id")
    summary["role"] = protocol.get("role")
    summary["success_rule"] = protocol.get("success_rule")
    summary["trained"] = any(row["status"] == "COMPLETE" for row in summary["rows"] if row["name"] in PRIMARY)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    destination = args.run_dir / "aggregation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
