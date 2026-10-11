"""Write the inverse-auxiliary sweep table. Missing runs stay unmeasured."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.inverse_aux import WEIGHTS, aggregate_inverse_sweep


def load_records(run_dir: Path) -> dict[str, dict | None]:
    records: dict[str, dict | None] = {}
    for weight in WEIGHTS:
        key = f"w{weight:g}"
        path = run_dir / key / "metrics.json"
        records[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    probe = run_dir / "probe" / "metrics.json"
    records["probe"] = json.loads(probe.read_text(encoding="utf-8")) if probe.is_file() else None
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/inverse_aux_v1.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "experiments/runs/inverse_aux_v1")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if protocol.get("select_weight_on_test"):
        raise SystemExit("refusing to aggregate a protocol that selects the weight on test.")
    summary = aggregate_inverse_sweep(load_records(args.run_dir), select_weight_on_test=False)
    summary["experiment_id"] = protocol.get("experiment_id")
    summary["weight_scale_note"] = protocol.get("weight_scale_note")
    summary["trained"] = any(row["status"] == "COMPLETE" for row in summary["rows"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    destination = args.run_dir / "aggregation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
