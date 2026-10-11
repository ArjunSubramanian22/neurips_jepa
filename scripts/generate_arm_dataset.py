"""Generate the modest arm dataset and refuse to continue if validation fails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.arm_data import generate_arm_dataset, validate_arm_dataset
from src.registry import to_jsonable

DATASET = ROOT / "outputs" / "mujoco_arm"
REPORT = ROOT / "experiments" / "runs" / "mujoco_arm_v1" / "validation.json"


def main() -> None:
    generate_arm_dataset(DATASET)
    report = validate_arm_dataset(DATASET)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(to_jsonable(report), indent=2) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
