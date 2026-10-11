"""Write the small MuJoCo fork audit. Does not train or load a checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.mujoco_audit import audit_task
from src.registry import to_jsonable

TASKS = ("slide_x", "slide_y", "glance_x")


def main() -> None:
    payload = {
        "experiment_id": "mujoco_box_push_v1",
        "environment_id": "mujoco_box_push",
        "dependency": {
            "package": "mujoco",
            "version": "3.3.7",
            "license": "Apache-2.0",
        },
        "learned_model": {
            "status": "not_measured",
            "reason": "no checkpoint was trained on mujoco_box_push observations",
        },
        "tasks": [audit_task(task, seed=0) for task in TASKS],
    }
    out = ROOT / "experiments" / "runs" / "mujoco_box_push_v1" / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
