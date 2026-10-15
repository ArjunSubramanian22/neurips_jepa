#!/usr/bin/env python3
"""Generate the arm campaign dataset and refuse it if the audit fails.

The probe is a small disjoint train/validation/test file used to time the
simulator. The full file is a separate directory and is not a test-set
evaluation of a trained model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.arm_campaign import (  # noqa: E402
    N_TEST,
    N_TRAIN,
    N_VAL,
    audit_campaign_dataset,
    generate_campaign_dataset,
)

PROBE = ROOT / "experiments" / "runs" / "arm_campaign_v2" / "probe"
FULL = ROOT / "outputs" / "mujoco_arm_campaign"


def _run(path: Path, n_train: int, n_val: int, n_test: int, length: int) -> dict:
    started = time.perf_counter()
    generate_campaign_dataset(path, n_train=n_train, n_val=n_val, n_test=n_test, length=length)
    seconds = time.perf_counter() - started
    report = audit_campaign_dataset(path)
    report["generation_seconds"] = seconds
    report["seconds_per_episode"] = seconds / float(n_train + n_val + n_test)
    if report["n_visible_interventions"] < 1:
        raise SystemExit("refusing: no reflected action changed both the object and the image.")
    if report["push_displacement_median"] is None or report["hold_displacement_median"] is None:
        raise SystemExit("refusing: the file has no push and hold episodes to compare.")
    if report["push_displacement_median"] <= report["hold_displacement_median"]:
        raise SystemExit("refusing: push episodes do not move the object more than hold episodes.")
    path.joinpath("audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "intervention_sample"}, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate or probe the arm campaign dataset.")
    parser.add_argument("--mode", choices=("probe", "full"), default="probe")
    args = parser.parse_args(argv)
    if args.mode == "probe":
        _run(PROBE, 6, 6, 6, 8)
        return
    _run(FULL, N_TRAIN, N_VAL, N_TEST, 16)


if __name__ == "__main__":
    main()
