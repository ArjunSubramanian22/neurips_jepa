#!/usr/bin/env python3
"""One-shot held-out evaluation for high-excitation episodes 1620-1799.

This file is not called by scripts/run_confirmatory_queue.py. It does not
score anything unless the protocol freeze exists, the acknowledgment flag is
set, and TEST_EVALUATED.lock is absent. Do not run it during development.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FREEZE_MD = ROOT / "experiments" / "MPM_HIGH_PROTOCOL_FREEZE.md"
FREEZE_JSON = ROOT / "experiments" / "mpm_high_protocol_freeze.json"
OUT = ROOT / "experiments" / "runs" / "mpm_high_test_final"
LOCK = OUT / "TEST_EVALUATED.lock"
TEST_EPISODES = list(range(1620, 1800))


def _warn() -> None:
    print("=" * 72, file=sys.stderr)
    print("THIS WILL OPEN THE HELD-OUT TEST SET.", file=sys.stderr)
    print("Episodes 1620-1799. No hyperparameter selection. No rerun without", file=sys.stderr)
    print("the dangerous override. The training queue must not call this file.", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Held-out high-excitation test. Refuses by default.")
    parser.add_argument(
        "--i-understand-this-opens-the-test-set",
        action="store_true",
        help="Required acknowledgment. Still requires the protocol freeze.",
    )
    parser.add_argument(
        "--override-existing-lock",
        action="store_true",
        help="Dangerous. Re-opens a test set that was already scored.",
    )
    args = parser.parse_args(argv)
    _warn()
    if not args.i_understand_this_opens_the_test_set:
        raise SystemExit("refusing: acknowledgment flag was not set.")
    if not FREEZE_MD.is_file() or not FREEZE_JSON.is_file():
        raise SystemExit(
            "refusing: experiments/MPM_HIGH_PROTOCOL_FREEZE.md and "
            "experiments/mpm_high_protocol_freeze.json must both exist."
        )
    if LOCK.exists() and not args.override_existing_lock:
        raise SystemExit(f"refusing: {LOCK} already exists. The test set was already opened.")
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    episodes = [int(episode) for episode in freeze["episodes"]]
    if episodes != TEST_EPISODES:
        raise SystemExit("refusing: the freeze episode list is not exactly 1620-1799.")
    if freeze.get("hyperparameter_selection"):
        raise SystemExit("refusing: the freeze requests hyperparameter selection.")
    if str(freeze.get("metric_version", "")) == "":
        raise SystemExit("refusing: the freeze does not name a metric version.")
    OUT.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if args.override_existing_lock:
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    try:
        descriptor = os.open(LOCK, flags, 0o644)
    except FileExistsError:
        raise SystemExit(f"refusing: {LOCK} already exists. The test set was already opened.") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            "TEST SET 1620-1799 HAS BEEN OPENED\n"
            f"git_commit: {freeze.get('git_commit')}\n"
            f"protocol_sha256: {freeze.get('protocol_sha256')}\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    from scripts.run_mpm_high_pretest import run_frozen_test
    run_frozen_test()
    print("Held-out test scoring finished. The lock remains.")


if __name__ == "__main__":
    main()
