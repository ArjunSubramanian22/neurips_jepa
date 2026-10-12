#!/usr/bin/env python3
"""Write one named MLS-MPM dump from an environment YAML.

The historical generator stays on ``MPMConfig()`` and ``command_hold=3``.
This script is for a calibrated regime. ``--n-episodes`` should be a multiple
of 60 when the dump will use ``fresh_pool_v2``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import generate_dataset
from src.environments import environment_from_dict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a named-environment MLS-MPM dump.")
    parser.add_argument("--env", required=True, help="Environment YAML.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-episodes", type=int, default=1800)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split-version", default="fresh_pool_v2")
    args = parser.parse_args(argv)
    payload = yaml.safe_load(Path(args.env).read_text(encoding="utf-8"))
    spec = environment_from_dict(payload)
    meta = generate_dataset(
        args.output,
        n_episodes=int(args.n_episodes),
        episode_length=int(args.episode_length),
        seed=int(args.seed),
        img_size=int(spec.mpm.img_size),
        mpm=spec.mpm,
        workers=None if int(args.workers) == 0 else int(args.workers),
        command_hold=int(spec.command_hold),
    )
    meta["environment_id"] = spec.environment_id
    meta["environment_role"] = spec.role
    meta["split_version"] = str(args.split_version)
    meta["source_config"] = str(args.env)
    meta["is_test_set"] = False
    meta["notes"] = (
        "Physics dump only. Model selection must use the train and validation "
        "blocks of split_version. The test block stays closed."
    )
    out = Path(args.output)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "environment_id": spec.environment_id, "seconds": meta["seconds"]}))


if __name__ == "__main__":
    main()
