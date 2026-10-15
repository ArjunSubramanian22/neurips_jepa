#!/usr/bin/env python3
"""Write deterministic simulator forks for one intervention family.

The environment config is required. This script does not default to the
legacy actuator.

    python3.13 scripts/generate_counterfactual_pairs.py \
      --env-config configs/environments/mpm_legacy.yaml \
      --dump outputs/physics_mpm \
      --episodes 0 --out outputs/counterfactual_pairs/legacy_ep0.npz

Do not point a high-excitation dump at the legacy config. The script refuses
that mix and does not write a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action_alignment import FAMILIES, PRIMARY_FAMILY, fork_episode
from src.environments import environment_from_dict

HIGH_GAIN = 3200.0
HIGH_DAMPING = 0.01
HIGH_HOLD = 8
LEGACY_GAIN = 55.0
LEGACY_DAMPING = 0.08
LEGACY_HOLD = 3
TEST_START = 1620


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _refuse_wrong_actuator(spec, *, dump: str | None, out: Path) -> None:
    gain = float(spec.mpm.action_gain)
    damping = float(spec.mpm.damping)
    hold = int(spec.command_hold)
    name = spec.environment_id
    high = (gain, damping, hold) == (HIGH_GAIN, HIGH_DAMPING, HIGH_HOLD)
    legacy = (gain, damping, hold) == (LEGACY_GAIN, LEGACY_DAMPING, LEGACY_HOLD)
    if name == "mpm_high_excitation" and not high:
        raise SystemExit(
            "refusing a high-excitation cache whose gain, damping, or hold "
            f"is {(gain, damping, hold)}, not {(HIGH_GAIN, HIGH_DAMPING, HIGH_HOLD)}."
        )
    if name == "mpm_legacy" and not legacy:
        raise SystemExit(
            "refusing a legacy cache whose gain, damping, or hold "
            f"is {(gain, damping, hold)}."
        )
    looks_high = "high" in name or "mpm_high" in str(out) or (dump is not None and "physics_mpm_high" in dump)
    if looks_high and legacy:
        raise SystemExit("refusing to build a high-excitation cache with the legacy actuator.")
    if looks_high and not high:
        raise SystemExit("refusing a high-excitation cache that does not resolve to gain 3200, damping 0.01, hold 8.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True, help="Environment YAML. There is no legacy default.")
    parser.add_argument("--dump", default=None, help="Dump directory whose metadata is hashed into the cache sidecar.")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--times", type=int, nargs="*", default=None)
    parser.add_argument("--family", choices=FAMILIES, default=PRIMARY_FAMILY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--dump-seed", type=int, default=0)
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and check the actuator, then write nothing.")
    args = parser.parse_args(argv)
    if any(int(episode) >= TEST_START for episode in args.episodes) and "mpm_high" in args.env_config:
        raise SystemExit("refusing high-excitation test episodes 1620-1799.")
    payload = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    spec = environment_from_dict(payload)
    _refuse_wrong_actuator(spec, dump=args.dump, out=args.out)
    if args.dry_run:
        print(json.dumps({
            "environment_id": spec.environment_id,
            "action_gain": float(spec.mpm.action_gain),
            "damping": float(spec.mpm.damping),
            "command_hold": int(spec.command_hold),
            "wrote": False,
        }))
        return
    if args.family != PRIMARY_FAMILY:
        print(f"note: primary family is {PRIMARY_FAMILY}; this file is {args.family} and must not be pooled with it.")
    times = args.times if args.times else list(range(args.episode_length - 1))
    caches = [
        fork_episode(
            episode,
            times,
            args.family,
            dump_seed=args.dump_seed,
            episode_length=args.episode_length,
            mpm=spec.mpm,
            hold=int(spec.command_hold),
        )
        for episode in args.episodes
    ]
    merged = caches[0]
    for extra in caches[1:]:
        if extra.family != merged.family:
            raise SystemExit("refusing to pool intervention families.")
        merged = type(merged)(
            merged.family,
            np.concatenate([merged.episode, extra.episode]),
            np.concatenate([merged.time, extra.time]),
            np.concatenate([merged.action_prime, extra.action_prime]),
            np.concatenate([merged.frame_prime, extra.frame_prime]),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(args.out)
    dump_meta = None
    if args.dump:
        meta_path = Path(args.dump) / "metadata.json"
        if meta_path.is_file():
            dump_meta = {"path": str(meta_path), "sha256": _sha256(meta_path)}
    sidecar = {
        "env_config": args.env_config,
        "environment_id": spec.environment_id,
        "action_gain": float(spec.mpm.action_gain),
        "damping": float(spec.mpm.damping),
        "command_hold": int(spec.command_hold),
        "contact_radius": float(spec.mpm.contact_radius),
        "dump": dump_meta,
        "episodes": [int(episode) for episode in args.episodes],
        "split": args.split,
        "family": args.family,
        "dump_seed": int(args.dump_seed),
        "git_commit": _git_commit(),
        "intervention": "see src.action_alignment.apply_intervention for this family",
    }
    args.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
