"""Write deterministic simulator forks for one intervention family.

Example:
    python3.13 scripts/generate_counterfactual_pairs.py --episodes 0 --out outputs/counterfactual_pairs/force_reversal_ep0.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MPMConfig
from src.action_alignment import FAMILIES, PRIMARY_FAMILY, fork_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--times", type=int, nargs="*", default=None)
    parser.add_argument("--family", choices=FAMILIES, default=PRIMARY_FAMILY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--dump-seed", type=int, default=0)
    args = parser.parse_args()
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
            mpm=MPMConfig(),
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
    merged.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
