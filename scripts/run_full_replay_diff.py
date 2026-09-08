#!/usr/bin/env python3
"""Byte-level original observations.npy vs local RNG replay for all 2000 episodes."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import replay_episode

DUMP = ROOT / "outputs" / "physics_mpm"
OUT = ROOT / "results" / "audit_phase2"


def _one(episode: int) -> dict[str, int | float]:
    rec = replay_episode(int(episode), dump_seed=0, episode_length=16)
    dump_frames = np.load(DUMP / "observations.npy", mmap_mode="r")[episode]
    dump_actions = np.load(DUMP / "actions.npy")[episode]
    delta = np.abs(dump_frames.astype(np.int16) - rec["frames"].astype(np.int16))
    per_ch = [(delta[:, c] > 0).sum() for c in range(3)]
    return {
        "episode": int(episode),
        "n_frames_diff": int(np.any(dump_frames != rec["frames"], axis=(1, 2, 3)).sum()),
        "n_pixels_diff": int((delta > 0).sum()),
        "max_abs": int(delta.max()),
        "sum_abs": float(delta.sum()),
        "n_pixels": int(delta.size),
        "ch0": int(per_ch[0]),
        "ch1": int(per_ch[1]),
        "ch2": int(per_ch[2]),
        "action_equal": int(np.array_equal(dump_actions, rec["actions"])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_ep = 2000
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, row in enumerate(pool.map(_one, range(n_ep), chunksize=40)):
            rows.append(row)
            if i % 200 == 0:
                print(f"replay-diff {i}/{n_ep}", flush=True)
    n_pixels = sum(r["n_pixels"] for r in rows)
    n_pixels_diff = sum(r["n_pixels_diff"] for r in rows)
    n_frames_diff = sum(r["n_frames_diff"] for r in rows)
    report = {
        "n_episodes": n_ep,
        "n_frames": n_ep * 16,
        "n_frames_any_pixel_diff": n_frames_diff,
        "n_pixels": n_pixels,
        "n_pixels_diff": n_pixels_diff,
        "frac_pixels_diff": n_pixels_diff / n_pixels,
        "max_abs_uint8": max(r["max_abs"] for r in rows),
        "mean_abs_uint8": sum(r["sum_abs"] for r in rows) / n_pixels,
        "per_channel_pixels_diff": [
            sum(r["ch0"] for r in rows),
            sum(r["ch1"] for r in rows),
            sum(r["ch2"] for r in rows),
        ],
        "n_episodes_any_diff": int(sum(r["n_pixels_diff"] > 0 for r in rows)),
        "n_action_mismatches": int(sum(r["action_equal"] == 0 for r in rows)),
        "method": "replay_episode dump_seed=0 (RNG resample, not Linux recorded-action replay)",
    }
    (OUT / "replay_diff_full.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "replay_diff_per_episode.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    csv_path = ROOT / "results" / "replay_diff_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
