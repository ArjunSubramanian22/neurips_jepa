#!/usr/bin/env python3
"""Seed-median summary of the corrected multi-horizon counterfactual rows.

Open-loop latent columns match the v1 files. Teacher-forced physical columns
come from the v2 correction. Cosine is undefined unless cosine_status is ok.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCES = (
    ROOT / "experiments" / "runs" / "mpm_high_multih" / "branch_rows_physindex_v2.csv",
    ROOT / "experiments" / "runs" / "mpm_high_test_final" / "branch_rows_physindex_v2.csv",
)
OUT = ROOT / "experiments" / "runs" / "canonical_counterfactual"


def _accumulate(path: Path, buckets: dict) -> int:
    n = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n += 1
            key = (
                row.get("regime") or row["fork_id"].split(":", 1)[0],
                row["model"],
                int(row["seed"]),
                row["protocol"],
                int(row["horizon"]),
                int(row["episode"]),
            )
            slot = buckets[key]
            if row["cosine_status"] == "ok" and row["cosine"] != "":
                slot["cosine"].append(float(row["cosine"]))
            if row["response_ratio"] != "":
                slot["response_ratio"].append(float(row["response_ratio"]))
            slot["rms_pos"].append(float(row["rms_pos"]))
            slot["raster_mse"].append(float(row["raster_mse"]))
    return n


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    buckets: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"cosine": [], "response_ratio": [], "rms_pos": [], "raster_mse": []}
    )
    counts = {}
    for path in SOURCES:
        if not path.is_file():
            raise SystemExit(f"missing corrected rows: {path}")
        counts[str(path.relative_to(ROOT))] = _accumulate(path, buckets)
    episode_level: dict[tuple, list[dict]] = defaultdict(list)
    for key, slot in buckets.items():
        regime, model, seed, protocol, horizon, episode = key
        episode_level[(regime, model, seed, protocol, horizon)].append({
            "episode": episode,
            "cosine": _median(slot["cosine"]),
            "response_ratio": _median(slot["response_ratio"]),
            "rms_pos": _median(slot["rms_pos"]),
            "raster_mse": _median(slot["raster_mse"]),
            "n_defined_cosine": len(slot["cosine"]),
        })
    seed_rows = []
    for key, episodes in sorted(episode_level.items()):
        regime, model, seed, protocol, horizon = key
        seed_rows.append({
            "regime": regime,
            "model": model,
            "seed": seed,
            "protocol": protocol,
            "horizon": horizon,
            "n_episodes": len(episodes),
            "cosine": _median([item["cosine"] for item in episodes if item["cosine"] is not None]),
            "n_defined_cosine_episodes": sum(item["cosine"] is not None for item in episodes),
            "response_ratio": _median([item["response_ratio"] for item in episodes if item["response_ratio"] is not None]),
            "rms_pos": _median([item["rms_pos"] for item in episodes]),
            "raster_mse": _median([item["raster_mse"] for item in episodes]),
        })
    headline = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["regime"], row["model"], row["protocol"], row["horizon"])].append(row)
    for key, rows in sorted(grouped.items()):
        regime, model, protocol, horizon = key
        headline.append({
            "regime": regime,
            "model": model,
            "protocol": protocol,
            "horizon": horizon,
            "n_seeds": len(rows),
            "cosine": _median([row["cosine"] for row in rows if row["cosine"] is not None]),
            "response_ratio": _median([row["response_ratio"] for row in rows if row["response_ratio"] is not None]),
            "rms_pos": _median([row["rms_pos"] for row in rows if row["rms_pos"] is not None]),
            "raster_mse": _median([row["raster_mse"] for row in rows if row["raster_mse"] is not None]),
            "seed_cosine": {str(row["seed"]): row["cosine"] for row in rows},
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seed_summary.json").write_text(json.dumps(seed_rows, indent=2) + "\n", encoding="utf-8")
    (OUT / "headline.json").write_text(json.dumps({
        "aggregation": "median across seeds of the median across episodes of the median across branch times",
        "cosine_rule": "branch rows with cosine_status other than ok are omitted",
        "sources": counts,
        "rows": headline,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": counts, "n_headline": len(headline)}, indent=2))


if __name__ == "__main__":
    main()
