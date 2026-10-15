#!/usr/bin/env python3
"""Rewrite teacher-forced JEPA physical and raster columns from cached forks.

Latent columns are copied. Open-loop rows and RSSM rows are checked against
the same cache and left unchanged. The original CSVs are not overwritten.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mpm_high_pretest import (  # noqa: E402
    BRANCH_TIMES,
    _raster_metrics,
    _state_metrics,
    physical_step,
)
from src.counterfactual_study import classify_bottleneck  # noqa: E402

CACHE_ROOT = ROOT / "outputs" / "counterfactual_cache"
PHYSICAL_KEYS = (
    "rms_pos",
    "rms_vel",
    "com_displacement",
    "raster_mse",
    "raster_l1",
    "identical_raster",
    "changed_pixel_fraction",
)
SOURCES = (
    ROOT / "experiments" / "runs" / "mpm_high_multih" / "branch_rows.csv",
    ROOT / "experiments" / "runs" / "mpm_high_test_final" / "branch_rows.csv",
    ROOT / "experiments" / "runs" / "mpm_high_bottleneck" / "branch_rows.csv",
)


def _regime(row: dict) -> str:
    regime = row.get("regime") or ""
    if regime:
        return regime
    return row["fork_id"].split(":", 1)[0]


def _needs_recompute(row: dict) -> bool:
    return row["protocol"] == "teacher_forced" and row["model"] != "rssm" and int(row["horizon"]) > 1


def _close(left: str, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)


def metrics_at(blob: np.lib.npyio.NpzFile, branch_index: int, frame: int) -> dict[str, float]:
    physical = _state_metrics(
        blob["pos_rec"][branch_index, frame],
        blob["vel_rec"][branch_index, frame],
        blob["pos_rev"][branch_index, frame],
        blob["vel_rev"][branch_index, frame],
    )
    raster = _raster_metrics(blob["frames_rec"][branch_index, frame], blob["frames_rev"][branch_index, frame])
    return {
        "rms_pos": physical["rms_pos"],
        "rms_vel": physical["rms_vel"],
        "com_displacement": physical["com_displacement"],
        "raster_mse": raster["raster_mse"],
        "raster_l1": raster["raster_l1"],
        "identical_raster": raster["identical_raster"],
        "changed_pixel_fraction": raster["changed_pixel_fraction"],
    }


def correct_rows(rows: list[dict], cache_root: Path) -> dict:
    """Return a summary. ``rows`` is updated in place."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (_regime(row), int(row["episode"]), int(row["branch_time"]), int(row["horizon"]))
        groups[key].append(index)
    by_episode: dict[tuple, list[tuple]] = defaultdict(list)
    for key in groups:
        by_episode[(key[0], key[1])].append(key)
    changed = 0
    checked = 0
    max_pos_delta = 0.0
    label_changes = 0
    for (regime, episode), keys in by_episode.items():
        path = cache_root / regime / f"ep{episode}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing fork cache {path}.")
        blob = np.load(path)
        try:
            for key in keys:
                _regime_name, _episode, branch_time, horizon = key
                frame = physical_step(horizon)
                if frame >= int(blob["pos_rec"].shape[1]):
                    raise RuntimeError(f"{path} has no frame index {frame}.")
                branch_index = BRANCH_TIMES.index(branch_time)
                expected = metrics_at(blob, branch_index, frame)
                for index in groups[key]:
                    row = rows[index]
                    if _needs_recompute(row):
                        before = float(row["rms_pos"])
                        max_pos_delta = max(max_pos_delta, abs(before - expected["rms_pos"]))
                        for name in PHYSICAL_KEYS:
                            row[name] = expected[name]
                        row["physical_step"] = frame
                        row["physical_metric_version"] = "v2_recomputed"
                        changed += 1
                    else:
                        for name in PHYSICAL_KEYS:
                            if not _close(row[name], expected[name]):
                                raise RuntimeError(
                                    f"cache mismatch for {row.get('fork_id')} {row['model']} "
                                    f"{row['protocol']} H{horizon} field {name}: "
                                    f"stored {row[name]} cache {expected[name]}."
                                )
                        row["physical_step"] = frame
                        row["physical_metric_version"] = "v1_unchanged"
                        checked += 1
                    if "bottleneck" in row:
                        previous = row["bottleneck"]
                        labeled = dict(row)
                        for key in ("cosine", "response_ratio", "projection"):
                            if labeled.get(key) == "":
                                labeled[key] = None
                        revised = classify_bottleneck(labeled)
                        row["bottleneck_v1"] = previous
                        row["bottleneck"] = revised
                        if previous != revised:
                            label_changes += 1
        finally:
            blob.close()
    return {
        "n_rows": len(rows),
        "n_recomputed": changed,
        "n_checked_unchanged": checked,
        "max_abs_rms_pos_change": max_pos_delta,
        "n_bottleneck_label_changes": label_changes,
    }


def _write(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    for name in ("physical_step", "physical_metric_version", "bottleneck_v1"):
        if any(name in row for row in rows) and name not in fieldnames:
            fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def correct_file(source: Path, cache_root: Path = CACHE_ROOT) -> dict:
    source = Path(source)
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{source} has no rows.")
    summary = correct_rows(rows, cache_root)
    destination = source.with_name("branch_rows_physindex_v2.csv")
    if destination.resolve() == source.resolve():
        raise RuntimeError("refusing to overwrite the source CSV.")
    _write(destination, rows)
    summary["source"] = str(source.relative_to(ROOT))
    summary["destination"] = str(destination.relative_to(ROOT))
    summary["latent_columns"] = "copied"
    note = destination.with_suffix(".json")
    note.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summaries = [correct_file(path) for path in SOURCES if path.is_file()]
    if not summaries:
        raise SystemExit("no branch_rows.csv files found.")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
