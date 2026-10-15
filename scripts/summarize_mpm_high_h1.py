#!/usr/bin/env python3
"""Aggregate high-excitation horizon-1 validation screens.

Reads existing screen.json files. Does not touch episodes 1620-1799.
Undefined cosines stay null. Seed-level medians are the replication unit.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
OUT_CSV = RUNS / "mpm_high_h1_summary.csv"
OUT_JSON = RUNS / "mpm_high_h1_summary.json"
OUT_MD = RUNS / "mpm_high_cross_model_screen_report.md"

EXPECTED = [
    ("vicreg_action", "action", 42, "mpm_high_jepa_s42_val_screen_v1"),
    ("vicreg_action", "action", 43, "mpm_high_jepa_s43_val_screen_v1"),
    ("vicreg_action", "action", 44, "mpm_high_jepa_s44_val_screen_v2"),
    ("vicreg_action", "action", 45, "mpm_high_jepa_s45_val_screen_v2"),
    ("vicreg_action", "action", 46, "mpm_high_jepa_s46_val_screen_v2"),
    ("vicreg_noaction", "no_action", 42, "mpm_high_noaction_s42_val_screen_v1"),
    ("vicreg_noaction", "no_action", 43, "mpm_high_noaction_s43_val_screen_v2"),
    ("vicreg_noaction", "no_action", 44, "mpm_high_noaction_s44_val_screen_v2"),
    ("vicreg_noaction", "no_action", 45, "mpm_high_noaction_s45_val_screen_v2"),
    ("vicreg_noaction", "no_action", 46, "mpm_high_noaction_s46_val_screen_v2"),
    ("mse_action", "action", 42, "mpm_high_mse_s42_val_screen_v2"),
    ("mse_action", "action", 43, "mpm_high_mse_s43_val_screen_v2"),
    ("mse_action", "action", 44, "mpm_high_mse_s44_val_screen_v2"),
    ("mse_action", "action", 45, "mpm_high_mse_s45_val_screen_v2"),
    ("mse_action", "action", 46, "mpm_high_mse_s46_val_screen_v2"),
    ("rssm", "action", 42, "mpm_high_rssm_s42_val_screen_v2"),
    ("rssm", "action", 43, "mpm_high_rssm_s43_val_screen_v2"),
    ("rssm", "action", 44, "mpm_high_rssm_s44_val_screen_v2"),
    ("rssm", "action", 45, "mpm_high_rssm_s45_val_screen_v2"),
    ("rssm", "action", 46, "mpm_high_rssm_s46_val_screen_v2"),
]


def _finite(values: list) -> np.ndarray:
    kept = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            kept.append(number)
    return np.asarray(kept, dtype=np.float64)


def _median(values: list):
    array = _finite(values)
    if array.size == 0:
        return None
    return float(np.median(array))


def _mean(values: list):
    array = _finite(values)
    if array.size == 0:
        return None
    return float(array.mean())


def _quantile(values: list, q: float):
    array = _finite(values)
    if array.size == 0:
        return None
    return float(np.quantile(array, q))


def _bootstrap_median(values: list, rng: np.random.Generator, n_resamples: int = 10000):
    array = _finite(values)
    if array.size == 0:
        return None, None, "undefined"
    draws = rng.integers(0, array.size, size=(n_resamples, array.size))
    medians = np.median(array[draws], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)), "episode_bootstrap_median"


def _row_from_screen(model: str, variant: str, seed: int, payload: dict) -> dict:
    summary = payload["summary"]
    rows = payload.get("rows", [])
    has_cosine = bool(rows) and "cosine" in rows[0]
    cosines = [row.get("cosine") for row in rows] if has_cosine else []
    if has_cosine:
        low, high, ci_status = _bootstrap_median(cosines, np.random.default_rng(int(seed)))
    else:
        low, high, ci_status = None, None, "not_in_source_rows"
    defined = int(summary.get("n_cosine_defined", 0))
    return {
        "model": model,
        "variant": variant,
        "seed": seed,
        "n_episodes": int(summary["n_episodes"]),
        "n_defined_cosine": defined,
        "median_cosine": summary.get("cosine_median"),
        "mean_cosine": _mean(cosines) if has_cosine else None,
        "cosine_q25": _quantile(cosines, 0.25) if has_cosine else None,
        "cosine_q75": _quantile(cosines, 0.75) if has_cosine else None,
        "cosine_ci_low": low,
        "cosine_ci_high": high,
        "cosine_ci_status": ci_status,
        "median_pred_contrast_norm": summary.get("median_pred_contrast_norm"),
        "median_target_contrast_norm": summary.get("median_target_contrast_norm"),
        "median_response_ratio": summary.get("median_response_ratio"),
        "median_one_step_error": summary.get("one_step_mse_median"),
        "median_raster_mse": summary.get("observation_mse_median"),
        "identical_raster_fraction": summary.get("fraction_identical_median"),
        "median_position_rms": summary.get("rms_pos_median"),
        "split": summary.get("split_role", "validation"),
        "episode_start": summary.get("episode_start"),
        "episode_end": None if summary.get("episode_stop") is None else int(summary["episode_stop"]) - 1,
        "source": None,
    }


def _markdown(table: list[dict], missing: list[str]) -> str:
    lines = [
        "# High-excitation horizon-1 screen",
        "",
        "Validation episodes 1440-1463. Time 0 only. Force reversal. Open loop.",
        "Undefined cosine is null, not zero. Raw latent MSE is not compared across families.",
        "",
    ]
    if missing:
        lines.append("Missing screens: " + ", ".join(missing))
        lines.append("")
    for model in ("vicreg_action", "vicreg_noaction", "mse_action", "rssm"):
        block = [row for row in table if row["model"] == model]
        if not block:
            continue
        medians = [row["median_cosine"] for row in block if row["median_cosine"] is not None]
        lines.append(f"## {model}")
        lines.append("")
        for row in block:
            lines.append(
                f"- seed {row['seed']}: cosine {row['median_cosine']} "
                f"defined {row['n_defined_cosine']}/{row['n_episodes']} "
                f"one-step error {row['median_one_step_error']} "
                f"pred norm {row['median_pred_contrast_norm']}"
            )
        if medians:
            lines.append(
                f"- seed-level cosine medians {medians}; "
                f"median of those medians {float(np.median(medians)):.6f}; "
                f"range {float(min(medians)):.6f} to {float(max(medians)):.6f}."
            )
        else:
            lines.append("- no defined cosine. Predictor contrast is the reported norm, not a cosine of 0.")
        lines.append("")
    lines.extend(["## Questions", ""])
    lines.extend(_answers(table, missing))
    lines.append("")
    return "\n".join(lines)


def _block(table: list[dict], model: str) -> list[dict]:
    return [row for row in table if row["model"] == model]


def _complete(block: list[dict], n: int = 5) -> bool:
    return len(block) == n and all(row["n_episodes"] == 24 for row in block)


def _answers(table: list[dict], missing: list[str]) -> list[str]:
    jepa = _block(table, "vicreg_action")
    noaction = _block(table, "vicreg_noaction")
    mse = _block(table, "mse_action")
    rssm = _block(table, "rssm")
    if missing:
        return [
            "The six questions stay unanswered until every expected screen.json exists.",
            "Missing: " + ", ".join(missing) + ".",
            "Undefined cosine is not zero.",
        ]
    jepa_values = ", ".join(f"seed {row['seed']} = {row['median_cosine']}" for row in jepa)
    no_values = ", ".join(
        f"seed {row['seed']} defined {row['n_defined_cosine']}/{row['n_episodes']} pred norm {row['median_pred_contrast_norm']}"
        for row in noaction
    )
    mse_values = ", ".join(f"seed {row['seed']} = {row['median_cosine']}" for row in mse)
    rssm_values = ", ".join(f"seed {row['seed']} = {row['median_cosine']}" for row in rssm)
    jepa_ok = _complete(jepa) and all(row["median_cosine"] is not None and row["n_defined_cosine"] == 24 for row in jepa)
    no_ok = _complete(noaction) and all(row["n_defined_cosine"] == 0 and row["median_cosine"] is None for row in noaction)
    mse_ok = _complete(mse) and all(row["median_cosine"] is not None and row["n_defined_cosine"] == 24 for row in mse)
    rssm_ok = _complete(rssm) and all(row["median_cosine"] is not None and row["n_defined_cosine"] == 24 for row in rssm)
    return [
        f"1. JEPA action seed medians: {jepa_values}. Directional on every seed: {jepa_ok}.",
        f"2. No-action: {no_values}. Cosine stays null with defined count 0: {no_ok}.",
        f"3. MSE action seed medians, compared only inside the MSE latent: {mse_values}. Defined on every seed: {mse_ok}.",
        f"4. RSSM seed medians in concat(h, stochastic mean), from one shared filtered posterior: {rssm_values}. Defined on every seed: {rssm_ok}.",
        f"5. The directional effect is present in JEPA, MSE, and RSSM, and absent for no-action: {jepa_ok and mse_ok and rssm_ok and no_ok}.",
        "6. Seed variation is the range of the five seed medians printed above. The 24 episodes inside one seed are not five training replicates.",
    ]


def _seed_aggregation(table: list[dict]) -> dict:
    grouped = {}
    for model in ("vicreg_action", "vicreg_noaction", "mse_action", "rssm"):
        block = _block(table, model)
        defined = [row["median_cosine"] for row in block if row["median_cosine"] is not None]
        grouped[model] = {
            "seeds": [row["seed"] for row in block],
            "seed_medians": [row["median_cosine"] for row in block],
            "median_of_seed_medians": None if not defined else float(np.median(defined)),
            "min_seed_median": None if not defined else float(min(defined)),
            "max_seed_median": None if not defined else float(max(defined)),
            "unit": "training seed",
        }
    return grouped


def main() -> int:
    table = []
    missing = []
    for model, variant, seed, experiment_id in EXPECTED:
        path = RUNS / experiment_id / "screen.json"
        if not path.is_file():
            missing.append(experiment_id)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = _row_from_screen(model, variant, seed, payload)
        row["source"] = str(path.relative_to(ROOT))
        table.append(row)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "variant", "seed", "n_episodes", "n_defined_cosine", "median_cosine",
        "mean_cosine", "cosine_q25", "cosine_q75", "cosine_ci_low", "cosine_ci_high",
        "cosine_ci_status", "median_pred_contrast_norm", "median_target_contrast_norm",
        "median_response_ratio", "median_one_step_error", "median_raster_mse",
        "identical_raster_fraction", "median_position_rms", "split", "episode_start",
        "episode_end", "source",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    OUT_JSON.write_text(
        json.dumps(
            {
                "rows": table,
                "missing": missing,
                "seed_aggregation": _seed_aggregation(table),
                "test_scored": False,
                "note": "Raw one-step error is not comparable across vicreg, mse, and rssm latent spaces.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_markdown(table, missing), encoding="utf-8")
    print(json.dumps({"rows": len(table), "missing": missing}))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
