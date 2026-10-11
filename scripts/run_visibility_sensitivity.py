#!/usr/bin/env python3
"""Visibility sensitivity for the multi-horizon counterfactual audit.

    python3.13 scripts/run_visibility_sensitivity.py

The primary raster-MSE cut is fit on cvpr_v1 validation forks. No predictor
is run on those episodes. Sensitivity cuts on the legacy reporting table are
reported in full and are not used to replace the primary cut.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, render_rgb, replay_episode
from src.environments import environment_from_dict
from src.eval_protocol import fit_visibility_threshold
from src.metrics.cascade import observation_visibility
from src.metrics.visibility_sensitivity import (
    equal_frequency_bins,
    integrate_top_fraction_curve,
    percentile_cutoff,
    select_visibility,
    spearman_visibility,
    summarize_selected,
    top_fraction_name,
)
from src.registry import RunRecorder, to_jsonable
from src.rgb_render import load_rgb_config, render_rgb_image, to_uint8
from src.splits import get_split
from src.visibility_sensitivity_plots import plot_visibility_sensitivity

VISIBILITY_METRICS = ("raster_mse", "rgb_mse", "fraction_affected_pixels")


def _affected(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.any(left != right, axis=0)))


def _fit_primary(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = get_split(config["split_version"])
    if split.headline_role == "test" and config["fit_role"] != "val":
        raise RuntimeError("the primary visibility cut is fit on validation.")
    spec = environment_from_dict(
        yaml.safe_load((ROOT / "configs" / "environments" / "mpm_legacy.yaml").read_text(encoding="utf-8"))
    )
    rgb_cfg = load_rgb_config(ROOT / "configs" / "rgb" / "rgb_v1_64.yaml")
    sim = MLSMPM2D(spec.mpm)
    measured: dict[int, float] = {}
    rows = []
    for episode in split.val_ids:
        print(f"validation visibility {episode}", flush=True)
        recorded = replay_episode(int(episode), dump_seed=0, episode_length=16, mpm=spec.mpm, hold=spec.command_hold)
        action = recorded["actions"][0].copy()
        action[2:4] *= -1.0
        sim.restore(recorded["snapshots"][0])
        apply_action_frames(sim, action, 1)
        frame_b = render_rgb(sim.x, sim.v, sim.cfg.img_size, sim.cfg.render_sigma)
        raster = observation_visibility(recorded["frames"][1], frame_b)
        rgb_a = to_uint8(render_rgb_image(recorded["snapshots"][1]["x"], recorded["snapshots"][1]["v"], rgb_cfg))
        rgb_b = to_uint8(render_rgb_image(sim.x, sim.v, rgb_cfg))
        rgb = observation_visibility(rgb_a, rgb_b)
        row = {
            "episode": int(episode),
            "split_version": split.split_version,
            "role": "val",
            "is_test_set": False,
            "raster_mse": raster["mse"],
            "fraction_affected_pixels": _affected(recorded["frames"][1], frame_b),
            "rgb_mse": rgb["mse"],
            "rgb_fraction_affected_pixels": _affected(rgb_a, rgb_b),
            "uses_predictor": False,
        }
        rows.append(row)
        measured[int(episode)] = float(raster["mse"])
    if set(measured) != set(split.val_ids):
        raise RuntimeError("validation visibility did not cover every validation episode.")
    nonzero = {episode: value for episode, value in measured.items() if value > 0.0}
    fitted = fit_visibility_threshold(
        nonzero,
        split,
        role=config["fit_role"],
        quantile=float(config["primary"]["quantile"]),
    )
    fitted["n_zero_excluded_from_nonzero_population"] = int(len(measured) - len(nonzero))
    fitted["n_validation_episodes"] = int(len(measured))
    fitted["visibility_metric"] = config["primary"]["visibility_metric"]
    fitted["population"] = "nonzero"
    fitted["uses_predictor"] = False
    fitted["selected_by_outcome"] = False
    fitted["fit_saw_predictor_metrics"] = False
    return fitted, rows


def _column(rows: list[dict[str, Any]], name: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(name)
        if value is None or value == "":
            values.append(np.nan)
        else:
            values.append(float(value))
    return values


def _ratio(rows: list[dict[str, Any]]) -> list[float | None]:
    values = []
    for row in rows:
        target = row.get("target_delta_l2")
        pred = row.get("pred_delta_l2")
        if target is None or pred is None or float(target) <= 1e-8:
            values.append(None)
        else:
            values.append(float(pred) / float(target))
    return values


def _cosine(rows: list[dict[str, Any]]) -> list[float | None]:
    return [float(row["cosine"]) if row.get("cosine_status") == "ok" and row.get("cosine") is not None else None for row in rows]


def _ranking(rows: list[dict[str, Any]]) -> list[float | None]:
    values = []
    for row in rows:
        if row.get("counterfactual_ranking_status") in {"scored", "distance_tie"} and row.get("counterfactual_ranking_top1") is not None:
            values.append(float(row["counterfactual_ranking_top1"]))
        else:
            values.append(None)
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in fields})


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiments" / "visibility_sensitivity_v1.yaml").read_text(encoding="utf-8"))
    reporting = get_split(config["reporting_split_version"])
    if reporting.headline_role != "legacy_validation":
        raise RuntimeError("historical checkpoint reporting is legacy validation, not a test set.")
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command="python3.13 scripts/run_visibility_sensitivity.py",
        seed="visibility_primary_not_fit_on_reporting",
        model={"source": "horizon_audit_v1 cached queries"},
        environment={"id": "mpm_legacy"},
        dataset_split={
            "fit_split_version": config["split_version"],
            "fit_role": config["fit_role"],
            "reporting_split_version": config["reporting_split_version"],
            "is_test_set": False,
        },
        checkpoint={"path": config["reporting_table"], "note": "no checkpoint is loaded for the threshold fit"},
    ) as run:
        primary, validation_rows = _fit_primary(config)
        _write_csv(run.run_dir / "validation_visibility.csv", validation_rows)
        (run.run_dir / "primary_threshold.json").write_text(json.dumps(to_jsonable(primary), indent=2) + "\n", encoding="utf-8")
        import pyarrow.parquet as pq

        reporting_rows = pq.read_table(ROOT / config["reporting_table"]).to_pylist()
        if any(row.get("is_test_set") for row in reporting_rows):
            raise RuntimeError("reporting rows include a test-set flag.")
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in reporting_rows:
            groups[(row["model"], row["protocol"], int(row["horizon"]), row["rollout_mode"])].append(row)
        reference_key = ("vicreg_action_s42", None, None, "open_loop")
        edges: dict[tuple, dict[str, dict[str, Any]]] = {}
        for (model, protocol, horizon, mode), group in groups.items():
            if model != reference_key[0] or mode != "open_loop":
                continue
            for metric in VISIBILITY_METRICS:
                edges[(protocol, horizon, metric)] = {
                    top_fraction_name(fraction): percentile_cutoff(_column(group, metric), fraction)
                    for fraction in config["top_fractions"]
                }
        slices = []
        free_rows = []
        bin_rows = []
        primary_threshold = float(primary["threshold"])
        for key, group in groups.items():
            model, protocol, horizon, mode = key
            base = {"model": model, "protocol": protocol, "horizon": horizon, "rollout_mode": mode}
            for metric in VISIBILITY_METRICS:
                visibility = _column(group, metric)
                for fraction in config["top_fractions"]:
                    cut = edges[(protocol, horizon, metric)][top_fraction_name(float(fraction))]
                    mask = select_visibility(visibility, threshold=cut["threshold"], comparison=cut["comparison"])
                    summary = summarize_selected(
                        group,
                        mask,
                        label=cut["name"],
                        visibility_metric=metric,
                        threshold=cut["threshold"],
                        comparison=cut["comparison"],
                        family="percentile",
                    )
                    summary.update(base)
                    summary["top_fraction"] = float(fraction)
                    slices.append(summary)
                for threshold in config["absolute_thresholds"][metric]:
                    mask = select_visibility(visibility, threshold=float(threshold), comparison=">=")
                    summary = summarize_selected(
                        group,
                        mask,
                        label=f"{metric}_ge_{threshold}",
                        visibility_metric=metric,
                        threshold=float(threshold),
                        comparison=">=",
                        family="absolute",
                    )
                    summary.update(base)
                    summary["top_fraction"] = None
                    slices.append(summary)
            raster = _column(group, "raster_mse")
            primary_mask = select_visibility(raster, threshold=primary_threshold, comparison=">=")
            primary_summary = summarize_selected(
                group,
                primary_mask,
                label="primary_validation_top_third",
                visibility_metric="raster_mse",
                threshold=primary_threshold,
                comparison=">=",
                family="primary",
            )
            primary_summary.update(base)
            primary_summary["top_fraction"] = float(config["primary"]["top_fraction"])
            slices.append(primary_summary)
            for metric in VISIBILITY_METRICS:
                for outcome_name, outcome in (
                    ("cosine", _cosine(group)),
                    ("response_ratio", _ratio(group)),
                    ("ranking_top1", _ranking(group)),
                ):
                    for nonzero_only in (False, True):
                        stat = spearman_visibility(_column(group, metric), outcome, nonzero_only=nonzero_only)
                        free_rows.append(
                            {
                                **base,
                                "statistic": "spearman",
                                "visibility_metric": metric,
                                "outcome": outcome_name,
                                **stat,
                                "integral": None,
                                "mean_over_fractions": None,
                            }
                        )
                curve_rows = [
                    item
                    for item in slices
                    if item["model"] == model
                    and item["protocol"] == protocol
                    and item["horizon"] == horizon
                    and item["rollout_mode"] == mode
                    and item["family"] == "percentile"
                    and item["visibility_metric"] == metric
                ]
                curve_rows.sort(key=lambda item: float(item["top_fraction"]))
                for outcome_name in ("cosine_median", "ranking_top1", "response_ratio_median"):
                    integrated = integrate_top_fraction_curve(
                        [float(item["top_fraction"]) for item in curve_rows],
                        [item[outcome_name] for item in curve_rows],
                    )
                    free_rows.append(
                        {
                            **base,
                            "statistic": "auc_top_fraction",
                            "visibility_metric": metric,
                            "outcome": outcome_name,
                            "spearman": None,
                            "n": integrated["n_points"],
                            "status": integrated["status"],
                            "n_total": len(group),
                            "n_excluded": len(integrated["missing_fractions"]),
                            "nonzero_only": True,
                            "integral": integrated["integral"],
                            "mean_over_fractions": integrated["mean_over_fractions"],
                        }
                    )
                labels, _bin_edges = equal_frequency_bins(_column(group, metric), n_bins=int(config["n_visibility_bins"]))
                for label in range(1, int(config["n_visibility_bins"]) + 1):
                    members = [row for row, assigned in zip(group, labels) if int(assigned) == label]
                    if not members:
                        continue
                    summary = summarize_selected(
                        members,
                        [True] * len(members),
                        label=f"bin_{label}",
                        visibility_metric=metric,
                        threshold=None,
                        comparison="bin",
                        family="bin",
                    )
                    summary.update(base)
                    summary["bin"] = label
                    summary["n_bin_total"] = len(group)
                    bin_rows.append(summary)
        _write_csv(run.run_dir / "sensitivity_slices.csv", slices)
        _write_csv(run.run_dir / "threshold_free.csv", free_rows)
        _write_csv(run.run_dir / "visibility_bins.csv", bin_rows)
        plot_visibility_sensitivity(
            slices,
            reporting_rows,
            ROOT / "figures" / "visibility_sensitivity",
            focus_protocol=config["focus_protocol"],
            focus_horizon=int(config["focus_horizon"]),
        )
        primary_rows = [row for row in slices if row["family"] == "primary" and row["protocol"] == config["focus_protocol"] and int(row["horizon"]) == int(config["focus_horizon"])]
        run.complete(
            to_jsonable(
                {
                    "primary_threshold": primary["threshold"],
                    "primary_quantile": primary["quantile"],
                    "primary_n_nonzero": primary["n_episodes"],
                    "primary_n_zero_excluded": primary["n_zero_excluded_from_nonzero_population"],
                    "fit_role": primary["role"],
                    "fit_split_version": primary["split_version"],
                    "saw_test": primary["saw_test"],
                    "selected_by_outcome": False,
                    "is_test_set": False,
                    "n_sensitivity_rows": len(slices),
                    "focus_primary": [
                        {
                            "model": row["model"],
                            "rollout_mode": row["rollout_mode"],
                            "n_selected": row["n_selected"],
                            "n_excluded": row["n_excluded"],
                            "ranking_top1": row["ranking_top1"],
                            "cosine_median": row["cosine_median"],
                            "n_cosine_defined": row["n_cosine_defined"],
                            "response_ratio_median": row["response_ratio_median"],
                            "state_action_ratio_status": row["state_action_ratio_status"],
                        }
                        for row in primary_rows
                    ],
                }
            )
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
