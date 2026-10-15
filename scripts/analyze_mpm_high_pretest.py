#!/usr/bin/env python3
"""Aggregate the pre-test counterfactual rows. Does not read the test split."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.counterfactual_study import (
    SUCCESS_COSINE,
    SUCCESS_RATIO,
    PHYS_TAU,
    bootstrap_median,
    classify_bottleneck,
    group_episode_rows,
    rank_correlation,
    visibility_threshold,
)
from src.registry import utc_now

ROWS = ROOT / "experiments" / "runs" / "mpm_high_multih" / "branch_rows.csv"
BOTTLENECK = ROOT / "experiments" / "runs" / "mpm_high_bottleneck"
FIGURES = ROOT / "experiments" / "runs" / "mpm_high_figures"
AGG_KEYS = (
    "cosine", "projection", "response_ratio", "pred_norm", "target_norm",
    "rms_pos", "rms_vel", "com_displacement", "raster_mse", "raster_l1", "changed_pixel_fraction",
)


def _load() -> list[dict]:
    rows = []
    with ROWS.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if int(raw["episode"]) >= 1620 and raw["regime"] == "mpm_high":
                raise RuntimeError("high-excitation test episode found in validation rows.")
            item = dict(raw)
            item["episode"] = int(raw["episode"])
            item["seed"] = int(raw["seed"])
            item["horizon"] = int(raw["horizon"])
            item["branch_time"] = int(raw["branch_time"])
            for key in (
                "rms_pos", "rms_vel", "com_displacement", "raster_mse", "raster_l1",
                "identical_raster", "changed_pixel_fraction", "pred_norm", "target_norm",
                "same_state_action_norm",
            ):
                item[key] = float(raw[key])
            for key in ("cosine", "response_ratio", "projection", "cosine_h", "cosine_z", "cosine_concat"):
                item[key] = None if raw[key] == "" else float(raw[key])
            rows.append(item)
    return rows


def _seed_table(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["model"], row["seed"], row["protocol"], row["horizon"])].append(row)
    table = []
    rng = np.random.default_rng(0)
    for key, group in sorted(groups.items()):
        episodes = group_episode_rows(group, AGG_KEYS)
        cosine = bootstrap_median([item["cosine"] for item in episodes], rng)
        projection = bootstrap_median([item["projection"] for item in episodes], rng)
        ratio = bootstrap_median([item["response_ratio"] for item in episodes], rng)
        physical = bootstrap_median([item["rms_pos"] for item in episodes], rng)
        raster = bootstrap_median([item["raster_mse"] for item in episodes], rng)
        target = bootstrap_median([item["target_norm"] for item in episodes], rng)
        pred = bootstrap_median([item["pred_norm"] for item in episodes], rng)
        table.append({
            "regime": key[0],
            "model": key[1],
            "seed": key[2],
            "protocol": key[3],
            "horizon": key[4],
            "n_episodes": cosine["n_episodes"],
            "cosine": cosine["estimate"],
            "cosine_ci_low": cosine["ci_low"],
            "cosine_ci_high": cosine["ci_high"],
            "projection": projection["estimate"],
            "projection_ci_low": projection["ci_low"],
            "projection_ci_high": projection["ci_high"],
            "response_ratio": ratio["estimate"],
            "response_ratio_ci_low": ratio["ci_low"],
            "response_ratio_ci_high": ratio["ci_high"],
            "rms_pos": physical["estimate"],
            "raster_mse": raster["estimate"],
            "target_norm": target["estimate"],
            "pred_norm": pred["estimate"],
            "cosine_status": cosine["status"],
        })
    return table


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bottleneck(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    labeled = []
    for row in rows:
        if row["regime"] != "mpm_high":
            continue
        item = dict(row)
        item["bottleneck"] = classify_bottleneck(row)
        labeled.append(item)
    counts = defaultdict(int)
    for row in labeled:
        counts[(row["model"], row["seed"], row["protocol"], row["horizon"], row["bottleneck"])] += 1
    summary = [
        {"model": key[0], "seed": key[1], "protocol": key[2], "horizon": key[3], "bottleneck": key[4], "n": value}
        for key, value in sorted(counts.items())
    ]
    return labeled, summary


def _visibility(rows: list[dict]) -> dict:
    reference = [
        row for row in rows
        if row["regime"] == "mpm_high" and row["model"] == "vicreg_action" and row["seed"] == 42
        and row["protocol"] == "open_loop" and row["horizon"] == 1
    ]
    threshold = visibility_threshold(np.asarray([row["raster_mse"] for row in reference], dtype=np.float64))
    cut = threshold["top_third_mse"]
    strata = {}
    for model in ("vicreg_action", "vicreg_noaction", "mse_action", "rssm"):
        for protocol in ("open_loop", "teacher_forced"):
            for horizon in (1, 5, 10):
                chosen = [
                    row for row in rows
                    if row["regime"] == "mpm_high" and row["model"] == model and row["protocol"] == protocol
                    and row["horizon"] == horizon
                ]
                masks = {
                    "all": chosen,
                    "nonidentical": [row for row in chosen if row["identical_raster"] == 0.0],
                    "top_third_raster": [row for row in chosen if cut is not None and row["raster_mse"] >= cut],
                }
                for name, subset in masks.items():
                    by_seed = defaultdict(list)
                    for row in subset:
                        by_seed[row["seed"]].append(row)
                    seed_cosines = []
                    for seed, group in sorted(by_seed.items()):
                        episodes = group_episode_rows(group, ("cosine", "raster_mse", "pred_norm", "target_norm"))
                        values = [item["cosine"] for item in episodes]
                        finite = [value for value in values if value is not None]
                        seed_cosines.append({
                            "seed": seed,
                            "cosine": None if not finite else float(np.median(finite)),
                            "n_episodes": len(episodes),
                        })
                    strata[f"{model}:{protocol}:H{horizon}:{name}"] = seed_cosines
    relations = []
    rng = np.random.default_rng(1)
    for model in ("vicreg_action", "mse_action", "rssm"):
        for seed in (42, 43, 44, 45, 46):
            group = [
                row for row in rows
                if row["regime"] == "mpm_high" and row["model"] == model and row["seed"] == seed
                and row["protocol"] == "open_loop" and row["horizon"] == 1
            ]
            episodes = group_episode_rows(group, ("raster_mse", "cosine", "pred_norm", "target_norm"))
            relations.append({
                "model": model,
                "seed": seed,
                "raster_vs_cosine": rank_correlation([item["raster_mse"] for item in episodes], [item["cosine"] for item in episodes]),
                "raster_vs_pred": rank_correlation([item["raster_mse"] for item in episodes], [item["pred_norm"] for item in episodes]),
                "raster_vs_target": rank_correlation([item["raster_mse"] for item in episodes], [item["target_norm"] for item in episodes]),
            })
    return {"threshold": threshold, "strata": strata, "relations": relations, "rng_note": "episode ranks, not branch-time ranks"}


def _headline(table: list[dict]) -> list[dict]:
    rows = []
    for regime in ("mpm_legacy", "mpm_high"):
        for horizon in (1, 5, 10):
            chosen = [
                row for row in table
                if row["regime"] == regime and row["model"] == "vicreg_action" and row["protocol"] == "open_loop"
                and row["horizon"] == horizon and row["seed"] in (42, 43, 44, 45)
            ]
            def median(key):
                values = [row[key] for row in chosen if row[key] is not None]
                return None if not values else float(np.median(values))
            rows.append({
                "regime": regime,
                "model": "vicreg_action",
                "seeds": "42-45",
                "protocol": "open_loop",
                "horizon": horizon,
                "physical_effect": median("rms_pos"),
                "observation_visibility": median("raster_mse"),
                "target_contrast": median("target_norm"),
                "predictor_contrast": median("pred_norm"),
                "cosine": median("cosine"),
                "response_ratio": median("response_ratio"),
                "projection": median("projection"),
                "n_seeds": len(chosen),
            })
    return rows


def _plot_horizon(table: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    colors = {"vicreg_action": "#1b4f72", "mse_action": "#b9770e", "rssm": "#196f3d", "vicreg_noaction": "#7b241c"}
    for ax, metric, title in (
        (axes[0], "cosine", "Cosine"),
        (axes[1], "projection", "Directional projection"),
    ):
        for model, color in colors.items():
            for seed in (42, 43, 44, 45, 46):
                xs, ys = [], []
                for horizon in (1, 5, 10):
                    match = [
                        row for row in table
                        if row["regime"] == "mpm_high" and row["model"] == model and row["seed"] == seed
                        and row["protocol"] == "open_loop" and row["horizon"] == horizon
                    ]
                    if match and match[0][metric] is not None:
                        xs.append(horizon)
                        ys.append(match[0][metric])
                if xs:
                    ax.plot(xs, ys, color=color, marker="o", linewidth=1, alpha=0.85, label=model if seed == 42 else None)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("Horizon")
        ax.set_title(title)
        ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("High excitation, open loop, one line per seed")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_horizon_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for protocol, style in (("open_loop", "-"), ("teacher_forced", "--")):
        for seed in (42, 43, 44, 45, 46):
            xs, ys = [], []
            for horizon in (1, 5, 10):
                match = [
                    row for row in table
                    if row["regime"] == "mpm_high" and row["model"] == "vicreg_action" and row["seed"] == seed
                    and row["protocol"] == protocol and row["horizon"] == horizon and row["cosine"] is not None
                ]
                if match:
                    xs.append(horizon)
                    ys.append(match[0]["cosine"])
            ax.plot(xs, ys, linestyle=style, marker="o", color="#1b4f72", alpha=0.8, label=protocol if seed == 42 else None)
    ax.set_xticks([1, 5, 10])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Episode-median cosine")
    ax.set_title("JEPA action, open loop vs teacher forced")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_open_vs_teacher.png", dpi=160)
    plt.close(fig)


def _plot_legacy(headline: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    metrics = ["physical_effect", "observation_visibility", "target_contrast", "predictor_contrast", "cosine", "response_ratio"]
    # Cosine and ratio are on different scales from norms. Plot H=1 cosine and ratio as a paired dot panel,
    # and the pathway norms in a second axis pair.
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, keys, title in (
        (axes[0], ["physical_effect", "observation_visibility"], "Physical and raster, H=1"),
        (axes[1], ["cosine", "response_ratio"], "Predictor alignment, H=1"),
    ):
        x = np.arange(len(keys))
        for offset, regime, color in ((-0.15, "mpm_legacy", "#7f8c8d"), (0.15, "mpm_high", "#1b4f72")):
            row = next(item for item in headline if item["regime"] == regime and item["horizon"] == 1)
            ax.bar(x + offset, [row[key] or 0 for key in keys], width=0.3, color=color, label=regime)
        ax.set_xticks(x, keys, rotation=15, ha="right")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("JEPA action seeds 42-45, open loop")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_legacy_vs_high.png", dpi=160)
    plt.close(fig)


def main() -> None:
    rows = _load()
    table = _seed_table(rows)
    labeled, summary = _bottleneck(rows)
    visibility = _visibility(rows)
    headline = _headline(table)
    multi = ROOT / "experiments" / "runs" / "mpm_high_multih"
    _write_csv(multi / "seed_summary.csv", table)
    _write_csv(multi / "legacy_vs_high.csv", headline)
    BOTTLENECK.mkdir(parents=True, exist_ok=True)
    branch_keep = [
        {key: row[key] for key in (
            "fork_id", "model", "seed", "protocol", "horizon", "episode", "branch_time",
            "rms_pos", "rms_vel", "com_displacement", "raster_mse", "raster_l1", "identical_raster",
            "changed_pixel_fraction", "pred_norm", "target_norm", "cosine", "cosine_status",
            "response_ratio", "projection", "bottleneck",
        )}
        for row in labeled
    ]
    _write_csv(BOTTLENECK / "branch_rows.csv", branch_keep)
    episodes = []
    for key, group in defaultdict(list, {}).items():
        pass
    by_episode = defaultdict(list)
    for row in labeled:
        if row["protocol"] == "open_loop":
            by_episode[(row["model"], row["seed"], row["horizon"], row["episode"])].append(row)
    for key, group in sorted(by_episode.items()):
        labels = [row["bottleneck"] for row in group]
        episodes.append({
            "model": key[0], "seed": key[1], "horizon": key[2], "episode": key[3],
            "majority_bottleneck": max(set(labels), key=labels.count),
            "n_branch": len(group),
        })
    _write_csv(BOTTLENECK / "episode_summary.csv", episodes)
    _write_csv(BOTTLENECK / "bottleneck_summary.csv", summary)
    (BOTTLENECK / "bottleneck_summary.json").write_text(json.dumps({
        "thresholds": {"phys_tau": PHYS_TAU, "success_cosine": SUCCESS_COSINE, "success_ratio": SUCCESS_RATIO},
        "summary": summary,
        "timestamp": utc_now(),
        "test_episodes_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    (BOTTLENECK / "report.md").write_text(_bottleneck_md(summary), encoding="utf-8")
    (multi / "visibility.json").write_text(json.dumps(visibility, indent=2) + "\n", encoding="utf-8")
    _plot_horizon(table)
    _plot_legacy(headline)
    _plot_visibility(visibility)
    print(json.dumps({"seed_rows": len(table), "bottleneck_rows": len(labeled), "headline": headline}, indent=2))


def _bottleneck_md(summary: list[dict]) -> str:
    lines = [
        "# Bottleneck labels",
        "",
        "Thresholds were declared from the validation scale before the horizon curves were interpreted.",
        f"Physical failure if position RMS <= {PHYS_TAU}. Observation bottleneck if the raster is identical.",
        f"Representation bottleneck if the target norm is <= 1e-8. Success if cosine >= {SUCCESS_COSINE} and response ratio >= {SUCCESS_RATIO}.",
        "Everything else with a visible target is a predictor bottleneck.",
        "",
    ]
    for row in summary:
        if row["protocol"] == "open_loop" and row["horizon"] in (1, 10):
            lines.append(
                f"- {row['model']} seed {row['seed']} H{row['horizon']} {row['bottleneck']}: {row['n']}"
            )
    lines.append("")
    return "\n".join(lines)


def _plot_visibility(visibility: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    strata = ("all", "nonidentical", "top_third_raster")
    x = np.arange(len(strata))
    for offset, model, color in (
        (-0.2, "vicreg_action", "#1b4f72"),
        (0.0, "mse_action", "#b9770e"),
        (0.2, "rssm", "#196f3d"),
    ):
        ys = []
        for name in strata:
            seed_rows = visibility["strata"][f"{model}:open_loop:H1:{name}"]
            values = [row["cosine"] for row in seed_rows if row["cosine"] is not None]
            ys.append(float(np.median(values)) if values else np.nan)
            ax.scatter(np.full(len(values), list(strata).index(name)) + offset, values, color=color, s=18, zorder=3)
        ax.plot(x + offset, ys, color=color, label=model)
    ax.set_xticks(x, strata)
    ax.set_ylabel("Seed cosine")
    ax.set_title("Visibility strata, open loop, H=1")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_visibility.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
