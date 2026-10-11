"""Plots of counterfactual future ranking. They read the per-query table."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def _mean_top1(rows: Sequence[dict[str, Any]], key: str) -> tuple[float | None, int, int]:
    scored = _finite(rows, key)
    ambiguous = sum(1 for row in rows if row.get(key.replace("_top1", "_status")) == "observationally_ambiguous")
    if scored.size == 0:
        return None, 0, ambiguous
    return float(scored.mean()), int(scored.size), ambiguous


def plot_future_ranking(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Draw retrieval summaries. Empty scored sets are skipped, not drawn as zero."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(row["model"] for row in rows))
    horizons = sorted({int(row["horizon"]) for row in rows})
    legacy = [row for row in rows if row["environment"] == "mpm_legacy"]
    _lines_by_horizon(legacy, models, horizons, "model_mse_top1", out_dir / "top1_by_horizon.png", "MSE top-1, model predictions, legacy environment")
    _control_bars(legacy, models, horizons, out_dir / "top1_controls.png")
    _binned(legacy, models, "visibility_bin", "model_mse_top1", out_dir / "top1_by_visibility.png", "MSE top-1 by observation visibility")
    _binned(legacy, models, "physical_bin", "model_mse_top1", out_dir / "top1_by_displacement.png", "MSE top-1 by physical displacement")
    _binned(legacy, models, "action_l2_quintile", "model_mse_top1", out_dir / "top1_by_action_distance.png", "MSE top-1 by action-distance quintile")
    _environment_bars(rows, models, out_dir / "top1_by_environment.png")
    _margin_boxes(legacy, models, out_dir / "margin_by_model.png")


def _lines_by_horizon(rows, models, horizons, key, path, title) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    drew = False
    for color, model in zip(colors, models):
        xs = []
        ys = []
        for horizon in horizons:
            taken = [row for row in rows if row["model"] == model and int(row["horizon"]) == horizon]
            mean, n_scored, _ambiguous = _mean_top1(taken, key)
            if mean is None:
                continue
            xs.append(horizon)
            ys.append(mean)
            drew = True
        if xs:
            ax.plot(xs, ys, marker="o", color=color, label=f"{model} (n={_finite([r for r in rows if r['model']==model], key).size})")
    if not drew:
        plt.close(fig)
        return
    ax.axhline(0.25, color="#888888", lw=1, label="chance 1/4")
    ax.set_xlabel("Horizon (held action)")
    ax.set_ylabel("Top-1 on scored queries")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _control_bars(rows, models, horizons, path) -> None:
    modes = (
        ("model_mse_top1", "model"),
        ("shuffled_predictions_mse_top1", "shuffled"),
        ("oracle_targets_mse_top1", "oracle"),
        ("constant_prediction_mse_top1", "constant"),
    )
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4.6), sharey=True, constrained_layout=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        subset = [row for row in rows if int(row["horizon"]) == horizon]
        x = np.arange(len(models))
        width = 0.18
        for index, (key, label) in enumerate(modes):
            heights = []
            for model in models:
                mean, _n, _a = _mean_top1([row for row in subset if row["model"] == model], key)
                heights.append(np.nan if mean is None else mean)
            ax.bar(x + (index - 1.5) * width, heights, width=width, label=label)
        ax.axhline(0.25, color="#888888", lw=1)
        ax.set_xticks(x, [model.replace("_", "\n") for model in models], fontsize=6)
        ax.set_title(f"Horizon {horizon}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("MSE top-1")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _binned(rows, models, bin_key, value_key, path, title) -> None:
    labels = list(dict.fromkeys(str(row[bin_key]) for row in rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    x = np.arange(len(labels))
    drew = False
    for color, model in zip(colors, models):
        ys = []
        for label in labels:
            taken = [row for row in rows if row["model"] == model and str(row[bin_key]) == label]
            mean, n_scored, _ambiguous = _mean_top1(taken, value_key)
            ys.append(np.nan if mean is None else mean)
            if n_scored:
                drew = True
        ax.plot(x, ys, marker="o", color=color, label=model)
    if not drew:
        plt.close(fig)
        return
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Top-1 on scored queries")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.25, color="#888888", lw=1)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _environment_bars(rows, models, path) -> None:
    environments = list(dict.fromkeys(row["environment"] for row in rows))
    fig, ax = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
    x = np.arange(len(models))
    width = 0.8 / max(len(environments), 1)
    for index, environment in enumerate(environments):
        heights = []
        for model in models:
            taken = [
                row
                for row in rows
                if row["model"] == model and row["environment"] == environment and int(row["horizon"]) == 1
            ]
            mean, _n, _a = _mean_top1(taken, "model_mse_top1")
            heights.append(np.nan if mean is None else mean)
        ax.bar(x + (index - (len(environments) - 1) / 2) * width, heights, width=width, label=environment)
    ax.axhline(0.25, color="#888888", lw=1)
    ax.set_xticks(x, [model.replace("_", "\n") for model in models], fontsize=7)
    ax.set_ylabel("Horizon-1 MSE top-1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Environment comparison, scored queries")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _margin_boxes(rows, models, path) -> None:
    series = []
    labels = []
    for model in models:
        taken = _finite(
            [row for row in rows if row["model"] == model and int(row["horizon"]) == 1 and row.get("model_mse_status") in {"scored", "distance_tie"}],
            "model_mse_margin",
        )
        if taken.size:
            series.append(taken)
            labels.append(model.replace("_", "\n") + f"\nn={taken.size}")
    if not series:
        return
    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    ax.boxplot(series, tick_labels=labels, showfliers=False)
    ax.axhline(0.0, color="#888888", lw=1)
    ax.set_ylabel("MSE margin at horizon 1")
    ax.set_title("Nearest incorrect minus correct. Positive means the true future is closer.")
    fig.savefig(path, dpi=140)
    plt.close(fig)
