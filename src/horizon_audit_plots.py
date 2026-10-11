"""Horizon curves for the counterfactual audit. Protocols are not pooled."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metrics.horizon_audit import PROTOCOLS, ROLLOUT_MODES


def _rows(rows: Sequence[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    chosen = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            chosen.append(row)
    return chosen


def _aggregate(rows: Sequence[dict[str, Any]], key: str, how: str) -> tuple[float | None, int]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    if not values:
        return None, 0
    array = np.asarray(values, dtype=np.float64)
    summary = float(array.mean()) if how == "mean" else float(np.median(array))
    return summary, int(array.size)


def plot_horizon_audit(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Draw horizon curves. A missing point has no defined values and is not drawn as zero."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_model = next(iter(dict.fromkeys(row["model"] for row in rows)))
    physical = _rows(rows, rollout_mode="open_loop", model=reference_model)
    _protocol_curves(physical, "rms_pos", "RMS position", out_dir / "physical_vs_horizon.png", "Physical separation")
    _protocol_curves(
        physical,
        "fraction_affected_pixels",
        "Affected-pixel fraction",
        out_dir / "visibility_vs_horizon.png",
        "Physical-raster visibility",
    )
    _protocol_curves(
        physical,
        "rgb_fraction_affected_pixels",
        "RGB affected-pixel fraction",
        out_dir / "rgb_visibility_vs_horizon.png",
        "RGB visibility",
    )
    _model_grid(rows, "cosine", "Median cosine", out_dir / "direction_vs_horizon.png", "Defined cosines only", how="median", ylim=(-1.05, 1.05))
    _model_grid(
        rows,
        "counterfactual_ranking_top1",
        "Top-1",
        out_dir / "ranking_vs_horizon.png",
        "Mean counterfactual retrieval on scored queries. Grey line is chance 1/2.",
        how="mean",
        chance=0.5,
        ylim=(-0.05, 1.05),
    )
    _rollout_curves(rows, out_dir / "rollout_vs_horizon.png")
    _model_bars(rows, out_dir / "model_comparison.png")


def _protocol_curves(rows, key, ylabel, path, title) -> None:
    horizons = sorted({int(row["horizon"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    drew = False
    for protocol in PROTOCOLS:
        xs, ys = [], []
        for horizon in horizons:
            median, count = _aggregate(_rows(rows, protocol=protocol, horizon=horizon), key, "median")
            if median is None:
                continue
            xs.append(horizon)
            ys.append(median)
            drew = True
        if xs:
            ax.plot(xs, ys, marker="o", label=protocol)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _model_grid(rows, key, ylabel, path, title, *, how: str, chance: float | None = None, ylim: tuple[float, float] | None = None) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    horizons = sorted({int(row["horizon"]) for row in rows})
    fig, axes = plt.subplots(len(ROLLOUT_MODES), len(PROTOCOLS), figsize=(14.5, 6.4), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    drew = False
    for row_index, mode in enumerate(ROLLOUT_MODES):
        for col_index, protocol in enumerate(PROTOCOLS):
            ax = axes[row_index, col_index]
            for color, model in zip(colors, models):
                xs, ys = [], []
                for horizon in horizons:
                    median, count = _aggregate(
                        _rows(rows, model=model, protocol=protocol, rollout_mode=mode, horizon=horizon),
                        key,
                        how,
                    )
                    if count == 0 or median is None:
                        continue
                    xs.append(horizon)
                    ys.append(median)
                    drew = True
                if xs:
                    ax.plot(xs, ys, marker="o", color=color, label=model)
            ax.set_title(f"{mode}\n{protocol}", fontsize=8)
            if chance is not None:
                ax.axhline(chance, color="#888888", lw=1)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if row_index == len(ROLLOUT_MODES) - 1:
                ax.set_xlabel("Horizon")
            if col_index == 0:
                ax.set_ylabel(ylabel)
    if not drew:
        plt.close(fig)
        return
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _rollout_curves(rows, path) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    horizons = sorted({int(row["horizon"]) for row in rows})
    recorded = _rows(rows, protocol="one_step_replacement")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    drew = False
    for ax, mode in zip(axes, ROLLOUT_MODES):
        for color, model in zip(colors, models):
            xs, ys = [], []
            for horizon in horizons:
                median, count = _aggregate(_rows(recorded, model=model, rollout_mode=mode, horizon=horizon), "rollout_mse_recorded", "median")
                if median is None:
                    continue
                xs.append(horizon)
                ys.append(median)
                drew = True
            if xs:
                ax.plot(xs, ys, marker="o", color=color, label=model)
        ax.set_title(mode)
        ax.set_xlabel("Horizon")
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel("Recorded-action rollout MSE")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Ordinary rollout error on the recorded branch")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _model_bars(rows, path) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    chosen = _rows(rows, rollout_mode="open_loop", horizon=10)
    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    x = np.arange(len(models))
    width = 0.18
    for index, protocol in enumerate(PROTOCOLS):
        heights = []
        for model in models:
            median, count = _aggregate(_rows(chosen, model=model, protocol=protocol), "counterfactual_ranking_top1", "mean")
            heights.append(np.nan if median is None or count == 0 else median)
        ax.bar(x + (index - 1.5) * width, heights, width=width, label=protocol)
    ax.axhline(0.5, color="#888888", lw=1)
    ax.set_xticks(x, [model.replace("_", "\n") for model in models], fontsize=7)
    ax.set_ylabel("Horizon-10 open-loop top-1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean counterfactual retrieval. Grey line is chance 1/2.")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)
