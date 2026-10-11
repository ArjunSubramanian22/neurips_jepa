"""Plots for visibility sensitivity. They read the saved tables and raw rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metrics.visibility_sensitivity import TARGET_TAU, equal_frequency_bins


def _selected(rows: Sequence[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    chosen = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            chosen.append(row)
    return chosen


def plot_visibility_sensitivity(
    slices: Sequence[dict[str, Any]],
    raw_rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    focus_protocol: str,
    focus_horizon: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus = _selected(list(slices), family="percentile", visibility_metric="raster_mse", protocol=focus_protocol, horizon=int(focus_horizon))
    _threshold_lines(focus, "ranking_top1", "Top-1", out_dir / "sensitivity_ranking.png", "Mean retrieval across visibility cuts", 0.5)
    _threshold_lines(focus, "cosine_median", "Median cosine", out_dir / "sensitivity_cosine.png", "Defined cosine across visibility cuts", None)
    _threshold_lines(focus, "rms_pos_median", "Median RMS position", out_dir / "sensitivity_physical.png", "Physical displacement across visibility cuts", None)
    raw = _selected(list(raw_rows), protocol=focus_protocol, horizon=int(focus_horizon))
    _scatter(raw, "response_ratio", "Predictor/target ratio", out_dir / "visibility_vs_response_ratio.png")
    _scatter(raw, "cosine", "Cosine", out_dir / "visibility_vs_cosine.png")


def _threshold_lines(rows, key, ylabel, path, title, chance: float | None) -> None:
    modes = list(dict.fromkeys(row["rollout_mode"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    if not modes or not models:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.4), sharey=True, constrained_layout=True)
    if len(modes) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    drew = False
    for ax, mode in zip(axes, modes):
        for color, model in zip(colors, models):
            taken = [row for row in rows if row["rollout_mode"] == mode and row["model"] == model and row.get(key) is not None]
            taken.sort(key=lambda row: float(row["top_fraction"]))
            if not taken:
                continue
            ax.plot(
                [float(row["top_fraction"]) for row in taken],
                [float(row[key]) for row in taken],
                marker="o",
                color=color,
                label=model,
            )
            drew = True
        if chance is not None:
            ax.axhline(chance, color="#888888", lw=1)
        ax.set_xlabel("Top fraction of nonzero raster MSE")
        ax.set_title(mode)
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _scatter(rows, outcome: str, ylabel: str, path: Path) -> None:
    modes = ("open_loop", "teacher_forced")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    drew = False
    for ax, mode in zip(axes, modes):
        subset = [row for row in rows if row["rollout_mode"] == mode]
        xs, ys = [], []
        undefined = 0
        for row in subset:
            visibility = row.get("raster_mse")
            value = _outcome(row, outcome)
            if visibility is None or not np.isfinite(float(visibility)):
                undefined += 1
                continue
            if value is None:
                undefined += 1
                continue
            xs.append(float(visibility))
            ys.append(float(value))
        if xs:
            ax.scatter(xs, ys, s=10, alpha=0.35, c="#2c5f8a")
            labels, edges = equal_frequency_bins(xs, n_bins=8)
            centers, medians = [], []
            array_x = np.asarray(xs)
            array_y = np.asarray(ys)
            for label in range(1, 9):
                taken = array_y[labels == label]
                if taken.size == 0:
                    continue
                centers.append(float(np.median(array_x[labels == label])))
                medians.append(float(np.median(taken)))
            if centers:
                ax.plot(centers, medians, color="#8a2c2c", marker="o", label="bin median")
            drew = True
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_title(f"{mode} (undefined {undefined})")
        ax.set_xlabel("Raster MSE")
        ax.legend(fontsize=7)
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel(ylabel)
    fig.suptitle("Raw pairs with equal-frequency bin medians")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _outcome(row: dict[str, Any], name: str) -> float | None:
    if name == "cosine":
        if row.get("cosine_status") != "ok" or row.get("cosine") is None:
            return None
        return float(row["cosine"])
    target = row.get("target_delta_l2")
    pred = row.get("pred_delta_l2")
    if target is None or pred is None or float(target) <= TARGET_TAU:
        return None
    return float(pred) / float(target)
