"""Plots of directional counterfactual metrics. They read the per-transition table."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _valid_cosine(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    values = [
        float(row["cosine"])
        for row in rows
        if row["cosine_status"] == "ok" and row["contrast_status"] == "valid_directional_contrast" and row["cosine"] != ""
    ]
    return np.asarray(values, dtype=np.float64)


def plot_directional(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Write the directional summary figures. Empty valid sets are skipped, not filled."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [
        row
        for row in rows
        if row["cosine_status"] == "ok" and row["contrast_status"] == "valid_directional_contrast" and row["cosine"] != ""
    ]
    if valid:
        cosines = np.asarray([float(row["cosine"]) for row in valid], dtype=np.float64)
        n_excluded = len(rows) - len(valid)
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        ax.hist(cosines, bins=40, range=(-1.0, 1.0), color="#2c5f8a")
        ax.set_xlabel("Counterfactual cosine")
        ax.set_ylabel("Valid transitions")
        ax.set_title(f"Valid directional contrasts (n={cosines.size}; excluded {n_excluded})")
        fig.savefig(out_dir / "cosine_histogram.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        ax.scatter(
            [float(row["fraction_affected_pixels"]) for row in valid],
            cosines,
            s=8,
            alpha=0.35,
            c="#2c5f8a",
        )
        ax.set_xlabel("Affected-pixel fraction")
        ax.set_ylabel("Cosine")
        ax.set_title("Cosine vs physical-raster visibility")
        fig.savefig(out_dir / "cosine_vs_visibility.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        ax.scatter(
            [float(row["action_l2"]) for row in valid],
            cosines,
            s=8,
            alpha=0.35,
            c="#2c5f8a",
        )
        ax.set_xlabel("Raw action L2")
        ax.set_ylabel("Cosine")
        ax.set_title("Cosine vs action distance")
        fig.savefig(out_dir / "cosine_vs_action_distance.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        ax.scatter(
            cosines,
            [float(row["magnitude_ratio"]) for row in valid],
            s=8,
            alpha=0.35,
            c="#2c5f8a",
        )
        ax.set_xlabel("Cosine")
        ax.set_ylabel("Magnitude ratio")
        ax.set_yscale("log")
        ax.set_title("Direction and magnitude (valid cosine only)")
        fig.savefig(out_dir / "magnitude_vs_direction.png", dpi=140)
        plt.close(fig)

    models = list(dict.fromkeys(row["model"] for row in rows))
    series = []
    positions = []
    for index, model in enumerate(models, start=1):
        taken = _valid_cosine([row for row in rows if row["model"] == model])
        if taken.size:
            series.append(taken)
            positions.append(index)
    if models:
        fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
        if series:
            ax.boxplot(series, positions=positions, showfliers=False)
        labels = []
        for index, model in enumerate(models, start=1):
            taken = _valid_cosine([row for row in rows if row["model"] == model])
            labels.append(model.replace("_", "\n") + f"\nn={taken.size}")
            if taken.size == 0:
                ax.text(index, 0.0, "no valid\ncosine", ha="center", va="center", fontsize=7, color="#8a2c2c")
        ax.set_xticks(range(1, len(models) + 1), labels)
        ax.set_ylabel("Valid cosine")
        ax.set_title("Per-model valid cosine. Models with none are marked, not imputed.")
        fig.savefig(out_dir / "cosine_by_model.png", dpi=140)
        plt.close(fig)
