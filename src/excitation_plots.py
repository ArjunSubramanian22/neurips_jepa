"""Plots of physical and raster response against the predeclared excitation index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_excitation(
    aggregates: Sequence[dict[str, Any]],
    out_path: Path,
    selected_ids: dict[str, str | None],
) -> None:
    """One panel per measurement. Selected regimes are marked, not used as the axis."""
    if not aggregates:
        raise ValueError("aggregates is empty.")
    xs = [float(row["excitation_index"]) for row in aggregates]
    panels = (
        ("median_pixel_rms", "Pixel RMS displacement"),
        ("median_raster_change", "Raster change fraction"),
        ("median_rms_pos", "RMS position"),
        ("median_fraction_identical", "Exact uint8 equality"),
        ("median_density_change_rate", "Density-channel change"),
        ("median_velocity_change_rate", "Velocity-channel change"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), constrained_layout=True)
    colors = {"baseline": "#b35900", "candidate": "#2c5f8a"}
    for ax, (key, title) in zip(axes.ravel(), panels, strict=True):
        for row, x in zip(aggregates, xs, strict=True):
            ax.scatter(
                x,
                float(row[key]),
                s=28,
                c=colors.get(str(row["role"]), "#2c5f8a"),
                alpha=0.85,
            )
        for name, env_id in selected_ids.items():
            match = [row for row in aggregates if row["environment_id"] == env_id]
            if not match:
                continue
            ax.scatter(
                [float(match[0]["excitation_index"])],
                [float(match[0][key])],
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Excitation index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Excitation sweep, force-reversed fork medians", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
