"""Figures for the material-identifiability study. They read saved probe rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_protocol(rows: Sequence[dict[str, Any]], out_path: Path, history_length: int) -> None:
    """Validation linear stiffness accuracy for each protocol and feature family."""
    families = (
        "privileged",
        "mechanical_state",
        "mechanical_history",
        "float_raster",
        "uint8_raster",
        "float_raster_history",
        "uint8_raster_history",
    )
    protocols = list(dict.fromkeys(row["protocol_id"] for row in rows))
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    width = 0.11
    x = np.arange(len(protocols))
    for index, family in enumerate(families):
        wanted = 1 if "history" not in family else history_length
        values = []
        for protocol in protocols:
            match = [
                row
                for row in rows
                if row["protocol_id"] == protocol
                and row["family"] == family
                and row["task"] == "stiffness"
                and row["probe"] == "linear"
                and row["split"] == "val"
                and int(row["history_length"]) == wanted
            ]
            values.append(float(match[0]["accuracy"]) if match else np.nan)
        ax.bar(x + (index - 3) * width, values, width=width, label=family)
    ax.axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xticks(x, protocols, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Stiffness probes on validation, linear readout")
    ax.legend(fontsize=7, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_history_overview(rows: Sequence[dict[str, Any]], out_path: Path, family: str) -> None:
    """Validation stiffness accuracy against history length, one line per protocol."""
    protocols = list(dict.fromkeys(row["protocol_id"] for row in rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for protocol in protocols:
        subset = [
            row
            for row in rows
            if row["protocol_id"] == protocol
            and row["family"] == family
            and row["task"] == "stiffness"
            and row["probe"] == "linear"
            and row["split"] == "val"
        ]
        subset.sort(key=lambda row: int(row["history_length"]))
        if not subset:
            continue
        ax.plot(
            [int(row["history_length"]) for row in subset],
            [float(row["accuracy"]) for row in subset],
            marker="o",
            label=protocol,
        )
    ax.axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("History length (frames)")
    ax.set_ylabel("Validation linear accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(family)
    ax.legend(fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_history_curve(rows: Sequence[dict[str, Any]], protocol_id: str, out_path: Path) -> None:
    """Validation and confirmatory test accuracy against history length."""
    families = ("mechanical_history", "uint8_raster_history", "float_raster_history")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True, sharey=True)
    for ax, split, title in (
        (axes[0], "val", "Validation, used for selection"),
        (axes[1], "test", "Test, confirmatory"),
    ):
        for family in families:
            subset = [
                row
                for row in rows
                if row["protocol_id"] == protocol_id
                and row["family"] == family
                and row["task"] == "stiffness"
                and row["probe"] == "linear"
                and row["split"] == split
            ]
            subset.sort(key=lambda row: int(row["history_length"]))
            ax.plot(
                [int(row["history_length"]) for row in subset],
                [float(row["accuracy"]) for row in subset],
                marker="o",
                label=family,
            )
        ax.axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("History length (frames)")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Linear stiffness accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle(protocol_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], out_path: Path, title: str) -> None:
    array = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    image = ax.imshow(array, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            ax.text(col, row, str(int(array[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
