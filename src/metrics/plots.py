"""Draw one panel per cascade layer. Mixed units are not placed on one axis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _values(pairs: Sequence[dict[str, Any]], reader) -> list[float | None]:
    return [reader(pair) for pair in pairs]


def plot_cascade_pairs(pairs: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Write a six-panel figure. Stand-in controls are labeled as stand-ins."""
    if not pairs:
        raise ValueError("pairs is empty.")
    labels = [str(pair["intervention"]) for pair in pairs]
    x = range(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), constrained_layout=True)

    def bars(ax, values: list[float | None], title: str, ylabel: str) -> None:
        numeric = [0.0 if value is None else value for value in values]
        colors = ["#bdbdbd" if value is None else "#2c5f8a" for value in values]
        ax.bar(list(x), numeric, color=colors)
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        if any(value is None for value in values):
            ax.text(0.02, 0.95, "grey = undefined", transform=ax.transAxes, va="top", fontsize=8)

    bars(
        axes[0, 0],
        _values(pairs, lambda pair: pair["cascade"]["layers"]["physical"]["rms_pos"]),
        "Physical response",
        "RMS position",
    )
    bars(
        axes[0, 1],
        _values(pairs, lambda pair: pair["cascade"]["layers"]["observation"]["fraction_identical"]),
        "Observation visibility",
        "Fraction identical uint8",
    )
    axes[0, 2].axis("off")
    axes[0, 2].set_title("Learned model layers", fontsize=10)
    learned = pairs[0]["cascade"]["layers"]
    text = "\n\n".join(
        f"{name}: {learned[name]['status']}\n{learned[name].get('reason', '')}"
        for name in ("representation", "predictor", "decision")
    )
    axes[0, 2].text(0.0, 0.95, text, va="top", fontsize=8, wrap=True)

    bars(
        axes[1, 0],
        _values(pairs, lambda pair: pair["controls"]["privileged_particle_coordinates"]["contrast"]["mse"]),
        "Privileged state stand-in",
        "Coordinate MSE",
    )
    bars(
        axes[1, 1],
        _values(pairs, lambda pair: pair["controls"]["constant_predictor_standin"]["contrast"]["mse"]),
        "Constant predictor stand-in",
        "Latent MSE",
    )
    bars(
        axes[1, 2],
        _values(
            pairs,
            lambda pair: pair["controls"]["oracle_privileged_standin"]["alignment"]["cosine"],
        ),
        "Oracle stand-in alignment",
        "Cosine",
    )
    fig.suptitle("Cascade layers are not combined into one score", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
