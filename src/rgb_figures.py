"""Counterfactual difference maps for paired rendered futures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.rgb_metrics import difference_maps


def _hwc(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    return np.clip(np.transpose(array, (1, 2, 0)), 0.0, 1.0)


def save_counterfactual_figure(
    recorded: np.ndarray,
    alternate: np.ndarray,
    out_path: Path,
    title: str,
    amplify: float = 8.0,
) -> None:
    """Recorded future, alternative future, absolute difference, amplified difference."""
    absolute, amplified = difference_maps(recorded, alternate, amplify=amplify)
    panels = (
        (_hwc(recorded), "Recorded future"),
        (_hwc(alternate), "Alternative-action future"),
        (_hwc(absolute), "Absolute difference"),
        (_hwc(amplified), f"Difference x{amplify:g}"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.2), constrained_layout=True)
    for axis, (image, label) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_title(label, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
