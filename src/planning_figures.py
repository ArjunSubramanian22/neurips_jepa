"""Rollout strips and distance curves for closed-loop planning."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def as_time_channel_frames(frames: np.ndarray) -> np.ndarray:
    """Return uint8 frames shaped ``(T, C, H, W)``."""
    array = np.asarray(frames)
    if array.ndim != 4:
        raise ValueError(f"frames must be 4D, got {array.shape}.")
    if array.shape[-1] in (1, 3) and array.shape[1] not in (1, 3):
        array = np.transpose(array, (0, 3, 1, 2))
    if array.dtype != np.uint8:
        scaled = array.astype(np.float64)
        if scaled.max(initial=0.0) <= 1.5:
            scaled = scaled * 255.0
        array = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def write_rollout_gif(named_frames: Mapping[str, np.ndarray], path: Path, *, duration_ms: int = 400) -> None:
    """Write one GIF. Each frame is a labeled row of controllers at that time."""
    from PIL import Image, ImageDraw

    names = list(named_frames)
    if not names:
        raise ValueError("named_frames is empty.")
    clips = [as_time_channel_frames(named_frames[name]) for name in names]
    height = clips[0].shape[-2]
    width = clips[0].shape[-1]
    steps = max(clip.shape[0] for clip in clips)
    label_height = 18
    canvas_w = width * 2
    panels: list[Image.Image] = []
    for step in range(steps):
        tiles = []
        for name, clip in zip(names, clips):
            frame = clip[min(step, clip.shape[0] - 1)]
            image = Image.fromarray(np.transpose(frame, (1, 2, 0)), mode="RGB")
            image = image.resize((width * 2, height * 2), Image.Resampling.NEAREST)
            tile = Image.new("RGB", (width * 2, height * 2 + label_height), "white")
            tile.paste(image, (0, label_height))
            draw = ImageDraw.Draw(tile)
            draw.text((1, 1), name[:18], fill=(0, 0, 0))
            tiles.append(tile)
        panel = Image.new("RGB", (canvas_w * len(tiles), height * 2 + label_height), "white")
        for index, tile in enumerate(tiles):
            panel.paste(tile, (index * canvas_w, 0))
        panels.append(panel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels[0].save(path, save_all=True, append_images=panels[1:], duration=int(duration_ms), loop=0)


def plot_distance_curves(
    named_distances: Mapping[str, Sequence[float]],
    path: Path,
    *,
    title: str,
    ylabel: str = "task distance",
) -> None:
    """Plot closed-loop task distance after each executed action."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not named_distances:
        raise ValueError("named_distances is empty.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    for name, values in named_distances.items():
        series = np.asarray(values, dtype=np.float64)
        axis.plot(np.arange(1, series.size + 1), series, marker="o", label=name)
    axis.set_xlabel("environment step")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(1, 1 + max(len(values) for values in named_distances.values())))
    axis.set_title(title, fontsize=9)
    axis.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_task_distances(
    table: Mapping[str, Mapping[str, float]],
    path: Path,
    *,
    title: str,
) -> None:
    """One panel per task so position and angle are not drawn on one axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(table)
    if not tasks:
        raise ValueError("table is empty.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(tasks), figsize=(3.1 * len(tasks), 3.8), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        pairs = list(table[task].items())
        axis.bar(np.arange(len(pairs)), [value for _, value in pairs], color="#4c78a8")
        axis.set_xticks(np.arange(len(pairs)), [name for name, _ in pairs], rotation=30, ha="right")
        axis.set_title(task, fontsize=9)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0][0].set_ylabel("mean final distance")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
