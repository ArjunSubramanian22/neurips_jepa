"""Publication panels drawn from arrays, not from hand-edited images.

Typography is shared. Family colors match the metric-planning figure.
Difference images use ``difference_maps`` rather than a new color scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.rgb_metrics import difference_maps

# Family colors are the ones in analysis/metric_planning_relationship.py.
# Oracle uses the completed-seed blue from the seed-uncertainty plots.
FAMILY_COLORS = {
    "vicreg_action": "#4c78a8",
    "vicreg_noaction": "#f58518",
    "mse_action": "#54a24b",
    "oracle": "#2c5f8a",
}
PANEL_INCH = 1.15
AMPLIFY = 8.0
PANEL_FIELDS = ("environment", "episode", "time", "seed", "checkpoint", "intervention")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_meta(**fields: Any) -> dict[str, Any]:
    """Every panel records the same identity fields. Missing facts are null."""
    record = {key: fields.get(key) for key in PANEL_FIELDS}
    record.update({key: value for key, value in fields.items() if key not in record})
    return record


def save_figure(figure: plt.Figure, stem: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Write PDF, SVG, PNG, and the panel metadata together."""
    from src.registry import to_jsonable

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(figure)
    payload = to_jsonable(metadata)
    payload.setdefault(
        "regeneration_command",
        "MPLCONFIGDIR=outputs/.mplconfig python3.13 scripts/render_qualitative_figures.py",
    )
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "png": str(stem.with_suffix(".png")),
        "metadata": str(stem.with_suffix(".json")),
    }


def _hwc(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    return np.clip(array, 0.0, 1.0)


def _blank(axis, text: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=7)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def _show(axis, image: np.ndarray, title: str) -> None:
    axis.imshow(_hwc(image), interpolation="nearest")
    axis.set_title(title, fontsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


def _pair(recorded: np.ndarray, alternate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute, amplified = difference_maps(recorded, alternate, amplify=AMPLIFY)
    absolute_u8 = np.clip(np.rint(absolute * 255.0), 0, 255).astype(np.uint8)
    amplified_u8 = np.clip(np.rint(amplified * 255.0), 0, 255).astype(np.uint8)
    return absolute_u8, amplified_u8


def fork_sequence_figure(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> plt.Figure:
    """Columns are current, recorded future, alternative, absolute, amplified."""
    apply_style()
    figure, axes = plt.subplots(
        len(rows),
        5,
        figsize=(PANEL_INCH * 5, PANEL_INCH * len(rows) + 0.45),
        squeeze=False,
        constrained_layout=True,
    )
    headers = ("Current", "Recorded future", "Alternative", "Absolute difference", f"Difference x{AMPLIFY:g}")
    for row_index, (label, current, recorded, alternative) in enumerate(rows):
        absolute, amplified = _pair(recorded, alternative)
        images = (current, recorded, alternative, absolute, amplified)
        for column, (image, header) in enumerate(zip(images, headers, strict=True)):
            _show(axes[row_index, column], image, header if row_index == 0 else "")
        axes[row_index, 0].set_ylabel(label, fontsize=8)
    figure.suptitle("Counterfactual fork")
    return figure


def bottleneck_figure(
    observation_difference: np.ndarray,
    scalars: Mapping[str, float | None],
    particles: np.ndarray | None,
    displacement: np.ndarray | None,
) -> plt.Figure:
    """One example. Latent panels are the saved scalars, not a spatial map."""
    apply_style()
    figure, axes = plt.subplots(1, 4, figsize=(PANEL_INCH * 4.4, PANEL_INCH + 0.7), constrained_layout=True)
    if particles is not None and displacement is not None:
        magnitude = np.linalg.norm(np.asarray(displacement), axis=-1)
        axes[0].scatter(particles[:, 0], particles[:, 1], c=magnitude, s=8, cmap="gray")
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_aspect("equal")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    else:
        _blank(axes[0], "particle positions\nnot in the cache")
    axes[0].set_title("Particle displacement")
    _show(axes[1], observation_difference, "Observation difference")
    _blank(axes[2], _scalar_text("Target latent L2", scalars.get("target_delta_l2")))
    axes[2].set_title("Target latent")
    _blank(axes[3], _scalar_text("Predicted latent L2", scalars.get("pred_delta_l2")))
    axes[3].set_title("Predicted latent")
    rms = scalars.get("rms_pos")
    affected = scalars.get("fraction_affected_pixels")
    figure.suptitle(
        f"rms {rms:.3g}   affected {affected:.3g}" if rms is not None and affected is not None else "Bottleneck"
    )
    return figure


def _scalar_text(label: str, value: float | None) -> str:
    if value is None:
        return f"{label}\nnot measured"
    return f"{label}\n{value:.3g}"


def example_grid(
    examples: Sequence[tuple[str, np.ndarray | None, np.ndarray | None, str]],
) -> plt.Figure:
    """Each row is one selected role: recorded, alternative, absolute, amplified."""
    apply_style()
    figure, axes = plt.subplots(
        len(examples),
        4,
        figsize=(PANEL_INCH * 4, PANEL_INCH * max(len(examples), 1) + 0.4),
        squeeze=False,
        constrained_layout=True,
    )
    headers = ("Recorded future", "Alternative", "Absolute difference", f"Difference x{AMPLIFY:g}")
    for row_index, (label, recorded, alternative, note) in enumerate(examples):
        if recorded is None or alternative is None:
            for column in range(4):
                _blank(axes[row_index, column], note if column == 0 else "")
            axes[row_index, 0].set_ylabel(label, fontsize=7)
            continue
        absolute, amplified = _pair(recorded, alternative)
        for column, image in enumerate((recorded, alternative, absolute, amplified)):
            _show(axes[row_index, column], image, headers[column] if row_index == 0 else "")
        axes[row_index, 0].set_ylabel(f"{label}\n{note}", fontsize=7)
    figure.suptitle("Selected forks")
    return figure


def planning_figure(rows: Sequence[tuple[str, np.ndarray | None, str]]) -> plt.Figure:
    """Each row is one controller. Columns are saved rollout frames."""
    apply_style()
    widths = [int(frames.shape[0]) if frames is not None else 1 for _, frames, _ in rows]
    columns = max(widths) if widths else 1
    figure, axes = plt.subplots(
        len(rows),
        columns,
        figsize=(PANEL_INCH * columns, PANEL_INCH * max(len(rows), 1) + 0.35),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (label, frames, color_key) in enumerate(rows):
        color = FAMILY_COLORS.get(color_key, "#333333")
        if frames is None:
            _blank(axes[row_index, 0], "frames not in the planning archive")
            for column in range(1, columns):
                _blank(axes[row_index, column], "")
        else:
            for column in range(columns):
                if column < frames.shape[0]:
                    _show(axes[row_index, column], frames[column], f"t{column}" if row_index == 0 else "")
                else:
                    _blank(axes[row_index, column], "")
        axes[row_index, 0].set_ylabel(label, fontsize=7, color=color)
    figure.suptitle("Planning rollouts")
    return figure


def comparison_figure(
    rows: Sequence[tuple[str, np.ndarray | None, np.ndarray | None]],
    title: str,
) -> plt.Figure:
    """Recorded, alternative, and absolute difference for each named condition."""
    apply_style()
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(PANEL_INCH * 3.2, PANEL_INCH * max(len(rows), 1) + 0.4),
        squeeze=False,
        constrained_layout=True,
    )
    headers = ("Recorded", "Alternative", "Absolute difference")
    for row_index, (label, recorded, alternative) in enumerate(rows):
        if recorded is None or alternative is None:
            for column in range(3):
                _blank(axes[row_index, column], "not verified" if column == 0 else "")
        else:
            absolute, _amplified = _pair(recorded, alternative)
            for column, image in enumerate((recorded, alternative, absolute)):
                _show(axes[row_index, column], image, headers[column] if row_index == 0 else "")
        axes[row_index, 0].set_ylabel(label, fontsize=7)
    figure.suptitle(title)
    return figure
