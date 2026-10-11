"""Per-seed headline plots. Missing seeds stay on the axis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_seed_values(blocks: Sequence[dict[str, Any]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        _one_metric(block, out_dir / f"{block['metric']}_{block['rollout_mode']}.png")
    paired = [block for block in blocks if block.get("paired")]
    if paired:
        _paired(paired[0]["paired"], out_dir / "paired_action_minus_noaction.png")


def _one_metric(block: dict[str, Any], path: Path) -> None:
    configurations = block["configurations"]
    seeds = block["seeds"]
    fig, ax = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    colors = {"COMPLETE": "#2c5f8a", "UNDEFINED": "#8a6a2c", "NOT_RUN": "#8a2c2c", "FAILED": "#8a2c2c"}
    for index, configuration in enumerate(configurations):
        for offset, seed_row in enumerate(configuration["aggregate"]["seeds"]):
            x = index + (offset - (len(seeds) - 1) / 2) * 0.12
            status = seed_row["status"]
            if status == "COMPLETE" and seed_row["value"] is not None:
                ax.scatter(x, seed_row["value"], color=colors["COMPLETE"], s=28, zorder=3)
                conditional = configuration["hierarchical"]["conditional_on_checkpoint"].get(str(seed_row["seed"]))
                if conditional is not None:
                    ax.plot(
                        [x, x],
                        [conditional["ci_low"], conditional["ci_high"]],
                        color="#9bb4c8",
                        lw=2,
                        zorder=2,
                    )
            else:
                ax.text(x, -0.12 if block["metric"] == "ranking_top1" else -1.15, seed_row["status"].replace("_", "\n"), ha="center", va="bottom", fontsize=5, color="#8a2c2c")
        ax.text(index, 1.02 if block["metric"] == "ranking_top1" else 1.08, configuration["id"].replace("_", "\n"), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(configurations)), [""] * len(configurations))
    missing_note = "Missing seeds are labeled on the axis and stay in the registry."
    ax.set_title(
        f"{block['metric'].replace('_', ' ')}, {block['rollout_mode'].replace('_', ' ')}, horizon {block['horizon']}\n"
        f"Vertical bars are episode intervals for that checkpoint. No training-seed interval is drawn.\n"
        f"{missing_note}"
    )
    ax.set_ylabel(block["metric"].replace("_", " "))
    if block["metric"] == "ranking_top1":
        ax.axhline(0.5, color="#888888", lw=1)
        ax.set_ylim(-0.15, 1.2)
    else:
        ax.axhline(0.0, color="#888888", lw=1)
        ax.set_ylim(-1.25, 1.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _paired(paired: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    differences = paired["action_minus_noaction"]["differences"]
    if differences:
        ax.scatter([item["seed"] for item in differences], [item["difference"] for item in differences], color="#2c5f8a")
        mean = paired["action_minus_noaction"]["effect_size"]["mean_difference"]
        if mean is not None:
            ax.axhline(mean, color="#2c5f8a", lw=1, label="mean of completed pairs")
    unpaired = paired["action_minus_noaction"]["unpaired_seeds"]
    for seed in unpaired:
        ax.text(seed, 0.02, "not paired", rotation=90, ha="center", va="bottom", fontsize=7, color="#8a2c2c")
    ax.axhline(0.0, color="#888888", lw=1)
    ax.set_xlim(41.5, 46.6)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Action minus no-action top-1")
    ax.set_title("Paired seeds only. The line is not a confidence interval.")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)
