"""Join the width sweep with the metrics that already exist.

Only the base width has trained seeds. Small and large stay empty. No line
is fit across a single trained scale.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.ac_jepa import ACJEPA
from src.models.capacity import capacity_report

PROTOCOL = ROOT / "configs" / "experiments" / "capacity_sweep_v1.yaml"
METRICS = ROOT / "analysis" / "metric_planning_models.csv"
HEADLINE = ROOT / "analysis" / "capacity_sweep_headline.json"
FIGURE_DIR = ROOT / "figures" / "capacity_sweep"

METRIC_FIELDS = {
    "rollout_error": "one_step_latent_mse",
    "effective_rank": "effective_rank",
    "action_response_magnitude": "magnitude_response_ratio",
    "directional_alignment": "directional_cosine",
    "retrieval": "retrieval_accuracy",
    "planning_success": "mpc_success_rate",
    "planning_distance": "final_goal_distance",
}


def scale_trend(scales_with_values: Sequence[dict[str, Any]]) -> str:
    """A scale effect needs two widths that each have measured values."""
    measured = {row["scale"] for row in scales_with_values if row.get("value") is not None}
    if len(measured) < 2:
        return "not_identified"
    return "estimated"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_base_metrics() -> list[dict[str, Any]]:
    with METRICS.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["family"] == "vicreg_action"]
    if len(rows) < 3:
        raise ValueError("the base scale needs several trained seeds.")
    return rows


def _history_seconds(checkpoint: str | None) -> float | None:
    if not checkpoint:
        return None
    history = ROOT / checkpoint
    history = history.with_name("history.json")
    if not history.is_file():
        return None
    payload = json.loads(history.read_text(encoding="utf-8"))
    seconds = payload.get("seconds")
    return None if seconds is None else float(seconds)


def _capacity(width: int) -> dict[str, int]:
    model = ACJEPA(
        ModelConfig(
            img_size=64,
            in_channels=3,
            action_dim=4,
            embed_dim=width,
            predictor_embed_dim=width,
            encoder_depth=4,
            predictor_depth=4,
            encoder_num_heads=4,
            predictor_num_heads=4,
            mlp_ratio=2.0,
            action_embed_dim=64,
            drop_path=0.05,
            fuse_mode="concat",
        )
    )
    report = capacity_report(model)
    return {
        "trainable_parameters": int(report["trainable_parameters"]),
        "train_step_forward_flops": int(report["train_step_forward_flops"]),
    }


def build_report() -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    registry = yaml.safe_load((ROOT / "configs" / "seeds" / "capacity_sweep_v1.yaml").read_text(encoding="utf-8"))
    base_paths = {
        int(seed): path
        for item in registry["configurations"]
        if item["id"] == "capacity_base"
        for seed, path in item["checkpoints"].items()
    }
    metrics = _load_base_metrics()
    scales = []
    for scale in protocol["scales"]:
        counts = _capacity(int(scale["width"]))
        if scale["name"] == "base":
            seed_rows = []
            for row in metrics:
                seed = int(row["training_seed"])
                checkpoint = base_paths.get(seed)
                seed_rows.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "training_seconds": _history_seconds(checkpoint),
                        **{name: _float_or_none(row[source]) for name, source in METRIC_FIELDS.items()},
                    }
                )
            summary = {
                name: {
                    "mean": _mean([item[name] for item in seed_rows if item[name] is not None]),
                    "n": sum(item[name] is not None for item in seed_rows),
                    "values": [item[name] for item in seed_rows],
                }
                for name in METRIC_FIELDS
            }
            status = "trained"
        else:
            seed_rows = []
            summary = {name: {"mean": None, "n": 0, "values": []} for name in METRIC_FIELDS}
            status = "not_trained"
        scales.append(
            {
                "scale": scale["name"],
                "width": int(scale["width"]),
                "status": status,
                **counts,
                "seeds": seed_rows,
                "metrics": summary,
            }
        )
    trends = {
        name: scale_trend([{"scale": row["scale"], "value": row["metrics"][name]["mean"]} for row in scales])
        for name in METRIC_FIELDS
    }
    return {
        "experiment_id": protocol["experiment_id"],
        "varied_factor": protocol["varied_factor"],
        "fixed": protocol["fixed"],
        "is_test_set": False,
        "split_version": "legacy_v1",
        "training_seconds_available_for_untrained_scales": False,
        "flops_are_forward_only": True,
        "scale_trends": trends,
        "conclusion": "not_identified",
        "headline": (
            "Only the base width has trained seeds. Small and large were not trained, "
            "so the sweep cannot tell whether action under-use is a capacity limit."
        ),
        "scales": scales,
    }


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _panel(axis, report: dict[str, Any], name: str, x_key: str, color: str) -> None:
    scales = report["scales"]
    for index, row in enumerate(scales):
        values = [value for value in row["metrics"][name]["values"] if value is not None]
        if values:
            axis.scatter([index] * len(values), values, s=28, color=color, zorder=3)
        else:
            axis.text(index, 0.5, "not trained", transform=axis.get_xaxis_transform(), ha="center", va="center", fontsize=7, color="#666666")
    axis.set_xticks(range(len(scales)), [f"{row['scale']}\n{row[x_key]:,}" for row in scales], fontsize=7)
    axis.set_title(name.replace("_", " "), fontsize=8)
    if name == "planning_success":
        axis.set_ylim(-0.05, 1.05)


def plot_report(report: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(METRIC_FIELDS)
    figure, axes = plt.subplots(2, 4, figsize=(12.8, 6.4), squeeze=False)
    for axis, name in zip(axes.ravel(), names):
        _panel(axis, report, name, "trainable_parameters", "#4c78a8")
        axis.set_xlabel("parameters")
    axes.ravel()[-1].axis("off")
    figure.suptitle(report["headline"], fontsize=9)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / "metrics_vs_parameters.png", dpi=140, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(12.8, 6.4), squeeze=False)
    for axis, name in zip(axes.ravel(), names):
        _panel(axis, report, name, "train_step_forward_flops", "#f58518")
        axis.set_xlabel("train-step forward FLOPs")
    axes.ravel()[-1].axis("off")
    figure.suptitle("Forward FLOPs per training step. Backward and wall-clock time are not the x-axis.", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "metrics_vs_forward_flops.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    report = build_report()
    if any(status != "not_identified" for status in report["scale_trends"].values()):
        raise SystemExit("refusing to claim a scale trend without two trained widths.")
    HEADLINE.parent.mkdir(parents=True, exist_ok=True)
    HEADLINE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_report(report)
    print(report["headline"])
    print(HEADLINE)


if __name__ == "__main__":
    main()
