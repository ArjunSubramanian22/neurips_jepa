#!/usr/bin/env python3
"""Write the confirmatory matrix from declared cells and files on disk.

A missing checkpoint stays NOT_RUN. This script does not launch training.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "experiments" / "confirmatory_matrix.csv"
OUT_MD = ROOT / "experiments" / "CONFIRMATORY_MATRIX.md"

# Checkpoint paths that already exist from the first wave. New runs use the
# output directories named in the notes column.
LEGACY = {
    ("vicreg_action", 42): "outputs/vicreg_nocrop/last.pt",
    ("vicreg_action", 43): "outputs/vicreg_nocrop_seed43/last.pt",
    ("vicreg_action", 44): "results/reviewer_strengthening/05_more_seeds/vicreg_action_s44/last.pt",
    ("vicreg_action", 45): "results/reviewer_strengthening/05_more_seeds/vicreg_action_s45/last.pt",
    ("vicreg_noaction", 42): "outputs/no_action/last.pt",
    ("vicreg_noaction", 43): "outputs/vicreg_nocrop_noaction_seed43/last.pt",
    ("vicreg_noaction", 44): "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s44/last.pt",
    ("vicreg_noaction", 45): "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s45/last.pt",
    ("mse_action", 42): "results/reviewer_strengthening/06_corrected_objective/mse_action_s42/last.pt",
    ("mse_action", 43): "results/reviewer_strengthening/06_corrected_objective/mse_action_s43/last.pt",
}
ARM = {
    ("jepa", 0): "outputs/mujoco_arm_runs/jepa/last.pt",
    ("rssm", 0): "outputs/mujoco_arm_runs/rssm/last.pt",
    ("latent_tf", 0): "outputs/mujoco_arm_runs/latent_tf/last.pt",
}

FIELDS = (
    "environment",
    "observation_modality",
    "resolution",
    "model",
    "conditioning",
    "loss",
    "seed",
    "training_status",
    "checkpoint_status",
    "counterfactual_eval_status",
    "planning_eval_status",
    "test_eval_status",
    "notes",
)


def _exists(rel: str | None) -> bool:
    return bool(rel) and (ROOT / rel).is_file() and (ROOT / rel).stat().st_size > 0


def _row(**kwargs: object) -> dict[str, str]:
    return {key: str(kwargs.get(key, "")) for key in FIELDS}


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    legacy_eval = "COMPLETE"
    legacy_plan = "COMPLETE_BELOW_TARGET_N"
    for model, seeds in (
        ("vicreg_action", (42, 43, 44, 45, 46)),
        ("vicreg_noaction", (42, 43, 44, 45, 46)),
        ("mse_action", (42, 43, 44, 45, 46)),
    ):
        loss = "mse" if model.startswith("mse") else "vicreg"
        for seed in seeds:
            path = LEGACY.get((model, seed))
            trained = _exists(path)
            rows.append(
                _row(
                    environment="mpm_legacy",
                    observation_modality="physical_raster",
                    resolution=64,
                    model=model,
                    conditioning="concat",
                    loss=loss,
                    seed=seed,
                    training_status="COMPLETE" if trained else "NOT_RUN",
                    checkpoint_status="PRESENT" if trained else "MISSING",
                    counterfactual_eval_status=legacy_eval if trained else "NOT_RUN",
                    planning_eval_status=legacy_plan if trained else "NOT_RUN",
                    test_eval_status="NOT_APPLICABLE",
                    notes="legacy_v1 validation episodes 1800-1999; no untouched test; planning report seeds are fewer than 20 scored goals",
                )
            )
    for model, loss in (("rssm", "rssm"), ("latent_tf", "latent_tf"), ("obs_pred", "obs_pred")):
        rows.append(
            _row(
                environment="mpm_legacy",
                observation_modality="physical_raster",
                resolution=64,
                model=model,
                conditioning="concat",
                loss=loss,
                seed=42,
                training_status="NOT_RUN",
                checkpoint_status="MISSING",
                counterfactual_eval_status="NOT_RUN",
                planning_eval_status="NOT_RUN",
                test_eval_status="NOT_APPLICABLE",
                notes="stage-2 screen on the legacy dump; not a fifth headline seed",
            )
        )
    for model, loss in (
        ("vicreg_action", "vicreg"),
        ("vicreg_noaction", "vicreg"),
        ("mse_action", "mse"),
        ("rssm", "rssm"),
        ("latent_tf", "latent_tf"),
    ):
        for seed in (42, 43, 44, 45, 46):
            high_paths = {
                ("vicreg_action", 42): (
                    "outputs/mpm_high_runs/jepa_s42/last.pt",
                    "experiments/runs/mpm_high_jepa_s42_val_screen_v1/screen.json",
                ),
                ("vicreg_action", 43): (
                    "outputs/mpm_high_runs/jepa_s43/last.pt",
                    "experiments/runs/mpm_high_jepa_s43_val_screen_v1/screen.json",
                ),
                ("vicreg_noaction", 42): (
                    "outputs/mpm_high_runs/noaction_s42/last.pt",
                    "experiments/runs/mpm_high_noaction_s42_val_screen_v1/screen.json",
                ),
            }
            ckpt, screen = high_paths.get((model, seed), (None, None))
            trained = _exists(ckpt)
            screened = trained and _exists(screen)
            rows.append(
                _row(
                    environment="mpm_high",
                    observation_modality="physical_raster",
                    resolution=64,
                    model=model,
                    conditioning="concat",
                    loss=loss,
                    seed=seed,
                    training_status="COMPLETE" if trained else "NOT_RUN",
                    checkpoint_status="PRESENT" if trained else "MISSING",
                    counterfactual_eval_status="COMPLETE_N24_VAL" if screened else "NOT_RUN",
                    planning_eval_status="NOT_RUN",
                    test_eval_status="NOT_RUN",
                    notes="fresh_pool_v2 on outputs/physics_mpm_high; test suffix closed; seed 42 screen is 24 validation episodes, horizon 1, open loop",
                )
            )
    for model in ("jepa", "rssm", "latent_tf"):
        path = ARM.get((model, 0))
        trained = _exists(path)
        rows.append(
            _row(
                environment="mujoco_arm",
                observation_modality="rgb",
                resolution=64,
                model=model,
                conditioning="concat",
                loss=model if model != "jepa" else "vicreg",
                seed=0,
                training_status="COMPLETE" if trained else "NOT_RUN",
                checkpoint_status="PRESENT" if trained else "MISSING",
                counterfactual_eval_status="COMPLETE" if trained else "NOT_RUN",
                planning_eval_status="COMPLETE_SMALL_N" if trained else "NOT_RUN",
                test_eval_status="NOT_APPLICABLE",
                notes="arm_tail_v1; 24 train episodes; validation tail; one seed is not a five-seed claim",
            )
        )
        for seed in (42, 43, 44, 45, 46):
            rows.append(
                _row(
                    environment="mujoco_arm",
                    observation_modality="rgb",
                    resolution=64,
                    model=model,
                    conditioning="concat",
                    loss=model if model != "jepa" else "vicreg",
                    seed=seed,
                    training_status="NOT_RUN",
                    checkpoint_status="MISSING",
                    counterfactual_eval_status="NOT_RUN",
                    planning_eval_status="NOT_RUN",
                    test_eval_status="NOT_APPLICABLE",
                    notes="five-seed arm replication; cheap relative to the MPM dump",
                )
            )
    for conditioning in ("film", "gate", "cross_attn"):
        rows.append(
            _row(
                environment="mpm_high",
                observation_modality="physical_raster",
                resolution=64,
                model="vicreg_action",
                conditioning=conditioning,
                loss="vicreg",
                seed=42,
                training_status="NOT_RUN",
                checkpoint_status="MISSING",
                counterfactual_eval_status="NOT_RUN",
                planning_eval_status="NOT_RUN",
                test_eval_status="NOT_RUN",
                notes="stage-2 conditioning screen; promote only if it changes action fidelity",
            )
        )
    for loss in ("inverse", "direction", "magnitude", "combined"):
        blocked = loss != "inverse"
        rows.append(
            _row(
                environment="mpm_high",
                observation_modality="physical_raster",
                resolution=64,
                model="vicreg_action",
                conditioning="concat",
                loss=loss,
                seed=42,
                training_status="BLOCKED" if blocked else "NOT_RUN",
                checkpoint_status="MISSING",
                counterfactual_eval_status="NOT_RUN",
                planning_eval_status="NOT_RUN",
                test_eval_status="NOT_RUN",
                notes=(
                    "counterfactual pair cache is not on disk"
                    if blocked
                    else "inverse-dynamics auxiliary; one-seed screen before any weight selection"
                ),
            )
        )
    return rows


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["training_status"]] = counts.get(row["training_status"], 0) + 1
    lines = [
        "# Confirmatory matrix",
        "",
        "Generated by `python3.13 scripts/write_confirmatory_matrix.py`.",
        "",
        f"Rows: {len(rows)}.",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "No row is omitted when its checkpoint is missing.", ""])
    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("| " + " | ".join("---" for _ in FIELDS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[key] for key in FIELDS) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, OUT_MD)
    print(f"rows {len(rows)} csv {OUT_CSV}")


if __name__ == "__main__":
    main()
