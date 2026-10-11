#!/usr/bin/env python3
"""Directional counterfactuals for the ten legacy no-crop checkpoints.

    python3.13 scripts/run_counterfactual_direction.py

The alternate action is a raw force reversal. Episodes 1800-1999 are legacy
validation, not a test set. Every transition is written, including exclusions.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, render_rgb, replay_episode
from src.config import ModelConfig
from src.counterfactual_direction_plots import plot_directional
from src.metrics.counterfactual_direction import (
    HIGH_VISIBILITY_AFFECTED,
    assign_quintiles,
    bootstrap_valid_cosine,
    directional_contrast,
    summarize_metric,
)
from src.models.ac_jepa import ACJEPA
from src.registry import RunRecorder, to_jsonable
from src.splits import get_split

CHECKPOINTS = (
    ("vicreg_action_s42", "outputs/vicreg_nocrop/last.pt", 42),
    ("vicreg_action_s43", "outputs/vicreg_nocrop_seed43/last.pt", 43),
    ("vicreg_noaction_s42", "outputs/no_action/last.pt", 42),
    ("vicreg_noaction_s43", "outputs/vicreg_nocrop_noaction_seed43/last.pt", 43),
    ("vicreg_action_s44", "results/reviewer_strengthening/05_more_seeds/vicreg_action_s44/last.pt", 44),
    ("vicreg_action_s45", "results/reviewer_strengthening/05_more_seeds/vicreg_action_s45/last.pt", 45),
    ("vicreg_noaction_s44", "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s44/last.pt", 44),
    ("vicreg_noaction_s45", "results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s45/last.pt", 45),
    ("mse_action_s42", "results/reviewer_strengthening/06_corrected_objective/mse_action_s42/last.pt", 42),
    ("mse_action_s43", "results/reviewer_strengthening/06_corrected_objective/mse_action_s43/last.pt", 43),
)


def _optional(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _affected(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.any(left != right, axis=0)))


def _collect_forks(episodes: range, hold: int) -> dict[str, Any]:
    current = []
    factual = []
    alternate = []
    action = []
    action_alt = []
    episode_ids = []
    times = []
    affected = []
    boundaries = []
    for episode in episodes:
        print(f"fork episode {episode}", flush=True)
        recorded = replay_episode(int(episode), dump_seed=0, episode_length=16, hold=hold)
        steps = int(recorded["actions"].shape[0])
        for step in range(steps):
            branch = MLSMPM2D()
            branch.restore(recorded["snapshots"][step])
            alt = recorded["actions"][step].copy()
            alt[2:4] *= -1.0
            apply_action_frames(branch, alt, 1)
            alt_frame = render_rgb(branch.x, branch.v, branch.cfg.img_size, branch.cfg.render_sigma)
            current.append(recorded["frames"][step])
            factual.append(recorded["frames"][step + 1])
            alternate.append(alt_frame)
            action.append(recorded["actions"][step])
            action_alt.append(alt)
            episode_ids.append(int(episode))
            times.append(step)
            affected.append(_affected(recorded["frames"][step + 1], alt_frame))
            boundaries.append(step % hold == 0)
    return {
        "current": np.stack(current),
        "factual": np.stack(factual),
        "alternate": np.stack(alternate),
        "action": np.stack(action).astype(np.float32),
        "action_alt": np.stack(action_alt).astype(np.float32),
        "episode": np.asarray(episode_ids, dtype=np.int64),
        "t": np.asarray(times, dtype=np.int64),
        "fraction_affected": np.asarray(affected, dtype=np.float64),
        "action_change_boundary": np.asarray(boundaries),
    }


def _encode(model: ACJEPA, frames: np.ndarray, target: bool) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], 64):
            batch = torch.from_numpy(frames[start : start + 64].astype(np.float32) / 255.0)
            outputs.append(model.encode(batch, target=target).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict(model: ACJEPA, latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, latent.shape[0], 64):
            z = torch.from_numpy(latent[start : start + 64])
            act = torch.from_numpy(actions[start : start + 64])
            outputs.append(model.predictor(z, act).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _slice_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    if not rows:
        return {"slice": name, "n_total": 0}
    return {
        "slice": name,
        "cosine": summarize_metric([row["cosine"] for row in rows], [row["cosine_status"] for row in rows], [row["contrast_status"] for row in rows]),
        "magnitude_ratio": summarize_metric(
            [row["magnitude_ratio"] for row in rows],
            ["ok" if row["magnitude_ratio"] is not None else "undefined_nonfinite" for row in rows],
            [row["contrast_status"] for row in rows],
        ),
        "normalized_vector_error": summarize_metric(
            [row["normalized_vector_error"] for row in rows],
            ["ok" if row["normalized_vector_error"] is not None else "undefined_nonfinite" for row in rows],
            [row["contrast_status"] for row in rows],
        ),
        "angular_error_rad": summarize_metric(
            [row["angular_error_rad"] for row in rows],
            [row["angular_status"] for row in rows],
            [row["contrast_status"] for row in rows],
        ),
        "signed_projection": summarize_metric(
            [row["signed_projection"] for row in rows],
            [row["projection_status"] for row in rows],
            [row["contrast_status"] for row in rows],
        ),
    }


def _analysis_slices(rows: list[dict[str, Any]], high_visibility: float) -> list[dict[str, Any]]:
    slices = [
        ("all", rows),
        ("nonzero_observable", [row for row in rows if float(row["fraction_affected_pixels"]) > 0.0]),
        ("high_visibility", [row for row in rows if float(row["fraction_affected_pixels"]) >= high_visibility]),
        ("action_change_boundary", [row for row in rows if row["action_change_boundary"]]),
    ]
    for quintile in range(1, 6):
        slices.append((f"action_l2_q{quintile}", [row for row in rows if int(row["action_l2_quintile"]) == quintile]))
    return [_slice_summary(group, name) for name, group in slices]


def main() -> None:
    config_path = ROOT / "configs" / "experiments" / "counterfactual_direction_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    legacy = get_split("legacy_v1")
    if legacy.headline_role != "legacy_validation":
        raise RuntimeError("directional eval of these checkpoints is legacy validation, not a test set.")
    command = "python3.13 scripts/run_counterfactual_direction.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed="legacy_val_force_reversed",
        model={"checkpoints": [name for name, _path, _seed in CHECKPOINTS]},
        environment={"simulator": "MLSMPM2D", "fork": "force_reversed"},
        dataset_split={
            "split_version": "legacy_v1",
            "headline_role": "legacy_validation",
            "is_test_set": False,
            "episodes": [legacy.val_ids[0], legacy.val_ids[-1] + 1],
        },
        checkpoint={"path": "ten no-crop last.pt", "note": "historical checkpoints"},
    ) as run:
        raw = np.load(ROOT / "outputs" / "physics_mpm" / "actions.npy")
        train_rows = raw[: legacy.val_ids[0]].reshape(-1, 4)
        mean = train_rows.mean(axis=0)
        std = np.maximum(train_rows.std(axis=0), 1e-6)
        forks = _collect_forks(range(int(legacy.val_ids[0]), int(legacy.val_ids[-1]) + 1), int(config["hold"]))
        action_l2 = np.linalg.norm(forks["action_alt"] - forks["action"], axis=1)
        quintile, edges = assign_quintiles(action_l2)
        norm_action = ((forks["action"] - mean) / std).astype(np.float32)
        norm_alt = ((forks["action_alt"] - mean) / std).astype(np.float32)
        rows: list[dict[str, Any]] = []
        by_model: dict[str, list[dict[str, Any]]] = {}
        for name, rel, seed in CHECKPOINTS:
            print("model", name, flush=True)
            payload = torch.load(ROOT / rel, map_location="cpu", weights_only=False)
            model = ACJEPA(ModelConfig(**payload["config"]["model"]))
            model.load_state_dict(payload["model"])
            model.eval()
            latent = _encode(model, forks["current"], False)
            target = _encode(model, forks["factual"], True)
            target_alt = _encode(model, forks["alternate"], True)
            pred = _predict(model, latent, norm_action)
            pred_alt = _predict(model, latent, norm_alt)
            scored = directional_contrast(
                pred_alt - pred,
                target_alt - target,
                fraction_affected=forks["fraction_affected"],
                norm_tau=float(config["norm_tau"]),
                epsilon=float(config["epsilon"]),
                invisible_affected_max=float(config["invisible_affected_max"]),
            )
            model_rows = []
            for index in range(scored["n"]):
                model_rows.append(
                    {
                        "model": name,
                        "seed": int(seed),
                        "objective": "mse" if name.startswith("mse_") else "vicreg",
                        "condition_on_action": "noaction" not in name,
                        "split_version": "legacy_v1",
                        "is_test_set": False,
                        "episode": int(forks["episode"][index]),
                        "t": int(forks["t"][index]),
                        "action_change_boundary": bool(forks["action_change_boundary"][index]),
                        "action_l2": float(action_l2[index]),
                        "action_l2_quintile": int(quintile[index]),
                        "fraction_affected_pixels": float(forks["fraction_affected"][index]),
                        "contrast_status": str(scored["contrast_status"][index]),
                        "target_near_zero": bool(scored["target_near_zero"][index]),
                        "target_invisible": bool(scored["target_invisible"][index]),
                        "target_norm": float(scored["target_norm"][index]),
                        "pred_norm": float(scored["pred_norm"][index]),
                        "cosine": _optional(float(scored["cosine"][index])),
                        "cosine_status": str(scored["cosine_status"][index]),
                        "normalized_vector_error": _optional(float(scored["normalized_vector_error"][index])),
                        "magnitude_ratio": _optional(float(scored["magnitude_ratio"][index])),
                        "angular_error_rad": _optional(float(scored["angular_error_rad"][index])),
                        "angular_status": str(scored["angular_status"][index]),
                        "signed_projection": _optional(float(scored["signed_projection"][index])),
                        "projection_status": str(scored["projection_status"][index]),
                    }
                )
            by_model[name] = model_rows
            rows.extend(model_rows)
        table = run.run_dir / "transitions.csv"
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if value is None else value for key, value in row.items()})
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(rows), run.run_dir / "transitions.parquet")
            parquet_status = "written"
        except ImportError:
            parquet_status = "pyarrow_unavailable"
        analyses = {
            name: _analysis_slices(model_rows, float(config["high_visibility_affected"]))
            for name, model_rows in by_model.items()
        }
        bootstraps = {
            name: bootstrap_valid_cosine(
                [row["episode"] for row in model_rows],
                [row["cosine"] for row in model_rows],
                [row["cosine_status"] for row in model_rows],
                [row["contrast_status"] for row in model_rows],
                n_resamples=int(config["bootstrap_resamples"]),
                rng=np.random.default_rng(int(config["bootstrap_seed"])),
            )
            for name, model_rows in by_model.items()
        }
        plot_directional(rows, ROOT / "figures" / "counterfactual_direction")
        summary = {
            "split_version": "legacy_v1",
            "is_test_set": False,
            "n_rows": len(rows),
            "action_l2_quintile_edges": edges,
            "high_visibility_affected": HIGH_VISIBILITY_AFFECTED,
            "analyses": analyses,
            "bootstrap": bootstraps,
            "parquet_status": parquet_status,
            "learned_model_evaluated": True,
        }
        (run.run_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
        run.complete(
            to_jsonable(
                {
                    "n_rows": len(rows),
                    "parquet_status": parquet_status,
                    "is_test_set": False,
                    "models": list(by_model),
                }
            )
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
