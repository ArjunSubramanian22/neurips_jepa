#!/usr/bin/env python3
"""Multi-horizon counterfactual audit for the ten legacy no-crop checkpoints.

    python3.13 scripts/run_horizon_audit.py

Branch A is the recorded action sequence. Branch B is one labeled intervention.
Open-loop forecasts recurse on predicted latents. Teacher-forced forecasts read
the true frame. Episodes 1800-1999 are legacy validation, not a test set.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
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
from src.environments import environment_from_dict
from src.horizon_audit_plots import plot_horizon_audit
from src.metrics.cascade import physical_response
from src.metrics.future_ranking import derangement
from src.metrics.horizon_audit import (
    bootstrap_labeled,
    counterfactual_actions,
    score_horizon_pair,
    summarize_labeled,
)
from src.models.ac_jepa import ACJEPA
from src.registry import RunRecorder, to_jsonable
from src.rgb_render import load_rgb_config, render_rgb_image, to_uint8
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
BOOL_FIELDS = {
    "condition_on_action",
    "is_test_set",
    "raster_identical",
    "rgb_identical",
    "target_near_zero",
    "target_invisible",
    "observationally_ambiguous",
    "physically_distinct_observationally_identical",
}
INT_FIELDS = {"seed", "episode", "t", "horizon"}
FLOAT_FIELDS = {
    "rms_pos",
    "rms_vel",
    "raster_fraction_identical",
    "raster_mse",
    "fraction_affected_pixels",
    "rgb_fraction_identical",
    "rgb_mse",
    "rgb_fraction_affected_pixels",
    "target_delta_l2",
    "target_delta_mse",
    "target_direction_cosine",
    "pred_delta_l2",
    "cosine",
    "magnitude_ratio",
    "normalized_vector_error",
    "recorded_ranking_top1",
    "counterfactual_ranking_top1",
    "counterfactual_ranking_margin",
    "rollout_mse_recorded",
    "rollout_mse_counterfactual",
}


def _encode(model: ACJEPA, frames: np.ndarray, target: bool) -> np.ndarray:
    flat = np.ascontiguousarray(frames).reshape(-1, *frames.shape[-3:])
    outputs = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], 64):
            batch = torch.from_numpy(flat[start : start + 64].astype(np.float32) / 255.0)
            outputs.append(model.encode(batch, target=target).cpu().numpy())
    encoded = np.concatenate(outputs, axis=0)
    return encoded.reshape(*frames.shape[:-3], encoded.shape[-1])


def _rollout(model: ACJEPA, latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, latent.shape[0], 32):
            z = torch.from_numpy(np.ascontiguousarray(latent[start : start + 32], dtype=np.float32))
            act = torch.from_numpy(np.ascontiguousarray(actions[start : start + 32], dtype=np.float32))
            outputs.append(model.rollout(z, act).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict(model: ACJEPA, latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, latent.shape[0], 64):
            z = torch.from_numpy(np.ascontiguousarray(latent[start : start + 64], dtype=np.float32))
            act = torch.from_numpy(np.ascontiguousarray(actions[start : start + 64], dtype=np.float32))
            outputs.append(model.predictor(z, act).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _zscore(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((actions - mean) / std).astype(np.float32)


def _simulate(config: dict[str, Any]) -> dict[str, Any]:
    legacy = get_split("legacy_v1")
    episodes = list(range(int(legacy.val_ids[0]), int(legacy.val_ids[-1]) + 1))
    spec = environment_from_dict(yaml.safe_load((ROOT / config["environment"]).read_text(encoding="utf-8")))
    horizons = [int(value) for value in config["horizons"]]
    if any(horizon < 1 for horizon in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be positive and unique.")
    max_horizon = max(horizons)
    start = int(config["start_time"])
    if start % int(spec.command_hold) != 0:
        raise ValueError("command-block replacement needs a start on a command boundary.")
    protocols = list(config["protocols"])
    dump_actions = np.load(ROOT / "outputs" / "physics_mpm" / "actions.npy")
    if start + max_horizon > dump_actions.shape[1]:
        raise ValueError("a requested horizon does not fit in the recorded action sequence.")
    rgb_cfg = load_rgb_config(ROOT / config["rgb_config"])
    n_episodes = len(episodes)
    n_protocols = len(protocols)
    frames_a = np.empty((n_episodes, max_horizon + 1, 3, 64, 64), dtype=np.uint8)
    actions_a = np.empty((n_episodes, max_horizon, 4), dtype=np.float64)
    positions_a: list[list[np.ndarray]] = []
    velocities_a: list[list[np.ndarray]] = []
    snaps0: list[dict[str, Any]] = []
    sim = MLSMPM2D(spec.mpm)
    for index, episode in enumerate(episodes):
        print(f"replay {episode}", flush=True)
        recorded = replay_episode(int(episode), dump_seed=0, episode_length=16, mpm=spec.mpm, hold=spec.command_hold)
        if not np.allclose(recorded["actions"], dump_actions[episode]):
            raise RuntimeError(f"legacy replay actions differ from the dump at episode {episode}.")
        frames_a[index] = recorded["frames"][: max_horizon + 1]
        actions_a[index] = recorded["actions"][start : start + max_horizon]
        positions_a.append([recorded["snapshots"][start + step]["x"] for step in range(max_horizon + 1)])
        velocities_a.append([recorded["snapshots"][start + step]["v"] for step in range(max_horizon + 1)])
        snaps0.append(recorded["snapshots"][start])
    partner = derangement(n_episodes, np.random.default_rng(int(config["shuffle_seed"])))
    actions_b = np.empty((n_episodes, n_protocols, max_horizon, 4), dtype=np.float64)
    for index in range(n_episodes):
        reversal = actions_a[index, 0].copy()
        reversal[2:4] *= -1.0
        shuffled = actions_a[int(partner[index]), 0]
        for protocol_index, protocol in enumerate(protocols):
            alternative = shuffled if protocol == "alternative_then_recorded" else reversal
            actions_b[index, protocol_index] = counterfactual_actions(
                actions_a[index],
                alternative,
                protocol,
                spec.command_hold,
            )
    frames_b = np.empty((n_episodes, n_protocols, max_horizon + 1, 3, 64, 64), dtype=np.uint8)
    rgb_a = np.empty((n_episodes, len(horizons), 3, 64, 64), dtype=np.uint8)
    rgb_b = np.empty((n_episodes, n_protocols, len(horizons), 3, 64, 64), dtype=np.uint8)
    rms_pos = np.empty((n_episodes, n_protocols, max_horizon), dtype=np.float64)
    rms_vel = np.empty((n_episodes, n_protocols, max_horizon), dtype=np.float64)
    horizon_index = {horizon: index for index, horizon in enumerate(horizons)}
    for index in range(n_episodes):
        print(f"fork {episodes[index]}", flush=True)
        for horizon, slot in horizon_index.items():
            rgb_a[index, slot] = to_uint8(render_rgb_image(positions_a[index][horizon], velocities_a[index][horizon], rgb_cfg))
        for protocol_index in range(n_protocols):
            sim.restore(snaps0[index])
            frames_b[index, protocol_index, 0] = frames_a[index, 0]
            for step in range(max_horizon):
                apply_action_frames(sim, actions_b[index, protocol_index, step], 1)
                if not np.isfinite(sim.x).all() or not np.isfinite(sim.v).all():
                    raise RuntimeError(f"nonfinite state at episode {episodes[index]} protocol {protocols[protocol_index]} step {step + 1}.")
                frames_b[index, protocol_index, step + 1] = render_rgb(sim.x, sim.v, sim.cfg.img_size, sim.cfg.render_sigma)
                physical = physical_response(positions_a[index][step + 1], sim.x, velocities_a[index][step + 1], sim.v)
                rms_pos[index, protocol_index, step] = physical["rms_pos"]
                rms_vel[index, protocol_index, step] = physical["rms_vel"]
                horizon = step + 1
                if horizon in horizon_index:
                    rgb_b[index, protocol_index, horizon_index[horizon]] = to_uint8(render_rgb_image(sim.x, sim.v, rgb_cfg))
    return {
        "episodes": np.asarray(episodes, dtype=np.int64),
        "horizons": horizons,
        "protocols": protocols,
        "command_hold": int(spec.command_hold),
        "environment": spec.environment_id,
        "frames_a": frames_a,
        "frames_b": frames_b,
        "rgb_a": rgb_a,
        "rgb_b": rgb_b,
        "rms_pos": rms_pos,
        "rms_vel": rms_vel,
        "actions_a": actions_a,
        "actions_b": actions_b,
        "start_time": start,
    }


def _arrow_schema(fieldnames: list[str]):
    import pyarrow as pa

    fields = []
    for name in fieldnames:
        if name in BOOL_FIELDS:
            typ = pa.bool_()
        elif name in INT_FIELDS:
            typ = pa.int64()
        elif name in FLOAT_FIELDS:
            typ = pa.float64()
        else:
            typ = pa.string()
        fields.append(pa.field(name, typ, nullable=True))
    return pa.schema(fields)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _save_cache(cache: Path, fork: dict[str, Any]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "episodes.npy", fork["episodes"])
    np.save(cache / "frames_a.npy", fork["frames_a"])
    np.save(cache / "frames_b.npy", fork["frames_b"])
    np.save(cache / "rgb_a.npy", fork["rgb_a"])
    np.save(cache / "rgb_b.npy", fork["rgb_b"])
    np.save(cache / "rms_pos.npy", fork["rms_pos"])
    np.save(cache / "rms_vel.npy", fork["rms_vel"])
    np.save(cache / "actions_a.npy", fork["actions_a"])
    np.save(cache / "actions_b.npy", fork["actions_b"])
    manifest = {
        "episodes": [int(fork["episodes"][0]), int(fork["episodes"][-1]) + 1],
        "horizons": fork["horizons"],
        "protocols": fork["protocols"],
        "command_hold": fork["command_hold"],
        "environment": fork["environment"],
        "start_time": fork["start_time"],
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_cache(cache: Path) -> dict[str, Any]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    return {
        "episodes": np.load(cache / "episodes.npy"),
        "horizons": manifest["horizons"],
        "protocols": manifest["protocols"],
        "command_hold": manifest["command_hold"],
        "environment": manifest["environment"],
        "frames_a": np.load(cache / "frames_a.npy"),
        "frames_b": np.load(cache / "frames_b.npy"),
        "rgb_a": np.load(cache / "rgb_a.npy"),
        "rgb_b": np.load(cache / "rgb_b.npy"),
        "rms_pos": np.load(cache / "rms_pos.npy"),
        "rms_vel": np.load(cache / "rms_vel.npy"),
        "actions_a": np.load(cache / "actions_a.npy"),
        "actions_b": np.load(cache / "actions_b.npy"),
        "start_time": manifest["start_time"],
    }


def _cache_matches(cache: Path, config: dict[str, Any]) -> bool:
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("horizons") == [int(value) for value in config["horizons"]]
        and manifest.get("protocols") == list(config["protocols"])
        and int(manifest.get("start_time", -1)) == int(config["start_time"])
    )


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiments" / "horizon_audit_v1.yaml").read_text(encoding="utf-8"))
    legacy = get_split("legacy_v1")
    if legacy.headline_role != "legacy_validation":
        raise RuntimeError("this audit of historical checkpoints is legacy validation, not a test set.")
    cache = ROOT / "experiments" / "caches" / "horizon_audit_legacy_v1"
    if _cache_matches(cache, config):
        print("loading fork cache", cache, flush=True)
        fork = _load_cache(cache)
    else:
        fork = _simulate(config)
    dump_actions = np.load(ROOT / "outputs" / "physics_mpm" / "actions.npy")
    train = dump_actions[: legacy.val_ids[0]].reshape(-1, 4)
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 1e-6)
    horizons = [int(value) for value in fork["horizons"]]
    protocols = list(fork["protocols"])
    horizon_index = {horizon: index for index, horizon in enumerate(horizons)}
    if horizons[0] != 1:
        raise RuntimeError("directional evolution uses horizon 1 as the reference.")
    command = "python3.13 scripts/run_horizon_audit.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed="legacy_val_multi_horizon",
        model={"checkpoints": [name for name, _path, _seed in CHECKPOINTS]},
        environment={"id": fork["environment"], "command_hold": fork["command_hold"]},
        dataset_split={
            "split_version": "legacy_v1",
            "headline_role": "legacy_validation",
            "is_test_set": False,
            "episodes": [int(fork["episodes"][0]), int(fork["episodes"][-1]) + 1],
        },
        checkpoint={"path": "ten no-crop last.pt", "note": "historical checkpoints"},
    ) as run:
        if not _cache_matches(cache, config):
            _save_cache(cache, fork)
        _save_cache(run.run_dir / "cache", fork)
        rows: list[dict[str, Any]] = []
        target_dir = run.run_dir / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, rel, seed in CHECKPOINTS:
            print("model", name, flush=True)
            payload = torch.load(ROOT / rel, map_location="cpu", weights_only=False)
            if int(payload["epoch"]) != 200 or float(payload["config"]["data"]["crop_scale_min"]) != 1.0:
                raise RuntimeError(f"{name} is not an epoch-200 no-crop checkpoint.")
            model = ACJEPA(ModelConfig(**payload["config"]["model"]))
            model.load_state_dict(payload["model"])
            model.eval()
            online_a = _encode(model, fork["frames_a"], False)
            target_a = _encode(model, fork["frames_a"], True)
            online_b = _encode(model, fork["frames_b"], False)
            target_b = _encode(model, fork["frames_b"], True)
            np.savez_compressed(
                target_dir / f"{name}.npz",
                online_a=online_a,
                target_a=target_a,
                online_b=online_b,
                target_b=target_b,
            )
            context = online_a[:, 0]
            pred_a = _rollout(model, context, _zscore(fork["actions_a"], mean, std))
            pred_b = np.stack(
                [_rollout(model, context, _zscore(fork["actions_b"][:, protocol_index], mean, std)) for protocol_index in range(len(protocols))],
                axis=1,
            )
            teacher_a = []
            teacher_b = []
            for horizon in horizons:
                teacher_a.append(_predict(model, online_a[:, horizon - 1], _zscore(fork["actions_a"][:, horizon - 1], mean, std)))
                teacher_b.append(
                    np.stack(
                        [
                            _predict(
                                model,
                                online_b[:, protocol_index, horizon - 1],
                                _zscore(fork["actions_b"][:, protocol_index, horizon - 1], mean, std),
                            )
                            for protocol_index in range(len(protocols))
                        ],
                        axis=1,
                    )
                )
            for index, episode in enumerate(fork["episodes"].tolist()):
                for protocol_index, protocol in enumerate(protocols):
                    reference = target_b[index, protocol_index, 1] - target_a[index, 1]
                    for horizon in horizons:
                        slot = horizon_index[horizon]
                        predictions = {
                            "open_loop": (pred_a[index, horizon - 1], pred_b[index, protocol_index, horizon - 1]),
                            "teacher_forced": (teacher_a[slot][index], teacher_b[slot][index, protocol_index]),
                        }
                        for mode, (hat_a, hat_b) in predictions.items():
                            scored = score_horizon_pair(
                                protocol=protocol,
                                rollout_mode=mode,
                                horizon=horizon,
                                position_a=np.zeros((1, 2)),
                                position_b=np.zeros((1, 2)),
                                velocity_a=np.zeros((1, 2)),
                                velocity_b=np.zeros((1, 2)),
                                raster_a=fork["frames_a"][index, horizon],
                                raster_b=fork["frames_b"][index, protocol_index, horizon],
                                target_a=target_a[index, horizon],
                                target_b=target_b[index, protocol_index, horizon],
                                pred_a=hat_a,
                                pred_b=hat_b,
                                target_delta_h1=None if horizon == 1 else reference,
                                rgb_a=fork["rgb_a"][index, slot],
                                rgb_b=fork["rgb_b"][index, protocol_index, slot],
                                norm_tau=float(config["norm_tau"]),
                                epsilon=float(config["epsilon"]),
                            )
                            scored["rms_pos"] = float(fork["rms_pos"][index, protocol_index, horizon - 1])
                            scored["rms_vel"] = float(fork["rms_vel"][index, protocol_index, horizon - 1])
                            scored["physically_distinct_observationally_identical"] = bool(
                                scored["observationally_ambiguous"] and scored["rms_pos"] > 0.0
                            )
                            scored.update(
                                {
                                    "model": name,
                                    "seed": int(seed),
                                    "objective": "mse" if name.startswith("mse_") else "vicreg",
                                    "condition_on_action": "noaction" not in name,
                                    "environment": fork["environment"],
                                    "episode": int(episode),
                                    "t": int(fork["start_time"]),
                                    "split_version": "legacy_v1",
                                    "is_test_set": False,
                                }
                            )
                            rows.append(scored)
            del model, online_a, target_a, online_b, target_b, pred_a, pred_b, teacher_a, teacher_b
        fieldnames = list(rows[0].keys())
        with (run.run_dir / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row[key]) for key in fieldnames})
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows, schema=_arrow_schema(fieldnames))
            pq.write_table(table, run.run_dir / "queries.parquet")
            parquet_status = "written"
        except ImportError:
            parquet_status = "pyarrow_unavailable"
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row["model"], row["protocol"], int(row["horizon"]), row["rollout_mode"])].append(row)
        slices = []
        bootstraps = []
        rng = np.random.default_rng(int(config["bootstrap_seed"]))
        for (model, protocol, horizon, mode), group in groups.items():
            summary = summarize_labeled(group)
            summary["model"] = model
            slices.append(summary)
            for metric, include_key in (("cosine", "cosine"), ("ranking", "counterfactual_ranking_top1")):
                if metric == "cosine":
                    include = [row["cosine_status"] == "ok" and row["cosine"] is not None for row in group]
                    values = [row["cosine"] for row in group]
                else:
                    include = [
                        row["counterfactual_ranking_status"] in {"scored", "distance_tie"} and row["counterfactual_ranking_top1"] is not None
                        for row in group
                    ]
                    values = [row["counterfactual_ranking_top1"] for row in group]
                boot = bootstrap_labeled(
                    [int(row["episode"]) for row in group],
                    values,
                    include,
                    n_resamples=int(config["bootstrap_resamples"]),
                    rng=rng,
                )
                bootstraps.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "horizon": horizon,
                        "rollout_mode": mode,
                        "metric": metric,
                        **boot,
                    }
                )
        recorded = {}
        for row in rows:
            if row["rollout_mode"] != "open_loop":
                continue
            recorded.setdefault((row["model"], row["episode"], row["horizon"]), []).append(float(row["rollout_mse_recorded"]))
        rollout_agrees = all(max(values) - min(values) <= 1e-8 for values in recorded.values())
        summary_payload = {
            "split_version": "legacy_v1",
            "is_test_set": False,
            "n_rows": len(rows),
            "recorded_rollout_identical_across_protocols": rollout_agrees,
            "parquet_status": parquet_status,
            "slices": slices,
            "bootstrap": bootstraps,
            "learned_model_evaluated": True,
        }
        (run.run_dir / "summary.json").write_text(json.dumps(to_jsonable(summary_payload), indent=2) + "\n", encoding="utf-8")
        with (run.run_dir / "summary_slices.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(slices[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in slices:
                writer.writerow({key: _csv_value(item[key]) for key in fields})
        plot_horizon_audit(rows, ROOT / "figures" / "horizon_audit")
        run.complete(
            to_jsonable(
                {
                    "n_rows": len(rows),
                    "parquet_status": parquet_status,
                    "is_test_set": False,
                    "recorded_rollout_identical_across_protocols": rollout_agrees,
                    "slices": [
                        {
                            "model": item["model"],
                            "protocol": item["protocol"],
                            "rollout_mode": item["rollout_mode"],
                            "horizon": item["horizon"],
                            "ranking_top1": item["ranking_top1"],
                            "n_ranking_scored": item["n_ranking_scored"],
                            "n_ranking_ambiguous": item["n_ranking_ambiguous"],
                            "cosine_median": item["cosine_median"],
                            "n_cosine_defined": item["n_cosine_defined"],
                            "n_cosine_undefined": item["n_cosine_undefined"],
                            "rollout_mse_recorded_median": item["rollout_mse_recorded_median"],
                            "rms_pos_median": item["rms_pos_median"],
                            "fraction_affected_median": item["fraction_affected_median"],
                        }
                        for item in slices
                    ],
                }
            )
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
