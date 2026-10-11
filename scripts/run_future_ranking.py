#!/usr/bin/env python3
"""Counterfactual future ranking for the ten legacy no-crop checkpoints.

    python3.13 scripts/run_future_ranking.py

Each candidate action is held from the shared state. Episodes 1800-1999 are
legacy validation, not a test set. Queries whose rasters match another
candidate stay in the table and are not counted as retrieval misses.
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
from src.future_ranking_plots import plot_future_ranking
from src.metrics.counterfactual_direction import assign_quintiles
from src.metrics.future_ranking import (
    DISTANCE_KINDS,
    MODES,
    STATUS_AMBIGUOUS,
    STATUS_SCORED,
    STATUS_SIM,
    STATUS_TIE,
    bootstrap_top1,
    candidate_actions,
    control_predictions,
    derangement,
    physical_bin,
    rank_candidates,
    visibility_bin,
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
METRIC_FIELDS = (
    "status",
    "top1",
    "topk",
    "reciprocal_rank",
    "margin",
    "pairwise_rate",
    "pairwise_n",
    "pairwise_n_ambiguous",
    "pairwise_n_tie",
)


class _Bucket:
    def __init__(self) -> None:
        self.n_total = 0
        self.n_scored = 0
        self.n_hit = 0
        self.n_ambiguous = 0
        self.n_phys = 0
        self.n_undefined = 0
        self.n_sim = 0
        self.n_tie = 0
        self.n_topk = 0
        self.n_topk_hit = 0
        self.rr_sum = 0.0
        self.margins: list[float] = []
        self.pair_correct = 0
        self.pair_tie = 0
        self.pair_incorrect = 0
        self.pair_ambiguous = 0
        self.pair_undefined = 0

    def add(self, query: dict[str, Any]) -> None:
        self.n_total += 1
        if query["physically_distinct_observationally_identical"]:
            self.n_phys += 1
        status = query["status"]
        if status == STATUS_AMBIGUOUS:
            self.n_ambiguous += 1
        elif status == STATUS_UNDEFINED:
            self.n_undefined += 1
        elif status == STATUS_SIM:
            self.n_sim += 1
        elif status in {STATUS_SCORED, STATUS_TIE}:
            self.n_scored += 1
            self.n_hit += int(query["top1"] == 1.0)
            self.n_tie += int(status == STATUS_TIE)
            self.rr_sum += float(query["reciprocal_rank"])
            self.margins.append(float(query["margin"]))
            if query["topk"] is not None:
                self.n_topk += 1
                self.n_topk_hit += int(query["topk"] == 1.0)
        self.pair_correct += int(query["pairwise_n_correct"])
        self.pair_tie += int(query["pairwise_n_tie"])
        self.pair_incorrect += int(query["pairwise_n_incorrect"])
        self.pair_ambiguous += int(query["pairwise_n_ambiguous"])
        self.pair_undefined += int(query["pairwise_n_undefined"])

    def summary(self, n_candidates: int, topk: int) -> dict[str, Any]:
        pair_scored = self.pair_correct + self.pair_tie + self.pair_incorrect
        margins = np.asarray(self.margins, dtype=np.float64)
        out = {
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_hit": self.n_hit,
            "n_ambiguous": self.n_ambiguous,
            "n_physically_distinct_observationally_identical": self.n_phys,
            "n_undefined": self.n_undefined,
            "n_simulation_nonfinite": self.n_sim,
            "n_tie": self.n_tie,
            "n_candidates": n_candidates,
            "chance_top1": 1.0 / float(n_candidates),
            "top1": None if self.n_scored == 0 else float(self.n_hit / self.n_scored),
            "topk": None if self.n_topk == 0 else float(self.n_topk_hit / self.n_topk),
            "topk_k": topk,
            "mean_reciprocal_rank": None if self.n_scored == 0 else float(self.rr_sum / self.n_scored),
            "margin_mean": None if margins.size == 0 else float(margins.mean()),
            "margin_median": None if margins.size == 0 else float(np.median(margins)),
            "pairwise_n_scored": pair_scored,
            "pairwise_n_correct": self.pair_correct,
            "pairwise_n_tie": self.pair_tie,
            "pairwise_n_incorrect": self.pair_incorrect,
            "pairwise_n_ambiguous": self.pair_ambiguous,
            "pairwise_n_undefined": self.pair_undefined,
            "pairwise_probability": None if pair_scored == 0 else float(self.pair_correct / pair_scored),
        }
        for percentile in (10, 90):
            out[f"margin_p{percentile}"] = None if margins.size == 0 else float(np.percentile(margins, percentile))
        return out


def _affected(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.any(left != right, axis=0)))


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    gap = np.linalg.norm(np.asarray(right) - np.asarray(left), axis=-1)
    return float(np.sqrt(np.mean(np.square(gap))))


def _obs_equal(frames: np.ndarray) -> np.ndarray:
    k = int(frames.shape[0])
    equal = np.eye(k, dtype=bool)
    for i in range(k):
        for j in range(i + 1, k):
            same = bool(np.array_equal(frames[i], frames[j]))
            equal[i, j] = same
            equal[j, i] = same
    return equal


def _n_unique(equal: np.ndarray) -> int:
    seen = np.zeros(equal.shape[0], dtype=bool)
    count = 0
    for index in range(equal.shape[0]):
        if seen[index]:
            continue
        count += 1
        seen[equal[index]] = True
    return count


def _roll_held(sim: MLSMPM2D, snap: dict[str, Any], action: np.ndarray, horizons: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    sim.restore(snap)
    frames: dict[int, np.ndarray] = {}
    positions: dict[int, np.ndarray] = {}
    wanted = set(horizons)
    for step in range(1, max(horizons) + 1):
        apply_action_frames(sim, action, 1)
        if not np.isfinite(sim.x).all() or not np.isfinite(sim.v).all():
            return None
        if step in wanted:
            frames[step] = render_rgb(sim.x, sim.v, sim.cfg.img_size, sim.cfg.render_sigma)
            positions[step] = sim.x.copy()
    if set(frames) != wanted:
        return None
    return frames, positions


def _simulate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    legacy = get_split("legacy_v1")
    episodes = list(range(int(legacy.val_ids[0]), int(legacy.val_ids[-1]) + 1))
    dump_actions = np.load(ROOT / "outputs" / "physics_mpm" / "actions.npy")
    train = dump_actions[: legacy.val_ids[0]].reshape(-1, 4)
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 1e-6)
    horizons = [int(value) for value in config["horizons"]]
    start_times = [int(value) for value in config["start_times"]]
    states: list[dict[str, Any]] = []
    for env_index, rel in enumerate(config["environments"]):
        spec = environment_from_dict(yaml.safe_load((ROOT / rel).read_text(encoding="utf-8")))
        print("simulate", spec.environment_id, flush=True)
        sim = MLSMPM2D(spec.mpm)
        replays = []
        recorded = []
        for episode in episodes:
            print(f"replay {spec.environment_id} {episode}", flush=True)
            rec = replay_episode(
                int(episode),
                dump_seed=0,
                episode_length=16,
                mpm=spec.mpm,
                hold=spec.command_hold,
            )
            if spec.environment_id == "mpm_legacy" and not np.allclose(rec["actions"], dump_actions[episode]):
                raise RuntimeError(f"legacy replay actions differ from the dump at episode {episode}.")
            replays.append(rec)
            recorded.append(rec["actions"])
        recorded_arr = np.stack(recorded)
        partner_perms = {}
        for start in start_times:
            partner_perms[start] = derangement(
                len(episodes),
                np.random.default_rng(int(config["shuffle_seed"]) + 10007 * start + 100003 * env_index),
            )
        for index, rec in enumerate(replays):
            for start in start_times:
                partner = int(partner_perms[start][index])
                actions, roles = candidate_actions(
                    recorded_arr[index, start],
                    recorded_arr[partner, start],
                    force_half_scale=float(config["force_half_scale"]),
                )
                rolled = [
                    _roll_held(sim, rec["snapshots"][start], action, horizons)
                    for action in actions
                ]
                sim_ok = all(item is not None for item in rolled)
                if not sim_ok:
                    frame_stack = np.zeros((len(horizons), actions.shape[0], 3, 64, 64), dtype=np.uint8)
                    rms = np.zeros((len(horizons), actions.shape[0], actions.shape[0]), dtype=np.float64)
                else:
                    frame_stack = np.stack(
                        [np.stack([item[0][horizon] for item in rolled], axis=0) for horizon in horizons],
                        axis=0,
                    )
                    rms = np.zeros((len(horizons), actions.shape[0], actions.shape[0]), dtype=np.float64)
                    for h_index, horizon in enumerate(horizons):
                        positions = [item[1][horizon] for item in rolled]
                        for left in range(actions.shape[0]):
                            for right in range(left + 1, actions.shape[0]):
                                gap = _rms(positions[left], positions[right])
                                rms[h_index, left, right] = gap
                                rms[h_index, right, left] = gap
                states.append(
                    {
                        "environment": spec.environment_id,
                        "env_index": env_index,
                        "episode": int(episodes[index]),
                        "t": int(start),
                        "shuffle_partner_episode": int(episodes[partner]),
                        "roles": roles,
                        "actions": actions,
                        "actions_norm": ((actions - mean) / std).astype(np.float32),
                        "context": rec["frames"][start],
                        "sim_ok": sim_ok,
                        "frames": frame_stack,
                        "physical_rms": rms,
                    }
                )
            replays[index] = None
    edges: dict[str, list[float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[state["environment"]].append(state)
    for environment, group in grouped.items():
        values = []
        owners = []
        for state in group:
            k = int(state["actions"].shape[0])
            state["action_l2_max"] = []
            for query in range(k):
                gaps = [
                    float(np.linalg.norm(state["actions"][query] - state["actions"][other]))
                    for other in range(k)
                    if other != query
                ]
                values.append(max(gaps))
                owners.append((state, query))
                state["action_l2_max"].append(max(gaps))
        labels, edge = assign_quintiles(values)
        edges[environment] = edge
        for state in group:
            state["quintile"] = [0] * len(state["action_l2_max"])
        for (state, query), label in zip(owners, labels):
            state["quintile"][query] = int(label)
    return states, edges


def _encode(model: ACJEPA, frames: np.ndarray, target: bool) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], 64):
            batch = torch.from_numpy(np.ascontiguousarray(frames[start : start + 64], dtype=np.float32) / 255.0)
            outputs.append(model.encode(batch, target=target).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _rollout(model: ACJEPA, latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, latent.shape[0], 32):
            z = torch.from_numpy(np.ascontiguousarray(latent[start : start + 32], dtype=np.float32))
            act = torch.from_numpy(np.ascontiguousarray(actions[start : start + 32], dtype=np.float32))
            outputs.append(model.rollout(z, act).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _blank_query(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "physically_distinct_observationally_identical": False,
        "top1": None,
        "topk": None,
        "reciprocal_rank": None,
        "margin": None,
        "pairwise_n": 0,
        "pairwise_n_ambiguous": 0,
        "pairwise_n_undefined": 0,
        "pairwise_n_correct": 0,
        "pairwise_n_tie": 0,
        "pairwise_n_incorrect": 0,
        "pairwise_rate": None,
    }


def _arrow_schema(fieldnames: list[str]):
    import pyarrow as pa

    bools = {
        "condition_on_action",
        "is_test_set",
        "correct_observation_unique",
        "duplicate_action",
        "sim_ok",
        "physically_distinct_observationally_identical",
    }
    ints = {
        "seed",
        "episode",
        "t",
        "horizon",
        "query_index",
        "shuffle_partner_episode",
        "n_candidates",
        "n_unique_observations",
        "action_l2_quintile",
    }
    floats = {"action_l2_max", "fraction_affected_max", "physical_rms_max"}
    fields = []
    for name in fieldnames:
        if name in bools:
            typ = pa.bool_()
        elif name in ints or name.endswith(("_pairwise_n", "_pairwise_n_ambiguous", "_pairwise_n_tie")):
            typ = pa.int64()
        elif name in floats or name.endswith(("_top1", "_topk", "_reciprocal_rank", "_margin", "_pairwise_rate")):
            typ = pa.float64()
        else:
            typ = pa.string()
        fields.append(pa.field(name, typ, nullable=True))
    return pa.schema(fields)


def _row_fieldnames() -> list[str]:
    meta = [
        "model",
        "seed",
        "objective",
        "condition_on_action",
        "environment",
        "episode",
        "t",
        "horizon",
        "query_index",
        "query_role",
        "shuffle_partner_episode",
        "split_version",
        "is_test_set",
        "n_candidates",
        "n_unique_observations",
        "correct_observation_unique",
        "duplicate_action",
        "sim_ok",
        "action_l2_max",
        "fraction_affected_max",
        "physical_rms_max",
        "visibility_bin",
        "physical_bin",
        "action_l2_quintile",
        "physically_distinct_observationally_identical",
    ]
    for mode in MODES:
        for kind in DISTANCE_KINDS:
            for field in METRIC_FIELDS:
                meta.append(f"{mode}_{kind}_{field}")
    return meta


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def main() -> None:
    config_path = ROOT / "configs" / "experiments" / "future_ranking_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    legacy = get_split("legacy_v1")
    if legacy.headline_role != "legacy_validation":
        raise RuntimeError("future ranking of these checkpoints is legacy validation, not a test set.")
    horizons = [int(value) for value in config["horizons"]]
    states, edges = _simulate(config)
    print(f"states {len(states)} sim_ok {sum(bool(state['sim_ok']) for state in states)}", flush=True)
    k_candidates = int(states[0]["actions"].shape[0])
    fieldnames = _row_fieldnames()
    buckets: dict[tuple, _Bucket] = defaultdict(_Bucket)
    boot: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    episodes_by_env: dict[str, set[int]] = defaultdict(set)
    for state in states:
        episodes_by_env[state["environment"]].add(int(state["episode"]))
    reference_rows = []
    l2_hits: dict[tuple, int] = defaultdict(int)
    mse_hits: dict[tuple, int] = defaultdict(int)
    command = "python3.13 scripts/run_future_ranking.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed="legacy_val_held_action",
        model={"checkpoints": [name for name, _path, _seed in CHECKPOINTS]},
        environment={"ids": [Path(path).stem for path in config["environments"]]},
        dataset_split={
            "split_version": "legacy_v1",
            "headline_role": "legacy_validation",
            "is_test_set": False,
            "episodes": [legacy.val_ids[0], legacy.val_ids[-1] + 1],
        },
        checkpoint={"path": "ten no-crop last.pt", "note": "historical checkpoints"},
    ) as run:
        table_path = run.run_dir / "queries.csv"
        parquet_path = run.run_dir / "queries.parquet"
        csv_handle = table_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        parquet_writer = None
        arrow_schema = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            arrow_schema = _arrow_schema(fieldnames)
        except ImportError:
            pa = None
            pq = None
        contexts = np.stack([state["context"] for state in states])
        action_norm = np.stack([state["actions_norm"] for state in states])
        max_horizon = max(horizons)
        n_states = len(states)
        for name, rel, seed in CHECKPOINTS:
            print("model", name, flush=True)
            payload = torch.load(ROOT / rel, map_location="cpu", weights_only=False)
            if int(payload["epoch"]) != 200 or float(payload["config"]["data"]["crop_scale_min"]) != 1.0:
                raise RuntimeError(f"{name} is not an epoch-200 no-crop checkpoint.")
            model = ACJEPA(ModelConfig(**payload["config"]["model"]))
            model.load_state_dict(payload["model"])
            model.eval()
            latent = _encode(model, contexts, False)
            repeated = np.repeat(action_norm[:, :, None, :], max_horizon, axis=2).reshape(n_states * k_candidates, max_horizon, 4)
            predicted = _rollout(model, np.repeat(latent, k_candidates, axis=0), repeated)
            predicted = predicted.reshape(n_states, k_candidates, max_horizon, -1)
            model_rows = []
            for h_index, horizon in enumerate(horizons):
                future_frames = np.stack([state["frames"][h_index] for state in states])
                encoded = _encode(model, future_frames.reshape(-1, 3, 64, 64), True).reshape(n_states, k_candidates, -1)
                variance = {}
                for environment in {state["environment"] for state in states}:
                        mask = np.asarray([state["environment"] == environment and state["sim_ok"] for state in states])
                        recorded_latents = encoded[mask, 0]
                        variance[environment] = None if recorded_latents.shape[0] < 2 else float(recorded_latents.var(axis=0).mean())
                        reference_rows.append(
                            {
                                "model": name,
                                "environment": environment,
                                "horizon": horizon,
                                "reference_variance": variance[environment],
                                "source": "legacy_validation_recorded_target_embeddings",
                                "used_for_selection": False,
                            }
                        )
                for state_index, state in enumerate(states):
                    if not state["sim_ok"]:
                        equal = np.eye(k_candidates, dtype=bool)
                        rms = np.zeros((k_candidates, k_candidates))
                        controls = None
                    else:
                        equal = _obs_equal(state["frames"][h_index])
                        rms = state["physical_rms"][h_index]
                        rng = np.random.default_rng(
                            int(config["prediction_shuffle_seed"])
                            + int(state["episode"])
                            + 10007 * int(state["t"])
                            + 17 * int(horizon)
                            + 100003 * int(state["env_index"])
                        )
                        controls = control_predictions(
                            predicted[state_index, :, horizon - 1],
                            encoded[state_index],
                            rng,
                        )
                    scored_by = {}
                    for mode in MODES:
                        for kind in DISTANCE_KINDS:
                            if controls is None:
                                queries = [_blank_query(STATUS_SIM) for _ in range(k_candidates)]
                            else:
                                queries = rank_candidates(
                                    controls[mode],
                                    encoded[state_index],
                                    equal,
                                    rms,
                                    distance=kind,
                                    ref_variance=variance[state["environment"]],
                                    topk=int(config["topk"]),
                                    norm_tau=float(config["norm_tau"]),
                                    physical_tau=float(config["physical_tau"]),
                                    simulation_ok=True,
                                )["queries"]
                            scored_by[(mode, kind)] = queries
                    for query_index in range(k_candidates):
                        if state["sim_ok"]:
                            others = [j for j in range(k_candidates) if j != query_index]
                            affected = max(_affected(state["frames"][h_index][query_index], state["frames"][h_index][j]) for j in others)
                            physical = float(np.max(state["physical_rms"][h_index][query_index, others]))
                            unique = not any(bool(equal[query_index, j]) for j in others)
                            vis = visibility_bin(unique, affected, high_visibility=float(config["high_visibility_affected"]))
                            phys = physical_bin(physical, small_rms=float(config["physical_small_rms"]))
                            n_unique = _n_unique(equal)
                        else:
                            affected = None
                            physical = None
                            unique = False
                            vis = "simulation_nonfinite"
                            phys = "simulation_nonfinite"
                            n_unique = None
                            equal = np.eye(k_candidates, dtype=bool)
                        duplicate = any(
                            query_index != other and np.allclose(state["actions"][query_index], state["actions"][other])
                            for other in range(k_candidates)
                        )
                        flat = {
                            "model": name,
                            "seed": int(seed),
                            "objective": "mse" if name.startswith("mse_") else "vicreg",
                            "condition_on_action": "noaction" not in name,
                            "environment": state["environment"],
                            "episode": int(state["episode"]),
                            "t": int(state["t"]),
                            "horizon": int(horizon),
                            "query_index": query_index,
                            "query_role": state["roles"][query_index],
                            "shuffle_partner_episode": int(state["shuffle_partner_episode"]),
                            "split_version": "legacy_v1",
                            "is_test_set": False,
                            "n_candidates": k_candidates,
                            "n_unique_observations": n_unique,
                            "correct_observation_unique": bool(unique),
                            "duplicate_action": bool(duplicate),
                            "sim_ok": bool(state["sim_ok"]),
                            "action_l2_max": float(state["action_l2_max"][query_index]),
                            "fraction_affected_max": affected,
                            "physical_rms_max": physical,
                            "visibility_bin": vis,
                            "physical_bin": phys,
                            "action_l2_quintile": int(state["quintile"][query_index]),
                            "physically_distinct_observationally_identical": bool(
                                scored_by[(MODES[0], DISTANCE_KINDS[0])][query_index][
                                    "physically_distinct_observationally_identical"
                                ]
                            ),
                        }
                        for mode in MODES:
                            for kind in DISTANCE_KINDS:
                                query = scored_by[(mode, kind)][query_index]
                                prefix = f"{mode}_{kind}"
                                for field in METRIC_FIELDS:
                                    flat[f"{prefix}_{field}"] = query[field]
                                slice_keys = (
                                    ("all", "all", "all", "all"),
                                    ("visibility", vis, "all", "all"),
                                    ("physical", "all", phys, "all"),
                                    ("action_distance", "all", "all", str(state["quintile"][query_index])),
                                )
                                for group, vis_label, phys_label, quintile in slice_keys:
                                    key = (name, state["environment"], horizon, kind, mode, group, vis_label, phys_label, quintile)
                                    buckets[key].add(query)
                                if query["status"] in {STATUS_SCORED, STATUS_TIE} and query["top1"] is not None:
                                    boot[(name, state["environment"], horizon, kind, mode)][int(state["episode"])].append(float(query["top1"]))
                                if mode == "model" and kind == "mse" and query["top1"] == 1.0:
                                    mse_hits[(name, state["environment"], horizon)] += 1
                                if mode == "model" and kind == "l2" and query["top1"] == 1.0:
                                    l2_hits[(name, state["environment"], horizon)] += 1
                        model_rows.append(flat)
                        writer.writerow({key: _csv_value(flat[key]) for key in fieldnames})
            if pq is not None:
                table = pa.Table.from_pylist(model_rows, schema=arrow_schema)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(parquet_path, arrow_schema)
                parquet_writer.write_table(table)
            del model_rows
        csv_handle.close()
        if parquet_writer is not None:
            parquet_writer.close()
            parquet_status = "written"
        else:
            parquet_status = "pyarrow_unavailable"
        slices = []
        for key, bucket in buckets.items():
            model, environment, horizon, kind, mode, group, vis_label, phys_label, quintile = key
            payload = {
                "model": model,
                "environment": environment,
                "horizon": horizon,
                "distance": kind,
                "mode": mode,
                "group": group,
                "visibility_bin": vis_label,
                "physical_bin": phys_label,
                "action_l2_quintile": quintile,
            }
            payload.update(bucket.summary(k_candidates, int(config["topk"])))
            slices.append(payload)
        bootstraps = []
        rng = np.random.default_rng(int(config["bootstrap_seed"]))
        for key, by_episode in boot.items():
            model, environment, horizon, kind, mode = key
            episode_ids = []
            top1_values = []
            statuses = []
            for episode, values in by_episode.items():
                for value in values:
                    episode_ids.append(episode)
                    top1_values.append(value)
                    statuses.append(STATUS_SCORED)
            missing = sorted(episodes_by_env[environment] - set(by_episode))
            for episode in missing:
                episode_ids.append(episode)
                top1_values.append(None)
                statuses.append(STATUS_AMBIGUOUS)
            summary = {
                "model": model,
                "environment": environment,
                "horizon": horizon,
                "distance": kind,
                "mode": mode,
            }
            if episode_ids:
                summary.update(
                    bootstrap_top1(
                        episode_ids,
                        top1_values,
                        statuses,
                        n_resamples=int(config["bootstrap_resamples"]),
                        rng=rng,
                    )
                )
            bootstraps.append(summary)
        agree = l2_hits == mse_hits
        headline = [
            item
            for item in slices
            if item["group"] == "all" and item["distance"] == "mse" and item["mode"] in {"model", "shuffled_predictions", "oracle_targets", "constant_prediction"}
        ]
        summary = {
            "split_version": "legacy_v1",
            "is_test_set": False,
            "n_states": len(states),
            "n_query_rows_per_model": len(states) * len(horizons) * k_candidates,
            "action_l2_quintile_edges": edges,
            "reference_variance": reference_rows,
            "l2_mse_top1_hits_agree": agree,
            "parquet_status": parquet_status,
            "slices": slices,
            "bootstrap": bootstraps,
            "learned_model_evaluated": True,
        }
        (run.run_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
        slice_path = run.run_dir / "summary_slices.csv"
        with slice_path.open("w", newline="", encoding="utf-8") as handle:
            slice_fields = list(slices[0].keys())
            slice_writer = csv.DictWriter(handle, fieldnames=slice_fields)
            slice_writer.writeheader()
            for item in slices:
                slice_writer.writerow({key: _csv_value(item[key]) for key in slice_fields})
        if parquet_status == "written":
            plot_rows = pq.read_table(parquet_path).to_pylist()
            plot_future_ranking(plot_rows, ROOT / "figures" / "future_ranking")
        run.complete(
            to_jsonable(
                {
                    "n_states": len(states),
                    "n_slices": len(slices),
                    "parquet_status": parquet_status,
                    "is_test_set": False,
                    "l2_mse_top1_hits_agree": agree,
                    "mse_all": [
                        {
                            "model": item["model"],
                            "environment": item["environment"],
                            "horizon": item["horizon"],
                            "mode": item["mode"],
                            "top1": item["top1"],
                            "n_scored": item["n_scored"],
                            "n_ambiguous": item["n_ambiguous"],
                            "n_physically_distinct_observationally_identical": item[
                                "n_physically_distinct_observationally_identical"
                            ],
                            "pairwise_probability": item["pairwise_probability"],
                        }
                        for item in headline
                    ],
                }
            )
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
