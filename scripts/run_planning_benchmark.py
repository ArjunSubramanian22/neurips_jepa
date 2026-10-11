"""Run the closed-loop planning benchmark on fresh simulator goals.

    python3.13 scripts/run_planning_benchmark.py

The CEM settings are the predeclared configuration. Learned-model highlight
uses the validation seeds only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MPMConfig, apply_action_frames, render_rgb
from src.metrics.planning_benchmark import (
    POOR_ACTION_FAMILY,
    PRINCIPAL_CHECKPOINTS,
    aggregate_controller,
    choose_highlight,
    population_action_moments,
    record_episode,
)
from src.mpc import heuristic_push, learned_propose, mean_force_action, oracle_propose, random_action, run_closed_loop
from src.planning_tasks import SUCCESS_FRACTION, make_reachable_goal, task_distance
from src.registry import to_jsonable
from src.train import load_model_from_checkpoint


class _SimWorld:
    def __init__(self, sim: object) -> None:
        self.sim = sim

    def snapshot(self) -> dict:
        return self.sim.snapshot()

    def restore(self, snap: dict) -> None:
        self.sim.restore(snap)

    def step_action(self, action: np.ndarray) -> None:
        apply_action_frames(self.sim, np.asarray(action, dtype=np.float32), 1)


def _cem(protocol: dict) -> dict:
    block = protocol["cem"]
    return {
        "mean": np.asarray(block["mean"], dtype=np.float64),
        "std": np.asarray(block["std"], dtype=np.float64),
        "horizon": int(block["horizon"]),
        "population": int(block["population"]),
        "n_iterations": int(block["n_iterations"]),
        "elite_fraction": float(block["elite_fraction"]),
        "action_dim": int(block["action_dim"]),
        "lower": np.asarray(block["lower"], dtype=np.float64),
        "upper": np.asarray(block["upper"], dtype=np.float64),
        "temporal_smooth": float(block["temporal_smooth"]),
    }


def _encode(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(frame)).float()
    if float(tensor.max()) > 1.5:
        tensor /= 255.0
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        return model.encode(tensor.to(device)).squeeze(0).detach().cpu().numpy()


def _rollout_fn(model: torch.nn.Module, device: torch.device):
    def rollout(latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(np.asarray(latent), dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(actions.shape[0], -1)
        command = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
        with torch.no_grad():
            return model.rollout(state, command).detach().cpu().numpy()

    return rollout


def _run_episode(sim, spec: dict, propose, max_steps: int, *, keep_frames: bool) -> tuple[dict, dict | None]:
    world = _SimWorld(sim)
    frames = [spec["start_frame"]] if keep_frames else None
    cfg = sim.cfg

    def step(action: np.ndarray) -> None:
        world.step_action(action)
        if frames is not None:
            frames.append(render_rgb(sim.x, sim.v, cfg.img_size, cfg.render_sigma))

    initial = float(spec["initial_distance"])
    threshold = SUCCESS_FRACTION * initial
    result = run_closed_loop(
        propose,
        step,
        lambda: task_distance(spec["task"], sim.x, spec["goal_positions"], spec["region"]),
        initial_distance=initial,
        max_steps=max_steps,
        success_distance=threshold,
        ambiguous=bool(spec["observationally_ambiguous"]),
    )
    trajectory = None
    if frames is not None:
        trajectory = {
            "frames": np.stack(frames, axis=0),
            "actions": np.stack(result["actions"], axis=0) if result["actions"] else np.zeros((0, 4)),
            "distances": np.asarray(result["distances"], dtype=np.float64),
            "status": result["status"],
        }
    return result, trajectory


def _summarize(records: list[dict], seeds: list[int]) -> dict[str, dict]:
    chosen = {int(seed) for seed in seeds}
    grouped: dict[str, list[dict]] = {}
    for row in records:
        if int(row["seed"]) not in chosen:
            continue
        grouped.setdefault(str(row["controller"]), []).append(row)
    return {name: aggregate_controller(rows) for name, rows in sorted(grouped.items())}


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs/experiments/planning_benchmark_v1.yaml").read_text(encoding="utf-8"))
    if protocol.get("select_on_test"):
        raise SystemExit("refusing a planning protocol that selects on test.")
    cem = _cem(protocol)
    max_steps = int(protocol["max_env_steps"])
    actions = np.load(ROOT / "outputs/physics_mpm/actions.npy")
    if actions.shape[0] != 2000:
        raise SystemExit(f"expected 2000 episodes of actions, found {actions.shape[0]}.")
    n_train = 1800
    action_mean, action_std = population_action_moments(actions, n_train)
    device = torch.device("cpu")
    mpm = MPMConfig()
    records: list[dict] = []
    qualitative: dict[str, dict] = {}
    qualitative_task = protocol["qualitative"]["task"]
    qualitative_seed = int(protocol["qualitative"]["seed"])
    nonlearned = ("random", "mean_action", "heuristic", "oracle")

    for task_index, task in enumerate(protocol["tasks"]):
        for seed in protocol["seeds"]:
            spec = make_reachable_goal(task, seed=int(seed), scripted_steps=max_steps, mpm=mpm)
            start = spec["sim"].snapshot()
            keep = task == qualitative_task and int(seed) == qualitative_seed
            rng_key = 10_000 + task_index * 100 + int(seed)
            random_rng = np.random.default_rng(rng_key + 1)
            oracle_rng = np.random.default_rng(rng_key)
            controllers: list[tuple[str, object]] = []

            def random_propose(_warm, rng=random_rng):
                action = random_action(rng, cem["lower"], cem["upper"])
                plan = np.repeat(action.reshape(1, -1), cem["horizon"], axis=0)
                return action, plan, 0.0

            def mean_propose(_warm):
                action = mean_force_action(cem["action_dim"])
                return action, np.repeat(action.reshape(1, -1), cem["horizon"], axis=0), 0.0

            def heuristic_propose(_warm, spec=spec):
                action = heuristic_push(spec["sim"].x, spec["goal_positions"])
                return action, np.repeat(action.reshape(1, -1), cem["horizon"], axis=0), 0.0

            def oracle_call(warm, spec=spec, rng=oracle_rng):
                world = _SimWorld(spec["sim"])
                return oracle_propose(
                    world,
                    lambda item: task_distance(task, item.sim.x, spec["goal_positions"], spec["region"]),
                    cem,
                    rng,
                    warm,
                )

            controllers.extend(
                [
                    ("random", random_propose),
                    ("mean_action", mean_propose),
                    ("heuristic", heuristic_propose),
                    ("oracle", oracle_call),
                ]
            )
            for name, propose in controllers:
                spec["sim"].restore(start)
                result, trajectory = _run_episode(spec["sim"], spec, propose, max_steps, keep_frames=keep)
                row = record_episode(name, task, int(seed), result)
                row["observationally_ambiguous"] = bool(spec["observationally_ambiguous"])
                records.append(row)
                if trajectory is not None:
                    qualitative[name] = trajectory
                print(f"{name} {task} seed {seed}: {row['status']}", flush=True)

            for item in PRINCIPAL_CHECKPOINTS:
                path = ROOT / item["path"]
                spec["sim"].restore(start)
                if not path.is_file():
                    records.append(
                        {
                            "controller": item["name"],
                            "task": task,
                            "seed": int(seed),
                            "status": "checkpoint_missing",
                            "scored": False,
                            "success": None,
                            "final_distance": None,
                            "initial_distance": float(spec["initial_distance"]),
                            "cumulative_cost": None,
                            "actions_to_success": None,
                            "planning_runtime_s": None,
                            "error": None,
                            "family": item["family"],
                        }
                    )
                    continue
                model, _loaded = load_model_from_checkpoint(path, device)
                model.eval()
                goal_latent = _encode(model, spec["goal_frame"], device)
                rollout = _rollout_fn(model, device)
                model_rng = np.random.default_rng(10_000 + task_index * 100 + int(seed))

                def learned_call(warm, model=model, goal_latent=goal_latent, rollout=rollout, model_rng=model_rng, spec=spec):
                    latent = _encode(model, render_rgb(spec["sim"].x, spec["sim"].v, mpm.img_size, mpm.render_sigma), device)
                    return learned_propose(
                        latent,
                        goal_latent,
                        rollout,
                        cem,
                        model_rng,
                        warm,
                        action_mean=action_mean,
                        action_std=action_std,
                    )

                result, trajectory = _run_episode(spec["sim"], spec, learned_call, max_steps, keep_frames=keep)
                row = record_episode(item["name"], task, int(seed), result)
                row["family"] = item["family"]
                row["observationally_ambiguous"] = bool(spec["observationally_ambiguous"])
                records.append(row)
                if trajectory is not None:
                    qualitative[item["name"]] = trajectory
                print(f"{item['name']} {task} seed {seed}: {row['status']}", flush=True)
                del model

    eligible = [item["name"] for item in PRINCIPAL_CHECKPOINTS if item["family"] != POOR_ACTION_FAMILY]
    highlight = choose_highlight(records, protocol["validation_seeds"], eligible)
    present = []
    for item in PRINCIPAL_CHECKPOINTS:
        rows = [row for row in records if row["controller"] == item["name"]]
        present.append(
            {
                "name": item["name"],
                "family": item["family"],
                "path": item["path"],
                "checkpoint_present": (ROOT / item["path"]).is_file(),
                "poor_action_control": item["family"] == POOR_ACTION_FAMILY,
                "report": aggregate_controller([row for row in rows if int(row["seed"]) in set(protocol["report_seeds"])]),
                "validation": aggregate_controller([row for row in rows if int(row["seed"]) in set(protocol["validation_seeds"])]),
            }
        )
    summary = {
        "experiment_id": protocol["experiment_id"],
        "split_version": protocol["split_version"],
        "is_test_set": False,
        "select_on_test": False,
        "same_config_for_every_model": True,
        "planning_config_status": protocol["planning_config_status"],
        "active_config": protocol["active_config"],
        "selected_config": None,
        "cem": protocol["cem"],
        "action_moments": {
            "n_train_episodes": n_train,
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "uses_validation_tail": False,
        },
        "highlight": highlight,
        "report_seeds": _summarize(records, protocol["report_seeds"]),
        "validation_seeds": _summarize(records, protocol["validation_seeds"]),
        "checkpoints": present,
        "nonlearned_controllers": nonlearned,
        "qualitative_comparison": {
            "task": qualitative_task,
            "seed": qualitative_seed,
            "oracle": "oracle",
            "highlight": highlight["highlight"],
            "poor_action_model": "vicreg_noaction_s42",
            "random": "random",
            "selection_uses_this_seed": qualitative_seed in protocol["validation_seeds"],
        },
    }
    out = ROOT / "experiments" / "runs" / "planning_benchmark_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.json").write_text(json.dumps(to_jsonable(records), indent=2) + "\n", encoding="utf-8")
    (out / "aggregation.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    if qualitative:
        payload = {f"{name}_frames": value["frames"] for name, value in qualitative.items()}
        payload.update({f"{name}_distances": value["distances"] for name, value in qualitative.items()})
        np.savez_compressed(out / "qualitative.npz", **payload)
        (out / "qualitative_status.json").write_text(
            json.dumps({name: value["status"] for name, value in qualitative.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(out / "aggregation.json")


if __name__ == "__main__":
    main()
