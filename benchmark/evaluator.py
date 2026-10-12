"""Map a fork audit into the shared result schema.

Numbers come from :func:`src.envs.fork_audit.audit_forks`. The methods below
are the same cascade functions that audit already calls.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from benchmark.environments import make_registered, resolve_split, task_spec
from benchmark.schema import SCHEMA_VERSION
from src.envs.fork_audit import audit_forks, retrieval
from src.metrics.cascade import (
    cosine_alignment,
    latent_contrast,
    observation_visibility,
    physical_response,
)


class CounterfactualEvaluator:
    """Layer methods plus one ``run`` that fills the result schema."""

    def physical_response(self, positions_a, positions_b, velocities_a=None, velocities_b=None) -> dict[str, Any]:
        return physical_response(positions_a, positions_b, velocities_a, velocities_b)

    def observation_visibility(self, obs_a, obs_b) -> dict[str, Any]:
        return observation_visibility(obs_a, obs_b)

    def target_response(self, left, right) -> dict[str, Any]:
        delta = np.asarray(right, dtype=np.float64).reshape(-1) - np.asarray(left, dtype=np.float64).reshape(-1)
        return {
            "status": "measured",
            "source": "tracked_positions",
            "delta_l2": float(np.linalg.norm(delta)),
        }

    def predictor_response(self, predicted_a, predicted_b) -> dict[str, Any]:
        return latent_contrast(predicted_a, predicted_b)

    def direction_metrics(self, predicted_delta, target_delta) -> dict[str, Any]:
        return cosine_alignment(predicted_delta, target_delta)

    def ranking_metrics(self, futures, predict) -> dict[str, Any]:
        return retrieval(futures, predict)

    def multi_step_metrics(self, horizon: int, physical: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "measured",
            "horizon": int(horizon),
            "final_physical_rms": physical.get("rms_pos"),
            "final_observation_mse": observation.get("mse"),
            "per_step": "not_measured",
            "per_step_reason": "the fork records the horizon outcome, not a curve",
        }

    def run(
        self,
        env_name: str,
        task: str,
        *,
        split: str,
        seed: int = 0,
        model: str | None = None,
        checkpoint: str | None = None,
        command: str = "audit",
        represent: Callable | None = None,
    ) -> dict[str, Any]:
        """Audit one task and return a schema record. Does not write a file."""
        split_info = resolve_split(env_name, split)
        if (model is None) ^ (checkpoint is None):
            raise ValueError("model and checkpoint are supplied together.")
        if represent is None and model is not None and checkpoint is not None:
            from benchmark.adapters import load_adapter

            represent = load_adapter(model, checkpoint).represent_fork
        report = _fork(env_name, task, seed, represent)
        return _schema(
            report,
            command=command,
            env_name=env_name,
            model=model,
            checkpoint=checkpoint,
            split_info=split_info,
            seed=seed,
        )


def _fork(env_name: str, task: str, seed: int, represent: Callable | None) -> dict[str, Any]:
    spec = task_spec(env_name, task)
    env = make_registered(env_name, task)
    try:
        report = audit_forks(
            env,
            spec["actions"],
            positive=spec["positive"],
            baseline=spec["baseline"],
            alternate=spec["alternate"],
            horizon=int(spec["horizon"]),
            goal_cost=spec["goal_cost"],
            learned_reason=spec["learned_reason"],
            seed=seed,
            represent=represent,
        )
    finally:
        env.close()
    report["task"] = task
    return report


def _layer_status(layer: Mapping[str, Any], default: str = "measured") -> dict[str, Any]:
    out = dict(layer)
    out.setdefault("status", default)
    return out


def _schema(
    report: dict[str, Any],
    *,
    command: str,
    env_name: str,
    model: str | None,
    checkpoint: str | None,
    split_info: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    pair = report["positive_vs_baseline"]
    physical = pair["physical"]
    observation = pair["observation"]
    predictor = dict(pair["predictor"])
    predictor["status"] = pair["predictor"]["oracle_copies_target"]["status"]
    direction = dict(pair["direction"])
    direction["status"] = direction.get("oracle_status", "not_measured")
    ranking = dict(report["retrieval"])
    ranking["status"] = report["retrieval"]["oracle_copies_target"]["status"]
    planning = dict(report["planning"])
    planning["status"] = planning["oracle"]["status"]
    horizon = int(report["horizon"])
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "env": env_name,
        "model": model,
        "checkpoint": checkpoint,
        "split": split_info["split"],
        "split_role": split_info["split_role"],
        "is_test_set": split_info["is_test_set"],
        "seed": int(seed),
        "results": [
            {
                "task": report["task"],
                "horizon": horizon,
                "restoration_exact": bool(report["restoration_exact"]),
                "positive": report["names"]["positive"],
                "layers": {
                    "physical_response": _layer_status(physical),
                    "observation_visibility": _layer_status(observation),
                    "target_response": _layer_status(pair["target_latent"]),
                    "predictor_response": predictor,
                    "direction_metrics": direction,
                    "ranking_metrics": ranking,
                    "multi_step_metrics": CounterfactualEvaluator().multi_step_metrics(horizon, physical, observation),
                    "planning": planning,
                },
            }
        ],
    }

