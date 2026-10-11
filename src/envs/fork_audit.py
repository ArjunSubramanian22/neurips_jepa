"""Environment-agnostic action-fork audit.

Physical and image layers call the shared cascade metrics. Privileged
tracked positions are the target latent until a checkpoint is attached.
``predict`` and ``encode`` are optional and must return vectors in one
shared space. The metric arithmetic is not reimplemented per environment.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from src.metrics.cascade import (
    cosine_alignment,
    counterfactual_alignment,
    not_measured,
    observation_visibility,
    physical_response,
)

Predict = Callable[[np.ndarray, np.ndarray], np.ndarray]
Encode = Callable[[np.ndarray], np.ndarray]
GoalCost = Callable[[str, Mapping[str, Any]], float]


def _as_image(env) -> np.ndarray:
    image = np.asarray(env.render())
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image)


def _roll(env, state: dict[str, Any], action: np.ndarray, horizon: int) -> dict[str, Any]:
    env.set_state(state)
    for _ in range(horizon):
        env.step(action)
    future: dict[str, Any] = {
        "positions": np.asarray(env.tracked_positions(), dtype=np.float64).copy(),
        "velocities": None
        if env.tracked_velocities() is None
        else np.asarray(env.tracked_velocities(), dtype=np.float64).copy(),
        "image": _as_image(env),
    }
    if hasattr(env, "control_state"):
        future["control"] = np.asarray(env.control_state(), dtype=np.float64).copy()
    return future


def _pair(left: Mapping[str, Any], right: Mapping[str, Any], learned_reason: str) -> dict[str, Any]:
    physical = physical_response(
        left["positions"],
        right["positions"],
        left["velocities"],
        right["velocities"],
    )
    image = observation_visibility(left["image"], right["image"])
    delta = right["positions"].reshape(-1) - left["positions"].reshape(-1)
    copied = cosine_alignment(delta, delta)
    ignored = cosine_alignment(np.zeros_like(delta), delta)
    return {
        "physical": physical,
        "observation": image,
        "target_latent": {
            "status": "measured",
            "source": "tracked_positions",
            "delta_l2": float(np.linalg.norm(delta)),
        },
        "predictor": {
            "learned": not_measured(learned_reason),
            "oracle_copies_target": copied,
            "constant_ignores_action": ignored,
        },
        "direction": {
            "learned": not_measured(learned_reason),
            "oracle_cosine": copied["cosine"],
            "oracle_status": copied["status"],
            "constant_cosine": ignored["cosine"],
            "constant_status": ignored["status"],
        },
    }


def retrieval(futures: Mapping[str, np.ndarray], predict) -> dict[str, Any]:
    """Nearest neighbor in a supplied vector space. A tie is not a hit."""
    names = list(futures)
    hits = 0
    ties = 0
    for name in names:
        predicted = np.asarray(predict(name), dtype=np.float64).reshape(-1)
        distances = []
        for other in names:
            target = np.asarray(futures[other], dtype=np.float64).reshape(-1)
            distances.append(float(np.mean((predicted - target) ** 2)))
        best = min(distances)
        winners = [names[i] for i, dist in enumerate(distances) if np.isclose(dist, best)]
        if len(winners) != 1:
            ties += 1
        elif winners[0] == name:
            hits += 1
    if ties == len(names):
        return {
            "top1": None,
            "n": len(names),
            "n_ties": ties,
            "status": "ambiguous_identical_target",
        }
    return {
        "top1": float(hits / len(names)),
        "n": len(names),
        "n_ties": ties,
        "status": "measured",
    }


def _choose(costs: Mapping[str, float]) -> dict[str, Any]:
    finite = {name: float(cost) for name, cost in costs.items()}
    best = min(finite.values())
    winners = [name for name, cost in finite.items() if np.isclose(cost, best)]
    if len(winners) != 1:
        return {"chosen": None, "costs": finite, "status": "ambiguous_tie"}
    return {"chosen": winners[0], "costs": finite, "status": "measured"}


def _float_image(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if np.issubdtype(frame.dtype, np.integer):
        return frame.astype(np.float64) / 255.0
    return frame.astype(np.float64)


def audit_forks(
    env,
    actions: Mapping[str, np.ndarray],
    *,
    positive: str,
    baseline: str,
    alternate: str,
    horizon: int,
    goal_cost: GoalCost,
    learned_reason: str,
    seed: int | None = None,
    predict: Predict | None = None,
    encode: Encode | None = None,
) -> dict[str, Any]:
    """Fork ``actions`` from one state and keep each diagnostic layer separate."""
    if seed is not None:
        env.reset(int(seed))
    origin = env.get_state()
    start_image = _as_image(env)
    env.set_state(origin)
    restored = env.get_state()
    if not np.array_equal(restored["vector"], origin["vector"]):
        raise RuntimeError("set_state did not restore the fork origin.")
    futures = {
        name: _roll(env, origin, np.asarray(action, dtype=np.float64), horizon)
        for name, action in actions.items()
    }
    env.set_state(origin)
    if not np.array_equal(env.get_state()["vector"], origin["vector"]):
        raise RuntimeError("set_state did not restore the fork origin after the rollouts.")
    images = {name: _float_image(value["image"]) for name, value in futures.items()}
    positions = {name: value["positions"].reshape(-1) for name, value in futures.items()}
    report: dict[str, Any] = {
        "environment_id": str(getattr(env, "environment_id", "")),
        "seed": None if seed is None else int(seed),
        "horizon": int(horizon),
        "restoration_exact": True,
        "names": {"positive": positive, "baseline": baseline, "alternate": alternate},
        "positions": {name: value.tolist() for name, value in positions.items()},
        "positive_vs_baseline": _pair(futures[baseline], futures[positive], learned_reason),
        "positive_vs_alternate": _pair(futures[alternate], futures[positive], learned_reason),
        "retrieval": {
            "learned": not_measured(learned_reason),
            "space": "rendered_rgb",
            "oracle_copies_target": retrieval(images, lambda name: images[name]),
            "constant_ignores_action": retrieval(images, lambda _name: images[baseline]),
            "privileged_positions": {
                "oracle_copies_target": retrieval(positions, lambda name: positions[name]),
                "note": "Tracked coordinates can tie when an action does not move them.",
            },
        },
        "planning": {
            "learned": not_measured(learned_reason),
            "oracle": _choose({name: goal_cost(name, futures[name]) for name in futures}),
        },
    }
    if predict is None or encode is None:
        return report
    start = _float_image(start_image)
    predicted = {
        name: np.asarray(predict(start, np.asarray(actions[name], dtype=np.float64)), dtype=np.float64)
        for name in futures
    }
    encoded = {name: np.asarray(encode(images[name]), dtype=np.float64) for name in futures}
    alignment = counterfactual_alignment(
        predicted[baseline],
        predicted[positive],
        encoded[baseline],
        encoded[positive],
    )
    report["positive_vs_baseline"]["predictor"]["learned"] = alignment["predictor"]
    report["positive_vs_baseline"]["direction"]["learned"] = alignment["alignment"]
    report["retrieval"]["learned"] = retrieval(encoded, lambda name: predicted[name])
    goal_latent = encoded[positive]
    learned_plan = _choose(
        {
            name: float(np.mean((predicted[name].reshape(-1) - goal_latent.reshape(-1)) ** 2))
            for name in futures
        }
    )
    report["planning"]["learned"] = learned_plan
    report["planning"]["learned_goal"] = positive
    return report
