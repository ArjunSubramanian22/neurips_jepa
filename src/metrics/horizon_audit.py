"""Multi-horizon counterfactual audit.

Each row is one protocol and one rollout mode. Protocols are different
action-sequence interventions. ``open_loop`` feeds the predictor its own
previous prediction. ``teacher_forced`` feeds it the true frame at that
step. Summaries refuse to pool those labels.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.metrics.cascade import cosine_alignment, observation_visibility, physical_response
from src.metrics.counterfactual_direction import directional_contrast
from src.metrics.future_ranking import rank_candidates
from src.stats import episode_bootstrap_mean

PROTOCOLS: tuple[str, ...] = (
    "one_step_replacement",
    "command_block_replacement",
    "persistent_alternative",
    "alternative_then_recorded",
)
ROLLOUT_MODES: tuple[str, ...] = ("open_loop", "teacher_forced")

PROTOCOL_SPECS: dict[str, dict[str, str]] = {
    "one_step_replacement": {"temporal": "first_step", "source": "force_reversal"},
    "command_block_replacement": {"temporal": "command_block", "source": "force_reversal"},
    "persistent_alternative": {"temporal": "all_steps", "source": "force_reversal"},
    "alternative_then_recorded": {"temporal": "first_step", "source": "shuffled_valid_action"},
}


def counterfactual_actions(
    recorded: Any,
    alternative: Any,
    protocol: str,
    command_hold: int,
) -> np.ndarray:
    """Action sequence for branch B. Branch A is ``recorded`` unchanged.

    ``one_step_replacement`` writes the force-reversed command at step 0 only.
    ``command_block_replacement`` writes it for the first ``command_hold``
    steps, which is the aligned command block when the fork starts on a block
    boundary. ``persistent_alternative`` writes it at every step.
    ``alternative_then_recorded`` uses the same first-step mask, but the
    caller supplies a shuffled valid action rather than a force reversal.
    """
    if protocol not in PROTOCOL_SPECS:
        raise ValueError(f"unknown protocol {protocol!r}.")
    base = np.asarray(recorded, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] < 1 or base.shape[1] < 1:
        raise ValueError("recorded actions must have shape (H, A).")
    other = np.asarray(alternative, dtype=np.float64).reshape(-1)
    if other.shape != base.shape[1:]:
        raise ValueError("alternative action shape does not match one recorded step.")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(other)):
        raise ValueError("actions must be finite.")
    hold = int(command_hold)
    if hold < 1:
        raise ValueError("command_hold must be >= 1.")
    out = base.copy()
    temporal = PROTOCOL_SPECS[protocol]["temporal"]
    if temporal == "first_step":
        out[0] = other
    elif temporal == "command_block":
        if hold > out.shape[0]:
            raise ValueError("command_hold is longer than the action sequence.")
        out[:hold] = other
    elif temporal == "all_steps":
        out[:] = other
    else:
        raise ValueError(f"unknown temporal pattern {temporal!r}.")
    return out


def open_loop_rollout(transition: Callable[[np.ndarray, np.ndarray], np.ndarray], state: Any, actions: Any) -> np.ndarray:
    """Iterate ``transition`` on its own output. This is not teacher forcing."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError("actions must have shape (H, A).")
    latent = np.asarray(state, dtype=np.float64).copy()
    outputs = []
    for action in actions:
        latent = np.asarray(transition(latent, action), dtype=np.float64)
        outputs.append(latent.copy())
    return np.stack(outputs, axis=0)


def teacher_forced_rollout(
    transition: Callable[[np.ndarray, np.ndarray], np.ndarray],
    states: Any,
    actions: Any,
) -> np.ndarray:
    """Predict each step from the true state, not from the previous prediction.

    ``states`` has length H+1 and ``states[0]`` is the shared start.
    """
    actions = np.asarray(actions, dtype=np.float64)
    truth = np.asarray(states, dtype=np.float64)
    if actions.ndim != 2 or truth.ndim != 2:
        raise ValueError("states and actions must be two-dimensional.")
    if truth.shape[0] != actions.shape[0] + 1:
        raise ValueError("teacher forcing needs one more state than actions.")
    return np.stack(
        [np.asarray(transition(truth[index], actions[index]), dtype=np.float64) for index in range(actions.shape[0])],
        axis=0,
    )


def _none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _pixel_fraction(left: np.ndarray, right: np.ndarray) -> tuple[bool, float]:
    identical = bool(np.array_equal(left, right))
    if left.ndim >= 1:
        spatial = np.any(left != right, axis=0)
        affected = float(np.mean(spatial))
    else:
        affected = 0.0 if identical else 1.0
    return identical, affected


def _direction_vs_reference(delta: np.ndarray, reference: np.ndarray | None, *, horizon: int, norm_tau: float) -> dict[str, Any]:
    if int(horizon) <= 1 or reference is None:
        return {"cosine": None, "status": "not_applicable"}
    compared = cosine_alignment(delta, reference)
    if compared["status"] != "ok":
        return {"cosine": None, "status": compared["status"]}
    if float(np.linalg.norm(delta)) <= float(norm_tau) or float(np.linalg.norm(reference)) <= float(norm_tau):
        return {"cosine": None, "status": "undefined_zero_vector"}
    return {"cosine": compared["cosine"], "status": "ok"}


def score_horizon_pair(
    *,
    protocol: str,
    rollout_mode: str,
    horizon: int,
    position_a: Any,
    position_b: Any,
    velocity_a: Any,
    velocity_b: Any,
    raster_a: Any,
    raster_b: Any,
    target_a: Any,
    target_b: Any,
    pred_a: Any,
    pred_b: Any,
    target_delta_h1: Any | None = None,
    rgb_a: Any | None = None,
    rgb_b: Any | None = None,
    norm_tau: float = 1e-8,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Score one labeled branch pair. The labels are stored on the result."""
    if protocol not in PROTOCOL_SPECS:
        raise ValueError(f"unknown protocol {protocol!r}.")
    if rollout_mode not in ROLLOUT_MODES:
        raise ValueError(f"unknown rollout mode {rollout_mode!r}.")
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1.")
    physical = physical_response(position_a, position_b, velocity_a, velocity_b)
    raster_left = np.asarray(raster_a)
    raster_right = np.asarray(raster_b)
    raster = observation_visibility(raster_left, raster_right)
    raster_equal, raster_affected = _pixel_fraction(raster_left, raster_right)
    if rgb_a is None and rgb_b is None:
        rgb_status = "not_measured"
        rgb_equal = None
        rgb_affected = None
        rgb_mse = None
        rgb_fraction_identical = None
    else:
        rgb = observation_visibility(rgb_a, rgb_b)
        rgb_equal, rgb_affected = _pixel_fraction(np.asarray(rgb_a), np.asarray(rgb_b))
        rgb_status = "measured"
        rgb_mse = rgb["mse"]
        rgb_fraction_identical = rgb["fraction_identical"]
    tgt_a = np.asarray(target_a, dtype=np.float64).reshape(-1)
    tgt_b = np.asarray(target_b, dtype=np.float64).reshape(-1)
    hat_a = np.asarray(pred_a, dtype=np.float64).reshape(-1)
    hat_b = np.asarray(pred_b, dtype=np.float64).reshape(-1)
    contrast = directional_contrast(
        hat_b - hat_a,
        tgt_b - tgt_a,
        fraction_affected=[raster_affected],
        norm_tau=norm_tau,
        epsilon=epsilon,
    )
    reference = None if target_delta_h1 is None else np.asarray(target_delta_h1, dtype=np.float64).reshape(-1)
    direction = _direction_vs_reference(tgt_b - tgt_a, reference, horizon=int(horizon), norm_tau=norm_tau)
    equal = np.array([[True, raster_equal], [raster_equal, True]])
    rms = np.array(
        [[0.0, physical["rms_pos"]], [physical["rms_pos"], 0.0]],
        dtype=np.float64,
    )
    ranked = rank_candidates(
        np.stack([hat_a, hat_b]),
        np.stack([tgt_a, tgt_b]),
        equal,
        rms,
        distance="mse",
        norm_tau=norm_tau,
    )
    recorded_query = ranked["queries"][0]
    counterfactual_query = ranked["queries"][1]
    return {
        "protocol": protocol,
        "intervention_source": PROTOCOL_SPECS[protocol]["source"],
        "temporal_pattern": PROTOCOL_SPECS[protocol]["temporal"],
        "rollout_mode": rollout_mode,
        "horizon": int(horizon),
        "rms_pos": physical["rms_pos"],
        "rms_vel": physical["rms_vel"],
        "raster_identical": raster_equal,
        "raster_fraction_identical": raster["fraction_identical"],
        "raster_mse": raster["mse"],
        "fraction_affected_pixels": raster_affected,
        "rgb_status": rgb_status,
        "rgb_identical": rgb_equal,
        "rgb_fraction_identical": rgb_fraction_identical,
        "rgb_mse": rgb_mse,
        "rgb_fraction_affected_pixels": rgb_affected,
        "target_delta_l2": float(np.linalg.norm(tgt_b - tgt_a)),
        "target_delta_mse": float(np.mean((tgt_b - tgt_a) ** 2)),
        "target_direction_cosine": direction["cosine"],
        "target_direction_status": direction["status"],
        "pred_delta_l2": _none(float(contrast["pred_norm"][0])),
        "cosine": _none(float(contrast["cosine"][0])),
        "cosine_status": str(contrast["cosine_status"][0]),
        "magnitude_ratio": _none(float(contrast["magnitude_ratio"][0])),
        "normalized_vector_error": _none(float(contrast["normalized_vector_error"][0])),
        "contrast_status": str(contrast["contrast_status"][0]),
        "target_near_zero": bool(contrast["target_near_zero"][0]),
        "target_invisible": bool(contrast["target_invisible"][0]),
        "recorded_ranking_status": recorded_query["status"],
        "recorded_ranking_top1": recorded_query["top1"],
        "counterfactual_ranking_status": counterfactual_query["status"],
        "counterfactual_ranking_top1": counterfactual_query["top1"],
        "counterfactual_ranking_margin": counterfactual_query["margin"],
        "observationally_ambiguous": counterfactual_query["status"] == "observationally_ambiguous",
        "physically_distinct_observationally_identical": bool(
            counterfactual_query["physically_distinct_observationally_identical"]
        ),
        "rollout_mse_recorded": float(np.mean((hat_a - tgt_a) ** 2)),
        "rollout_mse_counterfactual": float(np.mean((hat_b - tgt_b) ** 2)),
    }


def summarize_labeled(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one protocol, one rollout mode, and one horizon. Mixed labels raise."""
    chosen = list(rows)
    if not chosen:
        raise ValueError("cannot summarize an empty group.")
    protocols = {row["protocol"] for row in chosen}
    modes = {row["rollout_mode"] for row in chosen}
    horizons = {int(row["horizon"]) for row in chosen}
    if len(protocols) != 1 or len(modes) != 1 or len(horizons) != 1:
        raise ValueError("refusing to pool protocols, rollout modes, or horizons.")
    cosine_values = [row["cosine"] for row in chosen if row["cosine_status"] == "ok" and row["cosine"] is not None]
    cosine_arr = np.asarray(cosine_values, dtype=np.float64)
    ranking_scored = [
        row
        for row in chosen
        if row["counterfactual_ranking_status"] in {"scored", "distance_tie"} and row["counterfactual_ranking_top1"] is not None
    ]
    hits = [row for row in ranking_scored if row["counterfactual_ranking_top1"] == 1.0]
    margins = np.asarray(
        [float(row["counterfactual_ranking_margin"]) for row in ranking_scored if row["counterfactual_ranking_margin"] is not None],
        dtype=np.float64,
    )
    direction_values = [
        float(row["target_direction_cosine"])
        for row in chosen
        if row["target_direction_status"] == "ok" and row["target_direction_cosine"] is not None
    ]
    direction_arr = np.asarray(direction_values, dtype=np.float64)
    physical = np.asarray([float(row["rms_pos"]) for row in chosen], dtype=np.float64)
    visible = np.asarray([float(row["fraction_affected_pixels"]) for row in chosen], dtype=np.float64)
    rollout = np.asarray([float(row["rollout_mse_recorded"]) for row in chosen], dtype=np.float64)
    return {
        "protocol": next(iter(protocols)),
        "rollout_mode": next(iter(modes)),
        "horizon": next(iter(horizons)),
        "n_total": len(chosen),
        "n_cosine_defined": int(cosine_arr.size),
        "n_cosine_undefined": int(len(chosen) - cosine_arr.size),
        "cosine_median": None if cosine_arr.size == 0 else float(np.median(cosine_arr)),
        "cosine_mean": None if cosine_arr.size == 0 else float(cosine_arr.mean()),
        "n_ranking_scored": len(ranking_scored),
        "n_ranking_hit": len(hits),
        "n_ranking_ambiguous": int(sum(bool(row["observationally_ambiguous"]) for row in chosen)),
        "n_physically_distinct_observationally_identical": int(
            sum(bool(row["physically_distinct_observationally_identical"]) for row in chosen)
        ),
        "ranking_top1": None if not ranking_scored else float(len(hits) / len(ranking_scored)),
        "ranking_margin_median": None if margins.size == 0 else float(np.median(margins)),
        "n_direction_defined": int(direction_arr.size),
        "direction_cosine_median": None if direction_arr.size == 0 else float(np.median(direction_arr)),
        "rms_pos_median": float(np.median(physical)),
        "fraction_affected_median": float(np.median(visible)),
        "rollout_mse_recorded_median": float(np.median(rollout)),
        "raster_identical_count": int(sum(bool(row["raster_identical"]) for row in chosen)),
    }


def bootstrap_labeled(
    episode_ids: Sequence[int],
    values: Sequence[float | None],
    include: Sequence[bool],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Equal-episode interval. Episodes with no included row are listed, not imputed."""
    episodes = np.asarray(episode_ids)
    numeric = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    mask = np.asarray(list(include), dtype=bool) & np.isfinite(numeric)
    if not (episodes.shape[0] == numeric.shape[0] == mask.shape[0]):
        raise ValueError("bootstrap columns must have equal length.")
    means = []
    excluded = []
    for episode in np.unique(episodes).tolist():
        taken = numeric[(episodes == episode) & mask]
        if taken.size == 0:
            excluded.append(int(episode))
        else:
            means.append(float(taken.mean()))
    result: dict[str, Any] = {
        "n_episodes": int(np.unique(episodes).size),
        "n_episodes_with_values": len(means),
        "n_episodes_without_values": len(excluded),
        "excluded_episode_ids": excluded,
    }
    if not means:
        result["bootstrap"] = None
        return result
    result["bootstrap"] = episode_bootstrap_mean(np.asarray(means), n_resamples, rng)
    return result
