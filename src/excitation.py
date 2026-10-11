"""Physical and raster metrics for excitation forks. No learned model is involved.

Regime cuts are applied to these measurements. They are not fit to predictor
error or any checkpoint.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from scripts.generate_trajectories import (
    MLSMPM2D,
    apply_action_frames,
    render_rgb,
    sample_action,
    sample_geometry,
)
from src.environments import EnvironmentSpec, mpm_legacy
from src.metrics.cascade import action_delta, observation_visibility, physical_response

STABILITY_LIMITS: dict[str, float] = {
    "max_clip_fraction": 0.05,
    "max_svd_clip_fraction": 0.20,
    "max_speed": 25.0,
    "max_median_rms_pos": 0.20,
}

SELECTION_FORK = "force_reversed"


def excitation_index(action_gain: float, damping: float, command_hold: int, contact_radius: float) -> float:
    """Predeclared descriptive axis. Selection does not optimize this index."""
    if damping <= 0.0 or command_hold < 1 or contact_radius <= 0.0 or action_gain <= 0.0:
        raise ValueError("excitation_index inputs must be positive.")
    return float((action_gain / 55.0) * (0.08 / damping) * (command_hold / 3.0) * (contact_radius / 0.10))


def channel_change_rates(frame_a: Any, frame_b: Any) -> dict[str, float]:
    """Fraction of spatial sites whose uint8 channel changed.

    Density is channel 0. Velocity is channels 1 and 2: a site counts if either
    velocity channel differs.
    """
    left = np.asarray(frame_a)
    right = np.asarray(frame_b)
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] < 3:
        raise ValueError(f"expected (3, H, W) frames, got {left.shape} and {right.shape}.")
    differ = left != right
    return {
        "density_change_rate": float(differ[0].mean()),
        "velocity_change_rate": float(np.any(differ[1:3], axis=0).mean()),
    }


def correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Pearson and Spearman. A zero-variance sample is undefined, not zero."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("correlation needs at least two paired values.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("correlation inputs must be finite.")

    def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
        aa = a - a.mean()
        bb = b - b.mean()
        denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
        if denom == 0.0:
            return None
        return float(np.sum(aa * bb) / denom)

    order_x = np.argsort(left, kind="mergesort")
    order_y = np.argsort(right, kind="mergesort")
    rank_x = np.empty(left.size, dtype=np.float64)
    rank_y = np.empty(right.size, dtype=np.float64)
    rank_x[order_x] = np.arange(left.size, dtype=np.float64)
    rank_y[order_y] = np.arange(right.size, dtype=np.float64)
    pearson = _pearson(left, right)
    spearman = _pearson(rank_x, rank_y)
    status = "ok" if pearson is not None and spearman is not None else "undefined_zero_variance"
    return {
        "pearson": pearson,
        "spearman": spearman,
        "n": int(left.size),
        "status": status,
    }


def stability_report(sim: MLSMPM2D, limits: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Flag non-finite states, boundary clipping, and constitutive clipping."""
    caps = dict(STABILITY_LIMITS if limits is None else limits)
    finite = bool(np.all(np.isfinite(sim.x)) and np.all(np.isfinite(sim.v)) and np.all(np.isfinite(sim.F)))
    on_box = (sim.x <= np.float32(0.0201)) | (sim.x >= np.float32(0.9799))
    clip_fraction = float(np.any(on_box, axis=1).mean()) if sim.x.size else 1.0
    singular = np.linalg.svd(sim.F, compute_uv=False) if sim.F.size else np.zeros((0, 2))
    at_svd = (np.abs(singular - 0.2) <= 1e-3) | (np.abs(singular - 4.0) <= 1e-3)
    svd_fraction = float(at_svd.mean()) if singular.size else 1.0
    speeds = np.linalg.norm(sim.v, axis=1) if sim.v.size else np.zeros(1)
    max_speed = float(np.max(speeds)) if finite else float("inf")
    stable = bool(
        finite
        and clip_fraction <= float(caps["max_clip_fraction"])
        and svd_fraction <= float(caps["max_svd_clip_fraction"])
        and max_speed <= float(caps["max_speed"])
    )
    return {
        "finite": finite,
        "clip_fraction": clip_fraction,
        "svd_clip_fraction": svd_fraction,
        "max_speed": max_speed if finite else None,
        "stable": stable,
    }


def intervene(schedule: Sequence[np.ndarray], kind: str, hold: int, radius: float) -> list[np.ndarray]:
    """Replace the first hold block. Later resamples stay shared with the factual schedule."""
    out = [np.asarray(action, dtype=np.float32).copy() for action in schedule]
    if kind == "identical":
        return out
    span = min(int(hold), len(out))
    command = out[0].copy()
    if kind == "force_reversed":
        command[2:4] *= np.float32(-1.0)
    elif kind == "contact_shifted":
        command[0] = np.float32(np.clip(float(command[0]) + float(radius), 0.08, 0.92))
    elif kind == "zero_force":
        command[2:4] = np.float32(0.0)
    else:
        raise ValueError(f"Unknown fork {kind!r}.")
    for step in range(span):
        out[step] = command.copy()
    return out


def _roll(sim: MLSMPM2D, schedule: Sequence[np.ndarray]) -> None:
    for action in schedule:
        apply_action_frames(sim, action, 1)


def _fork_metrics(
    factual: MLSMPM2D,
    alternate: MLSMPM2D,
    factual_command: np.ndarray,
    alternate_command: np.ndarray,
) -> dict[str, Any]:
    physical = physical_response(factual.x, alternate.x, factual.v, alternate.v)
    frame_f = render_rgb(factual.x, factual.v, factual.cfg.img_size, factual.cfg.render_sigma)
    frame_a = render_rgb(alternate.x, alternate.v, alternate.cfg.img_size, alternate.cfg.render_sigma)
    visible = observation_visibility(frame_f, frame_a)
    rates = channel_change_rates(frame_f, frame_a)
    delta = action_delta(factual_command, alternate_command)
    pixel_rms = float(physical["rms_pos"]) * float(factual.cfg.img_size - 1)
    return {
        "action_l2": delta["l2"],
        "rms_pos": physical["rms_pos"],
        "rms_vel": physical["rms_vel"],
        "pixel_rms": pixel_rms,
        "raster_mse": visible["mse"],
        "fraction_identical": visible["fraction_identical"],
        "raster_change": float(1.0 - visible["fraction_identical"]),
        "density_change_rate": rates["density_change_rate"],
        "velocity_change_rate": rates["velocity_change_rate"],
    }


def measure_environment(
    spec: EnvironmentSpec,
    *,
    seed: int,
    horizon: int,
    forks: Sequence[str],
    limits: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Fork each class-geometry state under constant-command interventions."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    rows: list[dict[str, Any]] = []
    for class_id, youngs in enumerate((18.0, 70.0, 220.0)):
        for geom_id, geometry in enumerate(("disk", "bar")):
            state_id = class_id + 3 * geom_id
            rng = np.random.default_rng(int(seed) + 10007 * state_id)
            sim = MLSMPM2D(spec.mpm)
            points = sample_geometry(geometry, spec.mpm.particle_spacing, rng)
            sim.reset(points, youngs)
            apply_action_frames(sim, sample_action(rng, sim.x), 1)
            schedule: list[np.ndarray] = []
            current: np.ndarray | None = None
            for step in range(int(horizon)):
                if current is None or step % int(spec.command_hold) == 0:
                    current = sample_action(rng, sim.x)
                schedule.append(np.asarray(current, dtype=np.float32).copy())
            origin = sim.clone()
            factual = origin.clone()
            _roll(factual, schedule)
            factual_stable = stability_report(factual, limits)
            for kind in forks:
                alternate_schedule = intervene(schedule, kind, spec.command_hold, spec.mpm.contact_radius)
                branch = origin.clone()
                _roll(branch, alternate_schedule)
                branch_stable = stability_report(branch, limits)
                metrics = _fork_metrics(factual, branch, schedule[0], alternate_schedule[0])
                stable = bool(factual_stable["stable"] and branch_stable["stable"])
                rows.append(
                    {
                        "environment_id": spec.environment_id,
                        "role": spec.role,
                        "damping": float(spec.mpm.damping),
                        "action_gain": float(spec.mpm.action_gain),
                        "contact_radius": float(spec.mpm.contact_radius),
                        "command_hold": int(spec.command_hold),
                        "contact_weight_mode": spec.mpm.contact_weight_mode,
                        "center_influence": float(spec.mpm.center_influence),
                        "force_influence": float(spec.mpm.force_influence),
                        "excitation_index": excitation_index(
                            spec.mpm.action_gain,
                            spec.mpm.damping,
                            spec.command_hold,
                            spec.mpm.contact_radius,
                        ),
                        "state_id": state_id,
                        "geometry": geometry,
                        "youngs": float(youngs),
                        "fork": kind,
                        "stable": stable,
                        "clip_fraction": max(factual_stable["clip_fraction"], branch_stable["clip_fraction"]),
                        "svd_clip_fraction": max(
                            factual_stable["svd_clip_fraction"],
                            branch_stable["svd_clip_fraction"],
                        ),
                        "max_speed": None
                        if factual_stable["max_speed"] is None or branch_stable["max_speed"] is None
                        else max(factual_stable["max_speed"], branch_stable["max_speed"]),
                        **metrics,
                    }
                )
    return rows


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def aggregate_environment(rows: Sequence[Mapping[str, Any]], selection_fork: str = SELECTION_FORK) -> dict[str, Any]:
    """Median response of one fork, plus action-distance correlation on the others."""
    if not rows:
        raise ValueError("no rows to aggregate.")
    chosen = [row for row in rows if row["fork"] == selection_fork]
    if not chosen:
        raise ValueError(f"selection fork {selection_fork!r} is missing.")
    head = dict(chosen[0])
    summary: dict[str, Any] = {
        "environment_id": head["environment_id"],
        "role": head["role"],
        "damping": head["damping"],
        "action_gain": head["action_gain"],
        "contact_radius": head["contact_radius"],
        "command_hold": head["command_hold"],
        "contact_weight_mode": head["contact_weight_mode"],
        "center_influence": head["center_influence"],
        "force_influence": head["force_influence"],
        "excitation_index": head["excitation_index"],
        "selection_fork": selection_fork,
        "n_states": len(chosen),
        "stable": all(bool(row["stable"]) for row in rows),
        "median_clip_fraction": _median([float(row["clip_fraction"]) for row in chosen]),
        "median_svd_clip_fraction": _median([float(row["svd_clip_fraction"]) for row in chosen]),
        "max_speed": (
            None
            if not any(row["max_speed"] is not None for row in chosen)
            else max(float(row["max_speed"]) for row in chosen if row["max_speed"] is not None)
        ),
    }
    for key in (
        "rms_pos",
        "rms_vel",
        "pixel_rms",
        "raster_mse",
        "fraction_identical",
        "raster_change",
        "density_change_rate",
        "velocity_change_rate",
        "action_l2",
    ):
        summary[f"median_{key}"] = _median([float(row[key]) for row in chosen])
    paired = [row for row in rows if row["fork"] != "identical"]
    corr = correlation([float(row["action_l2"]) for row in paired], [float(row["rms_pos"]) for row in paired])
    summary["action_rms_pearson"] = corr["pearson"]
    summary["action_rms_spearman"] = corr["spearman"]
    summary["action_rms_correlation_status"] = corr["status"]
    summary["action_rms_n"] = corr["n"]
    return summary


def _within_band(row: Mapping[str, Any], band: Mapping[str, float], *, upper_only: bool = False) -> bool:
    change = float(row["median_raster_change"])
    pixel = float(row["median_pixel_rms"])
    if upper_only:
        return change <= float(band["raster_change_max"]) and pixel <= float(band["pixel_rms_max"])
    return (
        float(band["raster_change_min"]) <= change <= float(band["raster_change_max"])
        and float(band["pixel_rms_min"]) <= pixel <= float(band["pixel_rms_max"])
    )


def _eligible(row: Mapping[str, Any], limits: Mapping[str, float]) -> bool:
    return bool(row["stable"]) and float(row["median_rms_pos"]) <= float(limits["max_median_rms_pos"])


def _closest(rows: list[dict[str, Any]], target: float, used: set[str]) -> dict[str, Any] | None:
    pool = [row for row in rows if row["environment_id"] not in used]
    if not pool:
        return None
    pool.sort(
        key=lambda row: (
            abs(float(row["median_raster_change"]) - target),
            float(row["action_gain"]),
            int(row["command_hold"]),
            str(row["environment_id"]),
        )
    )
    return pool[0]


def select_regimes(
    aggregates: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any],
    limits: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Pick low, medium, high, and persistent regimes.

    Low prefers ``mpm_legacy`` when that baseline meets the low-visibility caps.
    Persistent is chosen first so a long-hold environment is not also reused as
    the medium or high regime. Empty bands stay ``UNFILLED``.
    """
    caps = dict(STABILITY_LIMITS if limits is None else limits)
    rows = [dict(row) for row in aggregates if _eligible(row, caps)]
    used: set[str] = set()
    selected: dict[str, Any] = {}

    legacy_rows = [row for row in rows if row["environment_id"] == mpm_legacy().environment_id]
    low_band = criteria["low"]
    if legacy_rows and _within_band(legacy_rows[0], low_band, upper_only=True):
        selected["low"] = {"status": "SELECTED", "environment_id": legacy_rows[0]["environment_id"], "source": "baseline"}
        used.add(legacy_rows[0]["environment_id"])
    else:
        low_pool = [row for row in rows if _within_band(row, low_band, upper_only=True)]
        choice = _closest(low_pool, float(low_band.get("target_raster_change", 0.05)), used)
        selected["low"] = (
            {"status": "UNFILLED", "environment_id": None, "source": None}
            if choice is None
            else {"status": "SELECTED", "environment_id": choice["environment_id"], "source": "candidate"}
        )
        if choice is not None:
            used.add(choice["environment_id"])

    persistent_band = criteria["persistent"]
    persistent_pool = [
        row
        for row in rows
        if int(row["command_hold"]) >= int(persistent_band["min_command_hold"])
        and float(row["damping"]) <= float(persistent_band["max_damping"])
        and float(row["median_raster_change"]) >= float(persistent_band["min_raster_change"])
        and float(row["median_pixel_rms"]) <= float(persistent_band["max_pixel_rms"])
    ]
    persistent = _closest(persistent_pool, float(persistent_band["target_raster_change"]), used)
    selected["persistent"] = (
        {"status": "UNFILLED", "environment_id": None}
        if persistent is None
        else {"status": "SELECTED", "environment_id": persistent["environment_id"]}
    )
    if persistent is not None:
        used.add(persistent["environment_id"])

    for name in ("high", "medium"):
        band = criteria[name]
        pool = [row for row in rows if _within_band(row, band)]
        choice = _closest(pool, float(band["target_raster_change"]), used)
        selected[name] = (
            {"status": "UNFILLED", "environment_id": None}
            if choice is None
            else {"status": "SELECTED", "environment_id": choice["environment_id"]}
        )
        if choice is not None:
            used.add(choice["environment_id"])
    return selected
