"""Regenerate qualitative figures from saved experiment tables and frame caches.

    python3.13 scripts/render_qualitative_figures.py

Selection is ``src.qualitative_select``. Pixels come from the horizon-audit
cache, the planning archive, or a re-render that is checked against
``visibility.csv``. Predicted RGB frames were not stored, so that row is
recorded as not measured.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

from src.rgb_metrics import difference_maps
from src.qualitative_figures import (
    bottleneck_figure,
    comparison_figure,
    example_grid,
    fork_sequence_figure,
    panel_meta,
    planning_figure,
    save_figure,
)
from src.qualitative_select import select_counterfactual_models, select_fork_examples

QUERIES = ROOT / "experiments/runs/horizon_audit_v1/queries.csv"
CACHE = ROOT / "experiments/runs/horizon_audit_v1/cache"
THRESHOLD = ROOT / "experiments/runs/visibility_sensitivity_v1/primary_threshold.json"
PLANNING = ROOT / "experiments/runs/planning_benchmark_v1"
VISIBILITY = ROOT / "experiments/runs/rgb_visibility_v1/visibility.csv"
REGIMES = ROOT / "experiments/runs/excitation_calibration_v1/regime_summary.json"
OUT = ROOT / "figures/qualitative"
EXCITATION = (
    ("weak", "low", "mpm_legacy", ROOT / "configs/environments/mpm_legacy.yaml"),
    ("medium", "medium", "mpm_medium_excitation", ROOT / "configs/environments/mpm_medium_excitation.yaml"),
    ("strong", "high", "mpm_high_excitation", ROOT / "configs/environments/mpm_high_excitation.yaml"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _locate(episodes: np.ndarray, episode: int) -> int:
    hits = np.flatnonzero(np.asarray(episodes) == int(episode))
    if hits.size != 1:
        raise KeyError(f"episode {episode} is not a unique cache row.")
    return int(hits[0])


def _identity(source: Mapping[str, str], **extra: Any) -> dict[str, Any]:
    return panel_meta(
        environment=source.get("environment", source.get("environment_id")),
        episode=int(source["episode"]) if source.get("episode") else None,
        time=int(source["t"]) if source.get("t") not in (None, "") else int(source.get("frame", 0) or 0),
        seed=int(source["seed"]) if source.get("seed") not in (None, "") else None,
        checkpoint=source.get("model") or source.get("checkpoint"),
        intervention=source.get("intervention_source") or source.get("intervention") or "force_reversal",
        **extra,
    )


def _frames_for(row: Mapping[str, str], cache: dict[str, Any]) -> dict[str, np.ndarray]:
    manifest = cache["manifest"]
    episode_index = _locate(cache["episodes"], int(row["episode"]))
    protocol_index = list(manifest["protocols"]).index(row["protocol"])
    horizon = int(row["horizon"])
    horizon_index = list(manifest["horizons"]).index(horizon)
    return {
        "raster_current": np.array(cache["frames_a"][episode_index, 0]),
        "raster_recorded": np.array(cache["frames_a"][episode_index, horizon]),
        "raster_alternative": np.array(cache["frames_b"][episode_index, protocol_index, horizon]),
        "rgb_recorded": np.array(cache["rgb_a"][episode_index, horizon_index]),
        "rgb_alternative": np.array(cache["rgb_b"][episode_index, protocol_index, horizon_index]),
    }


def _load_cache() -> dict[str, Any]:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "episodes": np.load(CACHE / "episodes.npy"),
        "frames_a": np.load(CACHE / "frames_a.npy", mmap_mode="r"),
        "frames_b": np.load(CACHE / "frames_b.npy", mmap_mode="r"),
        "rgb_a": np.load(CACHE / "rgb_a.npy", mmap_mode="r"),
        "rgb_b": np.load(CACHE / "rgb_b.npy", mmap_mode="r"),
    }


def _rgb_at_start(row: Mapping[str, str], raster_current: np.ndarray) -> np.ndarray:
    """RGB of the cached start frame. The horizon cache stores RGB only at future horizons."""
    from scripts.generate_trajectories import replay_episode
    from src.rgb_render import load_rgb_config, render_rgb_image, to_uint8

    recorded = replay_episode(int(row["episode"]), dump_seed=0, episode_length=16, hold=3)
    if not np.array_equal(recorded["frames"][0], raster_current):
        raise RuntimeError(f"replay frame 0 does not match the horizon cache at episode {row['episode']}.")
    cfg = load_rgb_config(ROOT / "configs/rgb/rgb_v1_64.yaml")
    snap = recorded["snapshots"][0]
    return to_uint8(render_rgb_image(snap["x"], snap["v"], cfg))


def _particle_displacement(row: Mapping[str, str], cache: dict[str, Any]) -> dict[str, Any]:
    from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, replay_episode
    from src.metrics.cascade import physical_response

    manifest = cache["manifest"]
    episode = int(row["episode"])
    horizon = int(row["horizon"])
    protocol_index = list(manifest["protocols"]).index(row["protocol"])
    episode_index = _locate(cache["episodes"], episode)
    actions = np.load(CACHE / "actions_b.npy")
    recorded = replay_episode(episode, dump_seed=0, episode_length=16, hold=int(manifest["command_hold"]))
    sim = MLSMPM2D()
    sim.restore(recorded["snapshots"][0])
    for step in range(horizon):
        apply_action_frames(sim, actions[episode_index, protocol_index, step], 1)
    positions = recorded["snapshots"][horizon]["x"]
    velocities = recorded["snapshots"][horizon]["v"]
    measured = physical_response(positions, sim.x, velocities, sim.v)
    archived = float(row["rms_pos"])
    matches = abs(float(measured["rms_pos"]) - archived) <= 1e-8
    return {
        "positions": positions,
        "displacement": sim.x - positions,
        "replay_rms": float(measured["rms_pos"]),
        "archived_rms": archived,
        "matches_artifact": matches,
    }


def _visibility_module():
    path = ROOT / "experiments/rgb_observations/run_visibility.py"
    spec = importlib.util.spec_from_file_location("rgb_visibility_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _render_pair(module, spec_path: Path, frame: int, pipeline: str) -> tuple[np.ndarray, np.ndarray, float]:
    from src.rgb_metrics import frame_visibility

    fork = module._fork(spec_path, "disk", 70.0, 0)
    recorded, alternate = fork["recorded"][frame], fork["alternate"][frame]
    cache = ROOT / "experiments/runs/rgb_visibility_v1/cache"
    left = module._render(pipeline, recorded[0], recorded[1], cache)
    right = module._render(pipeline, alternate[0], alternate[1], cache)
    mse = float(frame_visibility(left, right)["pixel_mse"])
    return left, right, mse


def _archived_mse(rows: list[dict[str, str]], environment: str, frame: int, pipeline: str) -> float | None:
    matches = [
        row
        for row in rows
        if row["environment_id"] == environment
        and row["geometry"] == "disk"
        and row["youngs"] == "70.0"
        and int(row["frame"]) == frame
        and row["pipeline"] == pipeline
    ]
    if len(matches) != 1:
        return None
    return float(matches[0]["pixel_mse"])


def _family(model: str) -> str:
    if model.startswith("mse_"):
        return "mse_action"
    if "noaction" in model:
        return "vicreg_noaction"
    if model.startswith("vicreg_"):
        return "vicreg_action"
    return model


def main() -> None:
    queries = _read_csv(QUERIES)
    threshold = float(json.loads(THRESHOLD.read_text(encoding="utf-8"))["threshold"])
    chosen = select_fork_examples(queries, threshold)
    ranked = select_counterfactual_models(queries)
    cache = _load_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "threshold_source": str(THRESHOLD.relative_to(ROOT)),
                "reasons": chosen["reasons"],
                "n_pool": chosen["n_pool"],
                "n_visible": chosen["n_visible"],
                "examples": {
                    role: None
                    if row is None
                    else {
                        "episode": int(row["episode"]),
                        "time": int(row["t"]),
                        "cosine": row.get("cosine"),
                        "raster_mse": row.get("raster_mse"),
                        "rms_pos": row.get("rms_pos"),
                        "model": row.get("model"),
                        "seed": row.get("seed"),
                    }
                    for role, row in chosen["selected"].items()
                },
                "models": ranked,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_fork(chosen, cache)
    _write_bottleneck(chosen, cache)
    _write_examples(chosen, cache)
    _write_planning(ranked)
    module = _visibility_module()
    visibility_rows = _read_csv(VISIBILITY)
    _write_resolution(module, visibility_rows)
    _write_excitation(module, visibility_rows)
    print(OUT)


def _write_fork(chosen: dict[str, Any], cache: dict[str, Any]) -> None:
    row = chosen["selected"]["high_visibility_good_alignment"]
    if row is None:
        return
    frames = _frames_for(row, cache)
    rgb_current = _rgb_at_start(row, frames["raster_current"])
    figure = fork_sequence_figure(
        [
            ("raster", frames["raster_current"], frames["raster_recorded"], frames["raster_alternative"]),
            ("rgb", rgb_current, frames["rgb_recorded"], frames["rgb_alternative"]),
        ]
    )
    panels = []
    for row_name in ("physical_raster", "rgb"):
        for column in ("current", "recorded_future", "alternative", "absolute_difference", "amplified_difference"):
            panels.append(_identity(row, row=row_name, column=column))
    panels.append(
        panel_meta(
            environment=row["environment"],
            episode=int(row["episode"]),
            time=int(row["t"]),
            seed=int(row["seed"]),
            checkpoint=row["model"],
            intervention=row["intervention_source"],
            row="model_prediction",
            status="not_measured",
            reason="predicted RGB frames were not stored",
        )
    )
    save_figure(figure, OUT / "fork_sequence", {"figure": "fork_sequence", "panels": panels})


def _write_bottleneck(chosen: dict[str, Any], cache: dict[str, Any]) -> None:
    row = chosen["selected"]["high_visibility_good_alignment"]
    if row is None:
        return
    frames = _frames_for(row, cache)
    absolute, _amplified = difference_maps(
        frames["raster_recorded"], frames["raster_alternative"]
    )
    particles = _particle_displacement(row, cache)
    use_particles = bool(particles["matches_artifact"])
    figure = bottleneck_figure(
        np.clip(np.rint(absolute * 255.0), 0, 255).astype(np.uint8),
        {
            "rms_pos": float(row["rms_pos"]),
            "fraction_affected_pixels": float(row["fraction_affected_pixels"]),
            "target_delta_l2": float(row["target_delta_l2"]) if row.get("target_delta_l2") else None,
            "pred_delta_l2": float(row["pred_delta_l2"]) if row.get("pred_delta_l2") else None,
        },
        particles["positions"] if use_particles else None,
        particles["displacement"] if use_particles else None,
    )
    base = _identity(row, replay_rms=particles["replay_rms"], matches_artifact=use_particles)
    panels = [
        {**base, "column": "particle_displacement"},
        {**base, "column": "observation_difference"},
        {**base, "column": "target_latent"},
        {**base, "column": "predicted_latent"},
    ]
    save_figure(figure, OUT / "bottleneck", {"figure": "bottleneck", "panels": panels})


def _write_examples(chosen: dict[str, Any], cache: dict[str, Any]) -> None:
    packed = []
    panels = []
    labels = {
        "high_visibility_good_alignment": "high visibility\ngood alignment",
        "high_visibility_poor_alignment": "high visibility\npoor alignment",
        "physical_but_invisible": "physical change\ninvisible raster",
    }
    for role in ("high_visibility_good_alignment", "high_visibility_poor_alignment", "physical_but_invisible"):
        row = chosen["selected"][role]
        if row is None:
            packed.append((labels[role], None, None, chosen["reasons"].get(role, "not selected")))
            panels.append(
                panel_meta(
                    environment="mpm_legacy",
                    episode=None,
                    time=None,
                    seed=42,
                    checkpoint="vicreg_action_s42",
                    intervention="force_reversal",
                    role=role,
                    status="not_selected",
                    reason=chosen["reasons"].get(role),
                )
            )
            continue
        frames = _frames_for(row, cache)
        cosine = row.get("cosine")
        note = f"ep {row['episode']}" if not cosine else f"ep {row['episode']}\ncos {float(cosine):.3g}"
        packed.append((labels[role], frames["raster_recorded"], frames["raster_alternative"], note))
        for column in ("recorded_future", "alternative", "absolute_difference", "amplified_difference"):
            panels.append(_identity(row, role=role, column=column))
    save_figure(example_grid(packed), OUT / "selected_examples", {"figure": "selected_examples", "panels": panels})


def _write_planning(ranked: dict[str, Any]) -> None:
    import yaml

    archive = np.load(PLANNING / "qualitative.npz")
    protocol = yaml.safe_load((ROOT / "configs/experiments/planning_benchmark_v1.yaml").read_text(encoding="utf-8"))
    qualitative = protocol["qualitative"]
    rows = []
    panels = []
    order = [("oracle", "oracle", None)]
    if ranked.get("strong"):
        order.append(("strong", ranked["strong"]["model"], ranked["strong"]["median_cosine"]))
    if ranked.get("poor"):
        order.append(("poor", ranked["poor"]["model"], ranked["poor"]["median_cosine"]))
    for role, model, median in order:
        key = f"{model}_frames"
        frames = np.array(archive[key]) if key in archive.files else None
        if role == "oracle":
            label = "oracle"
        else:
            label = f"{role}\n{model.replace('_action', '')}\n{median:.3g}"
        rows.append((label, frames, "oracle" if role == "oracle" else _family(model)))
        panels.append(
            panel_meta(
                environment="mpm_legacy",
                episode=None,
                time=0,
                seed=int(qualitative["seed"]),
                checkpoint=None if role == "oracle" else model,
                intervention="closed_loop",
                role=role,
                task=qualitative["task"],
                median_cosine=median,
                frames_present=frames is not None,
            )
        )
    save_figure(planning_figure(rows), OUT / "planning_rollouts", {"figure": "planning_rollouts", "panels": panels})


def _checked_pair(module, rows, environment: str, spec_path: Path, frame: int, pipeline: str):
    left, right, mse = _render_pair(module, spec_path, frame, pipeline)
    archived = _archived_mse(rows, environment, frame, pipeline)
    if archived is None:
        return left, right, mse, None, True
    matches = abs(mse - archived) <= 1e-8
    if not matches:
        return None, None, mse, archived, False
    return left, right, mse, archived, True


def _write_resolution(module, visibility_rows: list[dict[str, str]]) -> None:
    candidates = [
        row
        for row in visibility_rows
        if row["environment_id"] == "mpm_high_excitation"
        and row["geometry"] == "disk"
        and row["pipeline"] == "rgb_128"
    ]
    showcase = max(candidates, key=lambda row: (float(row["fraction_affected_pixels"]), int(row["frame"])))
    frame = int(showcase["frame"])
    spec = ROOT / "configs/environments/mpm_high_excitation.yaml"
    packed = []
    panels = []
    for pipeline in ("rgb_64", "rgb_128", "rgb_256"):
        left, right, mse, archived, ok = _checked_pair(module, visibility_rows, "mpm_high_excitation", spec, frame, pipeline)
        packed.append((pipeline, left, right))
        panels.append(
            panel_meta(
                environment="mpm_high_excitation",
                episode=None,
                time=frame,
                seed=0,
                checkpoint=None,
                intervention="force_reversal",
                pipeline=pipeline,
                geometry="disk",
                youngs=70.0,
                rendered_mse=mse,
                archived_mse=archived,
                matches_artifact=ok,
            )
        )
    save_figure(
        comparison_figure(packed, "Resolution, high-excitation disk"),
        OUT / "resolution",
        {"figure": "resolution", "showcase_rule": "max rgb_128 affected fraction, then latest frame", "panels": panels},
    )


def _write_excitation(module, visibility_rows: list[dict[str, str]]) -> None:
    summary = json.loads(REGIMES.read_text(encoding="utf-8"))
    packed = []
    panels = []
    for label, regime_key, environment_id, spec in EXCITATION:
        archived_id = summary["selection"][regime_key]["environment_id"]
        left, right, mse, archived, ok = _checked_pair(
            module, visibility_rows, environment_id, spec, 0, "physical_raster"
        )
        packed.append((label, left if ok or archived is None else None, right if ok or archived is None else None))
        panels.append(
            panel_meta(
                environment=environment_id,
                episode=None,
                time=0,
                seed=0,
                checkpoint=None,
                intervention="force_reversal",
                regime=label,
                selected_environment_id=archived_id,
                geometry="disk",
                youngs=70.0,
                rendered_mse=mse,
                archived_mse=archived,
                matches_artifact=ok if archived is not None else None,
            )
        )
    save_figure(
        comparison_figure(packed, "Excitation regimes"),
        OUT / "excitation",
        {"figure": "excitation", "panels": panels},
    )


if __name__ == "__main__":
    main()
