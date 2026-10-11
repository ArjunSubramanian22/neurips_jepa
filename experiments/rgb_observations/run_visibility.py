#!/usr/bin/env python3
"""Compare action-fork visibility in the physical raster and in RGB.

    python3.13 experiments/rgb_observations/run_visibility.py

The showcase frame is the RGB-128 high-excitation disk frame with the largest
affected-pixel fraction. Ties use the latest frame. That rule is fixed here
and does not use a learned model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import (
    MLSMPM2D,
    apply_action_frames,
    render_rgb,
    sample_action,
    sample_geometry,
)
from src.environments import environment_from_dict
from src.registry import RunRecorder, to_jsonable
from src.rgb_figures import save_counterfactual_figure
from src.rgb_metrics import frame_visibility
from src.rgb_render import load_rgb_config, render_rgb_cached

CONFIGS = {
    "physical_raster": None,
    "rgb_64": ROOT / "configs" / "rgb" / "rgb_v1_64.yaml",
    "rgb_128": ROOT / "configs" / "rgb" / "rgb_v1_128.yaml",
    "rgb_256": ROOT / "configs" / "rgb" / "rgb_v1_256.yaml",
}
ENVIRONMENTS = (
    ROOT / "configs" / "environments" / "mpm_legacy.yaml",
    ROOT / "configs" / "environments" / "mpm_high_excitation.yaml",
)
STATES = (("disk", 70.0, 0), ("bar", 70.0, 1))
HORIZON = 3


def _fork(spec_path: Path, geometry: str, youngs: float, seed: int) -> dict[str, Any]:
    spec = environment_from_dict(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    sim = MLSMPM2D(spec.mpm)
    rng = np.random.default_rng(int(seed) + 10007)
    points = sample_geometry(geometry, spec.mpm.particle_spacing, rng)
    sim.reset(points, youngs)
    apply_action_frames(sim, sample_action(rng, sim.x), 1)
    action = sample_action(rng, sim.x)
    alternate = action.copy()
    alternate[2:] *= -1.0
    recorded = sim.clone()
    other = sim.clone()
    recorded_frames = []
    alternate_frames = []
    for _ in range(HORIZON):
        apply_action_frames(recorded, action, 1)
        apply_action_frames(other, alternate, 1)
        recorded_frames.append((recorded.x.copy(), recorded.v.copy()))
        alternate_frames.append((other.x.copy(), other.v.copy()))
    return {
        "environment_id": spec.environment_id,
        "geometry": geometry,
        "youngs": youngs,
        "recorded": recorded_frames,
        "alternate": alternate_frames,
    }


def _render(kind: str, positions: np.ndarray, velocities: np.ndarray, cache: Path) -> np.ndarray:
    if kind == "physical_raster":
        digest = hashlib.sha256()
        digest.update(b"physical_raster_v1")
        digest.update(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(velocities, dtype=np.float32).tobytes())
        folder = cache / "physical_raster"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{digest.hexdigest()}.npy"
        if path.is_file():
            return np.load(path)
        image = render_rgb(positions, velocities, 64, 1.35)
        np.save(path, image)
        return image
    cfg = load_rgb_config(CONFIGS[kind])
    return render_rgb_cached(positions, velocities, cfg, cache / kind)


def main() -> None:
    command = "python3.13 experiments/rgb_observations/run_visibility.py"
    with RunRecorder(
        "rgb_visibility_v1",
        repo_root=ROOT,
        config={
            "pipelines": list(CONFIGS),
            "environments": [path.name for path in ENVIRONMENTS],
            "states": [list(item) for item in STATES],
            "horizon": HORIZON,
            "fork": "force_reversed",
            "showcase_rule": "max rgb_128 affected-pixel fraction, then latest frame, high-excitation disk",
        },
        command=command,
        seed=0,
        model={"name": None, "note": "no learned model"},
        environment={"simulator": "MLSMPM2D"},
        dataset_split={"split_version": "not_applicable", "evaluated": False},
        checkpoint={"path": None, "note": "no checkpoint"},
    ) as run:
        cache = run.run_dir / "cache"
        rows: list[dict[str, Any]] = []
        showcase: dict[str, Any] | None = None
        best_affected = -1.0
        for spec_path in ENVIRONMENTS:
            for geometry, youngs, seed in STATES:
                fork = _fork(spec_path, geometry, youngs, seed)
                for frame_index, (recorded, alternate) in enumerate(zip(fork["recorded"], fork["alternate"], strict=True)):
                    rendered = {}
                    for kind in CONFIGS:
                        rendered[kind] = (
                            _render(kind, recorded[0], recorded[1], cache),
                            _render(kind, alternate[0], alternate[1], cache),
                        )
                        metrics = frame_visibility(rendered[kind][0], rendered[kind][1])
                        rows.append(
                            {
                                "environment_id": fork["environment_id"],
                                "geometry": geometry,
                                "youngs": youngs,
                                "frame": frame_index,
                                "pipeline": kind,
                                **{key: value for key, value in metrics.items() if key != "lpips"},
                                "lpips_status": metrics["lpips"]["status"],
                                "lpips": metrics["lpips"]["value"],
                            }
                        )
                    rgb_metrics = next(
                        row
                        for row in reversed(rows)
                        if row["pipeline"] == "rgb_128" and row["frame"] == frame_index and row["geometry"] == geometry
                    )
                    if (
                        fork["environment_id"] == "mpm_high_excitation"
                        and geometry == "disk"
                        and (
                            float(rgb_metrics["fraction_affected_pixels"]) > best_affected
                            or (
                                float(rgb_metrics["fraction_affected_pixels"]) == best_affected
                                and showcase is not None
                                and frame_index > int(showcase["frame"])
                            )
                        )
                    ):
                        best_affected = float(rgb_metrics["fraction_affected_pixels"])
                        showcase = {
                            "frame": frame_index,
                            "rendered": rendered,
                            "environment_id": fork["environment_id"],
                        }
        figure_dir = ROOT / "figures" / "rgb_observations"
        if showcase is None:
            raise RuntimeError("no showcase frame was recorded.")
        save_counterfactual_figure(
            showcase["rendered"]["rgb_128"][0],
            showcase["rendered"]["rgb_128"][1],
            figure_dir / "rgb128_counterfactual_difference.png",
            f"RGB 128 high excitation, frame {showcase['frame']}",
        )
        save_counterfactual_figure(
            showcase["rendered"]["physical_raster"][0],
            showcase["rendered"]["physical_raster"][1],
            figure_dir / "physical_counterfactual_difference.png",
            f"Physical raster, same fork, frame {showcase['frame']}",
        )
        table_path = run.run_dir / "visibility.csv"
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(to_jsonable(rows))
        summary_rows = []
        for environment in {row["environment_id"] for row in rows}:
            for pipeline in CONFIGS:
                subset = [row for row in rows if row["environment_id"] == environment and row["pipeline"] == pipeline]
                summary_rows.append(
                    {
                        "environment_id": environment,
                        "pipeline": pipeline,
                        "n": len(subset),
                        "median_fraction_affected_pixels": float(np.median([row["fraction_affected_pixels"] for row in subset])),
                        "median_pixel_mse": float(np.median([row["pixel_mse"] for row in subset])),
                        "median_ssim": float(np.median([row["ssim"] for row in subset])),
                        "median_fraction_identical_uint8": float(
                            np.median([row["fraction_identical_uint8"] for row in subset])
                        ),
                    }
                )
        (run.run_dir / "visibility_summary.json").write_text(
            json.dumps(to_jsonable({"rows": summary_rows, "showcase_frame": showcase["frame"], "learned_model_evaluated": False}), indent=2)
            + "\n",
            encoding="utf-8",
        )
        run.complete(
            to_jsonable(
                {
                    "n_rows": len(rows),
                    "summary": summary_rows,
                    "showcase_frame": showcase["frame"],
                    "lpips_status": "unavailable",
                    "learned_model_evaluated": False,
                }
            )
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
