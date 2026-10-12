#!/usr/bin/env python3
"""Run the reduced-simulator cascade protocol check and register it.

    python3.13 scripts/run_cascade_protocol_smoke.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import MPMConfig
from src.cascade_protocol import measure_forks
from src.registry import RunRecorder, to_jsonable


def _mpm(raw: dict) -> MPMConfig:
    return MPMConfig(
        n_grid=int(raw["n_grid"]),
        img_size=int(raw["img_size"]),
        particle_spacing=float(raw["particle_spacing"]),
        substeps=int(raw["substeps"]),
        dt=float(raw["dt"]),
        damping=float(raw["damping"]),
        action_gain=float(raw["action_gain"]),
        contact_radius=float(raw["contact_radius"]),
        render_sigma=float(raw["render_sigma"]),
        gravity=float(raw["gravity"]),
    )


def _write_csv(path: Path, pairs: list[dict]) -> None:
    fields = [
        "intervention",
        "action_l2",
        "rms_pos",
        "rms_vel",
        "fraction_identical",
        "observation_mse",
        "privileged_mse",
        "constant_predictor_mse",
        "oracle_predictor_mse",
        "oracle_cosine",
        "oracle_cosine_status",
        "learned_representation_status",
        "learned_predictor_status",
        "decision_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            layers = pair["cascade"]["layers"]
            controls = pair["controls"]
            oracle = controls["oracle_privileged_standin"]["alignment"]
            writer.writerow(
                {
                    "intervention": pair["intervention"],
                    "action_l2": pair["cascade"]["action"]["l2"],
                    "rms_pos": layers["physical"]["rms_pos"],
                    "rms_vel": layers["physical"]["rms_vel"],
                    "fraction_identical": layers["observation"]["fraction_identical"],
                    "observation_mse": layers["observation"]["mse"],
                    "privileged_mse": controls["privileged_particle_coordinates"]["contrast"]["mse"],
                    "constant_predictor_mse": controls["constant_predictor_standin"]["contrast"]["mse"],
                    "oracle_predictor_mse": controls["oracle_privileged_standin"]["contrast"]["mse"],
                    "oracle_cosine": "" if oracle["cosine"] is None else oracle["cosine"],
                    "oracle_cosine_status": oracle["status"],
                    "learned_representation_status": layers["representation"]["status"],
                    "learned_predictor_status": layers["predictor"]["status"],
                    "decision_status": layers["decision"]["status"],
                }
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reduced-simulator cascade protocol check.")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Registry id. The default in the config is refused once that record exists.",
    )
    args = parser.parse_args(argv)
    config_path = ROOT / "configs" / "experiments" / "cascade_protocol_smoke.yaml"
    config = dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if args.experiment_id:
        config["experiment_id"] = str(args.experiment_id)
    command = "python3.13 scripts/run_cascade_protocol_smoke.py"
    if args.experiment_id:
        command = f"{command} --experiment-id {args.experiment_id}"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed=int(config["seed"]),
        model={
            "learned": None,
            "controls": [
                "privileged_particle_coordinates",
                "constant_predictor_standin",
                "oracle_privileged_standin",
            ],
        },
        environment={
            "simulator": "MLSMPM2D",
            "resolution": "reduced",
            "comparable_to_paper_dump": False,
        },
        dataset_split=config["dataset_split"],
        checkpoint={"path": None, "note": "no learned checkpoint"},
    ) as run:
        measured = measure_forks(
            seed=int(config["seed"]),
            n_frames=int(config["n_frames"]),
            youngs=float(config["youngs"]),
            geometry=str(config["geometry"]),
            interventions=list(config["interventions"]),
            mpm=_mpm(config["simulator"]),
        )
        payload = to_jsonable(measured)
        (run.run_dir / "pairs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(run.run_dir / "pairs.csv", payload["pairs"])
        run.complete(
            {
                "n_pairs": len(payload["pairs"]),
                "n_particles": payload["n_particles"],
                "simulator": payload["simulator"],
                "synthetic_controls": payload["synthetic_controls"],
                "learned_model_evaluated": False,
                "pairs_csv": "pairs.csv",
                "pairs_json": "pairs.json",
            }
        )
    print(run.run_dir)


if __name__ == "__main__":
    main()
