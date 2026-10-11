#!/usr/bin/env python3
"""Calibrate MPM excitation regimes from simulator forks.

    python3.13 experiments/calibrate_excitation.py

Writes raw tables, plots, and selected configs. Does not train a model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environments import calibration_candidates, mpm_legacy
from src.excitation import aggregate_environment, measure_environment, select_regimes
from src.excitation_plots import plot_excitation
from src.registry import RunRecorder, to_jsonable

CONFIG_PATH = ROOT / "configs" / "experiments" / "excitation_calibration_v1.yaml"
ENV_DIR = ROOT / "configs" / "environments"
SELECTED_NAMES = {
    "low": "mpm_low_excitation",
    "medium": "mpm_medium_excitation",
    "high": "mpm_high_excitation",
    "persistent": "mpm_persistent_excitation",
}


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return ""
    return value


def write_table(rows: list[dict[str, Any]], csv_path: Path, parquet_path: Path) -> dict[str, Any]:
    if not rows:
        raise ValueError("refusing to write an empty table.")
    fields = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row[key]) for key in fields})
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return {"csv": str(csv_path.relative_to(ROOT)), "parquet": None, "parquet_status": "pyarrow_unavailable"}
    table = pa.Table.from_pylist([{key: row[key] for key in fields} for row in rows])
    pq.write_table(table, parquet_path)
    return {
        "csv": str(csv_path.relative_to(ROOT)),
        "parquet": str(parquet_path.relative_to(ROOT)),
        "parquet_status": "written",
    }


def _selected_payload(spec_row: dict[str, Any], role_name: str) -> dict[str, Any]:
    environment_id = SELECTED_NAMES[role_name]
    if spec_row["environment_id"] == "mpm_legacy" and role_name == "low":
        environment_id = "mpm_legacy"
    return {
        "environment_id": environment_id,
        "source_candidate": spec_row["environment_id"],
        "role": role_name if environment_id != "mpm_legacy" else "baseline",
        "command_hold": spec_row["command_hold"],
        "damping": spec_row["damping"],
        "action_gain": spec_row["action_gain"],
        "contact_radius": spec_row["contact_radius"],
        "contact_weight_mode": spec_row["contact_weight_mode"],
        "center_influence": spec_row["center_influence"],
        "force_influence": spec_row["force_influence"],
        "n_grid": 32,
        "dt": 0.001,
        "substeps": 8,
        "img_size": 64,
        "particle_spacing": 0.026,
        "p_rho": 1.0,
        "gravity": 0.0,
        "in_channels": 3,
        "action_dim": 4,
        "render_sigma": 1.35,
        "calibration": {
            "selection_fork": spec_row["selection_fork"],
            "median_rms_pos": spec_row["median_rms_pos"],
            "median_rms_vel": spec_row["median_rms_vel"],
            "median_pixel_rms": spec_row["median_pixel_rms"],
            "median_raster_mse": spec_row["median_raster_mse"],
            "median_fraction_identical": spec_row["median_fraction_identical"],
            "median_raster_change": spec_row["median_raster_change"],
            "median_density_change_rate": spec_row["median_density_change_rate"],
            "median_velocity_change_rate": spec_row["median_velocity_change_rate"],
            "action_rms_pearson": spec_row["action_rms_pearson"],
            "action_rms_spearman": spec_row["action_rms_spearman"],
            "stable": spec_row["stable"],
        },
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = calibration_candidates(config)
    command = "python3.13 experiments/calibrate_excitation.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed=int(config["seed"]),
        model={"name": None, "note": "no learned model"},
        environment={"simulator": "MLSMPM2D", "baseline": "mpm_legacy"},
        dataset_split={"split_version": "not_applicable", "evaluated": False},
        checkpoint={"path": None, "note": "no checkpoint"},
    ) as run:
        raw: list[dict[str, Any]] = []
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {spec.environment_id}", flush=True)
            raw.extend(
                measure_environment(
                    spec,
                    seed=int(config["seed"]),
                    horizon=int(config["horizon"]),
                    forks=list(config["forks"]),
                    limits=config["stability"],
                )
            )
        aggregates = []
        for spec in specs:
            spec_rows = [row for row in raw if row["environment_id"] == spec.environment_id]
            aggregates.append(aggregate_environment(spec_rows, config["selection_fork"]))
        selection = select_regimes(aggregates, config["selection"], config["stability"])
        by_id = {row["environment_id"]: row for row in aggregates}
        selected_files = {}
        for role_name, decision in selection.items():
            if decision["status"] != "SELECTED" or decision["environment_id"] is None:
                continue
            payload = _selected_payload(by_id[decision["environment_id"]], role_name)
            if payload["environment_id"] == "mpm_legacy":
                continue
            text = yaml.safe_dump(payload, sort_keys=False)
            (run.run_dir / "selected").mkdir(parents=True, exist_ok=True)
            selected_path = run.run_dir / "selected" / f"{payload['environment_id']}.yaml"
            selected_path.write_text(text, encoding="utf-8")
            ENV_DIR.mkdir(parents=True, exist_ok=True)
            (ENV_DIR / f"{payload['environment_id']}.yaml").write_text(text, encoding="utf-8")
            selected_files[role_name] = payload["environment_id"]
        raw_info = write_table(to_jsonable(raw), run.run_dir / "forks.csv", run.run_dir / "forks.parquet")
        agg_info = write_table(
            to_jsonable(aggregates),
            run.run_dir / "candidates.csv",
            run.run_dir / "candidates.parquet",
        )
        selected_ids = {name: item["environment_id"] for name, item in selection.items()}
        plot_excitation(aggregates, run.run_dir / "excitation_curves.png", selected_ids)
        summary = {
            "n_environments": len(specs),
            "n_fork_rows": len(raw),
            "n_stable": sum(1 for row in aggregates if row["stable"]),
            "selection": selection,
            "selected_files": selected_files,
            "legacy_environment_id": mpm_legacy().environment_id,
            "tables": {"forks": raw_info, "candidates": agg_info},
            "learned_model_evaluated": False,
        }
        (run.run_dir / "regime_summary.json").write_text(
            json.dumps(to_jsonable(summary), indent=2) + "\n",
            encoding="utf-8",
        )
        lines = ["# Excitation regime summary", ""]
        for name, decision in selection.items():
            env_id = decision["environment_id"]
            if env_id is None:
                lines.append(f"- {name}: UNFILLED")
                continue
            row = by_id[env_id]
            lines.append(
                f"- {name}: {env_id} damping={row['damping']} gain={row['action_gain']} "
                f"radius={row['contact_radius']} hold={row['command_hold']} "
                f"raster_change={row['median_raster_change']} pixel_rms={row['median_pixel_rms']} "
                f"rms_pos={row['median_rms_pos']}"
            )
        (run.run_dir / "regime_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.complete(to_jsonable(summary))
    print(run.run_dir)


if __name__ == "__main__":
    main()
