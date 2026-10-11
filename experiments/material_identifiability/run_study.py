#!/usr/bin/env python3
"""Run the material-identifiability search and select a regime on validation.

    python3.13 experiments/material_identifiability/run_study.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environments import environment_from_dict
from src.material_excitation import simulate_episode
from src.material_figures import (
    plot_accuracy_vs_protocol,
    plot_confusion,
    plot_history_curve,
    plot_history_overview,
)
from src.material_probes import RasterProjector, evaluate_probe, history_matrix, select_protocol
from src.registry import RunRecorder, to_jsonable
from src.splits import fresh_pool_v2_split

CONFIG_PATH = ROOT / "configs" / "material_excitation" / "protocols.yaml"
FIGURE_DIR = ROOT / "figures" / "material_identifiability"


def _lookup(rows: list[dict[str, Any]], **query: Any) -> float:
    for row in rows:
        if all(row[key] == value for key, value in query.items()):
            return float(row["accuracy"])
    raise KeyError(query)


def _features_for(mech: np.ndarray, floats: np.ndarray, uint8: np.ndarray, length: int) -> dict[str, np.ndarray]:
    post_mech = mech[:, 1:, :]
    post_float = floats[:, 1:, :]
    post_uint8 = uint8[:, 1:, :]
    return {
        "mechanical_state": post_mech[:, -1, :],
        "mechanical_history": history_matrix(post_mech, length),
        "float_raster": post_float[:, -1, :],
        "uint8_raster": post_uint8[:, -1, :],
        "float_raster_history": history_matrix(post_float, length),
        "uint8_raster_history": history_matrix(post_uint8, length),
    }


def _probe_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ids: tuple[int, ...],
    val_ids: tuple[int, ...],
    test_ids: tuple[int, ...],
    n_classes: int,
    kind: str,
) -> dict[str, Any]:
    return evaluate_probe(
        kind,
        features[list(train_ids)],
        labels[list(train_ids)],
        features[list(val_ids)],
        labels[list(val_ids)],
        features[list(test_ids)],
        labels[list(test_ids)],
        n_classes,
    )


def _append_probe(
    rows: list[dict[str, Any]],
    matrices: dict[str, Any],
    *,
    protocol_id: str,
    family: str,
    history_length: int,
    task: str,
    result: dict[str, Any],
) -> None:
    for split in ("train", "val", "test"):
        rows.append(
            {
                "protocol_id": protocol_id,
                "family": family,
                "history_length": history_length,
                "task": task,
                "probe": result["probe"],
                "split": split,
                "accuracy": result[split]["accuracy"],
                "n": result[split]["n"],
                "d_in": result["d_in"],
            }
        )
        matrices[f"{protocol_id}|{family}|h{history_length}|{task}|{result['probe']}|{split}"] = result[split]["confusion"]


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split = fresh_pool_v2_split(int(config["n_episodes"]))
    split.assert_disjoint()
    projector = RasterProjector(64, 64, 3, int(config["raster_projection_dim"]), int(config["projection_seed"]))
    save_times = tuple(int(value) for value in config["save_times"])
    history_lengths = tuple(int(value) for value in config["history_lengths"])
    command = "python3.13 experiments/material_identifiability/run_study.py"
    with RunRecorder(
        config["experiment_id"],
        repo_root=ROOT,
        config=config,
        command=command,
        seed=int(config["dump_seed"]),
        model={"name": None, "note": "fixed probes, no world model"},
        environment={"protocols": [item["id"] for item in config["protocols"]]},
        dataset_split={
            "split_version": "fresh_pool_v2",
            "n_episodes": split.n_episodes,
            "train": list(split.train),
            "val": list(split.val),
            "test": list(split.test),
        },
        checkpoint={"path": None, "note": "no checkpoint"},
    ) as run:
        rows: list[dict[str, Any]] = []
        matrices: dict[str, Any] = {}
        protocol_meta: list[dict[str, Any]] = []
        for spec_cfg in config["protocols"]:
            protocol_id = spec_cfg["id"]
            print(protocol_id, flush=True)
            environment = environment_from_dict(
                yaml.safe_load((ROOT / spec_cfg["environment"]).read_text(encoding="utf-8"))
            )
            episodes = []
            for episode in range(int(config["n_episodes"])):
                episodes.append(
                    simulate_episode(
                        environment,
                        spec_cfg["schedule"],
                        episode,
                        episode_length=int(config["episode_length"]),
                        save_times=save_times,
                        dump_seed=int(config["dump_seed"]),
                        amplitude=float(spec_cfg["amplitude"]),
                        period=int(config["period"]),
                        bend_offset=float(config["bend_offset"]),
                        max_particles=int(config["max_particles"]),
                        projector=projector,
                    )
                )
            stable_fraction = float(np.mean([episode["stable"] for episode in episodes]))
            mech = np.stack([episode["mech"] for episode in episodes], axis=0)
            floats = np.stack([episode["float_raster"] for episode in episodes], axis=0)
            uint8 = np.stack([episode["uint8"] for episode in episodes], axis=0)
            stiffness = np.asarray([episode["class_id"] for episode in episodes], dtype=np.int64)
            geometry = np.asarray([episode["geom_id"] for episode in episodes], dtype=np.int64)
            privileged = np.stack(
                [np.array([episode["mu"], episode["lam"]], dtype=np.float32) for episode in episodes],
                axis=0,
            )
            finite = bool(np.all(np.isfinite(mech)) and np.all(np.isfinite(floats)) and np.all(np.isfinite(uint8)))
            protocol_meta.append(
                {
                    "protocol_id": protocol_id,
                    "selectable": bool(spec_cfg["selectable"]),
                    "stable_fraction": stable_fraction,
                    "stable": bool(finite and stable_fraction >= 0.95),
                    "schedule": spec_cfg["schedule"],
                    "amplitude": float(spec_cfg["amplitude"]),
                    "environment": spec_cfg["environment"],
                }
            )
            train_ids = tuple(range(*split.train))
            val_ids = tuple(range(*split.val))
            test_ids = tuple(range(*split.test))
            families_fixed = {
                "privileged": privileged,
                "geometry_uint8_t0": uint8[:, 0, :],
                "geometry_mechanical_t0": mech[:, 0, :],
            }
            for family, features in families_fixed.items():
                task = "stiffness" if family == "privileged" else "geometry"
                labels = stiffness if task == "stiffness" else geometry
                n_classes = 3 if task == "stiffness" else 2
                for kind in ("linear", "mlp"):
                    result = _probe_split(features, labels, train_ids, val_ids, test_ids, n_classes, kind)
                    stored_family = "privileged" if family == "privileged" else family
                    _append_probe(
                        rows,
                        matrices,
                        protocol_id=protocol_id,
                        family=stored_family,
                        history_length=1,
                        task=task,
                        result=result,
                    )
            for length in history_lengths:
                bundle = _features_for(mech, floats, uint8, length)
                for family, features in bundle.items():
                    tasks = ("stiffness",) if length < max(history_lengths) or "history" not in family else ("stiffness", "geometry")
                    if family in {"mechanical_state", "float_raster", "uint8_raster"} and length != 1:
                        continue
                    if family in {"mechanical_state", "float_raster", "uint8_raster"}:
                        tasks = ("stiffness",)
                    for task in tasks:
                        labels = stiffness if task == "stiffness" else geometry
                        n_classes = 3 if task == "stiffness" else 2
                        for kind in ("linear", "mlp"):
                            result = _probe_split(features, labels, train_ids, val_ids, test_ids, n_classes, kind)
                            _append_probe(
                                rows,
                                matrices,
                                protocol_id=protocol_id,
                                family=family,
                                history_length=1 if "history" not in family else length,
                                task=task,
                                result=result,
                            )
        max_history = max(history_lengths)
        val_records = []
        for meta in protocol_meta:
            protocol_id = meta["protocol_id"]
            val_records.append(
                {
                    "protocol_id": protocol_id,
                    "selectable": meta["selectable"],
                    "stable": meta["stable"],
                    "val": {
                        "privileged_stiffness_linear": _lookup(
                            rows, protocol_id=protocol_id, family="privileged", task="stiffness", probe="linear", split="val", history_length=1
                        ),
                        "geometry_uint8_linear": _lookup(
                            rows, protocol_id=protocol_id, family="geometry_uint8_t0", task="geometry", probe="linear", split="val", history_length=1
                        ),
                        "geometry_mechanical_linear": _lookup(
                            rows, protocol_id=protocol_id, family="geometry_mechanical_t0", task="geometry", probe="linear", split="val", history_length=1
                        ),
                        "mechanical_history_stiffness_linear": _lookup(
                            rows,
                            protocol_id=protocol_id,
                            family="mechanical_history",
                            task="stiffness",
                            probe="linear",
                            split="val",
                            history_length=max_history,
                        ),
                        "uint8_history_stiffness_linear": _lookup(
                            rows,
                            protocol_id=protocol_id,
                            family="uint8_raster_history",
                            task="stiffness",
                            probe="linear",
                            split="val",
                            history_length=max_history,
                        ),
                    },
                }
            )
        decision = select_protocol(
            val_records,
            chance=float(config["chance_stiffness"]),
            margin=float(config["margin_above_chance"]),
            privileged_min=float(config["privileged_min"]),
            geometry_min=float(config["geometry_min"]),
        )
        figure_dir = FIGURE_DIR
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_accuracy_vs_protocol(rows, figure_dir / "accuracy_vs_excitation.png", max_history)
        plot_history_overview(rows, figure_dir / "accuracy_vs_history_uint8_val.png", "uint8_raster_history")
        plot_history_overview(rows, figure_dir / "accuracy_vs_history_mechanical_val.png", "mechanical_history")
        if decision["protocol_id"] is not None:
            plot_history_curve(rows, decision["protocol_id"], figure_dir / "accuracy_vs_history.png")
            for family, stem in (
                ("uint8_raster_history", "confusion_uint8_history"),
                ("mechanical_history", "confusion_mechanical_history"),
            ):
                for split in ("val", "test"):
                    key = f"{decision['protocol_id']}|{family}|h{max_history}|stiffness|linear|{split}"
                    plot_confusion(
                        matrices[key],
                        figure_dir / f"{stem}_{split}.png",
                        f"{decision['protocol_id']} {family} {split}",
                    )
        summary = {
            "decision": decision,
            "protocols": protocol_meta,
            "val_records": val_records,
            "chance_stiffness": float(config["chance_stiffness"]),
            "margin_above_chance": float(config["margin_above_chance"]),
            "learned_model_evaluated": False,
            "split": {
                "split_version": "fresh_pool_v2",
                "train": list(split.train),
                "val": list(split.val),
                "test": list(split.test),
            },
        }
        (run.run_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
        (run.run_dir / "confusion.json").write_text(json.dumps(matrices) + "\n", encoding="utf-8")
        with (run.run_dir / "probe_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        run.complete(to_jsonable({"decision": decision, "n_rows": len(rows), "learned_model_evaluated": False}))
    print(run.run_dir)


if __name__ == "__main__":
    main()
