#!/usr/bin/env python3
"""Run the remaining confirmatory training cells one at a time.

Skips a cell when its last.pt already exists. A failed cell is logged and the
queue continues. Direction, magnitude, and combined alignment stay out of this
queue: their pair cache is not on disk.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get(
    "PYTHON",
    "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13",
)
LOG = ROOT / "experiments" / "confirmatory_queue" / "status.jsonl"
CONFIG_DIR = ROOT / "experiments" / "confirmatory_queue" / "configs"
MIN_FREE_BYTES = 4 * (1024**3)


def _log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def _free_bytes() -> int:
    return shutil.disk_usage(ROOT).free


def _ready(path: str) -> bool:
    file = ROOT / path
    return file.is_file() and file.stat().st_size > 0


def _write_config(name: str, payload: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _load(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def _high_from(base_rel: str, name: str, **updates: object) -> str:
    payload = _load(base_rel)
    payload["split_version"] = "fresh_pool_v2"
    payload.setdefault("is_test_set", False)
    data = payload.setdefault("data", {})
    data["split_version"] = "fresh_pool_v2"
    data["dataset_path"] = "outputs/physics_mpm_high"
    data["n_episodes"] = 1800
    training = payload.setdefault("training", {})
    training["output_dir"] = "outputs/mpm_high_runs"
    training["device"] = "mps"
    training["num_workers"] = 0
    model = payload.setdefault("model", {})
    for key, value in updates.items():
        if key.startswith("training."):
            training[key.split(".", 1)[1]] = value
        elif key.startswith("model."):
            model[key.split(".", 1)[1]] = value
        else:
            payload[key] = value
    return str(_write_config(name, payload).relative_to(ROOT))


def jobs() -> list[dict]:
    rows: list[dict] = []
    for seed in (42, 43, 44, 45, 46):
        for config, run in (
            ("configs/experiments/arm_jepa_v1.yaml", f"jepa_s{seed}"),
            ("configs/experiments/arm_rssm_v1.yaml", f"rssm_s{seed}"),
            ("configs/experiments/arm_latent_tf_v1.yaml", f"latent_tf_s{seed}"),
        ):
            rows.append(
                {
                    "id": f"mujoco_arm/{run}",
                    "kind": "train",
                    "config": config,
                    "seed": seed,
                    "run_name": run,
                    "output_dir": "outputs/mujoco_arm_runs",
                    "device": "cpu",
                    "checkpoint": f"outputs/mujoco_arm_runs/{run}/last.pt",
                }
            )
    high = "configs/experiments/mpm_high_jepa_v1.yaml"
    for seed in (44, 45, 46):
        rows.append(
            {
                "id": f"mpm_high/jepa_s{seed}",
                "kind": "train",
                "config": high,
                "seed": seed,
                "run_name": f"jepa_s{seed}",
                "output_dir": "outputs/mpm_high_runs",
                "device": "mps",
                "checkpoint": f"outputs/mpm_high_runs/jepa_s{seed}/last.pt",
            }
        )
    for seed in (43, 44, 45, 46):
        rows.append(
            {
                "id": f"mpm_high/noaction_s{seed}",
                "kind": "train",
                "config": high,
                "seed": seed,
                "run_name": f"noaction_s{seed}",
                "output_dir": "outputs/mpm_high_runs",
                "device": "mps",
                "no_action": True,
                "checkpoint": f"outputs/mpm_high_runs/noaction_s{seed}/last.pt",
            }
        )
    for seed in (42, 43, 44, 45, 46):
        rows.append(
            {
                "id": f"mpm_high/mse_s{seed}",
                "kind": "train",
                "config": high,
                "seed": seed,
                "run_name": f"mse_s{seed}",
                "output_dir": "outputs/mpm_high_runs",
                "device": "mps",
                "loss_type": "mse",
                "checkpoint": f"outputs/mpm_high_runs/mse_s{seed}/last.pt",
            }
        )
    for family, base in (
        ("rssm", "configs/experiments/rssm_v1.yaml"),
        ("latent_tf", "configs/experiments/latent_transformer_v1.yaml"),
    ):
        for seed in (42, 43, 44, 45, 46):
            rows.append(
                {
                    "id": f"mpm_high/{family}_s{seed}",
                    "kind": "train",
                    "config": _high_from(base, f"mpm_high_{family}.yaml"),
                    "seed": seed,
                    "run_name": f"{family}_s{seed}",
                    "output_dir": "outputs/mpm_high_runs",
                    "device": "mps",
                    "checkpoint": f"outputs/mpm_high_runs/{family}_s{seed}/last.pt",
                }
            )
    for fuse in ("film", "gate", "cross_attn"):
        rows.append(
            {
                "id": f"mpm_high/{fuse}_s42",
                "kind": "train",
                "config": _high_from(high, f"mpm_high_{fuse}.yaml", **{"model.fuse_mode": fuse}),
                "seed": 42,
                "run_name": f"{fuse}_s42",
                "output_dir": "outputs/mpm_high_runs",
                "device": "mps",
                "checkpoint": f"outputs/mpm_high_runs/{fuse}_s42/last.pt",
            }
        )
    rows.append(
        {
            "id": "mpm_high/inverse_s42",
            "kind": "train",
            "config": _high_from(
                high,
                "mpm_high_inverse.yaml",
                **{"training.inverse_aux_weight": 1.0, "training.inverse_mode": "joint"},
            ),
            "seed": 42,
            "run_name": "inverse_s42",
            "output_dir": "outputs/mpm_high_runs",
            "device": "mps",
            "checkpoint": "outputs/mpm_high_runs/inverse_s42/last.pt",
            "notes": "weight 1.0 is a screen, not a validation-selected weight",
        }
    )
    for loss, action, seed, out, run in (
        ("vicreg", "action", 46, "results/reviewer_strengthening/05_more_seeds", "vicreg_action_s46"),
        ("vicreg", "noaction", 46, "results/reviewer_strengthening/05_more_seeds", "vicreg_noaction_s46"),
        ("mse", "action", 44, "results/reviewer_strengthening/06_corrected_objective", "mse_action_s44"),
        ("mse", "action", 45, "results/reviewer_strengthening/06_corrected_objective", "mse_action_s45"),
        ("mse", "action", 46, "results/reviewer_strengthening/06_corrected_objective", "mse_action_s46"),
    ):
        rows.append(
            {
                "id": f"mpm_legacy/{run}",
                "kind": "reviewer",
                "seed": seed,
                "loss": loss,
                "action": action,
                "output_dir": out,
                "run_name": run,
                "checkpoint": f"{out}/{run}/last.pt",
            }
        )
    for family, config in (
        ("rssm", "configs/experiments/rssm_v1.yaml"),
        ("latent_tf", "configs/experiments/latent_transformer_v1.yaml"),
        ("obs_pred", "configs/experiments/obs_pred_v1.yaml"),
    ):
        rows.append(
            {
                "id": f"mpm_legacy/{family}_s42",
                "kind": "train",
                "config": config,
                "seed": 42,
                "run_name": f"{family}_s42",
                "output_dir": "outputs/confirmatory/mpm_legacy",
                "device": "mps",
                "checkpoint": f"outputs/confirmatory/mpm_legacy/{family}_s42/last.pt",
            }
        )
    return rows


def _train_command(job: dict) -> list[str]:
    if job["kind"] == "reviewer":
        return [
            "bash",
            "scripts/train_reviewer_seed.sh",
            str(job["seed"]),
            job["loss"],
            job["action"],
            job["output_dir"],
            job["run_name"],
        ]
    command = [
        PYTHON,
        "src/train.py",
        "--mode",
        "train",
        "--config",
        job["config"],
        "--seed",
        str(job["seed"]),
        "--run-name",
        job["run_name"],
        "--output-dir",
        job["output_dir"],
        "--device",
        job["device"],
        "--num-workers",
        "0",
    ]
    if job.get("no_action"):
        command.append("--no-action")
    if job.get("loss_type"):
        command.extend(["--loss-type", job["loss_type"]])
    return command


def _run(job: dict) -> None:
    if _ready(job["checkpoint"]):
        _log({"id": job["id"], "status": "SKIPPED", "reason": "last.pt exists", "checkpoint": job["checkpoint"]})
        return
    free = _free_bytes()
    if free < MIN_FREE_BYTES:
        _log({"id": job["id"], "status": "BLOCKED", "reason": "disk below 4GB", "free_bytes": free})
        return
    command = _train_command(job)
    _log({"id": job["id"], "status": "STARTED", "command": command})
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / "outputs" / ".mplconfig")
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, env=env)
    elapsed = round(time.time() - started, 1)
    status = "COMPLETE" if completed.returncode == 0 and _ready(job["checkpoint"]) else "FAILED"
    _log(
        {
            "id": job["id"],
            "status": status,
            "returncode": completed.returncode,
            "seconds": elapsed,
            "checkpoint": job["checkpoint"],
        }
    )
    if status == "FAILED":
        _log({"id": job["id"], "status": "RETRY", "reason": "one retry after a non-zero exit"})
        completed = subprocess.run(command, cwd=ROOT, env=env)
        status = "COMPLETE" if completed.returncode == 0 and _ready(job["checkpoint"]) else "FAILED"
        _log(
            {
                "id": job["id"],
                "status": status,
                "returncode": completed.returncode,
                "seconds": round(time.time() - started, 1),
                "checkpoint": job["checkpoint"],
                "attempt": 2,
            }
        )


def main() -> None:
    planned = jobs()
    _log(
        {
            "status": "QUEUE_START",
            "n_jobs": len(planned),
            "skipped_blocked": [
                "mpm_high direction seed 42",
                "mpm_high magnitude seed 42",
                "mpm_high combined seed 42",
            ],
            "blocked_reason": "outputs/counterfactual_pairs/force_reversal_train.npz is absent",
        }
    )
    for job in planned:
        _run(job)
    _log({"status": "QUEUE_DONE"})


if __name__ == "__main__":
    main()
