#!/usr/bin/env python3
"""High-excitation queue.

The active phase waits until rssm_s46/last.pt exists, screens the checkpoints
that already exist, writes the horizon-1 summary, and stops. It does not
start latent-transformer, fusion, inverse, alignment, or legacy training.
It never calls scripts/eval_mpm_high_test_once.py.

``jobs()`` is the deferred training list. ``main`` does not call it.
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


SCREEN_H1 = [
    ("jepa", 44, "vicreg_action"),
    ("jepa", 45, "vicreg_action"),
    ("jepa", 46, "vicreg_action"),
    ("noaction", 43, "vicreg_noaction"),
    ("noaction", 44, "vicreg_noaction"),
    ("noaction", 45, "vicreg_noaction"),
    ("noaction", 46, "vicreg_noaction"),
    ("mse", 42, "mse_action"),
    ("mse", 43, "mse_action"),
    ("mse", 44, "mse_action"),
    ("mse", 45, "mse_action"),
    ("mse", 46, "mse_action"),
    ("rssm", 42, "rssm"),
    ("rssm", 43, "rssm"),
    ("rssm", 44, "rssm"),
    ("rssm", 45, "rssm"),
    ("rssm", 46, "rssm"),
]
PHASE_FILE = ROOT / "experiments" / "confirmatory_queue" / "ACTIVE_PHASE.txt"
ENABLED_PHASE = "SCREEN_H1"


def planned_screen_commands() -> list[list[str]]:
    """Commands the active phase will run. The held-out test script is absent."""
    commands = []
    for family, seed, model_name in SCREEN_H1:
        commands.append(
            [
                PYTHON,
                "scripts/eval_mpm_high_screen.py",
                "--checkpoint",
                f"outputs/mpm_high_runs/{family}_s{seed}/last.pt",
                "--experiment-id",
                f"mpm_high_{family}_s{seed}_val_screen_v2",
                "--model-name",
                model_name,
                "--seed",
                str(seed),
                "--n-episodes",
                "24",
            ]
        )
    commands.append([PYTHON, "scripts/summarize_mpm_high_h1.py"])
    return commands


def rssm_run_complete(run_dir: Path, epochs: int = 200) -> bool:
    """A run is finished only when history.json records every epoch.

    ``last.pt`` is also written at ``ckpt_interval``, so its presence does
    not mean training finished.
    """
    history_path = run_dir / "history.json"
    checkpoint = run_dir / "last.pt"
    if not history_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history") or []
    return int(payload.get("epochs", 0)) == int(epochs) and len(history) >= int(epochs)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _rssm46_running() -> bool:
    pid_path = ROOT / "experiments" / "confirmatory_queue" / "rssm_s46.pid"
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if _pid_alive(pid):
            return True
    completed = subprocess.run(["ps", "-ax", "-o", "command="], capture_output=True, text=True, check=False)
    return any("src/train.py" in line and "rssm_s46" in line for line in completed.stdout.splitlines())


def _start_rssm46() -> None:
    if _rssm46_running() or rssm_run_complete(ROOT / "outputs" / "mpm_high_runs" / "rssm_s46"):
        return
    log_path = ROOT / "experiments" / "confirmatory_queue" / "rssm_s46_train.log"
    pid_path = ROOT / "experiments" / "confirmatory_queue" / "rssm_s46.pid"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / "outputs" / ".mplconfig")
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            PYTHON,
            "src/train.py",
            "--mode",
            "train",
            "--config",
            "experiments/confirmatory_queue/configs/mpm_high_rssm.yaml",
            "--seed",
            "46",
            "--run-name",
            "rssm_s46",
            "--output-dir",
            "outputs/mpm_high_runs",
            "--device",
            "mps",
            "--num-workers",
            "0",
        ],
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    _log({"status": "STARTED", "phase": "TRAIN_BASELINES", "name": "rssm_s46", "pid": process.pid})


def _wait_for_rssm46() -> None:
    run_dir = ROOT / "outputs" / "mpm_high_runs" / "rssm_s46"
    while not rssm_run_complete(run_dir):
        if not _rssm46_running():
            _start_rssm46()
        else:
            _log({
                "status": "WAITING",
                "phase": "TRAIN_BASELINES",
                "checkpoint": str(run_dir / "last.pt"),
                "history": str(run_dir / "history.json"),
                "complete": False,
            })
        time.sleep(60)
    _log({"status": "COMPLETE", "phase": "TRAIN_BASELINES", "checkpoint": str(run_dir / "last.pt")})


def _screen(family: str, seed: int, model_name: str) -> int:
    experiment_id = f"mpm_high_{family}_s{seed}_val_screen_v2"
    screen_path = ROOT / "experiments" / "runs" / experiment_id / "screen.json"
    if screen_path.is_file():
        _log({"status": "SKIPPED_EXISTING", "phase": "SCREEN_H1", "name": experiment_id})
        return 0
    checkpoint = f"outputs/mpm_high_runs/{family}_s{seed}/last.pt"
    if not _ready(checkpoint):
        _log({"status": "BLOCKED", "phase": "SCREEN_H1", "name": experiment_id, "reason": "checkpoint missing"})
        return 1
    command = [
        PYTHON,
        "scripts/eval_mpm_high_screen.py",
        "--checkpoint",
        checkpoint,
        "--experiment-id",
        experiment_id,
        "--model-name",
        model_name,
        "--seed",
        str(seed),
        "--n-episodes",
        "24",
    ]
    _log({"status": "STARTED", "phase": "SCREEN_H1", "name": experiment_id, "command": command})
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / "outputs" / ".mplconfig")
    completed = subprocess.run(command, cwd=ROOT, env=env)
    if completed.returncode == 0 and screen_path.is_file():
        _log({"status": "COMPLETE", "phase": "SCREEN_H1", "name": experiment_id})
        return 0
    _log({"status": "FAILED", "phase": "SCREEN_H1", "name": experiment_id, "exit_code": completed.returncode})
    return int(completed.returncode or 1)


def main() -> None:
    PHASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PHASE_FILE.write_text(ENABLED_PHASE + "\n", encoding="utf-8")
    commands = planned_screen_commands()
    if any("eval_mpm_high_test_once.py" in part for command in commands for part in command):
        raise RuntimeError("the active queue must not call the held-out test script.")
    _log(
        {
            "status": "QUEUE_START",
            "phase": ENABLED_PHASE,
            "n_commands": len(commands),
            "deferred": "experiments/confirmatory_queue/deferred.json",
            "stops_after": "scripts/summarize_mpm_high_h1.py",
            "does_not_call": "scripts/eval_mpm_high_test_once.py",
        }
    )
    _wait_for_rssm46()
    for family, seed, model_name in SCREEN_H1:
        code = _screen(family, seed, model_name)
        if code != 0:
            _log({"status": "QUEUE_STOPPED", "phase": "SCREEN_H1"})
            raise SystemExit(code)
    summary = subprocess.run([PYTHON, "scripts/summarize_mpm_high_h1.py"], cwd=ROOT)
    _log({"status": "STOP_FOR_REVIEW", "phase": "SCREEN_H1", "summary_exit": summary.returncode})
    print("STOP. Horizon-1 screens are done. Multi-horizon audit does not start.")
    raise SystemExit(summary.returncode)


if __name__ == "__main__":
    main()
