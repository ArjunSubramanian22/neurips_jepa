"""Queue local jobs without hiding failures or resizing a model.

    python -m experiments.queue --config configs/experiments/queue_example.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.registry import capture_git, to_jsonable, utc_now

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "runs" / "queue_v1" / "registry.jsonl"
OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "MPS backend out of memory",
)
TRANSIENT_MARKERS = (
    "Resource temporarily unavailable",
    "Connection reset by peer",
)
TERMINAL = frozenset({"COMPLETE", "FAILED", "INTERRUPTED"})


def config_hash(job: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(job), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def detect_devices() -> dict[str, Any]:
    """CUDA, then MPS, then CPU. This does not change a model."""
    from src.config import resolve_device

    selected = resolve_device("auto")
    cuda = False
    mps = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        backend = getattr(torch.backends, "mps", None)
        mps = bool(backend is not None and backend.is_available())
    except ImportError:
        pass
    return {"selected": selected.type, "cuda": cuda, "mps": mps}


def load_queue_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError("queue config needs a jobs list.")
    if not payload["jobs"]:
        raise ValueError("queue config has no jobs.")
    return payload


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_registry(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = to_jsonable(dict(record))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(encoded, sort_keys=True) + "\n")


def latest_status(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["job_id"])] = dict(record)
    return latest


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def is_oom(stderr: str) -> bool:
    return any(marker in stderr for marker in OOM_MARKERS)


def is_transient(stderr: str, exit_code: int, transient_exit_codes: Sequence[int]) -> bool:
    if int(exit_code) in {int(code) for code in transient_exit_codes}:
        return True
    return any(marker in stderr for marker in TRANSIENT_MARKERS)


def apply_fallback_batch(command: Sequence[str], flag: str, size: int) -> list[str]:
    """Replace or append one batch-size flag. No other argument changes."""
    updated = [str(part) for part in command]
    if flag in updated:
        index = updated.index(flag)
        if index + 1 >= len(updated):
            raise ValueError(f"{flag} is missing a value.")
        updated[index + 1] = str(int(size))
        return updated
    return updated + [str(flag), str(int(size))]


def resolve_job_device(job: Mapping[str, Any], devices: Mapping[str, Any]) -> str:
    requested = str(job.get("device", "auto"))
    if requested == "auto":
        return str(devices["selected"])
    if requested == "cuda" and not devices["cuda"]:
        raise RuntimeError("cuda was requested but is not available.")
    if requested == "mps" and not devices["mps"]:
        raise RuntimeError("mps was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"unknown device {requested}.")
    return requested


def _accelerator(device: str) -> bool:
    return device in {"cuda", "mps"}


class _Process:
    def __init__(self, popen: subprocess.Popen[bytes], started: float) -> None:
        self.popen = popen
        self.started = started

    @property
    def pid(self) -> int:
        return int(self.popen.pid)

    def poll(self) -> int | None:
        code = self.popen.poll()
        return None if code is None else int(code)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def _start_process(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> _Process:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        popen = subprocess.Popen(
            list(command),
            stdout=stdout_handle,
            stderr=stderr_handle,
            cwd=ROOT,
        )
    return _Process(popen, time.perf_counter())


def reconcile_interrupted(
    records: Sequence[Mapping[str, Any]],
    *,
    alive: Callable[[int], bool] = pid_alive,
) -> list[dict[str, Any]]:
    """Return INTERRUPTED lines for RUNNING jobs whose process is gone."""
    notes = []
    for job_id, record in latest_status(records).items():
        if record.get("status") != "RUNNING":
            continue
        pid = int(record.get("pid") or 0)
        if alive(pid):
            continue
        notes.append(
            {
                "job_id": job_id,
                "config_hash": record.get("config_hash"),
                "git_commit": record.get("git_commit"),
                "status": "INTERRUPTED",
                "exit_code": None,
                "elapsed_seconds": record.get("elapsed_seconds"),
                "stdout_path": record.get("stdout_path"),
                "stderr_path": record.get("stderr_path"),
                "device": record.get("device"),
                "attempt": record.get("attempt"),
                "oom": bool(record.get("oom")),
                "batch_adapted": bool(record.get("batch_adapted")),
                "batch_size": record.get("batch_size"),
                "failure_reason": "process disappeared before a terminal status",
                "pid": pid,
                "command": record.get("command"),
                "started_at": record.get("started_at"),
                "ended_at": utc_now(),
            }
        )
    return notes


def _skip_reason(record: Mapping[str, Any] | None, config: str, git_commit: str) -> str | None:
    if record is None:
        return None
    if record.get("status") == "RUNNING":
        return "already running"
    if record.get("status") != "COMPLETE":
        return None
    if record.get("config_hash") == config and record.get("git_commit") == git_commit:
        return "complete with matching config and git commit"
    return None


def _can_start(job: Mapping[str, Any], running: Sequence[Mapping[str, Any]], max_parallel: int) -> bool:
    if len(running) >= max_parallel:
        return False
    if any(item["exclusive"] for item in running) or job["exclusive"]:
        return len(running) == 0
    if _accelerator(str(job["device"])) and any(_accelerator(str(item["device"])) for item in running):
        return False
    return True


def _log_paths(registry: Path, job_id: str, attempt: int) -> tuple[Path, Path]:
    folder = registry.parent / "logs" / job_id
    return folder / f"attempt_{attempt}.out", folder / f"attempt_{attempt}.err"


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def run_queue(
    config: Mapping[str, Any],
    *,
    registry: Path,
    dry_run: bool = False,
    git_commit: str | None = None,
    devices: Mapping[str, Any] | None = None,
    alive: Callable[[int], bool] = pid_alive,
    starter: Callable[[Sequence[str], Path, Path], _Process] = _start_process,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Run or describe the queue. Failures stay in the registry."""
    hardware = dict(devices or detect_devices())
    commit = git_commit if git_commit is not None else capture_git(ROOT).get("commit")
    records = read_registry(registry)
    interrupted = reconcile_interrupted(records, alive=alive)
    if interrupted and not dry_run:
        for note in interrupted:
            append_registry(registry, note)
        records.extend(interrupted)
    latest = latest_status(records)
    prepared = []
    for raw in config["jobs"]:
        job = dict(raw)
        job["config_hash"] = config_hash(raw)
        try:
            job["device"] = resolve_job_device(job, hardware)
            job["device_error"] = None
        except RuntimeError as exc:
            job["device"] = str(raw.get("device", "auto"))
            job["device_error"] = str(exc)
        job["exclusive"] = bool(job.get("exclusive", _accelerator(str(job["device"]))))
        job["max_transient_retries"] = int(job.get("max_transient_retries", 2))
        job["transient_exit_codes"] = [int(code) for code in job.get("transient_exit_codes", [])]
        prepared.append(job)
    max_parallel = int(config.get("max_parallel", 1))
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1.")
    plan = []
    for job in prepared:
        reason = None if job["device_error"] else _skip_reason(latest.get(str(job["id"])), job["config_hash"], str(commit))
        if not retry_failed and latest.get(str(job["id"]), {}).get("status") == "FAILED" and reason is None:
            reason = "failed previously; pass --retry-failed to run it again"
        plan.append({"id": job["id"], "action": "skip" if reason else "run", "reason": reason, "device": job["device"]})
    if dry_run:
        return {
            "dry_run": True,
            "devices": hardware,
            "git_commit": commit,
            "max_parallel": max_parallel,
            "plan": plan,
            "failures": [],
        }
    pending = [job for job, item in zip(prepared, plan, strict=True) if item["action"] == "run"]
    running: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def launch(job: dict[str, Any], command: list[str], attempt: int, *, oom: bool, adapted: bool, batch_size: int | None) -> None:
        attempt = int(job.get("_attempt", attempt))
        adapted = bool(job.get("_adapted", adapted))
        oom = bool(job.get("_oom_seen", oom))
        batch_size = job.get("batch_size", batch_size)
        stdout_path, stderr_path = _log_paths(registry, str(job["id"]), attempt)
        process = starter(command, stdout_path, stderr_path)
        record = {
            "job_id": job["id"],
            "config_hash": job["config_hash"],
            "git_commit": commit,
            "status": "RUNNING",
            "exit_code": None,
            "elapsed_seconds": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "device": job["device"],
            "attempt": attempt,
            "oom": oom,
            "batch_adapted": adapted,
            "batch_size": batch_size,
            "failure_reason": None,
            "pid": process.pid,
            "command": command,
            "started_at": utc_now(),
            "started_unix": time.time(),
            "ended_at": None,
        }
        append_registry(registry, record)
        running.append(
            {
                "job": job,
                "process": process,
                "command": command,
                "attempt": attempt,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "oom_seen": oom,
                "adapted": adapted,
                "batch_size": batch_size,
                "transient_retries": job.get("_transient_retries", 0),
                "device": job["device"],
                "exclusive": job["exclusive"],
                "record": record,
            }
        )

    for job in list(pending):
        if job["device_error"]:
            pending.remove(job)
            _finish_without_process(registry, job, commit, job["device_error"], failures)
    while pending or running:
        still_running = []
        for item in running:
            code = item["process"].poll()
            if code is None:
                still_running.append(item)
                continue
            _finish_process(registry, item, code, commit, pending, failures)
        running = still_running
        started = False
        for job in list(pending):
            if not _can_start(job, running, max_parallel):
                continue
            pending.remove(job)
            command = [str(part) for part in job["command"]]
            launch(
                job,
                command,
                1,
                oom=False,
                adapted=False,
                batch_size=job.get("batch_size"),
            )
            started = True
        if not started and (pending or running):
            time.sleep(0.02)
    return {
        "dry_run": False,
        "devices": hardware,
        "git_commit": commit,
        "max_parallel": max_parallel,
        "plan": plan,
        "failures": failures,
    }


def _finish_without_process(registry: Path, job: Mapping[str, Any], commit: str | None, reason: str, failures: list[dict[str, Any]]) -> None:
    record = {
        "job_id": job["id"],
        "config_hash": job["config_hash"],
        "git_commit": commit,
        "status": "FAILED",
        "exit_code": None,
        "elapsed_seconds": 0.0,
        "stdout_path": None,
        "stderr_path": None,
        "device": job.get("device"),
        "attempt": 1,
        "oom": False,
        "batch_adapted": False,
        "batch_size": job.get("batch_size"),
        "failure_reason": reason,
        "pid": None,
        "command": list(job.get("command") or []),
        "started_at": utc_now(),
        "ended_at": utc_now(),
    }
    append_registry(registry, record)
    failures.append({"job_id": job["id"], "failure_reason": reason, "oom": False})


def _finish_process(
    registry: Path,
    item: dict[str, Any],
    code: int,
    commit: str | None,
    pending: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    job = item["job"]
    stderr = _read_text(item["stderr_path"])
    elapsed = float(item["process"].elapsed())
    oom = is_oom(stderr)
    adapted = bool(item["adapted"])
    status = "COMPLETE" if code == 0 else "FAILED"
    reason = None if code == 0 else (stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {code}")
    if oom and not adapted and job.get("adaptive_batch") and job.get("fallback_batch_size") and job.get("batch_flag"):
        command = apply_fallback_batch(item["command"], str(job["batch_flag"]), int(job["fallback_batch_size"]))
        record = _terminal_record(item, code, elapsed, "ATTEMPT_FAILED", reason, oom=True, adapted=False, commit=commit)
        record["failure_reason"] = "oom; retrying once with the configured smaller batch"
        append_registry(registry, record)
        pending.insert(0, _adapted_job(job, command, item["attempt"] + 1))
        return
    if code != 0 and not oom and is_transient(stderr, code, job.get("transient_exit_codes", [])):
        if int(item["transient_retries"]) < int(job["max_transient_retries"]):
            record = _terminal_record(item, code, elapsed, "ATTEMPT_FAILED", reason, oom=False, adapted=adapted, commit=commit)
            record["failure_reason"] = f"transient failure: {reason}"
            append_registry(registry, record)
            retry = dict(job)
            retry["_attempt"] = int(item["attempt"]) + 1
            retry["_transient_retries"] = int(item["transient_retries"]) + 1
            retry["_adapted"] = adapted
            retry["command"] = list(item["command"])
            pending.insert(0, retry)
            return
    record = _terminal_record(item, code, elapsed, status, reason, oom=oom, adapted=adapted, commit=commit)
    append_registry(registry, record)
    if status == "FAILED":
        failures.append({"job_id": job["id"], "failure_reason": record["failure_reason"], "oom": oom, "exit_code": code})


def _adapted_job(job: Mapping[str, Any], command: list[str], attempt: int) -> dict[str, Any]:
    adapted = dict(job)
    adapted["command"] = command
    adapted["_attempt"] = int(attempt)
    adapted["_adapted"] = True
    adapted["_oom_seen"] = True
    adapted["batch_size"] = int(job["fallback_batch_size"])
    return adapted


def _terminal_record(
    item: Mapping[str, Any],
    code: int,
    elapsed: float,
    status: str,
    reason: str | None,
    *,
    oom: bool,
    adapted: bool,
    commit: str | None,
) -> dict[str, Any]:
    job = item["job"]
    return {
        "job_id": job["id"],
        "config_hash": job["config_hash"],
        "git_commit": commit,
        "status": status,
        "exit_code": int(code),
        "elapsed_seconds": float(elapsed),
        "stdout_path": str(item["stdout_path"]),
        "stderr_path": str(item["stderr_path"]),
        "device": job["device"],
        "attempt": int(item["attempt"]),
        "oom": bool(oom),
        "batch_adapted": bool(adapted),
        "batch_size": job.get("batch_size"),
        "failure_reason": reason,
        "pid": item["process"].pid,
        "command": list(item["command"]),
        "started_at": item["record"]["started_at"],
        "started_unix": item["record"].get("started_unix"),
        "ended_at": utc_now(),
        "ended_unix": time.time(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue local experiment jobs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    config = load_queue_config(Path(args.config))
    result = run_queue(
        config,
        registry=Path(args.registry),
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    if result["dry_run"]:
        return 0
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
