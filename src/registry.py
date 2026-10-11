"""Append-only registry for one experiment run.

A run directory stores ``record.json``. The registry file
``experiments/registry.jsonl`` receives one line when the run reaches
``COMPLETE`` or ``FAILED``. A second launch of the same experiment id is
refused so a finished record is not overwritten.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"COMPLETE", "FAILED"})


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def capture_software() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "torch", "matplotlib", "yaml"):
        try:
            module = __import__(name if name != "yaml" else "yaml")
            versions[name] = getattr(module, "__version__", "present")
        except ImportError:
            versions[name] = None
    return versions


def capture_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
    }
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch_cuda"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    info["torch_mps"] = bool(mps is not None and mps.is_available())
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or size <= 0:
        return None
    return int(pages * size)


def capture_git(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": None,
            "dirty_count": None,
            "status": "unavailable",
        }
    lines = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(lines),
        "dirty_count": len(lines),
        "status": "measured",
    }


def _json_ready(payload: Any) -> None:
    json.dumps(payload)


class RunRecorder:
    """Write one run record. Use as a context manager to mark crashes FAILED."""

    def __init__(
        self,
        experiment_id: str,
        *,
        repo_root: Path,
        config: dict[str, Any],
        command: str,
        seed: int | str,
        model: dict[str, Any],
        environment: dict[str, Any],
        dataset_split: dict[str, Any],
        checkpoint: dict[str, Any] | None,
        git_state: dict[str, Any] | None = None,
    ) -> None:
        if not experiment_id or "/" in experiment_id or experiment_id in {".", ".."}:
            raise ValueError("experiment_id must be a single path segment.")
        self.repo_root = Path(repo_root)
        self.run_dir = self.repo_root / "experiments" / "runs" / experiment_id
        self.record_path = self.run_dir / "record.json"
        self.registry_path = self.repo_root / "experiments" / "registry.jsonl"
        if self.record_path.exists():
            raise FileExistsError(
                f"{self.record_path} already exists. Start a new experiment id "
                "instead of overwriting a previous run."
            )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        git = git_state if git_state is not None else capture_git(self.repo_root)
        self.record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"),
            "git_dirty_count": git.get("dirty_count"),
            "git_status": git.get("status"),
            "config": config,
            "seed": seed,
            "model": model,
            "environment": environment,
            "dataset_split": dataset_split,
            "checkpoint": checkpoint,
            "command": command,
            "software": capture_software(),
            "hardware": capture_hardware(),
            "started_at": utc_now(),
            "ended_at": None,
            "output_dir": str(self.run_dir.relative_to(self.repo_root)),
            "status": "RUNNING",
            "metrics": None,
            "failure_reason": None,
        }
        _json_ready(self.record)
        self._write_record()
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def complete(self, metrics: dict[str, Any]) -> None:
        if self.record["status"] != "RUNNING":
            raise RuntimeError(f"cannot complete a run in status {self.record['status']}.")
        self.record["status"] = "COMPLETE"
        self.record["metrics"] = metrics
        self.record["failure_reason"] = None
        self.record["ended_at"] = utc_now()
        _json_ready(self.record)
        self._write_record()
        self._append_registry()

    def fail(self, reason: str) -> None:
        if self.record["status"] in TERMINAL_STATUSES:
            raise RuntimeError(f"cannot fail a run in status {self.record['status']}.")
        if not str(reason).strip():
            raise ValueError("failure reason must be non-empty.")
        self.record["status"] = "FAILED"
        self.record["failure_reason"] = str(reason)
        self.record["ended_at"] = utc_now()
        _json_ready(self.record)
        self._write_record()
        self._append_registry()

    def _write_record(self) -> None:
        self.record_path.write_text(
            json.dumps(self.record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _append_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.record, sort_keys=True) + "\n")

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if exc_type is not None and self.record["status"] == "RUNNING":
            self.fail(f"{exc_type.__name__}: {exc}")
        return False
