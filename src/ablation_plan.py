"""Expand the declared ablation catalog. It does not cross every axis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

COMPLETE = "COMPLETE"
FAILED = "FAILED"
NOT_RUN = "NOT_RUN"
PARTIAL = "PARTIAL"
BLOCKED = "BLOCKED"
SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"
IDENTITY_FIELDS = (
    "version",
    "environment",
    "model",
    "conditioning",
    "loss",
    "resolution",
    "seed",
    "epochs",
    "dataset",
)


def load_catalog(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("is_test_set"):
        raise ValueError("the ablation catalog must not be marked as a test set.")
    if not payload.get("jobs"):
        raise ValueError("the catalog has no jobs.")
    return payload


def load_seed_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["seeds"] = [int(seed) for seed in payload["seeds"]]
    return payload


def measured_reference_seconds(catalog: Mapping[str, Any], repo_root: Path) -> float:
    """Mean wall clock of the measured JEPA histories. Missing files are omitted, not invented."""
    seconds = []
    for relative in catalog["cost"]["measured_histories"]:
        path = repo_root / relative
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("seconds")
        if isinstance(value, (int, float)) and value == value and value > 0:
            seconds.append(float(value))
    if not seconds:
        raise ValueError("no measured history seconds are available for the cost prior.")
    return sum(seconds) / len(seconds)


def _registry_checkpoint(registry: Mapping[str, Any], configuration: str, seed: int) -> str | None:
    for item in registry["configurations"]:
        if item["id"] != configuration:
            continue
        checkpoints = {int(key): value for key, value in item["checkpoints"].items()}
        if seed not in checkpoints:
            raise KeyError(f"{configuration} has no seed {seed}.")
        raw = checkpoints[seed]
        if raw in (None, ""):
            return None
        return str(raw)
    raise KeyError(configuration)


def _job_id(spec: Mapping[str, Any]) -> str:
    payload = json.dumps({key: spec[key] for key in IDENTITY_FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _seeds(template: Mapping[str, Any], registry: Mapping[str, Any]) -> list[int]:
    raw = template["seeds"]
    if raw == "registry":
        return list(registry["seeds"])
    return [int(seed) for seed in raw]


def _output_dir(version: str, template_id: str, seed: int) -> str:
    return f"results/ablations/{version}/runs/{template_id}_s{seed}"


def _command(template: Mapping[str, Any], seed: int, version: str) -> list[str]:
    kind = template.get("command_kind")
    run_name = f"{template['id']}_s{seed}"
    output_root = f"results/ablations/{version}/runs"
    if kind == "audit":
        return ["python3.13", "-m", "benchmark.audit", "--config", str(template["audit_config"])]
    if kind == "reviewer_seed":
        return [
            "bash",
            "scripts/train_reviewer_seed.sh",
            str(seed),
            str(template["objective"]),
            str(template["action"]),
            output_root,
            run_name,
        ]
    if kind == "train_config":
        if template.get("overrides"):
            config = f"results/ablations/{version}/configs/{run_name}.yaml"
        else:
            config = template.get("base_config") or f"results/ablations/{version}/configs/{run_name}.yaml"
        return [
            "python3.13",
            "-m",
            "src.train",
            "--mode",
            "train",
            "--config",
            str(config),
            "--seed",
            str(seed),
            "--output-dir",
            output_root,
            "--run-name",
            run_name,
            "--num-workers",
            "0",
        ]
    return []


def _cost_seconds(template: Mapping[str, Any], reference: float, multipliers: Mapping[str, float]) -> float:
    if "estimated_seconds" in template:
        return float(template["estimated_seconds"])
    model_factor = float(multipliers.get(str(template["model"]), 1.0))
    conditioning_factor = float(multipliers.get(str(template["conditioning"]), 1.0))
    resolution = int(template["resolution"])
    resolution_factor = float(multipliers.get(f"resolution_{resolution}", 1.0))
    epoch_factor = float(template.get("epochs", 200)) / 200.0
    return reference * model_factor * conditioning_factor * resolution_factor * epoch_factor


def expand_jobs(catalog: Mapping[str, Any], registry: Mapping[str, Any], *, reference_seconds: float) -> list[dict[str, Any]]:
    """One job per declared template and its own seed list."""
    multipliers = catalog["cost"]["multipliers"]
    version = str(catalog["version"])
    jobs = []
    seen: dict[str, str] = {}
    for template in catalog["jobs"]:
        for seed in _seeds(template, registry):
            spec = {
                "version": version,
                "template": template["id"],
                "tier": int(template["tier"]),
                "required": bool(template["required"]),
                "environment": template["environment"],
                "model": template["model"],
                "conditioning": template["conditioning"],
                "loss": template["loss"],
                "resolution": int(template["resolution"]),
                "seed": int(seed),
                "epochs": int(template.get("epochs", 200)),
                "dataset": template.get("dataset"),
                "depends_on": [str(item) for item in template.get("depends_on", [])],
                "command_kind": template.get("command_kind"),
                "feasible": bool(template.get("feasible", True)),
                "blocked_reason": template.get("blocked_reason"),
                "requires_file": template.get("requires_file"),
                "registry_configuration": template.get("registry_configuration"),
                "base_config": template.get("base_config"),
                "overrides": template.get("overrides") or {},
            }
            spec["job_id"] = _job_id(spec)
            spec["output_dir"] = _output_dir(version, template["id"], seed)
            spec["command"] = _command(template, seed, version)
            spec["estimated_seconds"] = _cost_seconds(template, reference_seconds, multipliers)
            spec["cost_is_measurement"] = (
                "estimated_seconds" not in template
                and template["model"] in ("jepa", "noaction", "mse")
                and template["conditioning"] == "concat"
                and template["loss"] == "baseline"
                and int(template["resolution"]) == 64
                and int(template.get("epochs", 200)) == 200
            )
            satisfied = template.get("satisfied_by") or {}
            if spec["registry_configuration"]:
                spec["satisfied_by"] = _registry_checkpoint(registry, spec["registry_configuration"], seed)
            else:
                raw = satisfied.get(seed, satisfied.get(str(seed)))
                spec["satisfied_by"] = None if raw in (None, "") else str(raw)
            if spec["job_id"] in seen:
                spec["status"] = SKIPPED_DUPLICATE
                spec["duplicate_of"] = seen[spec["job_id"]]
            else:
                spec["status"] = None
                seen[spec["job_id"]] = f"{template['id']}@{seed}"
            jobs.append(spec)
    return jobs


def _file_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def annotate_status(jobs: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    """Classify each job from the filesystem. This does not launch or delete anything."""
    by_slot = {f"{job['template']}@{job['seed']}": job for job in jobs if job["status"] != SKIPPED_DUPLICATE}
    annotated = []
    for job in jobs:
        current = dict(job)
        if current["status"] == SKIPPED_DUPLICATE:
            annotated.append(current)
            continue
        output = repo_root / current["output_dir"]
        last = output / "last.pt"
        failure = repo_root / "results" / "ablations" / current["version"] / "failures" / f"{current['job_id']}.json"
        if not current["feasible"]:
            current["status"] = BLOCKED
        elif current.get("requires_file") and not (repo_root / str(current["requires_file"])).is_file():
            current["status"] = BLOCKED
        elif unmet_dependencies(current, by_slot, repo_root):
            current["status"] = BLOCKED
            current["blocked_reason"] = "dependency is not complete"
        elif current.get("satisfied_by") and _file_ready(repo_root / str(current["satisfied_by"])):
            current["status"] = COMPLETE
            history = (repo_root / str(current["satisfied_by"])).parent / "history.json"
            if history.is_file():
                payload = json.loads(history.read_text(encoding="utf-8"))
                if isinstance(payload.get("seconds"), (int, float)):
                    current["measured_seconds"] = float(payload["seconds"])
                    current["cost_is_measurement"] = True
        elif _file_ready(last):
            current["status"] = COMPLETE
        elif failure.is_file():
            current["status"] = FAILED
        elif output.is_dir() and any(output.iterdir()):
            current["status"] = PARTIAL
        else:
            current["status"] = NOT_RUN
        annotated.append(current)
    return annotated


def unmet_dependencies(job: Mapping[str, Any], by_slot: Mapping[str, Mapping[str, Any]], repo_root: Path) -> bool:
    for slot in job.get("depends_on") or []:
        parent = by_slot.get(slot)
        if parent is None:
            return True
        if parent.get("status") == COMPLETE:
            continue
        satisfied = parent.get("satisfied_by")
        if satisfied and _file_ready(repo_root / str(satisfied)):
            continue
        output = repo_root / str(parent["output_dir"]) / "last.pt"
        if _file_ready(output):
            continue
        return True
    return False


def assert_output_available(path: Path) -> None:
    """Refuse to write into a directory that already has files."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite {path}. Pass a new --version.")


def dry_run_summary(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    remaining = [
        job["estimated_seconds"]
        for job in jobs
        if job["status"] == NOT_RUN and not job.get("cost_is_measurement")
    ]
    measured_remaining = [
        job["estimated_seconds"]
        for job in jobs
        if job["status"] == NOT_RUN and job.get("cost_is_measurement")
    ]
    return {
        "n_jobs": len(jobs),
        "n_by_status": {status: sum(job["status"] == status for job in jobs) for status in (COMPLETE, NOT_RUN, PARTIAL, FAILED, BLOCKED, SKIPPED_DUPLICATE)},
        "n_by_tier": {tier: sum(job["tier"] == tier for job in jobs) for tier in (1, 2, 3)},
        "n_required": sum(job["required"] for job in jobs),
        "n_optional": sum(not job["required"] for job in jobs),
        "estimated_seconds_remaining_measured_scale": sum(measured_remaining),
        "estimated_seconds_remaining_unmeasured_scale": sum(remaining),
        "commands": [
            {"job": f"{job['template']}@{job['seed']}", "tier": job["tier"], "status": job["status"], "command": job["command"]}
            for job in jobs
            if job["status"] == NOT_RUN and job["command"]
        ],
    }


def format_dry_run(summary: Mapping[str, Any]) -> str:
    lines = [
        f"jobs {summary['n_jobs']}",
        f"required {summary['n_required']}",
        f"optional {summary['n_optional']}",
        "status " + ", ".join(f"{key}={value}" for key, value in summary["n_by_status"].items()),
        "tiers " + ", ".join(f"{key}={value}" for key, value in summary["n_by_tier"].items()),
        f"remaining_seconds_on_measured_scale {summary['estimated_seconds_remaining_measured_scale']:.1f}",
        f"remaining_seconds_on_unmeasured_scale {summary['estimated_seconds_remaining_unmeasured_scale']:.1f}",
        f"commands {len(summary['commands'])}",
    ]
    for item in summary["commands"]:
        lines.append(f"[{item['status']}] tier {item['tier']} {item['job']}")
        lines.append("  " + " ".join(item["command"]))
    return "\n".join(lines) + "\n"


def materialize_config(job: Mapping[str, Any], repo_root: Path) -> Path:
    """Write one training config. An existing file is left untouched."""
    if not job.get("overrides"):
        raise ValueError("this job uses an existing config.")
    destination = repo_root / f"results/ablations/{job['version']}/configs/{job['template']}_s{job['seed']}.yaml"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}.")
    base = yaml.safe_load((repo_root / str(job["base_config"])).read_text(encoding="utf-8"))
    for section, values in job["overrides"].items():
        base.setdefault(section, {}).update(values)
    base.setdefault("training", {})["seed"] = int(job["seed"])
    base["training"]["output_dir"] = f"results/ablations/{job['version']}/runs"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return destination


def cartesian_upper_bound(catalog: Mapping[str, Any], registry: Mapping[str, Any]) -> int:
    """The product the catalog deliberately does not schedule."""
    environments = {job["environment"] for job in catalog["jobs"]}
    models = {job["model"] for job in catalog["jobs"]}
    conditionings = {job["conditioning"] for job in catalog["jobs"]}
    losses = {job["loss"] for job in catalog["jobs"]}
    resolutions = {int(job["resolution"]) for job in catalog["jobs"]}
    return len(environments) * len(models) * len(conditionings) * len(losses) * len(resolutions) * len(registry["seeds"])
