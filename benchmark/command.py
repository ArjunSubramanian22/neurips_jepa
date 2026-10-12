"""Shared argument parsing for the audit and plan commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def load_options(argv: list[str] | None) -> dict[str, Any]:
    """Flags override a YAML config. ``model`` and ``checkpoint`` are optional together."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {}
    if args.config:
        payload = _read_config(args.config)
    options = {
        "env": args.env if args.env is not None else payload.get("env"),
        "model": args.model if args.model is not None else payload.get("model"),
        "checkpoint": args.checkpoint if args.checkpoint is not None else payload.get("checkpoint"),
        "split": args.split if args.split is not None else payload.get("split"),
        "task": args.task if args.task is not None else payload.get("task"),
        "seed": args.seed if args.seed is not None else int(payload.get("seed", 0)),
        "output": args.output if args.output is not None else payload.get("output"),
        "device": args.device if args.device is not None else payload.get("device", "cpu"),
        "config": args.config,
    }
    missing = [key for key in ("env", "split", "output") if not options[key]]
    if missing:
        raise ValueError(f"missing required options: {missing}.")
    return options


def _read_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("YAML configs require pyyaml.") from exc
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a mapping.")
    return payload


def execute(options: dict[str, Any], command: str) -> dict[str, Any]:
    """Run one command. An unknown split is raised before the simulator is built."""
    from benchmark.environments import TASKS, resolve_split
    from benchmark.evaluator import CounterfactualEvaluator
    from benchmark.planner import Planner

    env_name = str(options["env"])
    resolve_split(env_name, str(options["split"]))
    tasks = [str(options["task"])] if options["task"] else list(TASKS[env_name])
    runner = CounterfactualEvaluator() if command == "audit" else Planner()
    records = [
        runner.run(
            env_name,
            task,
            split=str(options["split"]),
            seed=int(options["seed"]),
            model=options["model"],
            checkpoint=options["checkpoint"],
            command=command,
            device=str(options["device"]),
        )
        if command == "audit"
        else runner.evaluate(
            env_name,
            task,
            split=str(options["split"]),
            seed=int(options["seed"]),
            model=options["model"],
            checkpoint=options["checkpoint"],
            device=str(options["device"]),
        )
        for task in tasks
    ]
    combined = dict(records[0])
    combined["results"] = [item for record in records for item in record["results"]]
    combined["command"] = command
    from src.registry import capture_git

    git = capture_git(Path(__file__).resolve().parents[1])
    combined["git_commit"] = git.get("commit")
    combined["git_dirty"] = bool(git.get("dirty"))
    combined["config"] = options.get("config")
    return combined
