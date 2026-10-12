"""Choose an action from fork costs.

``plan`` is the same unique-minimum rule as the fork audit. ``evaluate``
returns a full schema record so a plan file can be read by the same checker
as an audit file.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.envs.fork_audit import _choose


class Planner:
    def plan(self, costs: Mapping[str, float]) -> dict[str, Any]:
        return _choose(costs)

    def evaluate(
        self,
        env: str,
        task: str,
        *,
        seed: int = 0,
        split: str,
        model: str | None = None,
        checkpoint: str | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from benchmark.evaluator import CounterfactualEvaluator

        record = CounterfactualEvaluator().run(
            env,
            task,
            split=split,
            seed=seed,
            model=model,
            checkpoint=checkpoint,
            command="plan",
            device=device,
        )
        return record
