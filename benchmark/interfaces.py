"""Interfaces a new environment or checkpoint implements.

The methods are the ones the audit calls. Existing simulators keep
``get_state`` and ``set_state``; :class:`benchmark.environments.ForkableEnvironment`
exposes those operations under the names below.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class WorldModelAdapter(Protocol):
    """Action-conditioned latent model. One checkpoint, one adapter."""

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Map one observation to a latent vector."""

    def predict(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One-step latent prediction. ``action`` is in the environment's raw units."""

    def rollout(self, latent: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Open-loop predictions. ``actions`` has shape ``(K, A)``."""

    def metadata(self) -> dict[str, Any]:
        """Architecture name and any declared widths. Not a score."""


class ForkableEnvironment(Protocol):
    """Simulator that can be copied, restored, and rendered."""

    def reset(self, seed: int) -> None:
        """Deterministic initial state. ``seed`` also seeds action sampling."""

    def clone_state(self) -> dict[str, Any]:
        """Copy the state required to reproduce the next step."""

    def restore_state(self, state: dict[str, Any]) -> None:
        """Replace the current state with a previous :meth:`clone_state`."""

    def step(self, action: np.ndarray) -> None:
        """Advance one observation step."""

    def render(self) -> np.ndarray:
        """RGB frame. Channel layout is the environment's native layout."""

    def physical_metrics(self, other: dict[str, Any]) -> dict[str, float]:
        """Distance from the current physical state to a saved state."""

    def sample_counterfactual_action(self) -> np.ndarray:
        """An action other than the one that produced the current state."""


class CounterfactualEvaluator(Protocol):
    """Layer-separated fork metrics. Each method returns its own record."""

    def physical_response(self, positions_a: Any, positions_b: Any, velocities_a: Any = None, velocities_b: Any = None) -> dict[str, Any]:
        """Tracked-point displacement."""

    def observation_visibility(self, obs_a: Any, obs_b: Any) -> dict[str, Any]:
        """Whether the two futures differ in the rendered frame."""

    def target_response(self, left: Any, right: Any) -> dict[str, Any]:
        """Privileged future displacement used as the target latent."""

    def predictor_response(self, predicted_a: Any, predicted_b: Any) -> dict[str, Any]:
        """Displacement of the model's own predictions."""

    def direction_metrics(self, predicted_delta: Any, target_delta: Any) -> dict[str, Any]:
        """Cosine of the predicted and target difference vectors."""

    def ranking_metrics(self, futures: Any, predict) -> dict[str, Any]:
        """Nearest-neighbor retrieval. A tie is not a hit."""

    def multi_step_metrics(self, horizon: int, physical: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        """Outcome after the declared open-loop horizon."""


class Planner(Protocol):
    """Choose one action from a finite set of costs, then record that choice."""

    def plan(self, costs: dict[str, float]) -> dict[str, Any]:
        """Unique lowest cost. A tie is ``ambiguous_tie`` and chooses nothing."""

    def evaluate(self, env: str, task: str, *, seed: int = 0, model: str | None = None, checkpoint: str | None = None) -> dict[str, Any]:
        """Run the fork and return a schema record whose command is ``plan``."""
