"""Common simulator surface for action forks.

The MLS-MPM dump and MuJoCo box push both implement these operations.
Learned-model metrics are not part of the interface.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ForkableEnv(Protocol):
    """Deterministic environment that can be restored and rendered."""

    def reset(self, seed: int) -> None:
        """Set a deterministic initial state and seed ``sample_action``."""

    def get_state(self) -> dict[str, Any]:
        """Copy the state needed to reproduce the next step exactly."""

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a state from :meth:`get_state`."""

    def step(self, action: np.ndarray) -> None:
        """Advance one observation step under a continuous action."""

    def render(self) -> np.ndarray:
        """Return an RGB frame. Layout is documented by the environment."""

    def sample_action(self) -> np.ndarray:
        """Draw one action from the seed set in :meth:`reset`."""

    def action_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        """Distance between two actions in the environment's action space."""

    def physical_state_distance(self, other: dict[str, Any]) -> dict[str, float]:
        """Distance between the current physical state and a saved state."""
