"""A free box and a mocap pusher in MuJoCo.

The scene is intentionally small: one contact body, one pusher, a fixed
timestep. ``mj_getState`` / ``mj_setState`` restore the physics and the mocap
pose. Rendering uses MuJoCo's offscreen renderer and needs a GL context.

This is not the MLS-MPM dump. No JEPA checkpoint was trained on these pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

XML = """
<mujoco model="box_push">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
  <visual>
    <global offwidth="64" offheight="64"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.75 0.75 0.75 1" friction="0.8 0.05 0.01"/>
    <body name="box" pos="0 0 0.05">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="0.3" rgba="0.85 0.25 0.2 1" friction="0.9 0.05 0.01"/>
    </body>
    <body name="pusher" mocap="true" pos="-0.16 0 0.05">
      <geom name="pusher_geom" type="sphere" size="0.035" mass="1" rgba="0.2 0.35 0.85 1" friction="1 0.05 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

# Pusher starts just outside the box. ``glance_x`` is off center, so the same
# x push rotates the box instead of driving it across the floor.
TASKS = {
    "slide_x": {"pusher_xy": (-0.16, 0.0), "axis": 0},
    "slide_y": {"pusher_xy": (0.0, -0.16), "axis": 1},
    "glance_x": {"pusher_xy": (-0.16, 0.06), "axis": 0},
}
ACTION_LIMIT = 0.01
SUBSTEPS = 8
RENDER_SIZE = 64


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("mujoco is required for the box-push environment.") from exc
    return mujoco


class MujocoBoxPush:
    """Continuous 2D pusher deltas. The third task shares the slide_x start."""

    environment_id = "mujoco_box_push"

    def __init__(self, task: str = "slide_x") -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}.")
        self.task = task
        mujoco = _mujoco()
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        physics = int(mujoco.mjtState.mjSTATE_PHYSICS)
        warm = int(mujoco.mjtState.mjSTATE_WARMSTART)
        mocap_pos = int(mujoco.mjtState.mjSTATE_MOCAP_POS)
        mocap_quat = int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
        self._spec = physics | warm | mocap_pos | mocap_quat
        self._state_size = int(mujoco.mj_stateSize(self.model, self._spec))
        self._rng = np.random.default_rng(0)
        self._renderer = None
        self.reset(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._mujoco.mj_resetData(self.model, self.data)
        pusher_x, pusher_y = TASKS[self.task]["pusher_xy"]
        self.data.mocap_pos[0, 0] = pusher_x
        self.data.mocap_pos[0, 1] = pusher_y
        self.data.mocap_pos[0, 2] = 0.05
        self._mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> dict[str, Any]:
        vector = np.empty(self._state_size, dtype=np.float64)
        self._mujoco.mj_getState(self.model, self.data, vector, self._spec)
        return {
            "spec": int(self._spec),
            "vector": vector.copy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        vector = np.asarray(state["vector"], dtype=np.float64)
        if vector.shape != (self._state_size,) or int(state["spec"]) != self._spec:
            raise ValueError("state does not match this MuJoCo model.")
        self._mujoco.mj_setState(self.model, self.data, vector, self._spec)

    def step(self, action: np.ndarray) -> None:
        delta = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -ACTION_LIMIT, ACTION_LIMIT)
        if delta.size < 2:
            raise ValueError("action must contain dx and dy.")
        for _ in range(SUBSTEPS):
            self.data.mocap_pos[0, 0] += delta[0]
            self.data.mocap_pos[0, 1] += delta[1]
            self._mujoco.mj_step(self.model, self.data)

    def render(self) -> np.ndarray:
        """RGB ``uint8`` image ``(H, W, 3)`` from the offscreen renderer."""
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.model, RENDER_SIZE, RENDER_SIZE)
        self._mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data)
        return np.ascontiguousarray(self._renderer.render())

    def sample_action(self) -> np.ndarray:
        return self._rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, size=2).astype(np.float64)

    def action_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        a = np.asarray(left, dtype=np.float64).reshape(-1)
        b = np.asarray(right, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("actions must share a shape.")
        return float(np.linalg.norm(a - b))

    def physical_state_distance(self, other: dict[str, Any]) -> dict[str, float]:
        from src.metrics.cascade import physical_response

        current = self.data.qpos[:3].reshape(1, 3)
        saved = np.asarray(other["qpos"], dtype=np.float64)[:3].reshape(1, 3)
        current_vel = self.data.qvel[:3].reshape(1, 3)
        saved_vel = np.asarray(other["qvel"], dtype=np.float64)[:3].reshape(1, 3)
        measured = physical_response(saved, current, saved_vel, current_vel)
        return {
            "box_rms_pos": float(measured["rms_pos"]),
            "box_rms_vel": float(measured["rms_vel"]),
        }

    def box_xy(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:2], dtype=np.float64).copy()

    def tracked_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:3], dtype=np.float64).reshape(1, 3).copy()

    def tracked_velocities(self) -> np.ndarray:
        return np.asarray(self.data.qvel[:3], dtype=np.float64).reshape(1, 3).copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
