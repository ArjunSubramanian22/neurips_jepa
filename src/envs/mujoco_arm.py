"""A two-link arm, a free object, and a static post.

The arm is a revolute chain under joint-position targets. That is a different
mechanism from the mocap pusher, and the post gives the camera something the
arm can cover without moving the object. Angles in the MJCF are radians.
Rendering uses the fixed ``side`` camera and needs a GL context.
"""

from __future__ import annotations

from typing import Any

import numpy as np

XML = """
<mujoco model="planar_arm">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="64" offheight="64"/>
  </visual>
  <worldbody>
    <light pos="0.0 -0.4 1.0" dir="0 0.35 -1"/>
    <camera name="side" pos="0.1 -0.48 0.28" xyaxes="1 0 0 0 0.45 0.89" fovy="55"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.86 0.84 0.78 1" friction="0.9 0.05 0.01"/>
    <body name="base" pos="-0.28 0 0.07">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="1.2" armature="0.05"/>
      <geom type="capsule" fromto="0.06 0 0 0.26 0 0" size="0.04" mass="0.5" rgba="0.95 0.2 0.05 1" contype="0" conaffinity="0"/>
      <body name="forearm" pos="0.26 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" damping="0.6" armature="0.03"/>
        <geom type="capsule" fromto="0.05 0 0 0.2 0 0" size="0.032" mass="0.3" rgba="0.08 0.2 0.92 1" contype="0" conaffinity="0"/>
        <geom name="finger" type="sphere" pos="0.22 0 0" size="0.045" mass="0.15" rgba="0.98 0.75 0.05 1" contype="2" conaffinity="1" friction="1 0.05 0.01"/>
      </body>
    </body>
    <body name="object" pos="0.2 0 0.045">
      <freejoint name="object_joint"/>
      <geom name="object_geom" type="box" size="0.05 0.05 0.04" mass="0.28" rgba="0.02 0.72 0.12 1" friction="1.1 0.08 0.01"/>
    </body>
    <body name="post" pos="-0.08 -0.123 0.07">
      <geom name="post_geom" type="box" size="0.025 0.025 0.07" rgba="0.1 0.85 0.95 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="18" kv="4" ctrlrange="-1.6 1.6"/>
    <position name="elbow_act" joint="elbow" kp="12" kv="2.5" ctrlrange="-2.4 2.2"/>
  </actuator>
</mujoco>
"""

HOME = np.array([0.8, -1.6], dtype=np.float64)
PUSH = np.array([-1.2, 1.8], dtype=np.float64)
FOLD = np.array([-1.2, -1.0], dtype=np.float64)
NAMED_ACTIONS = {"push": PUSH, "fold": FOLD, "hold": HOME}
TASKS = ("push_object", "fold_clear", "cover_post")
SUBSTEPS = 5
RENDER_SIZE = 64
# Positive command for each task, and the command it is contrasted with.
TASK_ACTIONS = {
    "push_object": {"positive": "push", "alternate": "fold"},
    "fold_clear": {"positive": "fold", "alternate": "push"},
    "cover_post": {"positive": "fold", "alternate": "push"},
}


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("mujoco is required for the arm environment.") from exc
    return mujoco


def post_visible_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that still show the cyan post.

    The arm is orange and blue and the object is green, so a pixel counts
    only when green and blue are both high and red is low.
    """
    frame = np.asarray(image)
    if frame.ndim != 3:
        raise ValueError("post visibility expects an RGB image.")
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if np.issubdtype(frame.dtype, np.floating) and float(np.max(frame)) <= 1.5:
        frame = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    visible = (blue > 140) & (green > 120) & (red < 100)
    return float(np.mean(visible))


class MujocoArm:
    """Joint-position targets for a planar arm. The object is a free box."""

    environment_id = "mujoco_arm"

    def __init__(self, task: str = "push_object") -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}.")
        self.task = task
        mujoco = _mujoco()
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        physics = int(mujoco.mjtState.mjSTATE_PHYSICS)
        warm = int(mujoco.mjtState.mjSTATE_WARMSTART)
        ctrl = int(mujoco.mjtState.mjSTATE_CTRL)
        self._spec = physics | warm | ctrl
        self._state_size = int(mujoco.mj_stateSize(self.model, self._spec))
        self._shoulder = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")])
        self._elbow = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")])
        self._object = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")])
        self._object_dof = int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")])
        self._rng = np.random.default_rng(0)
        self._renderer = None
        self.reset(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._shoulder] = HOME[0]
        self.data.qpos[self._elbow] = HOME[1]
        self.data.ctrl[:] = HOME
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
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.size < 2:
            raise ValueError("action must contain shoulder and elbow targets.")
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(command[:2], low, high)
        for _ in range(SUBSTEPS):
            self._mujoco.mj_step(self.model, self.data)

    def render(self) -> np.ndarray:
        """RGB ``uint8`` image ``(H, W, 3)`` from the side camera."""
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.model, RENDER_SIZE, RENDER_SIZE)
        self._mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera="side")
        return np.ascontiguousarray(self._renderer.render())

    def sample_action(self) -> np.ndarray:
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        return self._rng.uniform(low, high).astype(np.float64)

    def action_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        a = np.asarray(left, dtype=np.float64).reshape(-1)
        b = np.asarray(right, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("actions must share a shape.")
        return float(np.linalg.norm(a - b))

    def tracked_positions(self) -> np.ndarray:
        adr = self._object
        return np.asarray(self.data.qpos[adr : adr + 3], dtype=np.float64).reshape(1, 3).copy()

    def tracked_velocities(self) -> np.ndarray:
        adr = self._object_dof
        return np.asarray(self.data.qvel[adr : adr + 3], dtype=np.float64).reshape(1, 3).copy()

    def control_state(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self._shoulder], self.data.qpos[self._elbow]],
            dtype=np.float64,
        )

    def physical_state_distance(self, other: dict[str, Any]) -> dict[str, float]:
        from src.metrics.cascade import physical_response

        saved = np.asarray(other["qpos"], dtype=np.float64)[self._object : self._object + 3]
        saved_vel = np.asarray(other["qvel"], dtype=np.float64)[self._object_dof : self._object_dof + 3]
        measured = physical_response(
            saved.reshape(1, 3),
            self.tracked_positions(),
            saved_vel.reshape(1, 3),
            self.tracked_velocities(),
        )
        return {
            "object_rms_pos": float(measured["rms_pos"]),
            "object_rms_vel": float(measured["rms_vel"]),
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
