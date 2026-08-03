"""Multi-link mass robot simulation for PyBullet / Gymnasium.

This version models an upright planar chain with revolute joints around the y
axis, gravity along negative z, and configurable link masses plus an optional
payload on the final link.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover - dependency check happens at runtime
    p = None
    pybullet_data = None
    _PYBULLET_IMPORT_ERROR = exc
else:
    _PYBULLET_IMPORT_ERROR = None


class MassProfile:
    def __init__(self, link_masses: Iterable[float], payload_mass: float = 0.0):
        self.link_masses = tuple(float(mass) for mass in link_masses)
        self.payload_mass = float(payload_mass)

    def __repr__(self) -> str:
        return f"MassProfile(link_masses={self.link_masses}, payload_mass={self.payload_mass})"


def _ensure_pybullet():
    if p is None:
        raise RuntimeError(f"pybullet is required for MultiLinkMassRobotEnv: {_PYBULLET_IMPORT_ERROR}")


def _parse_floats(values: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, str):
        parts = [item.strip() for item in values.split(",") if item.strip()]
        return tuple(float(item) for item in parts)
    return tuple(float(item) for item in values)


def _default_mass_profile(n_links: int) -> MassProfile:
    link_masses = tuple(0.20 + 0.05 * index for index in range(n_links))
    return MassProfile(link_masses=link_masses, payload_mass=0.25)


def _mass_summary(profile: MassProfile) -> str:
    masses = ", ".join(f"{mass:.2f}" for mass in profile.link_masses)
    return f"links=[{masses}] payload={profile.payload_mass:.2f}"


class MultiLinkMassRobotEnv(gym.Env):
    """Upright multi-link robot with configurable attached masses."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        n_links: int = 4,
        mass_profile: MassProfile | None = None,
        segment_length: float = 0.18,
        segment_thickness: float = 0.045,
        max_torque: float = 24.0,
        max_steps: int = 400,
        render_mode: str | None = None,
    ):
        super().__init__()
        _ensure_pybullet()

        self.n_links = int(n_links)
        self.segment_length = float(segment_length)
        self.segment_thickness = float(segment_thickness)
        self.max_torque = float(max_torque)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.mass_profile = mass_profile or _default_mass_profile(self.n_links)
        if len(self.mass_profile.link_masses) != self.n_links:
            raise ValueError(f"mass_profile.link_masses must have length {self.n_links}")

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(self.n_links,),
            dtype=np.float32,
        )
        obs_dim = self.n_links * 2 + 6 + self.n_links + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        self.robot_id = None
        self._joint_indices: list[int] = []
        self._temp_urdf_path: Path | None = None
        self._step_count = 0

    def _make_urdf_text(self) -> str:
        link_blocks: list[str] = []
        joint_blocks: list[str] = []

        for index, base_mass in enumerate(self.mass_profile.link_masses):
            mass = float(base_mass + (self.mass_profile.payload_mass if index == self.n_links - 1 else 0.0))
            is_terminal = index == self.n_links - 1
            box_length = self.segment_length * (1.30 if is_terminal and self.mass_profile.payload_mass > 0.0 else 1.05)
            box_height = self.segment_thickness * (1.35 if is_terminal else 1.20)
            box_width = self.segment_thickness * (1.35 if is_terminal else 1.20)
            link_name = f"link_{index}"

            link_blocks.append(
                f"""
  <link name="{link_name}">
        <inertial>
            <origin xyz="0 0 {self.segment_length / 2:.6f}" rpy="0 0 0"/>
            <mass value="{mass:.6f}"/>
            <inertia ixx="0.010" ixy="0" ixz="0" iyy="0.010" iyz="0" izz="0.010"/>
        </inertial>
    <visual>
      <origin xyz="0 0 {self.segment_length / 2:.6f}" rpy="0 0 0"/>
      <geometry>
        <box size="{box_width:.6f} {box_height:.6f} {box_length:.6f}"/>
      </geometry>
      <material name="link_color">
        <color rgba="0.7 0.7 0.75 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {self.segment_length / 2:.6f}" rpy="0 0 0"/>
      <geometry>
        <box size="{box_width:.6f} {box_height:.6f} {box_length:.6f}"/>
      </geometry>
    </collision>
  </link>
""".strip()
            )

            parent_link = "base" if index == 0 else f"link_{index - 1}"
            joint_origin_z = 0.0 if index == 0 else self.segment_length
            joint_blocks.append(
                f"""
  <joint name="joint_{index}" type="revolute">
    <parent link="{parent_link}"/>
    <child link="{link_name}"/>
    <origin xyz="0 0 {joint_origin_z:.6f}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="{self.max_torque:.6f}" velocity="15.0"/>
        <dynamics damping="0.75" friction="0.06"/>
  </joint>
""".strip()
            )

        return f"""
<?xml version="1.0" ?>
<robot name="multi_link_mass_robot">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
{os.linesep.join(link_blocks)}
{os.linesep.join(joint_blocks)}
</robot>
""".strip()

    def _write_urdf(self) -> Path:
        if self._temp_urdf_path is not None and self._temp_urdf_path.exists():
            try:
                self._temp_urdf_path.unlink()
            except OSError:
                pass

        temp_dir = Path(tempfile.mkdtemp(prefix="multilink_mass_robot_"))
        urdf_path = temp_dir / "multilink_mass_robot.urdf"
        urdf_path.write_text(self._make_urdf_text(), encoding="utf-8")
        self._temp_urdf_path = urdf_path
        return urdf_path

    def _joint_state(self, index: int) -> tuple[float, float]:
        joint_state = p.getJointState(self.robot_id, index, physicsClientId=self.physics_client)
        return float(joint_state[0]), float(joint_state[1])

    def _tip_state(self) -> tuple[np.ndarray, np.ndarray]:
        link_index = self.n_links - 1
        link_state = p.getLinkState(self.robot_id, link_index, computeLinkVelocity=1, physicsClientId=self.physics_client)
        tip_pos = np.asarray(link_state[4], dtype=np.float32)
        tip_vel = np.asarray(link_state[6], dtype=np.float32)
        return tip_pos, tip_vel

    def _mass_vector(self) -> np.ndarray:
        return np.asarray(tuple(self.mass_profile.link_masses) + (float(self.mass_profile.payload_mass),), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        joint_positions = []
        joint_velocities = []
        for index in self._joint_indices:
            joint_pos, joint_vel = self._joint_state(index)
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)

        tip_pos, tip_vel = self._tip_state()
        obs = np.concatenate([
            np.asarray(joint_positions, dtype=np.float32),
            np.asarray(joint_velocities, dtype=np.float32),
            tip_pos,
            tip_vel,
            self._mass_vector(),
        ]).astype(np.float32)
        return obs

    def _reward(self, action: np.ndarray) -> float:
        tip_pos, tip_vel = self._tip_state()
        upright_bonus = float(max(tip_pos[2], 0.0))
        joint_center_penalty = 0.05 * float(np.sum(np.square(self._joint_positions())))
        torque_penalty = 0.01 * float(np.sum(np.square(action)))
        velocity_penalty = 0.01 * float(np.sum(np.square(self._joint_velocities())))
        tip_speed_bonus = 0.02 * float(np.linalg.norm(tip_vel))
        return upright_bonus + tip_speed_bonus - joint_center_penalty - torque_penalty - velocity_penalty

    def _joint_positions(self) -> np.ndarray:
        return np.asarray([self._joint_state(index)[0] for index in self._joint_indices], dtype=np.float32)

    def _joint_velocities(self) -> np.ndarray:
        return np.asarray([self._joint_state(index)[1] for index in self._joint_indices], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self.physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        urdf_path = self._write_urdf()
        self.robot_id = p.loadURDF(str(urdf_path), [0, 0, 0.02], useFixedBase=True, physicsClientId=self.physics_client)
        self._joint_indices = list(range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)))
        for joint_index in self._joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.physics_client,
            )

        self._step_count = 0
        obs = self._get_obs()
        info = {"mass_profile": self.mass_profile, "urdf_path": str(urdf_path)}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_links)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        for joint_index, torque in zip(self._joint_indices, action):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=float(torque),
                physicsClientId=self.physics_client,
            )

        p.stepSimulation(physicsClientId=self.physics_client)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._reward(action)
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {
            "tip_position": obs[self.n_links * 2:self.n_links * 2 + 3],
            "mass_profile": self.mass_profile,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            width, height, view_matrix, projection_matrix, renderer = self._camera_config()
            image = p.getCameraImage(
                width,
                height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=renderer,
                physicsClientId=self.physics_client,
            )
            rgba = np.reshape(image[2], (height, width, 4))
            return rgba[:, :, :3]
        return None

    def _camera_config(self):
        width, height = 960, 720
        target = [0.0, 0.0, self.segment_length * self.n_links * 0.45]
        distance = self.segment_length * self.n_links * 2.0 + 0.6
        yaw, pitch, roll = 20, -35, 0
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
            physicsClientId=self.physics_client,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=55,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=30.0,
        )
        return width, height, view_matrix, projection_matrix, p.ER_BULLET_HARDWARE_OPENGL

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None

        if self._temp_urdf_path is not None:
            try:
                temp_dir = self._temp_urdf_path.parent
                self._temp_urdf_path.unlink(missing_ok=True)
                temp_dir.rmdir()
            except Exception:
                pass
            self._temp_urdf_path = None


def run_demo(mass_profile: MassProfile, n_links: int, steps: int, render_mode: str):
    env = MultiLinkMassRobotEnv(
        n_links=n_links,
        mass_profile=mass_profile,
        render_mode=render_mode,
    )
    obs, info = env.reset()
    log.info("Mass profile: %s", _mass_summary(mass_profile))
    log.info("Observation dim: %d | Action dim: %d", env.observation_space.shape[0], env.action_space.shape[0])

    for step in range(steps):
        phase = 0.06 * step
        joint_ids = np.arange(n_links, dtype=np.float32)
        joint_positions = obs[:n_links]
        joint_velocities = obs[n_links:2 * n_links]
        tip_pos = obs[2 * n_links:2 * n_links + 3]
        tip_vel = obs[2 * n_links + 3:2 * n_links + 6]

        balance = -0.45 * joint_positions - 0.18 * joint_velocities
        height_correction = 0.08 * (n_links * env.segment_length - float(tip_pos[2])) + 0.03 * float(tip_vel[2])
        shaping = 0.12 * np.sin(phase + joint_ids * 0.4)
        action = balance + height_correction + shaping
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 50 == 0:
            tip = info["tip_position"]
            log.info("step=%d reward=%.3f tip=(%.3f, %.3f, %.3f)", step, reward, float(tip[0]), float(tip[1]), float(tip[2]))
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-link mass robot demo")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--masses", type=str, default="0.35,0.45,0.55,0.70")
    parser.add_argument("--payload-mass", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"], default="human")
    args = parser.parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode
    mass_profile = MassProfile(link_masses=_parse_floats(args.masses), payload_mass=float(args.payload_mass))
    run_demo(mass_profile=mass_profile, n_links=args.links, steps=args.steps, render_mode=render_mode)