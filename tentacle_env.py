"""
tentacle_env.py

A minimal Gymnasium environment for a 16-channel soft tentacle robot using PyBullet.
This is a scaffold for RL/CPG control experiments.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

class TentacleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.n_channels = 16
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_channels,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_channels + 6,), dtype=np.float32)
        self.render_mode = render_mode
        self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        self.timestep = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        # Load a plane and a simple tentacle (placeholder: chain of links)
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        try:
            self.tentacle_id = p.loadURDF("soft_tentacle.urdf", [0, 0, 0.05], physicsClientId=self.physics_client)
        except Exception as e:
            print("[ERROR] Failed to load soft_tentacle.urdf:", e)
            raise
        self.timestep = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Placeholder: apply action to tentacle joints
        for i in range(self.n_channels):
            try:
                p.setJointMotorControl2(self.tentacle_id, i, p.POSITION_CONTROL, targetPosition=float(action[i]), physicsClientId=self.physics_client)
            except Exception:
                pass  # Ignore if joint index out of range in placeholder
        p.stepSimulation(physicsClientId=self.physics_client)
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        self.timestep += 1
        terminated = self.timestep >= self.max_steps
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Placeholder: joint positions + base position/velocity
        joint_states = [0.0] * self.n_channels
        try:
            joint_states = [p.getJointState(self.tentacle_id, i, physicsClientId=self.physics_client)[0] for i in range(self.n_channels)]
        except Exception:
            pass
        base_pos, base_orn = p.getBasePositionAndOrientation(self.tentacle_id, physicsClientId=self.physics_client)
        base_vel, _ = p.getBaseVelocity(self.tentacle_id, physicsClientId=self.physics_client)
        obs = np.array(joint_states + list(base_pos) + list(base_vel), dtype=np.float32)
        return obs

    def _compute_reward(self, obs, action):
        # Placeholder: reward for moving tip forward (x direction)
        # Replace with your own reward function
        return float(obs[self.n_channels])  # x position of base

    def render(self):
        pass  # Handled by PyBullet GUI

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
