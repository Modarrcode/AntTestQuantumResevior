"""
rl_cpg_agent.py

RL agent that modulates a CPG controller for a multi-actuator soft robot.
This is a scaffold for integration with a Gym environment (e.g., 16-channel tentacle).
"""
import numpy as np
import gymnasium as gym
from typing import Any
import argparse
import time

# Import your CPGController from AntWithRes.py (or refactor to a shared module)
from AntWithRes import CPGController
# Import TentacleEnv
from tentacle_env import TentacleEnv

class RLCPGWrapper:
    """
    Wraps a CPGController, allowing an RL agent to modulate its parameters (amplitudes, phases, offsets, omega).
    """
    def __init__(self, n_channels: int):
        self.n = n_channels
        self.base_cpg = CPGController(n_channels)
        self.last_params = self.base_cpg.vector()

    def set_params(self, param_vec: np.ndarray):
        """Set CPG parameters from RL agent output."""
        self.last_params = param_vec
        self.base_cpg = CPGController.from_vector(param_vec, self.n)

    def step(self, t: float) -> np.ndarray:
        return self.base_cpg.step(t)

# Example RL agent (random policy for scaffold)
class RandomRLAgent:
    def __init__(self, n_params: int):
        self.n_params = n_params
    def act(self, obs: Any) -> np.ndarray:
        # For now, output random CPG params (omega, amplitudes, phases, offsets)
        omega = np.random.uniform(0.5, 4.0, 1)
        A = np.random.uniform(0.0, 1.0, self.n_params//3)
        phi = np.random.uniform(-np.pi, np.pi, self.n_params//3)
        off = np.random.uniform(-0.5, 0.5, self.n_params//3)
        return np.concatenate([omega, A, phi, off])

# Training loop scaffold
def train(env_name: str = None, n_channels: int = 16, n_episodes: int = 10, episode_length: int = 500, render: bool = False, sleep_time: float = 0.02):
    if env_name is not None:
        env = gym.make(env_name, render_mode="human" if render else None)
    else:
        env = TentacleEnv(render_mode="human" if render else None)
    agent = RandomRLAgent(1 + 3 * n_channels)
    cpg = RLCPGWrapper(n_channels)
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        for t in range(episode_length):
            params = agent.act(obs)
            cpg.set_params(params)
            action = cpg.step(t * 0.02)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            if render and sleep_time > 0:
                time.sleep(sleep_time)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep+1}: total_reward={total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable PyBullet GUI visualization")
    parser.add_argument("--sleep", type=float, default=0.02, help="Seconds to sleep after each step when rendering")
    args = parser.parse_args()
    train(env_name=None, n_channels=16, render=args.render, sleep_time=args.sleep)
