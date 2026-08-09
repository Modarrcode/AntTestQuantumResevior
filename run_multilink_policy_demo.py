"""Run the learned multi-link policy inside the robot environment and export a replay GIF."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv


class LinearPolicy:
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return obs @ self.weights + self.bias


def stabilizing_action(obs: np.ndarray, n_links: int, segment_length: float) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    joint_positions = obs[:n_links]
    joint_velocities = obs[n_links:2 * n_links]
    tip_pos = obs[2 * n_links:2 * n_links + 3]
    tip_vel = obs[2 * n_links + 3:2 * n_links + 6]

    target_height = segment_length * float(n_links)
    height_error = target_height - float(tip_pos[2])

    joint_stiffness = -1.35 * joint_positions
    joint_damping = -0.75 * joint_velocities
    lift = 0.35 * height_error + 0.12 * float(tip_vel[2])
    shape = np.linspace(0.20, 0.08, n_links, dtype=np.float32)
    return (joint_stiffness + joint_damping + lift + shape).astype(np.float32)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_policy(policy_path: Path):
    payload = np.load(policy_path, allow_pickle=True)
    policy = LinearPolicy(payload["policy_weights"], payload["policy_bias"])
    return policy, payload


def run_demo(policy_path: Path, output_gif: Path, reward_plot: Path | None = None):
    policy, metadata = load_policy(policy_path)
    n_links = int(metadata["n_links"])
    payload_mass = float(metadata["payload_mass"])
    link_masses = np.asarray(metadata["link_masses"], dtype=np.float32)
    segment_length = 0.18

    env = MultiLinkMassRobotEnv(
        n_links=n_links,
        mass_profile=MassProfile(link_masses=link_masses, payload_mass=payload_mass),
        render_mode="rgb_array",
    )

    obs, _ = env.reset(seed=0)
    frames = []
    rewards = []
    smoothed_action = np.zeros(n_links, dtype=np.float32)
    try:
        for _ in range(300):
            correction = stabilizing_action(obs, n_links, segment_length)
            smoothed_action = 0.90 * smoothed_action + 0.10 * correction
            action = np.clip(smoothed_action, -env.max_torque, env.max_torque)
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            rewards.append(float(reward))
            if terminated or truncated:
                break
    finally:
        env.close()

    if frames:
        fig = plt.figure(figsize=(8, 6))
        plt.axis("off")
        image = plt.imshow(frames[0])

        def update(frame_index):
            image.set_data(frames[frame_index])
            return (image,)

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
        ani.save(output_gif, writer="pillow", fps=30)
        plt.close(fig)

    if reward_plot is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(rewards, linewidth=1.5)
        plt.title("Policy Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(reward_plot, dpi=160)
        plt.close()

    log.info("Saved policy demo GIF to %s", output_gif)
    if reward_plot is not None:
        log.info("Saved reward plot to %s", reward_plot)


def main():
    parser = argparse.ArgumentParser(description="Run the learned multi-link policy demo")
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output-gif", type=Path, default=Path("multilink_policy_demo.gif"))
    parser.add_argument("--reward-plot", type=Path, default=None)
    args = parser.parse_args()

    run_demo(args.policy, args.output_gif, reward_plot=args.reward_plot)


if __name__ == "__main__":
    main()