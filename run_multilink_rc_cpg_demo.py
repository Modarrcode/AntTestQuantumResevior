"""Run the trained RC/CPG policy for the upright multi-link mass robot."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv
from multilink_rc_cpg_backend import EchoStateNetwork, MassCondition


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_model(model_path: Path):
    payload = np.load(model_path, allow_pickle=True)
    return payload


def build_features(obs: np.ndarray, prev_obs: np.ndarray, mass_condition: MassCondition) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    prev_obs = np.asarray(prev_obs, dtype=np.float32)
    delta = obs - prev_obs
    return np.concatenate([obs, delta, mass_condition.vector]).astype(np.float32)


def rigid_hold_torque(obs: np.ndarray, n_links: int, max_torque: float) -> np.ndarray:
    joint_positions = np.asarray(obs[:n_links], dtype=np.float32)
    joint_velocities = np.asarray(obs[n_links:2 * n_links], dtype=np.float32)
    pd_torque = -(12.0 * joint_positions + 2.2 * joint_velocities)
    return np.clip(pd_torque, -max_torque, max_torque).astype(np.float32)


def run_demo(model_path: Path, output_gif: Path, reward_plot: Path | None = None):
    payload = load_model(model_path)
    n_links = int(payload["n_links"])
    link_masses = tuple(np.asarray(payload["link_masses"], dtype=np.float32).tolist())
    payload_mass = float(payload["payload_mass"])
    mass_condition = MassCondition(link_masses=link_masses, payload_mass=payload_mass)

    esn = EchoStateNetwork(
        n_inputs=int(payload["feature_dim"]),
        n_reservoir=int(payload["reservoir_n"]),
        leak_rate=float(payload["reservoir_leak"]),
    )
    esn.Win = np.asarray(payload["reservoir_Win"], dtype=np.float64)
    esn.W = np.asarray(payload["reservoir_W"], dtype=np.float64)
    esn.W_out = np.asarray(payload["reservoir_W_out"], dtype=np.float64)

    env = MultiLinkMassRobotEnv(
        n_links=n_links,
        mass_profile=MassProfile(link_masses=link_masses, payload_mass=payload_mass),
        render_mode="rgb_array",
    )

    obs, _ = env.reset(seed=0)
    prev_obs = obs.copy()
    frames = []
    rewards = []
    filtered_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    delayed_posture = np.zeros(env.action_space.shape[0], dtype=np.float32)
    try:
        esn.reset()
        for step in range(300):
            features = build_features(obs, prev_obs, mass_condition)
            action = np.asarray(esn.predict(features), dtype=np.float32)
            action = np.tanh(action / 14.0) * (0.16 * env.max_torque)
            filtered_action = 0.98 * filtered_action + 0.02 * action
            posture_action = rigid_hold_torque(obs, n_links, env.max_torque)
            delayed_posture = 0.90 * delayed_posture + 0.10 * posture_action
            blend = 0.05 if step < 80 else 0.10
            action = np.clip(0.96 * filtered_action + blend * delayed_posture, env.action_space.low, env.action_space.high)
            action[np.abs(action) < 0.60] = 0.0
            prev_obs = obs.copy()
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
        plt.title("RC/CPG Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(reward_plot, dpi=160)
        plt.close()

    log.info("Saved RC/CPG demo GIF to %s", output_gif)
    if reward_plot is not None:
        log.info("Saved reward plot to %s", reward_plot)


def main():
    parser = argparse.ArgumentParser(description="Run the multi-link RC/CPG demo")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-gif", type=Path, default=Path("multilink_rc_cpg_demo.gif"))
    parser.add_argument("--reward-plot", type=Path, default=None)
    args = parser.parse_args()

    run_demo(args.model, args.output_gif, reward_plot=args.reward_plot)


if __name__ == "__main__":
    main()