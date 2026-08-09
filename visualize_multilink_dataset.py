"""Visualize and replay multi-link robot training data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv


def _save_or_default_plot(plot_fn, save_path: Path | None, fallback_name: str, *args):
    if save_path is None:
        save_path = fallback_name
    plot_fn(*args, save_path=save_path)
    return save_path


def plot_rewards(rewards: np.ndarray, save_path: Path | None = None):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(rewards)), rewards, linewidth=1.5)
    plt.title("Episode Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_actions(actions: np.ndarray, save_path: Path | None = None):
    plt.figure(figsize=(10, 4))
    for joint_index in range(actions.shape[1]):
        plt.plot(actions[:, joint_index], label=f"Joint {joint_index}")
    plt.title("Actions Over Time")
    plt.xlabel("Step")
    plt.ylabel("Torque")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def replay_episode(dataset_path: Path, episode_index: int, save_gif: Path | None = None):
    data = np.load(dataset_path, allow_pickle=True)
    observations = data["observations"][episode_index]
    actions = data["actions"][episode_index]
    rewards = data["rewards"][episode_index]
    mass_profile_row = data["mass_profiles"][episode_index]
    n_links = int(data["n_links"])
    payload_mass = float(data["payload_mass"])

    frames = []
    env = None
    if save_gif:
        env = MultiLinkMassRobotEnv(
            n_links=n_links,
            mass_profile=MassProfile(link_masses=mass_profile_row[:n_links], payload_mass=payload_mass),
            render_mode="rgb_array",
        )
        try:
            env.reset(seed=episode_index)
            for action in actions:
                env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        finally:
            env.close()

    if save_gif and frames:
        fig = plt.figure(figsize=(8, 6))
        plt.axis("off")
        image = plt.imshow(frames[0])

        def update(frame_index):
            image.set_data(frames[frame_index])
            return (image,)

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
        ani.save(save_gif, writer="pillow", fps=30)
        plt.close(fig)

    return observations, rewards, frames


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-link robot training data")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--save-reward-plot", type=Path, default=None)
    parser.add_argument("--save-action-plot", type=Path, default=None)
    parser.add_argument("--save-gif", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    observations = data["observations"][args.episode]
    actions = data["actions"][args.episode]
    rewards = data["rewards"][args.episode]

    reward_plot_path = args.save_reward_plot or args.dataset.with_name(f"{args.dataset.stem}_reward_plot.png")
    action_plot_path = args.save_action_plot or args.dataset.with_name(f"{args.dataset.stem}_action_plot.png")

    plot_rewards(np.asarray(rewards, dtype=np.float32), save_path=reward_plot_path)
    plot_actions(np.asarray(actions, dtype=np.float32), save_path=action_plot_path)

    if args.save_gif:
        _, _, _ = replay_episode(args.dataset, args.episode, save_gif=args.save_gif)

    print(f"Loaded {len(observations)} steps from episode {args.episode}")
    print(f"Saved reward plot to {reward_plot_path}")
    print(f"Saved action plot to {action_plot_path}")


if __name__ == "__main__":
    main()