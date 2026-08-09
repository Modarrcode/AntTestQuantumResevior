"""Generate rollout data for the multi-link mass robot.

The resulting NPZ dataset is intended for imitation learning, regression, or
offline policy fitting. Each episode stores observations, actions, rewards,
and the mass profile used for that rollout.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv, _parse_floats


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def scripted_policy(obs: np.ndarray, n_links: int, step_index: int, segment_length: float = 0.18) -> np.ndarray:
    """State-feedback torque policy used to generate learnable training data."""
    obs = np.asarray(obs, dtype=np.float32)
    joint_positions = obs[:n_links]
    joint_velocities = obs[n_links:2 * n_links]
    tip_pos = obs[2 * n_links:2 * n_links + 3]
    tip_vel = obs[2 * n_links + 3:2 * n_links + 6]

    target_height = segment_length * float(n_links)
    height_error = target_height - float(tip_pos[2])
    lift = 0.18 * height_error + 0.06 * float(tip_vel[2])
    damping = 0.50 * joint_velocities
    stiffness = 0.75 * joint_positions
    joint_profile = np.linspace(0.12, 0.04, n_links, dtype=np.float32)
    phase = 0.02 * np.sin(0.05 * step_index + np.arange(n_links, dtype=np.float32) * 0.25)
    action = -stiffness - damping + lift + joint_profile + phase
    return action.astype(np.float32)


def collect_dataset(
    output_path: Path,
    n_links: int,
    masses: tuple[float, ...],
    payload_mass: float,
    episodes: int,
    steps_per_episode: int,
    render_mode: str | None,
):
    env = MultiLinkMassRobotEnv(
        n_links=n_links,
        mass_profile=MassProfile(link_masses=masses, payload_mass=payload_mass),
        render_mode=render_mode,
    )

    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_lengths = []
    episode_mass_profiles = []

    try:
        for episode_index in range(episodes):
            obs, info = env.reset(seed=episode_index)
            mass_profile = info["mass_profile"]
            episode_mass_profiles.append([*mass_profile.link_masses, mass_profile.payload_mass])

            ep_obs = []
            ep_actions = []
            ep_rewards = []
            ep_dones = []

            for step_index in range(steps_per_episode):
                action = scripted_policy(obs, n_links, step_index)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                ep_obs.append(np.asarray(obs, dtype=np.float32))
                ep_actions.append(np.asarray(action, dtype=np.float32))
                ep_rewards.append(float(reward))
                ep_dones.append(bool(terminated or truncated))

                obs = next_obs
                if terminated or truncated:
                    break

            all_obs.append(np.asarray(ep_obs, dtype=np.float32))
            all_actions.append(np.asarray(ep_actions, dtype=np.float32))
            all_rewards.append(np.asarray(ep_rewards, dtype=np.float32))
            all_dones.append(np.asarray(ep_dones, dtype=np.bool_))
            episode_lengths.append(len(ep_obs))
            log.info("Collected episode %d/%d with %d steps", episode_index + 1, episodes, len(ep_obs))
    finally:
        env.close()

    np.savez_compressed(
        output_path,
        observations=np.array(all_obs, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
        dones=np.array(all_dones, dtype=object),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        mass_profiles=np.asarray(episode_mass_profiles, dtype=np.float32),
        n_links=np.int32(n_links),
        payload_mass=np.float32(payload_mass),
    )
    log.info("Saved dataset to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate training data for the multi-link mass robot")
    parser.add_argument("--output", type=Path, default=Path("multilink_mass_robot_data.npz"))
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--masses", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--payload-mass", type=float, default=0.25)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="none")
    args = parser.parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode
    collect_dataset(
        output_path=args.output,
        n_links=args.links,
        masses=_parse_floats(args.masses),
        payload_mass=float(args.payload_mass),
        episodes=int(args.episodes),
        steps_per_episode=int(args.steps),
        render_mode=render_mode,
    )


if __name__ == "__main__":
    main()